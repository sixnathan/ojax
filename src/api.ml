open Types
module Ad = Interpreters.Ad
module Batching = Interpreters.Batching
module Pe = Interpreters.Partial_eval
module Tree = Tree_util

let aval_key (avals : aval list) : string =
  String.concat ";"
    (List.map
       (fun a ->
         Dtype.short_name a.dtype ^ "|"
         ^ String.concat "," (Array.to_list (Array.map string_of_int a.shape))
         ^ "|"
         ^ if a.weak_type then "w" else "s")
       avals)

let rec treedef_key (td : Tree.treedef) : string =
  match td with
  | Tree.Leaf_def -> "L"
  | Tree.Null_def -> "N"
  | Tree.List_def ds -> "[" ^ String.concat "," (List.map treedef_key ds) ^ "]"
  | Tree.Tuple_def ds -> "(" ^ String.concat "," (List.map treedef_key ds) ^ ")"
  | Tree.Custom_def (name, aux, ds) ->
      name ^ "<" ^ aux ^ ">("
      ^ String.concat "," (List.map treedef_key ds)
      ^ ")"
  | Tree.Dict_def kvs ->
      "{"
      ^ String.concat "," (List.map (fun (k, d) -> k ^ ":" ^ treedef_key d) kvs)
      ^ "}"

let ones_like_value (v : value) : value =
  let a = Core.get_aval v in
  let n = Array.fold_left ( * ) 1 a.shape in
  Concrete (Ndarray.of_floats a.dtype a.shape (Array.make n 1.0))

let compiled : (int, value list -> value list) Hashtbl.t = Hashtbl.create 16

let compile (cj : closed_jaxpr) : value list -> value list =
  match Hashtbl.find_opt compiled cj.jid with
  | Some executable -> executable
  | None ->
      let executable = Backend.executor cj in
      Hashtbl.replace compiled cj.jid executable;
      executable

let jit_flat (f : value list -> value list) : value list -> value list =
  let cache : (string, closed_jaxpr) Hashtbl.t = Hashtbl.create 8 in
  fun args ->
    if Config.value Config.disable_jit then f args
    else
      let avals = List.map Core.get_aval args in
      let key = aval_key avals in
      let cj =
        match Hashtbl.find_opt cache key with
        | Some cj -> cj
        | None ->
            let cj = Jaxpr.make_jaxpr avals f in
            Hashtbl.replace cache key cj;
            cj
      in
      (compile cj) args

type out_slot = Known of value | Unknown

let stage (f : value list -> value list) (args : value list) :
    jaxpr * value list * out_slot list =
  let pvals_in =
    List.map (fun a -> Pe.partial_val_unknown (Core.get_aval a)) args
  in
  let jaxpr, consts, pvals_out = Pe.partial_eval_flat f pvals_in in
  let plan =
    List.map
      (fun pv -> match pv.pv_const with Some c -> Known c | None -> Unknown)
      pvals_out
  in
  (jaxpr, consts, plan)

let weave (plan : out_slot list) (unknowns : value list) : value list =
  let rem = ref unknowns in
  let result =
    List.map
      (function
        | Known c -> c
        | Unknown -> (
            match !rem with
            | u :: r ->
                rem := r;
                u
            | [] -> failwith "api.call: output plan underflow"))
      plan
  in
  if !rem <> [] then failwith "api.call: output plan overflow";
  result

let call (f : value list -> value list) (args : value list) : value list =
  let jaxpr, consts, plan = stage f args in
  let unknowns = Jaxpr.eval_jaxpr jaxpr (consts @ args) in
  weave plan unknowns

let flatten_args (args : value Tree.t list) : value list * Tree.treedef =
  Tree.tree_flatten (Tree.Tuple args)

let make_jaxpr (f : value Tree.t list -> value Tree.t)
    (args : value Tree.t list) : closed_jaxpr =
  let flat_args, in_tree = flatten_args args in
  let avals = List.map Core.get_aval flat_args in
  let flat_fun, _ = Api_util.flatten_fun f in_tree in
  Jaxpr.make_jaxpr avals (Linear_util.call_wrapped flat_fun)

let jit (f : value Tree.t list -> value Tree.t) :
    value Tree.t list -> value Tree.t =
  let cache : (string, closed_jaxpr * Tree.treedef) Hashtbl.t =
    Hashtbl.create 8
  in
  fun args ->
    if Config.value Config.disable_jit then f args
    else
      let flat_args, in_tree = flatten_args args in
      let avals = List.map Core.get_aval flat_args in
      let key = aval_key avals ^ "#" ^ treedef_key in_tree in
      let cj, out_tree =
        match Hashtbl.find_opt cache key with
        | Some entry -> entry
        | None ->
            let flat_fun, out_tree_get = Api_util.flatten_fun f in_tree in
            let cj =
              Jaxpr.make_jaxpr avals (Linear_util.call_wrapped flat_fun)
            in
            let ot = out_tree_get () in
            Hashtbl.replace cache key (cj, ot);
            (cj, ot)
      in
      Tree.tree_unflatten out_tree ((compile cj) flat_args)

let device_put (x : value Tree.t) : value Tree.t =
  Tree.tree_map Backend.of_host_value x

let to_host (x : value Tree.t) : value Tree.t =
  Tree.tree_map Backend.to_host_value x

let jvp (f : value Tree.t list -> value Tree.t) (primals : value Tree.t list)
    (tangents : value Tree.t list) : value Tree.t * value Tree.t =
  let pf, in_tree = flatten_args primals in
  let tf, _ = flatten_args tangents in
  let flat_fun, out_tree = Api_util.flatten_fun f in_tree in
  let po, to_ = Ad.jvp (Linear_util.call_wrapped flat_fun) pf tf in
  let ot = out_tree () in
  (Tree.tree_unflatten ot po, Tree.tree_unflatten ot to_)

let linearize (f : value Tree.t list -> value Tree.t)
    (primals : value Tree.t list) :
    value Tree.t * (value Tree.t list -> value Tree.t) =
  let pf, in_tree = flatten_args primals in
  let flat_fun, out_tree = Api_util.flatten_fun f in_tree in
  let outs, f_lin = Ad.linearize (Linear_util.call_wrapped flat_fun) pf in
  let ot = out_tree () in
  let lin (tangents : value Tree.t list) : value Tree.t =
    let tf, _ = flatten_args tangents in
    Tree.tree_unflatten ot (f_lin tf)
  in
  (Tree.tree_unflatten ot outs, lin)

let vjp (f : value Tree.t list -> value Tree.t) (primals : value Tree.t list) :
    value Tree.t * (value Tree.t -> value Tree.t list) =
  let pf, in_tree = flatten_args primals in
  let flat_fun, out_tree = Api_util.flatten_fun f in_tree in
  let outs, f_vjp = Ad.vjp (Linear_util.call_wrapped flat_fun) pf in
  let ot = out_tree () in
  let out = Tree.tree_unflatten ot outs in
  let vjp_fn (cotangent : value Tree.t) : value Tree.t list =
    let gs = f_vjp (Tree.tree_leaves cotangent) in
    match Tree.tree_unflatten in_tree gs with
    | Tree.Tuple xs | Tree.List xs -> xs
    | other -> [ other ]
  in
  (out, vjp_fn)

let value_and_grad (f : value Tree.t list -> value Tree.t)
    (args : value Tree.t list) : value Tree.t * value Tree.t =
  match args with
  | [] -> failwith "value_and_grad: no arguments"
  | x0 :: rest ->
      let flat0, tree0 = Tree.tree_flatten x0 in
      let flat_f (flat0' : value list) : value list =
        let x0' = Tree.tree_unflatten tree0 flat0' in
        match f (x0' :: rest) with
        | Tree.Leaf v -> [ v ]
        | _ -> failwith "value_and_grad: function must return a scalar leaf"
      in
      let outs, f_vjp = Ad.vjp flat_f flat0 in
      let y =
        match outs with
        | [ y ] -> y
        | _ -> failwith "value_and_grad: expected a single output"
      in
      if (Core.get_aval y).shape <> [||] then
        invalid_arg
          (Printf.sprintf
             "Gradient only defined for scalar-output functions. Output had \
              shape: %s."
             (aval_key [ Core.get_aval y ]));
      let g = f_vjp [ ones_like_value y ] in
      (Tree.Leaf y, Tree.tree_unflatten tree0 g)

let grad (f : value Tree.t list -> value Tree.t) (args : value Tree.t list) :
    value Tree.t =
  snd (value_and_grad f args)

let vmap (f : value Tree.t list -> value Tree.t) (in_axes : int option list)
    (args : value Tree.t list) : value Tree.t =
  let flat_args, in_tree = flatten_args args in
  let flat_axes =
    List.concat
      (List.map2
         (fun ax arg -> List.map (fun _ -> ax) (Tree.tree_leaves arg))
         in_axes args)
  in
  let flat_fun, out_tree = Api_util.flatten_fun f in_tree in
  let outs =
    Batching.vmap (Linear_util.call_wrapped flat_fun) flat_axes flat_args
  in
  Tree.tree_unflatten (out_tree ()) outs

let leaf = function
  | Tree.Leaf v -> v
  | _ -> invalid_arg "expected an array leaf"

let split_args argnums args =
  let selected =
    List.map
      (fun i ->
        match List.nth_opt args i with
        | Some a -> a
        | None -> invalid_arg "argnums out of range")
      argnums
  in
  let rebuild replacements =
    List.mapi
      (fun i a ->
        match List.find_index (fun j -> j = i) argnums with
        | Some k -> List.nth replacements k
        | None -> a)
      args
  in
  (selected, rebuild)

let value_and_grad_argnums ?(has_aux = false) ~argnums f args =
  let selected, rebuild = split_args argnums args in
  let flat, tree = Tree.tree_flatten (Tree.Tuple selected) in
  let aux = ref Tree.Null in
  let flat_f leaves =
    let parts =
      match Tree.tree_unflatten tree leaves with
      | Tree.Tuple xs -> xs
      | _ -> failwith "value_and_grad: argument split"
    in
    let result = f (rebuild parts) in
    if has_aux then
      match result with
      | Tree.Tuple [ y; a ] ->
          aux := a;
          [ leaf y ]
      | _ -> invalid_arg "has_aux requires a (value, aux) tuple"
    else [ leaf result ]
  in
  let outs, f_vjp = Ad.vjp flat_f flat in
  let y = List.hd outs in
  if (Core.get_aval y).shape <> [||] then
    invalid_arg
      (Printf.sprintf
         "Gradient only defined for scalar-output functions. Output had shape: \
          %s."
         (aval_key [ Core.get_aval y ]));
  let grads =
    match Tree.tree_unflatten tree (f_vjp [ ones_like_value y ]) with
    | Tree.Tuple xs -> xs
    | _ -> failwith "value_and_grad: gradient split"
  in
  (y, grads, !aux)

let basis (a : aval) =
  let n = Array.fold_left ( * ) 1 a.shape in
  Concrete
    (Ndarray.of_floats a.dtype [| n; n |]
       (Array.init (n * n) (fun i -> if i mod (n + 1) = 0 then 1.0 else 0.0)))

let size_of (a : aval) = Array.fold_left ( * ) 1 a.shape
let reshape_to shape v = Core.bind1 (Reshape shape) [ v ]

let move_first_to_last_and_reshape out_shape in_shape v =
  let ndim = Array.length (Core.get_aval v).shape in
  let perm = Array.init ndim (fun i -> if i = ndim - 1 then 0 else i + 1) in
  let moved = Core.bind1 (Transpose perm) [ v ] in
  reshape_to (Array.append out_shape in_shape) moved

let zeros_for (a : aval) = Ad_util.zeros_like_aval a

let nested_jacobian out_tree in_tree (entries : value list list) =
  let per_output =
    List.map (fun column -> Tree.tree_unflatten in_tree column) entries
  in
  Tree.tree_unflatten out_tree per_output |> fun tree ->
  match tree with t -> t

let flatten_nested (tree : value Tree.t Tree.t) : value Tree.t =
  let rec go = function
    | Tree.Leaf inner -> inner
    | Tree.Null -> Tree.Null
    | Tree.List xs -> Tree.List (List.map go xs)
    | Tree.Tuple xs -> Tree.Tuple (List.map go xs)
    | Tree.Dict kvs -> Tree.Dict (List.map (fun (k, x) -> (k, go x)) kvs)
    | Tree.Custom (n, a, xs) -> Tree.Custom (n, a, List.map go xs)
  in
  go tree

let jacfwd ?(argnums = 0) f args =
  let selected, rebuild = split_args [ argnums ] args in
  let x0 = List.hd selected in
  let leaves, in_def = Tree.tree_flatten x0 in
  let avals = List.map Core.get_aval leaves in
  let out_def = ref Tree.Null_def in
  let call flat =
    let out = f (rebuild [ Tree.tree_unflatten in_def flat ]) in
    let o, d = Tree.tree_flatten out in
    out_def := d;
    o
  in
  let columns =
    List.mapi
      (fun k (a : aval) ->
        let push b =
          let tangents =
            List.mapi
              (fun j (aj : aval) ->
                if j = k then reshape_to a.shape b else zeros_for aj)
              avals
          in
          snd (Ad.jvp call leaves tangents)
        in
        Batching.vmap (fun bs -> push (List.hd bs)) [ Some 0 ] [ basis a ])
      avals
  in
  let n_out = List.length (List.hd columns) in
  let entries =
    List.init n_out (fun j ->
        List.map2
          (fun column (a : aval) ->
            let raw = List.nth column j in
            let full = (Core.get_aval raw).shape in
            let out_shape = Array.sub full 1 (Array.length full - 1) in
            move_first_to_last_and_reshape out_shape a.shape raw)
          columns avals)
  in
  flatten_nested (nested_jacobian !out_def in_def entries)

let jacrev ?(argnums = 0) f args =
  let selected, rebuild = split_args [ argnums ] args in
  let x0 = List.hd selected in
  let leaves, in_def = Tree.tree_flatten x0 in
  let avals = List.map Core.get_aval leaves in
  let out_def = ref Tree.Null_def in
  let call flat =
    let out = f (rebuild [ Tree.tree_unflatten in_def flat ]) in
    let o, d = Tree.tree_flatten out in
    out_def := d;
    o
  in
  let outs, f_vjp = Ad.vjp call leaves in
  let out_avals = List.map Core.get_aval outs in
  let entries =
    List.mapi
      (fun j (oa : aval) ->
        let pull bs =
          let ct =
            List.mapi
              (fun i (ai : aval) ->
                if i = j then reshape_to oa.shape (List.hd bs) else zeros_for ai)
              out_avals
          in
          f_vjp ct
        in
        let grads = Batching.vmap pull [ Some 0 ] [ basis oa ] in
        List.map2
          (fun g (a : aval) -> reshape_to (Array.append oa.shape a.shape) g)
          grads avals)
      out_avals
  in
  flatten_nested (nested_jacobian !out_def in_def entries)

let jacobian = jacrev

let hessian ?(argnums = 0) f args =
  jacfwd ~argnums (fun a -> jacrev ~argnums f a) args

let eval_shape f args =
  let flat, in_tree = flatten_args args in
  let flat_fun, out_tree = Api_util.flatten_fun f in_tree in
  let closed, _ =
    Jaxpr.make_jaxpr_hoisted
      (List.map Core.get_aval flat)
      (Linear_util.call_wrapped flat_fun)
  in
  let out_avals =
    List.map
      (function
        | A_var v -> v.vaval
        | A_lit nd ->
            {
              shape = Ndarray.shape nd;
              dtype = Ndarray.dtype nd;
              weak_type = false;
            }
        | DropVar a -> a)
      closed.jaxpr.outs
  in
  Tree.tree_unflatten (out_tree ()) out_avals

let linear_transpose f primals =
  let leaves, in_def = Tree.tree_flatten (Tree.Tuple primals) in
  let zeros = List.map (fun v -> zeros_for (Core.get_aval v)) leaves in
  let call flat =
    match Tree.tree_unflatten in_def flat with
    | Tree.Tuple xs -> Tree.tree_leaves (f xs)
    | _ -> failwith "linear_transpose: argument split"
  in
  let _, f_vjp = Ad.vjp call zeros in
  fun cotangent ->
    match Tree.tree_unflatten in_def (f_vjp (Tree.tree_leaves cotangent)) with
    | Tree.Tuple xs -> xs
    | _ -> failwith "linear_transpose: result split"

let checkpoint f = f
let remat = checkpoint
let named_call f = f
let named_scope _name f = f ()
let device_get = to_host
let block_until_ready x = x
let devices () = [ "cpu:0" ]
let local_devices = devices
let device_count () = 1
let local_device_count = device_count
let process_count () = 1
let process_index () = 0
let process_indices () = [ 0 ]
let host_count = process_count
let host_id = process_index
let host_ids = process_indices
let default_backend () = "cpu"
let clear_caches () = Hashtbl.reset compiled
let disable_jit thunk = Config.with_value Config.disable_jit true thunk
let ensure_compile_time_eval thunk = thunk ()

let pmap f args =
  let flat, _ = Tree.tree_flatten (Tree.Tuple args) in
  (match flat with
  | v :: _ ->
      let shape = (Core.get_aval v).shape in
      if Array.length shape = 0 || shape.(0) > device_count () then
        invalid_arg
          "pmap got more replicas than the number of available devices"
  | [] -> invalid_arg "pmap requires at least one argument");
  vmap f (List.map (fun _ -> Some 0) args) args

let pure_callback host_fun ~result_avals args =
  Custom_derivatives.pure_callback host_fun ~result_avals args
