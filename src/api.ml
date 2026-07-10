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
    let flat_args, in_tree = flatten_args args in
    let avals = List.map Core.get_aval flat_args in
    let key = aval_key avals ^ "#" ^ treedef_key in_tree in
    let cj, out_tree =
      match Hashtbl.find_opt cache key with
      | Some entry -> entry
      | None ->
          let flat_fun, out_tree_get = Api_util.flatten_fun f in_tree in
          let cj = Jaxpr.make_jaxpr avals (Linear_util.call_wrapped flat_fun) in
          let ot = out_tree_get () in
          Hashtbl.replace cache key (cj, ot);
          (cj, ot)
    in
    Tree.tree_unflatten out_tree ((compile cj) flat_args)

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
