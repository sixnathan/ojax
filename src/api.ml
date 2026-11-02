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
      if Config.value Config.enable_checks then Jaxpr.typecheck_jaxpr cj.jaxpr;
      let executable = Backend.executor cj in
      Hashtbl.replace compiled cj.jid executable;
      executable

let trace_for_jit avals f =
  match Jaxpr.make_jaxpr_stateful avals f with
  | closed, [], cells -> (closed, cells)
  | _ -> invalid_arg "make_jaxpr: function closed over a tracer"

let staged_under_transform cells args =
  cells = []
  && (not (Config.value Config.no_execution))
  &&
  match (Core.find_top_trace args).kind with
  | KJVP | KBatch | KPE -> true
  | KEval | KJaxpr -> false

let keep_out_weak (cj : closed_jaxpr) results =
  List.mapi
    (fun i result ->
      match (result, List.nth_opt cj.jaxpr.outs i) with
      | Concrete nd, Some (A_var { vaval = { weak_type = true; _ }; _ }) ->
          Concrete (Ndarray.as_weak nd)
      | _ -> result)
    results

let run_with_refs cj cells args =
  if staged_under_transform cells args then Core.bind (Xla_call cj) args
  else
    let state = List.map Ref_state.read cells in
    let outs = (compile cj) (args @ state) in
    let n_out = List.length outs - List.length cells in
    match Util.split_list outs [ n_out ] with
    | [ results; finals ] ->
        List.iter2 Ref_state.write cells finals;
        keep_out_weak cj results
    | _ -> failwith "jit: output arity"

let context_key () =
  String.concat ","
    (List.map (fun (k, v) -> k ^ "=" ^ v) (Config.trace_context ()))

type seen_key = {
  seen_context : (string * string) list;
  seen_avals : aval list;
  seen_tree : string;
}

let current_seen avals tree =
  {
    seen_context = Config.trace_context ();
    seen_avals = avals;
    seen_tree = tree;
  }

let diff_context now before =
  let changed =
    List.filter_map
      (fun (name, value) ->
        match List.assoc_opt name before with
        | Some old when old <> value ->
            Some
              (Printf.sprintf "  * %s differs: now %s and before %s" name value
                 old)
        | _ -> None)
      now
  in
  match changed with
  | [] -> None
  | lines ->
      Some
        ("Tracing context doesn't match, e.g. due to config or context manager."
       :: lines)

let diff_types now before =
  let lines =
    List.concat
      (List.mapi
         (fun i (a : aval) ->
           match List.nth_opt before i with
           | Some (b : aval) when b <> a ->
               [
                 Printf.sprintf "  * at args[%d], now %s and before %s" i
                   (Pretty_printer.aval_short a)
                   (Pretty_printer.aval_short b);
               ]
           | _ -> [])
         now)
  in
  match lines with [] -> None | _ -> Some ("different input types:" :: lines)

let explain_miss seen (current : seen_key) =
  if Config.value Config.explain_cache_misses then
    let header = "TRACING CACHE MISS at <ocaml>:" in
    let body =
      match seen with
      | [] -> [ "  never seen function:"; "    <fun>" ]
      | last :: _ ->
          let diff =
            match diff_context current.seen_context last.seen_context with
            | Some lines -> lines
            | None ->
                if current.seen_tree <> last.seen_tree then
                  [ "different input pytree:" ]
                else
                  Option.value
                    (diff_types current.seen_avals last.seen_avals)
                    ~default:[ "cache miss explanation unavailable" ]
          in
          "  for <fun>"
          :: "  all previously seen cache keys differ. For the closest \
              previous key:"
          :: List.map (fun line -> "  " ^ line) diff
    in
    Config.log (String.concat "\n" (header :: body))

let timed_trace thunk =
  let start = Sys.time () in
  let result = thunk () in
  Monitoring.record_event_duration_secs
    ~kwargs:[ ("fun_name", Monitoring.Text "<fun>") ]
    "/jax/core/compile/jaxpr_trace_duration"
    (Sys.time () -. start);
  result

let bind_named_call ~name (cj : closed_jaxpr) args =
  Pretty_printer.register_call_name cj.jid name;
  Core.bind (Xla_call cj) args

let jit_flat ?(name = "<lambda>") (f : value list -> value list) :
    value list -> value list =
  let cache :
      ( string,
        closed_jaxpr * Ref_state.cell list * Key_reuse.signature )
      Hashtbl.t =
    Hashtbl.create 8
  in
  let seen = ref [] in
  fun args ->
    if Config.value Config.disable_jit then f args
    else
      let avals = List.map Core.get_aval args in
      let kinds = Key_reuse.arg_kinds args in
      let key =
        aval_key avals ^ "#" ^ Key_reuse.kinds_key kinds ^ "#" ^ context_key ()
      in
      let cj, cells, signature =
        match Hashtbl.find_opt cache key with
        | Some entry -> entry
        | None ->
            if Config.value Config.no_tracing then
              failwith "jit tracing is disabled by no_tracing";
            let current = current_seen avals "" in
            explain_miss !seen current;
            seen := current :: !seen;
            let signature = Key_reuse.new_signature () in
            let cj, cells =
              timed_trace (fun () ->
                  trace_for_jit avals (Key_reuse.instrument signature kinds f))
            in
            let entry = (cj, cells, signature) in
            Hashtbl.replace cache key entry;
            entry
      in
      if cells = [] && Core.top_is_staging args then
        Key_reuse.call_with_signature signature args (fun () ->
            bind_named_call ~name cj args)
      else
        Key_reuse.call_with_signature signature args (fun () ->
            run_with_refs cj cells args)

type out_slot = Known of value | Unknown

let stage (f : value list -> value list) (args : value list) :
    jaxpr * value list * out_slot list =
  let pvals_in =
    List.map (fun a -> Pe.partial_val_unknown (Core.get_aval a)) args
  in
  let jaxpr, consts, pvals_out = Pe.partial_eval_flat f pvals_in in
  if Config.value Config.enable_checks then Jaxpr.typecheck_jaxpr jaxpr;
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

let split_static static_argnums args =
  let is_static i = List.mem i static_argnums in
  let dynamic = List.filteri (fun i _ -> not (is_static i)) args in
  let merge dyn =
    let remaining = ref dyn in
    List.mapi
      (fun i a ->
        if is_static i then a
        else
          match !remaining with
          | x :: rest ->
              remaining := rest;
              x
          | [] -> invalid_arg "static_argnums: arity")
      args
  in
  (dynamic, merge)

let make_jaxpr_shape ?(axis_env = []) ?(static_argnums = [])
    (f : value Tree.t list -> value Tree.t) (args : value Tree.t list) =
  let dynamic, merge = split_static static_argnums args in
  let flat_args, in_tree = flatten_args dynamic in
  let avals = List.map Core.get_aval flat_args in
  let flat_fun, out_tree =
    Api_util.flatten_fun (fun xs -> f (merge xs)) in_tree
  in
  let closed =
    Core.with_mesh_axes (axis_env @ !Core.mesh_axes) (fun () ->
        Jaxpr.make_jaxpr avals (Linear_util.call_wrapped flat_fun))
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
  (closed, Tree.tree_unflatten (out_tree ()) out_avals)

let make_jaxpr ?axis_env ?static_argnums f args =
  fst (make_jaxpr_shape ?axis_env ?static_argnums f args)

let jit_plain (f : value Tree.t list -> value Tree.t) :
    value Tree.t list -> value Tree.t =
  let cache :
      ( string,
        closed_jaxpr * Ref_state.cell list * Tree.treedef * Key_reuse.signature
      )
      Hashtbl.t =
    Hashtbl.create 8
  in
  let seen = ref [] in
  fun args ->
    if Config.value Config.disable_jit then f args
    else
      let flat_args, in_tree = flatten_args args in
      let avals = List.map Core.get_aval flat_args in
      let kinds = Key_reuse.arg_kinds flat_args in
      let tree_text = treedef_key in_tree in
      let key =
        aval_key avals ^ "#" ^ tree_text ^ "#" ^ Key_reuse.kinds_key kinds ^ "#"
        ^ context_key ()
      in
      let cj, cells, out_tree, signature =
        match Hashtbl.find_opt cache key with
        | Some entry -> entry
        | None ->
            let flat_fun, out_tree_get = Api_util.flatten_fun f in_tree in
            if Config.value Config.no_tracing then
              failwith "jit tracing is disabled by no_tracing";
            let current = current_seen avals tree_text in
            explain_miss !seen current;
            seen := current :: !seen;
            let signature = Key_reuse.new_signature () in
            let cj, cells =
              timed_trace (fun () ->
                  trace_for_jit avals
                    (Key_reuse.instrument signature kinds
                       (Linear_util.call_wrapped flat_fun)))
            in
            let entry = (cj, cells, out_tree_get (), signature) in
            Hashtbl.replace cache key entry;
            entry
      in
      Tree.tree_unflatten out_tree
        (Key_reuse.call_with_signature signature flat_args (fun () ->
             run_with_refs cj cells flat_args))

let static_key tree =
  String.concat ","
    (List.map
       (function
         | Concrete nd when Ndarray.shape nd = [||] ->
             Printf.sprintf "%s:%h"
               (Ref_state.dtype_name (Ndarray.dtype nd))
               (Ndarray.get_f nd [||])
         | Device b when Ndarray.shape (Pjrt.Buffer.to_host b) = [||] ->
             let nd = Pjrt.Buffer.to_host b in
             Printf.sprintf "%s:%h"
               (Ref_state.dtype_name (Ndarray.dtype nd))
               (Ndarray.get_f nd [||])
         | _ ->
             invalid_arg
               "Non-hashable static arguments are used in a jit function")
       (Tree.tree_leaves tree))

let jit ?(static_argnums = []) f =
  if static_argnums = [] then jit_plain f
  else
    let per_key = Hashtbl.create 8 in
    fun args ->
      let dynamic, merge = split_static static_argnums args in
      let key =
        String.concat ";"
          (List.mapi
             (fun i a -> if List.mem i static_argnums then static_key a else "")
             args)
      in
      let compiled =
        match Hashtbl.find_opt per_key key with
        | Some c -> c
        | None ->
            let c = jit_plain (fun xs -> f (merge xs)) in
            Hashtbl.replace per_key key c;
            c
      in
      compiled dynamic

let keep_key_kind source moved =
  if Key_reuse.is_typed source then Key_reuse.mark_typed moved else moved

let device_put (x : value Tree.t) : value Tree.t =
  Tree.tree_map (fun v -> keep_key_kind v (Backend.of_host_value v)) x

let to_host (x : value Tree.t) : value Tree.t =
  Tree.tree_map (fun v -> keep_key_kind v (Backend.to_host_value v)) x

let float0 = Dtypes.float0
let scalar_type_of v = Dtypes.scalar_type_of (Core.get_aval v).dtype
let is_integral_dtype dt = dt = Dtype.Bool || Dtypes.issubdtype dt `Integer

let float0_like v =
  let shape = (Core.get_aval v).shape in
  let n = Array.fold_left ( * ) 1 shape in
  Concrete (Ndarray.of_floats Dtypes.float0 shape (Array.make n 0.0))

let tangent_of_float0 primal tangent =
  let pd = (Core.get_aval primal).dtype in
  if (Core.get_aval tangent).dtype <> Dtypes.float0 then tangent
  else if is_integral_dtype pd then Ad_util.zeros_like_value primal
  else
    invalid_arg
      (Printf.sprintf
         "primal and tangent arguments to jax.jvp do not match; dtypes must be \
          equal, or in case of int/bool primal dtype the tangent dtype must be \
          float0.Got primal dtype %s and so expected tangent dtype %s, but got \
          tangent dtype float0 instead."
         (Dtype.numpy_name pd) (Dtype.numpy_name pd))

let float0_for_integral primal tangent =
  if is_integral_dtype (Core.get_aval primal).dtype then float0_like primal
  else tangent

let jvp (f : value Tree.t list -> value Tree.t) (primals : value Tree.t list)
    (tangents : value Tree.t list) : value Tree.t * value Tree.t =
  let pf, in_tree = flatten_args primals in
  let tf, _ = flatten_args tangents in
  let tf = List.map2 tangent_of_float0 pf tf in
  let flat_fun, out_tree = Api_util.flatten_fun f in_tree in
  let po, to_ = Ad.jvp_instantiated (Linear_util.call_wrapped flat_fun) pf tf in
  let ot = out_tree () in
  let to_ = List.map2 float0_for_integral po to_ in
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
    let cts = List.map2 tangent_of_float0 outs (Tree.tree_leaves cotangent) in
    let gs = List.map2 float0_for_integral pf (f_vjp cts) in
    match Tree.tree_unflatten in_tree gs with
    | Tree.Tuple xs | Tree.List xs -> xs
    | other -> [ other ]
  in
  (out, vjp_fn)

let check_grad_input allow_int v =
  let dt = (Core.get_aval v).dtype in
  if is_integral_dtype dt then
    begin if not allow_int then
      invalid_arg
        (Printf.sprintf
           "grad requires real- or complex-valued inputs (input dtype that is \
            a sub-dtype of np.inexact), but got %s. If you want to use \
            Boolean- or integer-valued inputs, use vjp or set allow_int to \
            True."
           (Dtype.numpy_name dt))
    end
  else if not (Dtypes.issubdtype dt `Inexact) then
    invalid_arg
      (Printf.sprintf
         "grad requires numerical-valued inputs (input dtype that is a \
          sub-dtype of np.bool_ or np.number), but got %s."
         (Dtype.numpy_name dt))

let value_and_grad ?(allow_int = false) (f : value Tree.t list -> value Tree.t)
    (args : value Tree.t list) : value Tree.t * value Tree.t =
  match args with
  | [] -> failwith "value_and_grad: no arguments"
  | x0 :: rest ->
      let flat0, tree0 = Tree.tree_flatten x0 in
      List.iter (check_grad_input allow_int) flat0;
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
      let g =
        List.map2 float0_for_integral flat0 (f_vjp [ ones_like_value y ])
      in
      (Tree.Leaf y, Tree.tree_unflatten tree0 g)

let move_argnum k args =
  if k = 0 then (args, fun xs -> xs)
  else (
    if k < 0 || k >= List.length args then invalid_arg "argnums out of range";
    let chosen = List.nth args k in
    let others = List.filteri (fun i _ -> i <> k) args in
    let restore = function
      | x :: rest ->
          let before = List.filteri (fun i _ -> i < k) rest in
          let after = List.filteri (fun i _ -> i >= k) rest in
          before @ (x :: after)
      | [] -> []
    in
    (chosen :: others, restore))

let value_and_grad ?allow_int ?(argnums = 0) f args =
  let moved, restore = move_argnum argnums args in
  value_and_grad ?allow_int (fun xs -> f (restore xs)) moved

let grad ?allow_int ?argnums (f : value Tree.t list -> value Tree.t)
    (args : value Tree.t list) : value Tree.t =
  snd (value_and_grad ?allow_int ?argnums f args)

let vmap ?axis_name ?out_axes ?axis_size ?(spmd_axis_name : string list option)
    ?(sum_match : bool option) (f : value Tree.t list -> value Tree.t)
    (in_axes : int option list) (args : value Tree.t list) : value Tree.t =
  let flat_args, in_tree = flatten_args args in
  let flat_axes =
    List.concat
      (List.map2
         (fun ax arg -> List.map (fun _ -> ax) (Tree.tree_leaves arg))
         in_axes args)
  in
  let flat_fun, out_tree = Api_util.flatten_fun f in_tree in
  let outs =
    Batching.vmap_flat ?axis_name ?out_axis:out_axes ?axis_size
      (Linear_util.call_wrapped flat_fun)
      flat_axes flat_args
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
          aux := Tree.tree_map Ad.aux_primal a;
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

type fwd_bwd_residuals = {
  residual_values : value list;
  backward : value list -> value list;
  selected_def : Tree.treedef;
  out_def : Tree.treedef;
}

let residual_values r = r.residual_values

let split_prefix l n =
  match Util.split_list l [ n ] with
  | [ a; b ] -> (a, b)
  | _ -> invalid_arg "fwd_and_bwd: arity"

type fwd_stage = {
  stage_out_def : Tree.treedef;
  stage_aux_def : Tree.treedef;
  stage_selected_def : Tree.treedef;
  n_out : int;
  n_residuals : int;
  stage_backward : value list -> value list -> value list;
}

let fwd_and_bwd ?(has_aux = false) ?(jitted = true) ~argnums f =
  let stage_call args stage =
    let selected, rebuild = split_args argnums args in
    let leaves, selected_def = Tree.tree_flatten (Tree.Tuple selected) in
    let out_def = ref Tree.Null_def and aux_def = ref Tree.Null_def in
    let aux_leaves = ref [] in
    let body inner =
      let parts =
        match Tree.tree_unflatten selected_def inner with
        | Tree.Tuple xs -> xs
        | _ -> failwith "fwd_and_bwd: argument split"
      in
      let result = f (rebuild parts) in
      let primal =
        if has_aux then
          match result with
          | Tree.Tuple [ y; a ] ->
              let a_leaves, a_def = Tree.tree_flatten a in
              aux_leaves := List.map Ad.aux_primal a_leaves;
              aux_def := a_def;
              y
          | _ -> invalid_arg "has_aux requires a (value, aux) tuple"
        else result
      in
      let out, def = Tree.tree_flatten primal in
      out_def := def;
      out
    in
    let outs, residuals, backward = Ad.vjp_staged body leaves in
    stage :=
      Some
        {
          stage_out_def = !out_def;
          stage_aux_def = !aux_def;
          stage_selected_def = selected_def;
          n_out = List.length outs;
          n_residuals = List.length residuals;
          stage_backward = backward;
        };
    outs @ residuals @ !aux_leaves
  in
  let finish entry outs back =
    let results, rest = split_prefix outs entry.n_out in
    let residuals, aux = split_prefix rest entry.n_residuals in
    ( Tree.tree_unflatten entry.stage_out_def results,
      {
        residual_values = residuals;
        backward = back residuals;
        selected_def = entry.stage_selected_def;
        out_def = entry.stage_out_def;
      },
      Tree.tree_unflatten entry.stage_aux_def aux )
  in
  let direct args =
    let stage = ref None in
    let outs = stage_call args stage in
    match !stage with
    | Some entry -> finish entry outs entry.stage_backward
    | None -> failwith "fwd_and_bwd: stage"
  in
  let cache = Hashtbl.create 4 in
  let via_jit args =
    let flat, in_tree = flatten_args args in
    let avals = List.map Core.get_aval flat in
    let key = aval_key avals ^ "#" ^ treedef_key in_tree in
    let cj, cells, entry, back =
      match Hashtbl.find_opt cache key with
      | Some cached -> cached
      | None ->
          if Config.value Config.no_tracing then
            failwith "jit tracing is disabled by no_tracing";
          let stage = ref None in
          let cj, cells =
            trace_for_jit avals (fun inner ->
                match Tree.tree_unflatten in_tree inner with
                | Tree.Tuple xs -> stage_call xs stage
                | _ -> failwith "fwd_and_bwd: argument split")
          in
          let entry =
            match !stage with
            | Some entry -> entry
            | None -> failwith "fwd_and_bwd: stage"
          in
          let compiled_back =
            jit_flat (fun all ->
                let residuals, cts = split_prefix all entry.n_residuals in
                entry.stage_backward residuals cts)
          in
          let back residuals cts = compiled_back (residuals @ cts) in
          let cached = (cj, cells, entry, back) in
          Hashtbl.replace cache key cached;
          cached
    in
    finish entry (run_with_refs cj cells flat) back
  in
  let fwd args =
    if jitted && not (Config.value Config.disable_jit) then via_jit args
    else direct args
  in
  let bwd residuals cotangent =
    let cts, def = Tree.tree_flatten cotangent in
    if def <> residuals.out_def then
      invalid_arg
        "unexpected tree structure. The argument to a VJP function must match \
         the pytree structure of the differentiated function";
    match
      Tree.tree_unflatten residuals.selected_def (residuals.backward cts)
    with
    | Tree.Tuple xs -> xs
    | _ -> failwith "fwd_and_bwd: gradient split"
  in
  (fwd, bwd)

let size_of (a : aval) = Array.fold_left ( * ) 1 a.shape
let reshape_to shape v = Core.bind1 (Reshape shape) [ v ]
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

let split_along arr sizes axis =
  Core.bind (Split { sizes = Np_ints.mark (Array.of_list sizes); axis }) [ arr ]

let reshape_if_needed v shape =
  if (Core.get_aval v).shape = shape then v else reshape_to shape v

let std_basis (leaves : value list) =
  let avals = List.map Core.get_aval leaves in
  let total = List.fold_left (fun acc a -> acc + size_of a) 0 avals in
  let dt =
    List.fold_left
      (fun acc (a : aval) -> Dtypes.promote_types acc a.dtype)
      (List.hd avals).dtype (List.tl avals)
  in
  let flat_basis =
    let idt = Dtypes.default_int_dtype () in
    let iota dimension =
      Core.bind1
        (Iota { dtype = idt; shape = [| total; total |]; dimension })
        []
    in
    let rows = iota 0 in
    let cols = iota 1 in
    let shifted = Core.bind1 Add [ rows; Core.py_scalar idt 0.0 ] in
    Core.bind1 (Convert_element_type dt) [ Core.bind1 Eq [ shifted; cols ] ]
  in
  let parts = split_along flat_basis (List.map size_of avals) 1 in
  List.map2
    (fun part (a : aval) ->
      let shaped = reshape_if_needed part (Array.append [| total |] a.shape) in
      Core.bind1 (Convert_element_type a.dtype) [ shaped ])
    parts avals

let downcast v (example : aval) =
  let a = Core.get_aval v in
  let v =
    match (a.dtype, example.dtype) with
    | (Dtype.Complex64 | Dtype.Complex128), (Dtype.Complex64 | Dtype.Complex128)
      ->
        v
    | (Dtype.Complex64 | Dtype.Complex128), _ -> Core.bind1 Real [ v ]
    | _ -> v
  in
  if (Core.get_aval v).dtype = example.dtype && not a.weak_type then v
  else Core.bind1 (Convert_element_type example.dtype) [ v ]

let is_complex_dtype dt = dt = Dtype.Complex64 || dt = Dtype.Complex128

let check_input_jacfwd holomorphic v =
  let dt = (Core.get_aval v).dtype in
  let name = Dtype.numpy_name dt in
  if holomorphic then
    begin if not (is_complex_dtype dt) then
      invalid_arg
        ("jacfwd with holomorphic=True requires inputs with complex dtype, but \
          got " ^ name ^ ".")
    end
  else if not (Dtypes.issubdtype dt `Floating) then
    invalid_arg
      ("jacfwd requires real-valued inputs (input dtype that is a sub-dtype of \
        np.floating), but got " ^ name
     ^ ". For holomorphic differentiation, pass holomorphic=True. For \
        differentiation of non-holomorphic functions involving complex inputs \
        or integer inputs, use jax.jvp directly.")

let check_output_jacfwd holomorphic v =
  let dt = (Core.get_aval v).dtype in
  if holomorphic && not (is_complex_dtype dt) then
    invalid_arg
      ("jacfwd with holomorphic=True requires outputs with complex dtype, but \
        got " ^ Dtype.numpy_name dt ^ ".")

let check_input_jacrev holomorphic allow_int v =
  let dt = (Core.get_aval v).dtype in
  let name = Dtype.numpy_name dt in
  if holomorphic && not (is_complex_dtype dt) then
    invalid_arg
      ("jacrev with holomorphic=True requires inputs with complex dtype, but \
        got " ^ name ^ ".");
  if is_integral_dtype dt && not allow_int then
    invalid_arg
      ("jacrev requires real- or complex-valued inputs (input dtype that is a \
        sub-dtype of np.inexact), but got " ^ name
     ^ ". If you want to use Boolean- or integer-valued inputs, use vjp or set \
        allow_int to True.")

let check_output_jacrev holomorphic v =
  let dt = (Core.get_aval v).dtype in
  let name = Dtype.numpy_name dt in
  if holomorphic then
    begin if not (is_complex_dtype dt) then
      invalid_arg
        ("jacrev with holomorphic=True requires outputs with complex dtype, \
          but got " ^ name ^ ".")
    end
  else if is_complex_dtype dt then
    invalid_arg
      ("jacrev requires real-valued outputs (output dtype that is a sub-dtype \
        of np.floating), but got " ^ name
     ^ ". For holomorphic differentiation, pass holomorphic=True. For \
        differentiation of non-holomorphic functions involving complex \
        outputs, use jax.vjp directly.")
  else if not (Dtypes.issubdtype dt `Floating) then
    invalid_arg
      ("jacrev requires real-valued outputs (output dtype that is a sub-dtype \
        of np.floating), but got " ^ name
     ^ ". For differentiation of functions with integer outputs, use jax.vjp \
        directly.")

let jacfwd ?(argnums = 0) ?(holomorphic = false) f args =
  let selected, rebuild = split_args [ argnums ] args in
  let x0 = List.hd selected in
  let leaves, in_def = Tree.tree_flatten x0 in
  List.iter (check_input_jacfwd holomorphic) leaves;
  let avals = List.map Core.get_aval leaves in
  let out_def = ref Tree.Null_def in
  let call flat =
    let out = f (rebuild [ Tree.tree_unflatten in_def flat ]) in
    let o, d = Tree.tree_flatten out in
    out_def := d;
    o
  in
  let primal_outs = ref [] in
  let tangents = std_basis leaves in
  let moved_all =
    Batching.vmap_flat ~out_axis:(-1)
      (fun ts ->
        let po, to_ = Ad.jvp_instantiated call leaves ts in
        primal_outs := po;
        to_)
      (List.map (fun _ -> Some 0) leaves)
      tangents
  in
  let sizes = List.map size_of avals in
  let entries =
    List.map
      (fun moved ->
        let out_shape =
          let s = (Core.get_aval moved).shape in
          Array.sub s 0 (Array.length s - 1)
        in
        let example = Core.get_aval (List.hd !primal_outs) in
        let parts = split_along moved sizes (Array.length out_shape) in
        List.map2
          (fun part (a : aval) ->
            downcast
              (reshape_if_needed part (Array.append out_shape a.shape))
              { example with dtype = (Core.get_aval moved).dtype })
          parts avals)
      moved_all
  in
  flatten_nested (nested_jacobian !out_def in_def entries)

let jacrev ?(argnums = 0) ?(holomorphic = false) ?(allow_int = false) f args =
  let selected, rebuild = split_args [ argnums ] args in
  let x0 = List.hd selected in
  let leaves, in_def = Tree.tree_flatten x0 in
  List.iter (check_input_jacrev holomorphic allow_int) leaves;
  let avals = List.map Core.get_aval leaves in
  let out_def = ref Tree.Null_def in
  let call flat =
    let out = f (rebuild [ Tree.tree_unflatten in_def flat ]) in
    let o, d = Tree.tree_flatten out in
    out_def := d;
    o
  in
  let outs, f_vjp = Ad.vjp call leaves in
  List.iter (check_output_jacrev holomorphic) outs;
  let out_avals = List.map Core.get_aval outs in
  let basis_trees = std_basis outs in
  let grads =
    Batching.vmap f_vjp (List.map (fun _ -> Some 0) outs) basis_trees
  in
  let sizes = List.map size_of out_avals in
  let per_input =
    List.map2
      (fun g (a : aval) ->
        let parts = split_along g sizes 0 in
        List.map2
          (fun part (oa : aval) ->
            downcast (reshape_if_needed part (Array.append oa.shape a.shape)) a)
          parts out_avals)
      grads avals
  in
  let entries =
    List.mapi
      (fun j _ -> List.map (fun column -> List.nth column j) per_input)
      out_avals
  in
  flatten_nested (nested_jacobian !out_def in_def entries)

let jacobian = jacrev

let hessian ?(argnums = 0) ?holomorphic f args =
  jacfwd ~argnums ?holomorphic (fun a -> jacrev ~argnums ?holomorphic f a) args

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

let eval_shape_structs f structs =
  let flat, in_tree = Tree.tree_flatten (Tree.Tuple structs) in
  let avals = List.map Shape_dtype_struct.traced_aval flat in
  let flat_fun, out_tree = Api_util.flatten_fun f in_tree in
  let closed, _ =
    Jaxpr.make_jaxpr_hoisted avals (Linear_util.call_wrapped flat_fun)
  in
  Tree.tree_unflatten (out_tree ())
    (List.map
       (function
         | A_var v -> v.vaval
         | A_lit nd ->
             {
               shape = Ndarray.shape nd;
               dtype = Ndarray.dtype nd;
               weak_type = false;
             }
         | DropVar a -> a)
       closed.jaxpr.outs)

type tree_fn = value Tree.t list -> value Tree.t

let split_aux f =
  let aux = ref Tree.Null in
  ( (fun args ->
      match f args with
      | Tree.Tuple [ y; a ] ->
          aux := Tree.tree_map Ad.aux_primal a;
          y
      | _ -> invalid_arg "has_aux requires a (value, aux) tuple"),
    aux )

let check_holomorphic name v =
  match (Core.get_aval v).dtype with
  | Dtype.Complex64 | Dtype.Complex128 -> ()
  | dt ->
      invalid_arg
        (Printf.sprintf
           "%s with holomorphic=True requires inputs with complex dtype, but \
            got %s."
           name (Dtype.numpy_name dt))

let value_and_grad_holomorphic f args =
  match args with
  | [] -> failwith "value_and_grad: no arguments"
  | x0 :: rest ->
      let flat0, tree0 = Tree.tree_flatten x0 in
      List.iter (check_holomorphic "grad") flat0;
      let flat_f flat0' =
        match f (Tree.tree_unflatten tree0 flat0' :: rest) with
        | Tree.Leaf v -> [ v ]
        | _ -> failwith "value_and_grad: function must return a scalar leaf"
      in
      let outs, f_vjp = Ad.vjp flat_f flat0 in
      let y = List.hd outs in
      (Tree.Leaf y, Tree.tree_unflatten tree0 (f_vjp [ ones_like_value y ]))

let grad_holomorphic f args = snd (value_and_grad_holomorphic f args)

let grad_aux ?allow_int f args =
  let g, aux = split_aux f in
  let out = grad ?allow_int g args in
  (out, !aux)

let value_and_grad_aux ?allow_int f args =
  let g, aux = split_aux f in
  let y, out = value_and_grad ?allow_int g args in
  (y, out, !aux)

let jvp_aux f primals tangents =
  let g, aux = split_aux f in
  let y, t = jvp g primals tangents in
  (y, t, !aux)

let vjp_aux f primals =
  let g, aux = split_aux f in
  let y, back = vjp g primals in
  (y, back, !aux)

let linearize_aux f primals =
  let g, aux = split_aux f in
  let y, lin = linearize g primals in
  (y, lin, !aux)

let jacobian_aux ?argnums f args =
  let g, aux = split_aux f in
  let jac = jacrev ?argnums g args in
  (jac, !aux)

let jacfwd_aux ?argnums f args =
  let g, aux = split_aux f in
  let jac = jacfwd ?argnums g args in
  (jac, !aux)

let jacrev_aux ?argnums f args =
  let g, aux = split_aux f in
  let jac = jacrev ?argnums g args in
  (jac, !aux)

let hessian_aux ?argnums f args =
  jacfwd_aux ?argnums
    (fun a ->
      let jac, aux = jacrev_aux ?argnums f a in
      Tree.Tuple [ jac; aux ])
    args

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

let concrete_removed =
  "The `concrete` option to `jax.checkpoint` was deprecated in JAX v0.8.2, and \
   removed in JAX v0.10.0. In its place please use `static_argnums`; for \
   details refer to \
   https://docs.jax.dev/en/latest/jep/11830-new-remat-checkpoint.html."

let checkpoint ?policy ?(prevent_cse : bool option)
    ?(static_argnums = ([] : int list)) ?(concrete : bool option) f =
  if concrete <> None then raise (Py_exn.Not_implemented_error concrete_removed);
  match static_argnums with
  | [] -> Checkpoint.checkpoint ?policy ?prevent_cse f
  | static ->
      fun args ->
        let dynamic = List.filteri (fun i _ -> not (List.mem i static)) args in
        let merged values =
          let remaining = ref values in
          List.mapi
            (fun i a ->
              if List.mem i static then a
              else
                match !remaining with
                | v :: tl ->
                    remaining := tl;
                    v
                | [] -> invalid_arg "checkpoint: arity")
            args
        in
        Checkpoint.checkpoint ?policy (fun d -> f (merged d)) dynamic

let remat = checkpoint
let named_call ?name:_ f = f
let named_scope _name f = f ()
let device_get = to_host
let block_until_ready x = x
let devices ?backend () = Backend.devices ?backend ()

let local_devices ?process_index ?backend ?host_id () =
  let process_index =
    match host_id with Some h -> Some h | None -> process_index
  in
  (match process_index with
  | Some p when p <> 0 ->
      raise (Py_exn.Value_error (Printf.sprintf "Unknown process_index %d" p))
  | _ -> ());
  devices ?backend ()

let device_count ?backend () = List.length (devices ?backend ())
let local_device_count = device_count
let process_count () = 1
let process_index () = 0
let process_indices () = [ 0 ]
let host_count = process_count
let host_id = process_index
let host_ids = process_indices
let default_backend = Backend.default_backend

let clear_caches () =
  Hashtbl.reset compiled;
  Ad_util.clear_symbolic_zeros ();
  Backend.clear_placed_programs ();
  Backend.clear_region_programs ();
  Jaxpr.clear_call_cache ()

let disable_jit thunk = Config.with_value Config.disable_jit true thunk
let enable_checks v thunk = Config.with_value Config.enable_checks v thunk
let debug_key_reuse v thunk = Config.with_value Config.debug_key_reuse v thunk

let check_tracer_leaks v thunk =
  Config.with_value Config.check_tracer_leaks v thunk

let checking_leaks thunk = check_tracer_leaks true thunk

let explain_cache_misses v thunk =
  Config.with_value Config.explain_cache_misses v thunk

let enable_custom_prng ?(new_val = true) thunk =
  Config.with_value Config.enable_custom_prng new_val thunk

let softmax_custom_jvp v thunk =
  Config.with_value Config.softmax_custom_jvp v thunk

let threefry_partitionable v thunk =
  Config.with_value Config.threefry_partitionable v thunk

let allow_f16_reductions v thunk =
  Config.with_value Config.allow_f16_reductions v thunk

let thread_guard v thunk = Config.with_value Config.thread_guard v thunk

let jax2tf_associative_scan_reductions v thunk =
  Config.with_value Config.jax2tf_associative_scan_reductions v thunk

let numpy_dtype_promotion v thunk =
  Config.with_value Config.numpy_dtype_promotion v thunk

let legacy_prng_key v thunk = Config.with_value Config.legacy_prng_key v thunk

let default_matmul_precision v thunk =
  Config.with_value Config.default_matmul_precision (Some v) thunk

let transfer_guard = Config.with_transfer_guard

let transfer_guard_host_to_device v thunk =
  Config.with_value Config.transfer_guard_host_to_device (Some v) thunk

let transfer_guard_device_to_device v thunk =
  Config.with_value Config.transfer_guard_device_to_device (Some v) thunk

let transfer_guard_device_to_host v thunk =
  Config.with_value Config.transfer_guard_device_to_host (Some v) thunk

let array_garbage_collection_guard v thunk =
  Config.with_value Config.array_garbage_collection_guard (Some v) thunk

let ensure_compile_time_eval thunk = Core.with_eval_dynamic thunk

let pmap ?axis_name ?in_axes ?out_axes
    ?(static_broadcasted_argnums = ([] : int list)) ?axis_size f args =
  let static = static_broadcasted_argnums in
  let dynamic = List.filteri (fun i _ -> not (List.mem i static)) args in
  let in_axes =
    Option.map (List.filteri (fun i _ -> not (List.mem i static))) in_axes
  in
  let merged values =
    let remaining = ref values in
    List.mapi
      (fun i a ->
        if List.mem i static then a
        else
          match !remaining with
          | v :: tl ->
              remaining := tl;
              v
          | [] -> invalid_arg "pmap: arity")
      args
  in
  Spmd.pmap ?axis_name ?in_axes ?out_axes ?axis_size
    (fun d -> f (merged d))
    dynamic

let pure_callback host_fun ~result_avals args =
  Custom_derivatives.pure_callback host_fun ~result_avals args

let typeof v = Core.get_aval v
let effects_barrier () = ()
let copy_to_host_async x = x

let default_device ?(new_val : string option) () thunk =
  match new_val with
  | None ->
      raise
        (Py_exn.Type_error
           "Context manager for default_device config option requires an \
            argument representing the new value for the config option.")
  | Some _ -> thunk ()

let live_arrays ?(platform : string option) () =
  ignore platform;
  ([] : value list)

let make_array_from_callback ?(dtype : Dtype.t option) (shape : int array)
    callback =
  ignore dtype;
  callback (Array.map (fun s -> (0, s)) shape)

let print_environment_info ?(return_string = false) () =
  let info =
    Printf.sprintf "ojax: backend=%s devices=%d x64=%b" (default_backend ())
      (device_count ()) (Config.x64_enabled ())
  in
  if return_string then Some info
  else begin
    print_endline info;
    None
  end
