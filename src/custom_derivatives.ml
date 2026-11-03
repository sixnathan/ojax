open Types

let instantiate_zero primal t =
  if Ad_util.is_symbolic_zero t then
    let a = Core.get_aval t in
    match Core.splat a.dtype a.shape 0.0 with
    | Concrete nd when (Core.get_aval primal).weak_type ->
        Concrete (Ndarray.as_weak nd)
    | v -> v
  else t

let split_nondiff nondiff args =
  ( List.filteri (fun i _ -> List.mem i nondiff) args,
    List.filteri (fun i _ -> not (List.mem i nondiff)) args )

let merge_nondiff nondiff kept diff =
  let total = List.length kept + List.length diff in
  let k = ref kept and d = ref diff in
  List.init total (fun i ->
      let source = if List.mem i nondiff then k else d in
      match !source with
      | v :: tl ->
          source := tl;
          v
      | [] -> invalid_arg "custom derivative: arity")

let rec custom_jvp ?(nondiff_argnums = []) ?(name = "f") ?(jvp_name = "jvp")
    primal ~jvp =
  if nondiff_argnums <> [] then fun args ->
    let kept, diff = split_nondiff nondiff_argnums args in
    custom_jvp ~name ~jvp_name
      (fun d -> primal (merge_nondiff nondiff_argnums kept d))
      ~jvp:(fun primals tangents -> jvp (kept @ primals) tangents)
      diff
  else
    let rule primals tangents =
      let tangents = List.map2 instantiate_zero primals tangents in
      jvp primals tangents
    in
    let id = Core.register_custom_rule (Core.Jvp_rule { primal; jvp = rule }) in
    Core.register_custom_names id ~name ~jvp_name;
    fun args -> Core.bind (Custom_jvp_call id) args

let add_tangents x y =
  if Ad_util.is_symbolic_zero x then y
  else if Ad_util.is_symbolic_zero y then x
  else Ad_util.add_any x y

let defjvps ?name primal rules =
  let self = ref (fun _ -> []) in
  let jvp primals tangents =
    let out = !self primals in
    let terms =
      List.map2
        (fun rule t ->
          match rule with Some r -> Some (r t out primals) | None -> None)
        rules tangents
    in
    let tangent_for i po =
      let pick = function
        | Some term -> List.nth term i
        | None -> Ad_util.symbolic_zero_like_value po
      in
      match List.map pick terms with
      | [] -> Ad_util.symbolic_zero_like_value po
      | first :: rest -> List.fold_left add_tangents first rest
    in
    (out, List.mapi tangent_for out)
  in
  let call = custom_jvp ?name primal ~jvp in
  self := call;
  call

let aval_str = Ref_state.aval_string
let same_type (a : aval) (b : aval) = a.shape = b.shape && a.dtype = b.dtype

let mismatch_extra (a : aval) (b : aval) =
  let mismatches =
    (if a.dtype <> b.dtype then [ "the dtypes do not match" ] else [])
    @ if a.shape <> b.shape then [ "the shapes do not match" ] else []
  in
  match List.rev mismatches with
  | [] -> ""
  | [ only ] -> ", so " ^ only
  | last :: rest ->
      ", so " ^ String.concat ", " (List.rev rest) ^ ", and " ^ last

let tuple_def n =
  Tree_util.Tuple_def (List.init n (fun _ -> Tree_util.Leaf_def))

let check_bwd_count outs avals =
  let n_out = List.length outs and n_in = List.length avals in
  if n_out <> n_in then
    raise
      (Py_exn.Type_error
         (Printf.sprintf
            "Custom VJP bwd rule must produce an output with the same \
             container (pytree) structure as the args tuple of the primal \
             function, and in particular must produce a tuple of length equal \
             to the number of arguments to the primal function, but got bwd \
             output structure %s for primal input structure %s."
            (Tree_util.treedef_to_string (tuple_def n_out))
            (Tree_util.treedef_to_string (tuple_def n_in))))

let check_bwd_tangent index (input : aval) v =
  let got = Core.get_aval v in
  if Dtypes.issubdtype input.dtype `Inexact && got.shape <> input.shape then
    raise
      (Py_exn.Value_error
         (Printf.sprintf
            "Custom VJP bwd rule must produce an output with the same type as \
             the args tuple of the primal function, but at output[%d] the bwd \
             rule produced an output of type %s corresponding to an input of \
             type %s%s"
            index (aval_str got) (aval_str input) (mismatch_extra input got)))

let float0_tangent (input : aval) =
  input.dtype = Dtype.Bool || Dtypes.issubdtype input.dtype `Integer

let rec custom_vjp ?(nondiff_argnums = []) ?(name = "f") ?(fwd_name = "fwd")
    ?(bwd_name = "bwd") ?(optimize_remat = false) primal ~fwd ~bwd =
  let fwd =
    if optimize_remat then Remat_opt.optimize_remat_fwd primal fwd else fwd
  in
  if nondiff_argnums <> [] then fun args ->
    let kept, diff = split_nondiff nondiff_argnums args in
    custom_vjp ~name ~fwd_name ~bwd_name
      (fun d -> primal (merge_nondiff nondiff_argnums kept d))
      ~fwd:(fun d -> fwd (merge_nondiff nondiff_argnums kept d))
      ~bwd:(fun res ct -> bwd (kept @ res) ct)
      diff
  else
    let rules : (aval list, int) Hashtbl.t = Hashtbl.create 4 in
    let rule_for avals =
      match Hashtbl.find_opt rules avals with
      | Some id -> id
      | None ->
          let bwd res cts =
            let outs = bwd res cts in
            check_bwd_count outs avals;
            List.mapi
              (fun i (out, input) ->
                if float0_tangent input then Ad_util.symbolic_zero_of_aval input
                else begin
                  check_bwd_tangent i input out;
                  out
                end)
              (List.combine outs avals)
          in
          let id =
            Core.register_custom_rule
              (Core.Vjp_rule { primal; fwd; bwd; fwd_perturbed = None })
          in
          Core.register_custom_names id ~name ~jvp_name:fwd_name;
          if bwd_name <> "bwd" then
            Hashtbl.replace Pretty_printer.custom_bwd_names id bwd_name;
          Hashtbl.replace rules avals id;
          id
    in
    fun args ->
      Core.bind (Custom_vjp_call (rule_for (List.map Core.get_aval args))) args

type symbolic_zero = { aval : aval }
type tangent = Symbolic_zero of symbolic_zero | Tangent of value
type custom_vjp_primal = { value : value; perturbed : bool }

let symbolic_zero_from_primal_value v = { aval = Core.get_aval v }

let instantiate_tangent = function
  | Symbolic_zero { aval } -> Ad_util.zeros_like_aval aval
  | Tangent v -> v

let custom_vjp_primal_tree_values tree =
  Tree_util.tree_map (fun p -> p.value) tree

let tangent_of_internal t =
  if Ad_util.is_symbolic_zero t then Symbolic_zero { aval = Core.get_aval t }
  else Tangent t

let internal_of_tangent = function
  | Symbolic_zero { aval } -> Ad_util.symbolic_zero_of_aval aval
  | Tangent v -> v

let custom_jvp_symbolic ?(name = "f") ?(jvp_name = "jvp") primal ~jvp =
  let jvp_name = jvp_name ^ "\000symbolic" in
  let register primal ~jvp =
    let id = Core.register_custom_rule (Core.Jvp_rule { primal; jvp }) in
    Core.register_custom_names id ~name ~jvp_name;
    fun args -> Core.bind (Custom_jvp_call id) args
  in
  let rule primals tangents =
    let po, to_ = jvp primals (List.map tangent_of_internal tangents) in
    if List.compare_lengths po to_ <> 0 then
      invalid_arg
        "Custom JVP rule must produce primal and tangent outputs with equal \
         container (pytree) structures";
    let check p t =
      let expected = Core.get_aval p in
      let got =
        match t with
        | Symbolic_zero { aval } -> aval
        | Tangent v -> Core.get_aval v
      in
      if not (same_type expected got) then
        raise
          (Py_exn.Type_error
             (Printf.sprintf
                "Custom JVP rule must produce primal and tangent outputs with \
                 corresponding shapes and dtypes. Expected %s (tangent type of \
                 %s) but got %s."
                (aval_str expected) (aval_str expected) (aval_str got)))
    in
    List.iter2 check po to_;
    (po, List.map internal_of_tangent to_)
  in
  register primal ~jvp:rule

let bwd_output index (input : aval) = function
  | None -> Ad_util.symbolic_zero_of_aval input
  | Some _ when float0_tangent input -> Ad_util.symbolic_zero_of_aval input
  | Some (Symbolic_zero { aval }) ->
      if not (same_type aval input) then
        invalid_arg
          (Printf.sprintf
             "Custom VJP bwd rule produced a SymbolicZero with a shape/dtype \
              that does not match the corresponding input tangent shape/dtype: \
              at output[%d] the SymbolicZero had shape/dtype %s while the \
              corresponding input had shape/dtype %s. Consider just returning \
              a None here instead of a SymbolicZero object."
             index (aval_str aval) (aval_str input));
      Ad_util.symbolic_zero_of_aval aval
  | Some (Tangent v) ->
      check_bwd_tangent index input v;
      v

let custom_vjp_symbolic primal ~fwd ~bwd =
  let rules : (aval list, int) Hashtbl.t = Hashtbl.create 4 in
  let rule_for avals =
    match Hashtbl.find_opt rules avals with
    | Some id -> id
    | None ->
        let fwd_perturbed perturbed primals =
          fwd
            (List.map2
               (fun value perturbed -> { value; perturbed })
               primals perturbed)
        in
        let rule_bwd res cts =
          let outs = bwd res (List.map tangent_of_internal cts) in
          check_bwd_count outs avals;
          List.mapi
            (fun i (out, input) -> bwd_output i input out)
            (List.combine outs avals)
        in
        let id =
          Core.register_custom_rule
            (Core.Vjp_rule
               {
                 primal;
                 fwd = fwd_perturbed (List.map (fun _ -> true) avals);
                 bwd = rule_bwd;
                 fwd_perturbed = Some fwd_perturbed;
               })
        in
        Hashtbl.replace rules avals id;
        id
  in
  fun args ->
    Core.bind (Custom_vjp_call (rule_for (List.map Core.get_aval args))) args

let custom_gradient f =
  let rules = ref [] in
  let primal args = fst (f args) in
  let fwd args =
    let outs, rule = f args in
    let avals = List.map Core.get_aval outs in
    let closed, hoisted = Jaxpr.make_jaxpr_hoisted avals rule in
    rules := (List.map Core.get_aval hoisted, closed) :: !rules;
    (outs, hoisted)
  in
  let bwd res cts =
    let avals = List.map Core.get_aval res in
    let strip (a : aval) = (a.shape, a.dtype) in
    let closed =
      match
        List.find_opt
          (fun (k, _) -> List.map strip k = List.map strip avals)
          !rules
      with
      | Some (_, c) -> c
      | None -> snd (List.hd !rules)
    in
    Jaxpr.eval_closed_jaxpr closed (cts @ res)
  in
  custom_vjp ~name:"wrapped_fun" primal ~fwd ~bwd

let linear_call ~forward ~transpose =
  let id =
    Core.register_custom_rule
      (Core.Linear_rule { forward; transpose; res_differentiable = false })
  in
  fun res linear ->
    let out_avals =
      let traced =
        Jaxpr.make_jaxpr
          (List.map Core.get_aval (res @ linear))
          (fun args ->
            match Util.split_list args [ List.length res ] with
            | [ r; x ] -> forward r x
            | _ -> failwith "linear_call: arity")
      in
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
        traced.jaxpr.outs
    in
    Core.bind
      (Custom_lin
         { rule = id; n_res = List.length res; out_avals; in_zeros = [] })
      (res @ linear)

let rec stop_gradient v =
  match v with
  | Tracer { payload = JVP { primal; _ }; _ } -> stop_gradient primal
  | _ -> Core.bind1 Stop_gradient [ v ]

let closure_convert f avals =
  let closed, hoisted = Jaxpr.make_jaxpr_hoisted avals f in
  ((fun args -> Jaxpr.eval_closed_jaxpr closed args), hoisted)

let callback_id cb_effect host_fun ~result_avals =
  let host args =
    host_fun
      (List.map
         (function
           | Concrete nd -> Concrete nd
           | Device b -> Concrete (Pjrt.Buffer.to_host b)
           | Tracer _ -> failwith "pure_callback: expected concrete arguments")
         args)
  in
  Core.register_custom_rule
    (Core.Callback_rule { host; result_avals; cb_effect })

let pure_callback host_fun ~result_avals =
  let id = callback_id Core.Callback_pure host_fun ~result_avals in
  fun args -> Core.bind (Custom_vjp_call id) args

let shaped_array (a : aval) =
  let dims =
    String.concat "," (List.map string_of_int (Array.to_list a.shape))
  in
  Printf.sprintf "ShapedArray(%s[%s])" (Ref_state.dtype_name a.dtype) dims

let py_tuple = function
  | [ one ] -> "(" ^ one ^ ",)"
  | items -> "(" ^ String.concat ", " items ^ ")"

let io_callback ?(ordered = false) host_fun ~result_avals =
  let id = callback_id (Core.Callback_io { ordered }) host_fun ~result_avals in
  Pretty_printer.register_callback_view id "io_callback"
    [
      ("callback", "<host callback>");
      ("ordered", if ordered then "True" else "False");
      ("result_avals", py_tuple (List.map shaped_array result_avals));
      ("sharding", "None");
    ];
  fun args -> Core.bind (Custom_vjp_call id) args

let callback_with_view ~ordered view params host_fun ~result_avals =
  let id =
    callback_id (Core.Callback_debug { ordered }) host_fun ~result_avals
  in
  Pretty_printer.register_callback_view id view params;
  fun args -> Core.bind (Custom_vjp_call id) args

let callback_view_id ~ordered view params host_fun ~result_avals =
  let id =
    callback_id (Core.Callback_debug { ordered }) host_fun ~result_avals
  in
  Pretty_printer.register_callback_view id view params;
  (id, fun args -> Core.bind (Custom_vjp_call id) args)

let zero_from_primal v = Ad_util.zeros_like_aval (Core.get_aval v)
