open Types

type rule = int -> bool list -> value list -> value list * bool list

let rules : (int, rule * string * int) Hashtbl.t = Hashtbl.create 16

let tree_text = function
  | [ _ ] -> "*"
  | items -> "(" ^ String.concat ", " (List.map (fun _ -> "*") items) ^ ")"

let in_tree_text args =
  let star = match args with [ _ ] -> "(*,)" | _ -> tree_text args in
  "PyTreeDef(([], " ^ star ^ "))"

let custom_vmap ?(rule_name = "rule") f ~(rule : rule) args =
  let avals = List.map Core.get_aval args in
  let call = Jaxpr.make_jaxpr avals f in
  let primal = Jaxpr.eval_closed_jaxpr call in
  let jvp _ _ =
    raise
      (Py_exn.Not_implemented_error
         "Differentiation rule for 'custom_vmap_call' not implemented")
  in
  let id = Core.register_custom_rule (Core.Jvp_rule { primal; jvp }) in
  Hashtbl.replace rules id (rule, rule_name, List.length call.jaxpr.outs);
  Hashtbl.replace Jaxpr.opaque_dce_calls id ();
  Pretty_printer.register_custom_view id "custom_vmap_call"
    [
      ("call", Pretty_printer.Sub call);
      ("in_tree", Pretty_printer.Plain (in_tree_text args));
      ( "out_tree",
        Pretty_printer.Plain ("PyTreeDef(" ^ tree_text call.jaxpr.outs ^ ")") );
      ( "rule",
        Pretty_printer.Plain ("<function " ^ rule_name ^ " at 0xXXXXXXXXXXXX>")
      );
    ];
  Core.bind (Custom_jvp_call id) args

let short_name qualname =
  match String.rindex_opt qualname '.' with
  | Some i -> String.sub qualname (i + 1) (String.length qualname - i - 1)
  | None -> qualname

let out_tree_text items = "PyTreeDef(" ^ tree_text items ^ ")"

let check_vmap_rule_trees rule_name n_out outs batched =
  let name = short_name rule_name in
  if List.compare_lengths outs batched <> 0 then
    raise
      (Py_exn.Value_error
         ("structure of output value and output batching specification \
           returned by custom vmap rule (" ^ name
        ^ ") do not match.\nOutput values: " ^ out_tree_text outs
        ^ "\nBatching spec: " ^ out_tree_text batched));
  if List.length outs <> n_out then
    raise
      (Py_exn.Value_error
         ("structure of output returned by custom vmap rule (" ^ name
        ^ ") does not match that of original custom-vmapped function.\n\
           Original output: "
         ^ out_tree_text (List.init n_out Fun.id)
         ^ "\nRule output: " ^ out_tree_text outs))

let batch id axis_size vals bdims =
  match Hashtbl.find_opt rules id with
  | None -> None
  | Some (rule, rule_name, n_out) ->
      let front =
        List.map2
          (fun v d ->
            match d with
            | None -> v
            | Some _ -> Interpreters.Batching.move_batch_axis axis_size d 0 v)
          vals bdims
      in
      let outs, batched =
        rule axis_size (List.map Option.is_some bdims) front
      in
      check_vmap_rule_trees rule_name n_out outs batched;
      Some (outs, List.map (fun b -> if b then Some 0 else None) batched)

let () =
  let previous = !Interpreters.Batching.custom_call_batch in
  Interpreters.Batching.custom_call_batch :=
    fun id axis_size vals bdims ->
      match batch id axis_size vals bdims with
      | Some r -> Some r
      | None -> previous id axis_size vals bdims

let rec sequential_vmap f args =
  let rule _ in_batched args =
    let mapped = List.filteri (fun i _ -> List.nth in_batched i) args in
    let merge m =
      let rest = ref m in
      List.mapi
        (fun i a ->
          if List.nth in_batched i then begin
            let x = List.hd !rest in
            rest := List.tl !rest;
            x
          end
          else a)
        args
    in
    let outs = Lax_api.map (fun m -> sequential_vmap f (merge m)) mapped in
    (outs, List.map (fun _ -> true) outs)
  in
  custom_vmap ~rule_name:"sequential_vmap.<locals>.rule" f ~rule args
