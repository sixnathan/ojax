open Types

let custom_jvp ~primal ~jvp args =
  let top = Core.find_top_trace args in
  let tracers = List.map (Core.full_raise top) args in
  let outs = Core.process_custom_jvp top primal ~jvp tracers in
  List.map Core.full_lower outs

let custom_vjp ~primal ~fwd ~bwd args =
  let top = Core.find_top_trace args in
  let tracers = List.map (Core.full_raise top) args in
  let outs = Core.process_custom_vjp top primal ~fwd ~bwd tracers in
  List.map Core.full_lower outs

let custom_gradient fun_ args =
  let rule_ref = ref None in
  let primal a = fst (fun_ a) in
  let fwd a =
    let ans, rule = fun_ a in
    rule_ref := Some rule;
    ans
  in
  let bwd inputs =
    match !rule_ref with
    | Some rule -> rule inputs
    | None -> failwith "custom_gradient: fwd was not run before bwd"
  in
  custom_vjp ~primal ~fwd ~bwd args

let closure_convert fun_ example_args =
  let avals = List.map Core.get_aval example_args in
  let cj = Jaxpr.make_jaxpr avals fun_ in
  let consts = List.map (fun c -> Concrete c) cj.consts in
  let n_args = List.length avals in
  let converted all =
    let args, cs =
      match Util.split_list all [ n_args ] with
      | [ a; c ] -> (a, c)
      | _ -> failwith "closure_convert: argument count mismatch"
    in
    Jaxpr.eval_jaxpr cj.jaxpr (cs @ args)
  in
  (converted, consts)
