open Types

let rec cond (pred : value) (true_fun : value list -> value list)
    (false_fun : value list -> value list) (operands : value list) : value list
    =
  let avals = List.map Core.get_aval operands in
  let t, hoisted_t, cells_t = Jaxpr.make_jaxpr_stateful avals true_fun in
  let f, hoisted_f, cells_f = Jaxpr.make_jaxpr_stateful avals false_fun in
  let cells =
    cells_t
    @ List.filter
        (fun (c : Ref_state.cell) -> not (List.memq c cells_t))
        cells_f
  in
  if cells <> [] then begin
    let n = List.length operands in
    let stateful branch args =
      match Util.split_list args [ n ] with
      | [ ops; state ] ->
          Ref_state.pin cells state;
          let outs = branch ops in
          outs @ Ref_state.contents cells
      | _ -> failwith "cond: arity"
    in
    let outs =
      cond pred (stateful true_fun) (stateful false_fun)
        (operands @ List.map Ref_state.read cells)
    in
    let n_out = List.length outs - List.length cells in
    match Util.split_list outs [ n_out ] with
    | [ results; finals ] ->
        List.iter2 Ref_state.write cells finals;
        results
    | _ -> failwith "cond: arity"
  end
  else if hoisted_t = [] && hoisted_f = [] then
    Core.bind (Cond { t; f }) (pred :: operands)
  else
    let n = List.length operands in
    let n_t = List.length hoisted_t in
    let hoisted = hoisted_t @ hoisted_f in
    let all_avals = avals @ List.map Core.get_aval hoisted in
    let split args =
      match Util.split_list args [ n; n_t ] with
      | [ ops; h_t; h_f ] -> (ops, h_t, h_f)
      | _ -> failwith "cond: arity"
    in
    let t' =
      Jaxpr.make_jaxpr all_avals (fun args ->
          let ops, h_t, _ = split args in
          Jaxpr.eval_closed_jaxpr t (ops @ h_t))
    in
    let f' =
      Jaxpr.make_jaxpr all_avals (fun args ->
          let ops, _, h_f = split args in
          Jaxpr.eval_closed_jaxpr f (ops @ h_f))
    in
    Core.bind (Cond { t = t'; f = f' }) ((pred :: operands) @ hoisted)

let platform_index ~(platforms : string array option array) : value =
  Core.bind1 (Platform_index platforms) []
