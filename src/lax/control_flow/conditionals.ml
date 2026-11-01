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

let rec case (index : value) (branches : (value list -> value list) list)
    (operands : value list) : value list =
  let avals = List.map Core.get_aval operands in
  let traced =
    List.map (fun branch -> Jaxpr.make_jaxpr_stateful avals branch) branches
  in
  let cells =
    List.fold_left
      (fun acc (_, _, branch_cells) ->
        acc
        @ List.filter
            (fun (c : Ref_state.cell) -> not (List.memq c acc))
            branch_cells)
      [] traced
  in
  let n = List.length operands in
  if cells <> [] then begin
    let stateful branch args =
      match Util.split_list args [ n ] with
      | [ ops; state ] ->
          Ref_state.pin cells state;
          let outs = branch ops in
          outs @ Ref_state.contents cells
      | _ -> failwith "switch: arity"
    in
    let outs =
      case index
        (List.map stateful branches)
        (operands @ List.map Ref_state.read cells)
    in
    let n_out = List.length outs - List.length cells in
    match Util.split_list outs [ n_out ] with
    | [ results; finals ] ->
        List.iter2 Ref_state.write cells finals;
        results
    | _ -> failwith "switch: arity"
  end
  else
    let common =
      List.fold_left
        (fun acc (_, hoisted, _) ->
          acc @ List.filter (fun h -> not (List.memq h acc)) hoisted)
        [] traced
    in
    if common = [] then
      Core.bind (Case (List.map (fun (c, _, _) -> c) traced)) (index :: operands)
    else
      let all_avals = avals @ List.map Core.get_aval common in
      let rebuilt =
        List.map
          (fun (closed, hoisted, _) ->
            Jaxpr.make_jaxpr all_avals (fun args ->
                match Util.split_list args [ n ] with
                | [ ops; shared ] ->
                    let lookup h =
                      let rec go cs vs =
                        match (cs, vs) with
                        | c :: _, v :: _ when c == h -> v
                        | _ :: ct, _ :: vt -> go ct vt
                        | _ -> failwith "switch: hoisted value"
                      in
                      go common shared
                    in
                    Jaxpr.eval_closed_jaxpr closed
                      (ops @ List.map lookup hoisted)
                | _ -> failwith "switch: arity"))
          traced
      in
      Core.bind (Case rebuilt) ((index :: operands) @ common)

let platform_index ~(platforms : string array option array) : value =
  Core.bind1 (Platform_index platforms) []
