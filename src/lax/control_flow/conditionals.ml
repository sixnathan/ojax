open Types

let cond (pred : value) (true_fun : value list -> value list)
    (false_fun : value list -> value list) (operands : value list) : value list
    =
  let avals = List.map Core.get_aval operands in
  let t, hoisted_t = Jaxpr.make_jaxpr_hoisted avals true_fun in
  let f, hoisted_f = Jaxpr.make_jaxpr_hoisted avals false_fun in
  if hoisted_t = [] && hoisted_f = [] then
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
