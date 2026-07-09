open Types

let cond (pred : value) (true_fun : value list -> value list)
    (false_fun : value list -> value list) (operands : value list) : value list
    =
  let avals = List.map Core.get_aval operands in
  let t = Jaxpr.make_jaxpr avals true_fun in
  let f = Jaxpr.make_jaxpr avals false_fun in
  Core.bind (Cond { t; f }) (pred :: operands)

let platform_index ~(platforms : string array option array) : value =
  Core.bind1 (Platform_index platforms) []
