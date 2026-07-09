open Types

let aval_of_atom = function
  | A_var v -> v.vaval
  | A_lit nd ->
      { shape = Ndarray.shape nd; dtype = Ndarray.dtype nd; weak_type = false }
  | DropVar a -> a

let solve_out_avals (solve : closed_jaxpr) : aval list =
  List.map aval_of_atom solve.jaxpr.outs

let solve_impl (solve : closed_jaxpr) (b : value list) : value list =
  Jaxpr.eval_closed_jaxpr solve b

let custom_linear_solve ?(symmetric = false)
    ?(transpose_solve :
       ((value list -> value list) -> value list -> value list) option)
    (matvec : value list -> value list) (b : value list)
    (solve : (value list -> value list) -> value list -> value list) :
    value list =
  let b_avals = List.map Core.get_aval b in
  let solve_jaxpr = Jaxpr.make_jaxpr b_avals (fun bs -> solve matvec bs) in
  let ts_jaxpr =
    match transpose_solve with
    | Some ts -> Some (Jaxpr.make_jaxpr b_avals (fun bs -> ts matvec bs))
    | None -> if symmetric then Some solve_jaxpr else None
  in
  Core.bind
    (Custom_linear_solve { solve = solve_jaxpr; transpose_solve = ts_jaxpr })
    b
