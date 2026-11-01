open Types

let aval_of_atom = function
  | A_var v -> v.vaval
  | A_lit nd ->
      {
        shape = Ndarray.shape nd;
        dtype = Ndarray.dtype nd;
        weak_type = Ndarray.weak nd;
      }
  | DropVar a -> a

let solve_out_avals (solve : closed_jaxpr) : aval list =
  List.map aval_of_atom solve.jaxpr.outs

let split_args const_lengths args =
  let rec take k l =
    if k = 0 then ([], l)
    else
      match l with
      | x :: r ->
          let a, b = take (k - 1) r in
          (x :: a, b)
      | [] -> invalid_arg "custom_linear_solve: arity"
  in
  let rec go i l =
    if i = Array.length const_lengths then ([], l)
    else
      let part, rest = take const_lengths.(i) l in
      let parts, b = go (i + 1) rest in
      (part :: parts, b)
  in
  go 0 args

let solve_impl const_lengths (solve : closed_jaxpr) (args : value list) :
    value list =
  let parts, b = split_args const_lengths args in
  Jaxpr.eval_closed_jaxpr solve (List.nth parts 2 @ b)

let trace_separating_consts b_avals f =
  let closed, hoisted = Jaxpr.make_jaxpr_hoisted b_avals f in
  match hoisted with
  | [] -> (closed, [])
  | _ ->
      let n = List.length hoisted in
      let moved =
        Jaxpr.make_jaxpr
          (List.map Core.get_aval hoisted @ b_avals)
          (fun all ->
            let consts = List.filteri (fun i _ -> i < n) all in
            let x = List.filteri (fun i _ -> i >= n) all in
            Jaxpr.eval_closed_jaxpr closed (x @ consts))
      in
      (moved, hoisted)

let custom_linear_solve ?(symmetric = false)
    ?(transpose_solve :
       ((value list -> value list) -> value list -> value list) option)
    ?(linear_matvec : (value list -> value list) option)
    (matvec : value list -> value list) (b : value list)
    (solve : (value list -> value list) -> value list -> value list) :
    value list =
  let transpose_solve =
    match transpose_solve with
    | None when symmetric -> Some solve
    | other -> other
  in
  let b_avals = List.map Core.get_aval b in
  let matvec_jaxpr, matvec_consts = trace_separating_consts b_avals matvec in
  let solve_jaxpr, solve_consts =
    trace_separating_consts b_avals (solve matvec)
  in
  let vecmat_jaxpr, vecmat_consts, tr_jaxpr, tr_consts =
    match transpose_solve with
    | None -> (None, [], None, [])
    | Some ts ->
        let vecmat =
          if symmetric then matvec
          else fun cotangent ->
            let zeros =
              List.map
                (fun (a : aval) -> Core.splat a.dtype a.shape 0.0)
                b_avals
            in
            let linear = Option.value linear_matvec ~default:matvec in
            let _, f_vjp = Interpreters.Ad.vjp linear zeros in
            f_vjp cotangent
        in
        let vj, vc =
          if symmetric then (matvec_jaxpr, matvec_consts)
          else
            let c, h = trace_separating_consts b_avals vecmat in
            (Jaxpr.dce_keep_args c, h)
        in
        let tj, tc = trace_separating_consts b_avals (ts vecmat) in
        (Some vj, vc, Some tj, tc)
  in
  let const_lengths =
    [|
      List.length matvec_consts;
      List.length vecmat_consts;
      List.length solve_consts;
      List.length tr_consts;
    |]
  in
  Core.bind
    (Custom_linear_solve
       {
         matvec = matvec_jaxpr;
         vecmat = vecmat_jaxpr;
         solve = solve_jaxpr;
         transpose_solve = tr_jaxpr;
         const_lengths;
       })
    (matvec_consts @ vecmat_consts @ solve_consts @ tr_consts @ b)
