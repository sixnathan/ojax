open Types

let rules : (int, int * int * closed_jaxpr * (unit -> closed_jaxpr)) Hashtbl.t =
  Hashtbl.create 16

let memoized_repr = "<function _memoize.<locals>.memoized at 0xXXXXXXXXXXXX>"

let close_consts (c : closed_jaxpr) =
  (List.map (fun nd -> Concrete nd) c.consts, { c with consts = [] })

let register ~num_consts ~num_res (fwd_jaxpr : closed_jaxpr) fun_thunk =
  let primal xs = Jaxpr.eval_closed_jaxpr fwd_jaxpr xs in
  let jvp _ _ =
    raise
      (Py_exn.Not_implemented_error
         "Differentiation rule for 'remat_opt' not implemented")
  in
  let id = Core.register_custom_rule (Core.Jvp_rule { primal; jvp }) in
  Hashtbl.replace rules id (num_consts, num_res, fwd_jaxpr, fun_thunk);
  Hashtbl.replace Jaxpr.opaque_dce_calls id ();
  Pretty_printer.register_custom_view id "remat_opt"
    [
      ("fun_jaxpr_thunk", Pretty_printer.Plain memoized_repr);
      ("fwd_jaxpr", Pretty_printer.Sub fwd_jaxpr);
      ("num_consts", Pretty_printer.Plain (string_of_int num_consts));
      ("num_res", Pretty_printer.Plain (string_of_int num_res));
    ];
  id

let bind_remat_opt ~num_consts ~num_res fwd_jaxpr fun_thunk args =
  Core.bind
    (Custom_jvp_call (register ~num_consts ~num_res fwd_jaxpr fun_thunk))
    args

let is_remat_opt id = Hashtbl.mem rules id
let count l = List.length (List.filter Fun.id l)

let dce ~dce_closed id used_outs =
  match Hashtbl.find_opt rules id with
  | None -> None
  | Some (num_consts, num_res, fwd_jaxpr, fun_thunk) ->
      let used_res = List.filteri (fun i _ -> i < num_res) used_outs in
      if List.exists Fun.id used_res then
        let n = List.length fwd_jaxpr.jaxpr.in_binders in
        let instantiate = List.init n (fun i -> i >= num_consts) in
        let fwd', used_ins = dce_closed fwd_jaxpr used_outs instantiate in
        let num_consts' =
          count (List.filteri (fun i _ -> i < num_consts) used_ins)
        in
        let id' =
          register ~num_consts:num_consts' ~num_res:(count used_res) fwd'
            fun_thunk
        in
        Some (Some (id', used_ins))
      else Some None

let memoize f =
  let cell = ref None in
  fun () ->
    match !cell with
    | Some v -> v
    | None ->
        let v = f () in
        cell := Some v;
        v

let optimize_remat_fwd primal fwd args =
  let avals = List.map Core.get_aval args in
  let forwards = ref [] in
  let traced =
    Jaxpr.make_jaxpr avals (fun inputs ->
        let outs, res = fwd inputs in
        let index r =
          let rec go i = function
            | [] -> None
            | x :: rest -> if x == r then Some i else go (i + 1) rest
          in
          go 0 inputs
        in
        forwards := List.map index res;
        List.filter (fun r -> index r = None) res @ outs)
  in
  let consts, fwd_jaxpr = close_consts traced in
  let num_res = List.length (List.filter Option.is_none !forwards) in
  let fun_thunk =
    memoize (fun () ->
        let c = Jaxpr.make_jaxpr avals primal in
        c)
  in
  let out =
    bind_remat_opt ~num_consts:(List.length consts) ~num_res fwd_jaxpr fun_thunk
      (consts @ args)
  in
  let res, outs =
    match Util.split_list out [ num_res ] with
    | [ r; o ] -> (r, o)
    | _ -> failwith "remat_opt: arity"
  in
  let rest = ref res in
  let res =
    List.map
      (function
        | Some i -> List.nth args i
        | None -> (
            match !rest with
            | r :: tl ->
                rest := tl;
                r
            | [] -> failwith "remat_opt: residual arity"))
      !forwards
  in
  (outs, res)

let batch ~batch_jaxpr ~move id axis_size vals bdims =
  match Hashtbl.find_opt rules id with
  | None -> None
  | Some (num_consts, num_res, fwd_jaxpr, fun_thunk) ->
      let front =
        List.map2
          (fun v d ->
            match d with
            | Some d when d <> 0 -> move axis_size (Some d) 0 v
            | _ -> v)
          vals bdims
      in
      let in_dims = List.map (Option.map (fun _ -> 0)) bdims in
      let avals = List.map Core.get_aval front in
      let batched, out_batched =
        batch_jaxpr axis_size fwd_jaxpr avals in_dims
      in
      let extra, batched = close_consts batched in
      let prim_avals = List.filteri (fun i _ -> i >= num_consts) avals in
      let prim_dims = List.filteri (fun i _ -> i >= num_consts) in_dims in
      let batched_fun =
        memoize (fun () ->
            fst (batch_jaxpr axis_size (fun_thunk ()) prim_avals prim_dims))
      in
      let outs =
        bind_remat_opt
          ~num_consts:(num_consts + List.length extra)
          ~num_res batched batched_fun (extra @ front)
      in
      Some (outs, List.map (fun b -> if b then Some 0 else None) out_batched)
