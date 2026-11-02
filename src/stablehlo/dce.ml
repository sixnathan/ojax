open Types

let split_consts (c : closed_jaxpr) =
  match Util.split_list c.jaxpr.in_binders [ List.length c.consts ] with
  | [ consts; args ] -> (consts, args)
  | _ -> invalid_arg "Stablehlo.Dce: binder split"

let keep_where xs keeps =
  List.filter_map
    (fun (x, k) -> if k then Some x else None)
    (List.combine xs keeps)

let rec has_effects = function
  | Ffi_call { ffi_has_side_effect; _ } -> ffi_has_side_effect
  | Dce_sink _ | Create_token | After_all -> true
  | Xla_call c -> closed_effects c
  | Cond { branches; _ } -> List.exists closed_effects branches
  | While { cond; body; _ } -> closed_effects cond || closed_effects body
  | Scan { jaxpr; _ } -> closed_effects jaxpr
  | Custom_jvp_call id | Custom_vjp_call id -> (
      match Core.custom_rule id with
      | Core.Callback_rule
          { cb_effect = Core.Callback_io _ | Core.Callback_debug _; _ } ->
          true
      | Core.Callback_rule { result_avals; _ } -> result_avals = []
      | Core.Jvp_rule _ | Core.Vjp_rule _ | Core.Linear_rule _ -> (
          match Core.custom_call_jaxprs_of id with
          | [] -> false
          | cs -> List.exists closed_effects cs))
  | _ -> false

and closed_effects (c : closed_jaxpr) =
  List.exists (fun (e : eqn) -> has_effects e.prim) c.jaxpr.eqns

let call_cache :
    (int * bool list * bool list, closed_jaxpr * bool list) Hashtbl.t =
  Hashtbl.create 64

let custom_cache : (int * aval list * bool list * bool, int) Hashtbl.t =
  Hashtbl.create 64

let eqn_avals (e : eqn) =
  List.map
    (function
      | A_var (v : var) -> v.vaval
      | A_lit nd ->
          {
            shape = Ndarray.shape nd;
            dtype = Ndarray.dtype nd;
            weak_type = Ndarray.weak nd;
          }
      | DropVar a -> a)
    e.inputs

let rec dce_closed (c : closed_jaxpr) used_outputs ~instantiate =
  let key = (c.jid, used_outputs, instantiate) in
  match Hashtbl.find_opt call_cache key with
  | Some hit -> hit
  | None ->
      let result = dce_uncached c used_outputs ~instantiate in
      Hashtbl.replace call_cache key result;
      result

and dce_uncached (c : closed_jaxpr) used_outputs ~instantiate =
  let const_binders, args = split_consts c in
  let env = Hashtbl.create 64 in
  let read (v : var) = Hashtbl.mem env v.vid in
  let write atom used =
    match atom with
    | A_var (v : var) when used -> Hashtbl.replace env v.vid ()
    | A_var _ | A_lit _ | DropVar _ -> ()
  in
  List.iter2 write c.jaxpr.outs used_outputs;
  let kept =
    List.fold_left
      (fun acc (e : eqn) ->
        let used_ins, replacement = rule (List.map read e.outs) e in
        List.iter2 write e.inputs used_ins;
        match replacement with Some e' -> e' :: acc | None -> acc)
      [] (List.rev c.jaxpr.eqns)
  in
  let used_inputs =
    List.map2 (fun (v : var) inst -> read v || inst) args instantiate
  in
  let in_binders = const_binders @ keep_where args used_inputs in
  let outs = keep_where c.jaxpr.outs used_outputs in
  let jid = Core.fresh_id () in
  Core.inherit_inline ~from:c.jid jid;
  (match Pretty_printer.registered_call_name c.jid with
  | Some name -> Pretty_printer.register_call_name jid name
  | None -> ());
  ({ c with jid; jaxpr = { in_binders; eqns = kept; outs } }, used_inputs)

and rule used_outs (e : eqn) =
  let none = List.map (fun _ -> false) e.inputs in
  let everything = List.map (fun _ -> true) e.inputs in
  let dead = (not (List.exists Fun.id used_outs)) && not (has_effects e.prim) in
  if dead then (none, None)
  else
    let with_prim prim used_ins =
      Some
        {
          e with
          prim;
          inputs = keep_where e.inputs used_ins;
          outs = keep_where e.outs used_outs;
        }
    in
    match e.prim with
    | Xla_call c ->
        let n = List.length c.jaxpr.in_binders - List.length c.consts in
        let c', used_ins =
          dce_closed c used_outs ~instantiate:(List.init n (fun _ -> false))
        in
        ( used_ins,
          if
            (not (List.exists Fun.id used_ins))
            && (not (List.exists Fun.id used_outs))
            && not (closed_effects c')
          then None
          else with_prim (Xla_call c') used_ins )
    | Cond { branches; platforms } ->
        let n = List.length e.inputs - 1 in
        let nothing = List.init n (fun _ -> false) in
        let used =
          List.fold_left
            (fun acc b ->
              let _, u = dce_closed b used_outs ~instantiate:nothing in
              List.map2 ( || ) acc u)
            nothing branches
        in
        let branches' =
          List.map
            (fun b -> fst (dce_closed b used_outs ~instantiate:used))
            branches
        in
        let used_ins = true :: used in
        (used_ins, with_prim (Cond { branches = branches'; platforms }) used_ins)
    | Scan { length; reverse; num_consts; num_carry; jaxpr; unroll } ->
        scan_rule e used_outs ~length ~reverse ~num_consts ~num_carry ~jaxpr
          ~unroll
    | Custom_jvp_call id when !Jaxpr.remat_body id <> None ->
        let body = Option.get (!Jaxpr.remat_body id) in
        let n = List.length e.inputs in
        let body', used_ins =
          dce_closed body used_outs ~instantiate:(List.init n (fun _ -> false))
        in
        ( used_ins,
          if
            (not (List.exists Fun.id used_ins))
            && (not (List.exists Fun.id used_outs))
            && not (closed_effects body')
          then None
          else
            with_prim (Custom_jvp_call (!Jaxpr.remat_rebuild id body')) used_ins
        )
    | Custom_jvp_call id when Core.shard_rule id <> None ->
        shard_map_rule e used_outs id (Option.get (Core.shard_rule id))
    | Custom_jvp_call id when Hashtbl.mem Jaxpr.opaque_dce_calls id ->
        (everything, Some e)
    | (Custom_jvp_call id | Custom_vjp_call id)
      when Core.constraint_rule id <> None ->
        (everything, Some e)
    | Custom_jvp_call id | Custom_vjp_call id -> (
        let avals = eqn_avals e in
        match Core.find_custom_call_jaxpr id avals with
        | None -> (everything, Some e)
        | Some c ->
            let n = List.length e.inputs in
            let c', _ =
              dce_closed c used_outs ~instantiate:(List.init n (fun _ -> true))
            in
            let key = (id, avals, used_outs, Config.x64_enabled ()) in
            if c' == c then (everything, with_prim e.prim everything)
            else
              let id' =
                match Hashtbl.find_opt custom_cache key with
                | Some id' -> id'
                | None ->
                    let id' = Core.register_custom_rule (Core.custom_rule id) in
                    Core.note_custom_call_jaxpr id' avals c';
                    if Hashtbl.mem Jaxpr.barrier_calls id then
                      Hashtbl.replace Jaxpr.barrier_calls id' ();
                    (match Core.custom_names_of id with
                    | Some (name, jvp_name) ->
                        Core.register_custom_names id' ~name ~jvp_name
                    | None -> ());
                    Hashtbl.replace custom_cache key id';
                    id'
              in
              let prim =
                match e.prim with
                | Custom_jvp_call _ -> Custom_jvp_call id'
                | _ -> Custom_vjp_call id'
              in
              (everything, with_prim prim everything))
    | _ -> (everything, Some e)

and shard_map_rule (e : eqn) used_outs id (r : Core.shard_rule) =
  let n = List.length r.sr_in in
  let body', used_ins =
    dce_closed r.sr_body used_outs ~instantiate:(List.init n (fun _ -> false))
  in
  if
    (not (List.exists Fun.id used_ins))
    && (not (List.exists Fun.id used_outs))
    && not (closed_effects body')
  then (used_ins, None)
  else
    let key = (id, eqn_avals e, used_outs, Config.x64_enabled ()) in
    let id' =
      match Hashtbl.find_opt custom_cache key with
      | Some id' -> id'
      | None ->
          let avals = eqn_avals e in
          let primal kept =
            let rec refill keep avals kept =
              match (keep, avals, kept) with
              | [], _, _ -> []
              | true :: ks, _ :: az, x :: xs -> x :: refill ks az xs
              | false :: ks, (a : aval) :: az, xs ->
                  Concrete
                    (Ndarray.of_floats a.dtype a.shape
                       (Array.make (Array.fold_left ( * ) 1 a.shape) 0.))
                  :: refill ks az xs
              | _ -> invalid_arg "Stablehlo.Dce: shard_map inputs"
            in
            let outs =
              match Core.custom_rule id with
              | Core.Jvp_rule { primal; _ } ->
                  primal (refill used_ins avals kept)
              | _ -> invalid_arg "Stablehlo.Dce: shard_map rule"
            in
            keep_where outs used_outs
          in
          let id' =
            Core.register_custom_rule
              (Core.Jvp_rule
                 {
                   primal;
                   jvp =
                     (fun _ _ ->
                       invalid_arg "Stablehlo.Dce: jvp of a pruned shard_map");
                 })
          in
          Core.register_shard_rule id'
            {
              r with
              sr_in = keep_where r.sr_in used_ins;
              sr_out = keep_where r.sr_out used_outs;
              sr_body = body';
            };
          Hashtbl.replace custom_cache key id';
          id'
    in
    ( used_ins,
      Some
        {
          e with
          prim = Custom_jvp_call id';
          inputs = keep_where e.inputs used_ins;
          outs = keep_where e.outs used_outs;
        } )

and scan_rule (e : eqn) used_outputs ~length ~reverse ~num_consts ~num_carry
    ~jaxpr ~unroll =
  let n_in = List.length jaxpr.jaxpr.in_binders - List.length jaxpr.consts in
  let num_xs = n_in - num_consts - num_carry in
  let carry_out, extensive_out =
    match Util.split_list used_outputs [ num_carry ] with
    | [ a; b ] -> (a, b)
    | _ -> invalid_arg "Stablehlo.Dce: scan outputs"
  in
  let falses k = List.init k (fun _ -> false) in
  let rec fixpoint carry_out =
    let used = carry_out @ extensive_out in
    let c', used_inputs =
      dce_closed jaxpr used
        ~instantiate:(falses num_consts @ carry_out @ falses num_xs)
    in
    let carry_in =
      match Util.split_list used_inputs [ num_consts; num_carry ] with
      | [ _; a; _ ] -> a
      | _ -> invalid_arg "Stablehlo.Dce: scan inputs"
    in
    if carry_in = carry_out then (c', used_inputs, used)
    else fixpoint (List.map2 ( || ) carry_out carry_in)
  in
  let c', used_inputs, used = fixpoint carry_out in
  let num_consts', num_carry' =
    match Util.split_list used_inputs [ num_consts; num_carry ] with
    | [ k; a; _ ] ->
        (List.length (List.filter Fun.id k), List.length (List.filter Fun.id a))
    | _ -> invalid_arg "Stablehlo.Dce: scan inputs"
  in
  ( used_inputs,
    Some
      {
        e with
        prim =
          Scan
            {
              length;
              reverse;
              num_consts = num_consts';
              num_carry = num_carry';
              jaxpr = c';
              unroll;
            };
        inputs = keep_where e.inputs used_inputs;
        outs = keep_where e.outs used;
      } )

let prune_unused_inputs (cj : closed_jaxpr) =
  let const_binders, args = split_consts cj in
  let n_consts = List.length const_binders in
  let used_outs = List.map (fun _ -> true) cj.jaxpr.outs in
  let n_all = n_consts + List.length args in
  let pruned, used =
    dce_closed { cj with consts = [] } used_outs
      ~instantiate:(List.init n_all (fun _ -> false))
  in
  let used_consts, used_args =
    match Util.split_list used [ n_consts ] with
    | [ a; b ] -> (a, b)
    | _ -> invalid_arg "Stablehlo.Dce: prune split"
  in
  let kept =
    List.filter_map
      (fun (i, k) -> if k then Some i else None)
      (List.mapi (fun i k -> (i, k)) used_args)
  in
  ({ pruned with consts = keep_where cj.consts used_consts }, kept)

let () =
  Interpreters.Ad.tape_dce :=
    fun c used ->
      fst
        (dce_closed c used
           ~instantiate:(List.map (fun _ -> false) c.jaxpr.in_binders))
