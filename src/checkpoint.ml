open Types
module Ad = Interpreters.Ad

type policy = primitive -> aval list -> bool

let everything_saveable _ _ = true
let nothing_saveable _ _ = false

let dots_saveable prim _ =
  match prim with Dot_general _ | Conv_general_dilated _ -> true | _ -> false

let checkpoint_dots = dots_saveable

let dots_with_no_batch_dims_saveable prim _ =
  match prim with
  | Dot_general { lhs_batch = [||]; rhs_batch = [||]; _ } -> true
  | _ -> false

let checkpoint_dots_with_no_batch_dims = dots_with_no_batch_dims_saveable
let save_anything_except_these_names _ _ _ = true

let save_any_names_but_these names prim _ =
  match prim with
  | Checkpoint_name name -> not (List.mem name names)
  | _ -> false

let save_only_these_names names prim _ =
  match prim with Checkpoint_name name -> List.mem name names | _ -> false

let save_from_both_policies first second prim avals =
  let a = first prim avals and b = second prim avals in
  a || b

let checkpoint_name x name = Core.bind1 (Checkpoint_name name) [ x ]

let checkpoint_name_tree tree name =
  Tree_util.tree_map (fun x -> checkpoint_name x name) tree

let is_effectful = function
  | Custom_vjp_call id -> (
      match Core.custom_rule id with Core.Callback_rule _ -> true | _ -> false)
  | _ -> false

let atom_aval = function
  | A_var v -> v.vaval
  | A_lit nd ->
      {
        shape = Ndarray.shape nd;
        dtype = Ndarray.dtype nd;
        weak_type = Ndarray.weak nd;
      }
  | DropVar a -> a

let atom_vars atoms =
  List.filter_map (function A_var v -> Some v.vid | _ -> None) atoms

let live_eqns roots eqns =
  let live = Hashtbl.create 64 in
  List.iter (fun vid -> Hashtbl.replace live vid ()) roots;
  List.rev
    (List.filter
       (fun (e : eqn) ->
         let keep =
           is_effectful e.prim
           || List.exists (fun (v : var) -> Hashtbl.mem live v.vid) e.outs
         in
         if keep then
           List.iter
             (fun vid -> Hashtbl.replace live vid ())
             (atom_vars e.inputs);
         keep)
       (List.rev eqns))

type staging = {
  known : jaxpr;
  staged : jaxpr;
  consts : value list;
  used_consts : value list;
  used_primals : int list;
  tangent_avals : aval list;
  placeholders : value list;
}

let partition policy (cj : closed_jaxpr) ~n_primal ~n_out ~n_tangent =
  let jx = cj.jaxpr in
  let consts = List.map (fun c -> Concrete c) cj.consts in
  let const_binders, rest =
    match Util.split_list jx.in_binders [ List.length consts ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "checkpoint: binder arity"
  in
  let primal_binders, tangent_binders =
    match Util.split_list rest [ n_primal ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "checkpoint: binder arity"
  in
  assert (List.length tangent_binders = n_tangent);
  let unknown = Hashtbl.create 64 in
  List.iter (fun (v : var) -> Hashtbl.replace unknown v.vid ()) tangent_binders;
  let saved_producer = Hashtbl.create 16 in
  let known_eqns = ref [] and staged_eqns = ref [] in
  List.iter
    (fun (e : eqn) ->
      let depends = List.exists (Hashtbl.mem unknown) (atom_vars e.inputs) in
      if depends then begin
        staged_eqns := e :: !staged_eqns;
        List.iter (fun (v : var) -> Hashtbl.replace unknown v.vid ()) e.outs
      end
      else begin
        known_eqns := e :: !known_eqns;
        let saveable =
          is_effectful e.prim || policy e.prim (List.map atom_aval e.inputs)
        in
        if saveable then
          List.iter
            (fun (v : var) -> Hashtbl.replace saved_producer v.vid v)
            e.outs
        else staged_eqns := e :: !staged_eqns
      end)
    jx.eqns;
  let primal_outs, tangent_outs =
    match Util.split_list jx.outs [ n_out ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "checkpoint: output arity"
  in
  let staged_outs =
    List.map
      (function
        | A_var v when Hashtbl.mem unknown v.vid -> A_var v
        | atom -> (
            match Ad_util.zeros_like_aval (atom_aval atom) with
            | Concrete nd -> A_lit nd
            | _ -> failwith "checkpoint: zero output"))
      tangent_outs
  in
  let staged_live = live_eqns (atom_vars staged_outs) (List.rev !staged_eqns) in
  let defined = Hashtbl.create 64 in
  let used = Hashtbl.create 64 in
  List.iter
    (fun (e : eqn) ->
      List.iter
        (fun vid ->
          if not (Hashtbl.mem defined vid) then Hashtbl.replace used vid ())
        (atom_vars e.inputs);
      List.iter (fun (v : var) -> Hashtbl.replace defined v.vid ()) e.outs)
    staged_live;
  let saved_vars =
    List.concat_map
      (fun (e : eqn) ->
        List.filter
          (fun (v : var) ->
            Hashtbl.mem saved_producer v.vid && Hashtbl.mem used v.vid)
          e.outs)
      (List.rev !known_eqns)
  in
  let used_primals =
    List.filteri
      (fun _ i -> Hashtbl.mem used (List.nth primal_binders i).vid)
      (List.init n_primal Fun.id)
  in
  let used_primal_binders =
    List.map (fun i -> List.nth primal_binders i) used_primals
  in
  let used_consts_with_binders =
    List.filter
      (fun ((v : var), _) -> Hashtbl.mem used v.vid)
      (List.combine const_binders consts)
  in
  let known_roots =
    atom_vars primal_outs @ List.map (fun (v : var) -> v.vid) saved_vars
  in
  let known =
    {
      in_binders = jx.in_binders;
      eqns = live_eqns known_roots (List.rev !known_eqns);
      outs = primal_outs @ List.map (fun v -> A_var v) saved_vars;
    }
  in
  let staged =
    {
      in_binders =
        saved_vars
        @ List.map fst used_consts_with_binders
        @ used_primal_binders @ tangent_binders;
      eqns = staged_live;
      outs = staged_outs;
    }
  in
  let tangent_avals = List.map (fun (v : var) -> v.vaval) tangent_binders in
  {
    known;
    staged;
    consts;
    used_consts = List.map snd used_consts_with_binders;
    used_primals;
    tangent_avals;
    placeholders = List.map Ad_util.symbolic_zero_of_aval tangent_avals;
  }

let staged_rule policy (primal_jaxpr : closed_jaxpr) (avals : aval list) =
  let n = List.length avals in
  let by_nonzero : (bool list, staging * bool list ref * int) Hashtbl.t =
    Hashtbl.create 4
  in
  let build nonzero =
    let tangent_avals =
      List.concat
        (List.map2 (fun nz a -> if nz then [ a ] else []) nonzero avals)
    in
    let out_zero = ref [] in
    let n_out = ref 0 in
    let traced =
      Jaxpr.make_jaxpr (avals @ tangent_avals) (fun all ->
          match Util.split_list all [ n ] with
          | [ primals; tangents ] ->
              let rest = ref tangents in
              let full =
                List.map2
                  (fun nz (a : aval) ->
                    if nz then begin
                      let t = List.hd !rest in
                      rest := List.tl !rest;
                      t
                    end
                    else Ad_util.symbolic_zero_of_aval a)
                  nonzero avals
              in
              let po, to_ =
                Ad.jvp (Jaxpr.eval_closed_jaxpr primal_jaxpr) primals full
              in
              out_zero := List.map Ad_util.is_symbolic_zero to_;
              n_out := List.length po;
              po @ to_
          | _ -> failwith "checkpoint: arity")
    in
    let staging =
      partition policy traced ~n_primal:n ~n_out:!n_out
        ~n_tangent:(List.length tangent_avals)
    in
    let linear =
      Core.register_custom_rule
        (Core.Linear_rule
           {
             forward =
               (fun res ts -> Jaxpr.eval_jaxpr staging.staged (res @ ts));
             transpose =
               (fun res cts ->
                 Ad.eval_jaxpr_transposed staging.staged
                   (List.map (fun v -> Ad.Prim v) res
                   @ List.map (fun a -> Ad.Undef a) staging.tangent_avals)
                   cts);
             res_differentiable = true;
           })
    in
    (staging, out_zero, linear)
  in
  let jvp primals tangents =
    let nonzero =
      List.map (fun t -> not (Ad_util.is_symbolic_zero t)) tangents
    in
    let staging, out_zero, linear =
      match Hashtbl.find_opt by_nonzero nonzero with
      | Some entry -> entry
      | None ->
          let entry = build nonzero in
          Hashtbl.replace by_nonzero nonzero entry;
          entry
    in
    let nz_tangents =
      List.filter (fun t -> not (Ad_util.is_symbolic_zero t)) tangents
    in
    let outs =
      Jaxpr.eval_jaxpr staging.known
        (staging.consts @ primals @ staging.placeholders)
    in
    let n_out = List.length !out_zero in
    let primal_out, saved =
      match Util.split_list outs [ n_out ] with
      | [ a; b ] -> (a, b)
      | _ -> failwith "checkpoint: output arity"
    in
    let residuals =
      saved @ staging.used_consts
      @ List.map (fun i -> List.nth primals i) staging.used_primals
    in
    let out_avals = List.map Core.get_aval primal_out in
    let tangent_out =
      Core.bind
        (Custom_lin { rule = linear; n_res = List.length residuals; out_avals })
        (residuals @ nz_tangents)
    in
    ( primal_out,
      List.map2
        (fun zero t -> if zero then Ad_util.symbolic_zero_like_value t else t)
        !out_zero tangent_out )
  in
  Core.Jvp_rule { primal = Jaxpr.eval_closed_jaxpr primal_jaxpr; jvp }

let refs_forbidden =
  "a Ref closed over by a function under checkpoint is not supported"

let legacy_rule f =
  let linear =
    Core.register_custom_rule
      (Core.Linear_rule
         {
           forward = (fun res ts -> snd (Ad.jvp f res ts));
           transpose =
             (fun res cts ->
               let _, pull = Ad.vjp f res in
               pull cts);
           res_differentiable = true;
         })
  in
  Core.Jvp_rule
    {
      primal = f;
      jvp =
        (fun primals tangents ->
          let outs = f primals in
          let out_avals = List.map Core.get_aval outs in
          ( outs,
            Core.bind
              (Custom_lin
                 { rule = linear; n_res = List.length primals; out_avals })
              (primals @ tangents) ));
    }

let checkpoint ?policy f =
  let legacy = lazy (Core.register_custom_rule (legacy_rule f)) in
  let closed_rules : (aval list, int) Hashtbl.t = Hashtbl.create 4 in
  fun args ->
    let avals = List.map Core.get_aval args in
    match Hashtbl.find_opt closed_rules avals with
    | Some id -> Core.bind (Custom_jvp_call id) args
    | None -> (
        let traced, hoisted =
          Jaxpr.make_jaxpr_forbidding_refs ~message:refs_forbidden avals f
        in
        let chosen = Option.value policy ~default:nothing_saveable in
        match (hoisted, policy) with
        | [], None ->
            let id = Lazy.force legacy in
            Hashtbl.replace closed_rules avals id;
            Core.bind (Custom_jvp_call id) args
        | [], Some _ ->
            let id =
              Core.register_custom_rule (staged_rule chosen traced avals)
            in
            Hashtbl.replace closed_rules avals id;
            Core.bind (Custom_jvp_call id) args
        | _ ->
            let hoisted_avals = List.map Core.get_aval hoisted in
            let id =
              Core.register_custom_rule
                (staged_rule chosen traced (avals @ hoisted_avals))
            in
            Core.bind (Custom_jvp_call id) (args @ hoisted))

let saved_residuals f args =
  let avals = List.map Core.get_aval args in
  let traced =
    Jaxpr.make_jaxpr avals (fun inputs ->
        let _, residuals, _ = Ad.vjp_staged f inputs in
        residuals)
  in
  let jx = traced.jaxpr in
  let n_const = List.length traced.consts in
  let wanted = Hashtbl.create 16 in
  List.iter (fun vid -> Hashtbl.replace wanted vid ()) (atom_vars jx.outs);
  let literals =
    List.filter_map
      (function
        | A_lit nd -> Some (atom_aval (A_lit nd), "from a literal") | _ -> None)
      jx.outs
  in
  let seen_consts = ref [] in
  let fresh_const i =
    let nd = List.nth traced.consts i in
    if List.memq nd !seen_consts then false
    else begin
      seen_consts := nd :: !seen_consts;
      true
    end
  in
  let from_binders =
    List.concat
      (List.mapi
         (fun i (v : var) ->
           if not (Hashtbl.mem wanted v.vid) then []
           else if i < n_const then
             if fresh_const i then [ (v.vaval, "from a constant") ] else []
           else
             [
               ( v.vaval,
                 Printf.sprintf "from the argument at flattened index %d"
                   (i - n_const) );
             ])
         jx.in_binders)
  in
  let from_eqns =
    List.concat_map
      (fun (e : eqn) ->
        List.filter_map
          (fun (v : var) ->
            if Hashtbl.mem wanted v.vid then
              Some
                ( v.vaval,
                  match e.prim with
                  | Checkpoint_name name -> Printf.sprintf "named '%s'" name
                  | prim -> "output of " ^ Pretty_printer.primitive_name prim )
            else None)
          e.outs)
      jx.eqns
  in
  literals @ from_binders @ from_eqns
