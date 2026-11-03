open Types
module Ad = Interpreters.Ad
module Partial_eval = Interpreters.Partial_eval

type policy = primitive -> aval list -> bool

let policy_names : (policy * string) list ref = ref []

let named name (p : policy) =
  policy_names := (p, name) :: !policy_names;
  p

let function_repr name = "<function " ^ name ^ " at 0xXXXXXXXXXXXX>"

let policy_repr p =
  match List.assq_opt p !policy_names with
  | Some s -> s
  | None -> function_repr "policy"

let frozenset names =
  match names with
  | [] -> "frozenset()"
  | _ ->
      "frozenset({"
      ^ String.concat ", " (List.map (fun n -> "'" ^ n ^ "'") names)
      ^ "})"

let everything_saveable =
  named (function_repr "everything_saveable") (fun _ _ -> true)

let nothing_saveable =
  named (function_repr "nothing_saveable") (fun _ _ -> false)

let dots_saveable =
  named "DotsSaveable(only_if_no_batch_dims=False)" (fun prim _ ->
      match prim with
      | Dot_general _ | Conv_general_dilated _ -> true
      | _ -> false)

let checkpoint_dots = dots_saveable

let dots_with_no_batch_dims_saveable =
  named "DotsSaveable(only_if_no_batch_dims=True)" (fun prim _ ->
      match prim with
      | Dot_general { lhs_batch = [||]; rhs_batch = [||]; _ } -> true
      | _ -> false)

let checkpoint_dots_with_no_batch_dims = dots_with_no_batch_dims_saveable

let save_anything_except_these_names _ =
  named (function_repr "save_anything_except_these_names.<locals>.<lambda>")
    (fun _ _ -> true)

let save_any_names_but_these names =
  named
    ("SaveAnyNamesButThese(names=" ^ frozenset names ^ ")")
    (fun prim _ ->
      match prim with
      | Checkpoint_name name -> not (List.mem name names)
      | _ -> false)

let save_only_these_names names =
  named
    ("SaveOnlyTheseNames(saveable_names=" ^ frozenset names ^ ")")
    (fun prim _ ->
      match prim with Checkpoint_name name -> List.mem name names | _ -> false)

let save_from_both_policies first second =
  named (function_repr "save_from_both_policies.<locals>.policy")
    (fun prim avals ->
      let a = first prim avals and b = second prim avals in
      a || b)

type remat_case = Recompute | Saveable | Offloadable of string * string

let offloads : (policy * (primitive -> aval list -> remat_case)) list ref =
  ref []

let offload_policy name decide =
  let p prim avals = decide prim avals <> Recompute in
  offloads := (p, decide) :: !offloads;
  named (function_repr (name ^ ".<locals>.policy")) p

let remat_case (p : policy) prim avals =
  match List.assq_opt p !offloads with
  | Some decide -> decide prim avals
  | None -> if p prim avals then Saveable else Recompute

let offload_dot_with_no_batch_dims ~offload_src ~offload_dst =
  offload_policy "offload_dot_with_no_batch_dims" (fun prim _ ->
      match prim with
      | Dot_general { lhs_batch = [||]; rhs_batch = [||]; _ } ->
          Offloadable (offload_src, offload_dst)
      | _ -> Recompute)

let save_and_offload_only_these_names ~names_which_can_be_saved
    ~names_which_can_be_offloaded ~offload_src ~offload_dst =
  offload_policy "save_and_offload_only_these_names" (fun prim _ ->
      match prim with
      | Checkpoint_name n when List.mem n names_which_can_be_saved -> Saveable
      | Checkpoint_name n when List.mem n names_which_can_be_offloaded ->
          Offloadable (offload_src, offload_dst)
      | _ -> Recompute)

let checkpoint_name x name = Core.bind1 (Checkpoint_name name) [ x ]

let checkpoint_name_tree tree name =
  Tree_util.tree_map (fun x -> checkpoint_name x name) tree

let is_effectful = function
  | Custom_vjp_call id -> (
      match Core.custom_rule id with Core.Callback_rule _ -> true | _ -> false)
  | _ -> false

let is_debug_callback = function
  | Custom_vjp_call id -> (
      match Core.custom_rule id with
      | Core.Callback_rule { cb_effect = Core.Callback_debug _; _ } -> true
      | _ -> false)
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
        let policy_saves = policy e.prim (List.map atom_aval e.inputs) in
        if is_debug_callback e.prim && not policy_saves then
          staged_eqns := e :: !staged_eqns;
        let saveable = is_effectful e.prim || policy_saves in
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
        (Custom_lin
           {
             rule = linear;
             n_res = List.length residuals;
             out_avals;
             in_zeros = [];
           })
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

type cfg = { policy : policy option; prevent_cse : bool }

let default_cfg = { policy = None; prevent_cse = true }

type remat = { body : closed_jaxpr; differentiated : bool; cfg : cfg }

let remats : (int, remat) Hashtbl.t = Hashtbl.create 16

let remat_view id ~cfg ~differentiated traced =
  Pretty_printer.register_custom_view id "remat2"
    [
      ( "differentiated",
        Pretty_printer.Plain (if differentiated then "True" else "False") );
      ("jaxpr", Pretty_printer.Sub traced);
      ( "policy",
        Pretty_printer.Plain
          (match cfg.policy with None -> "None" | Some p -> policy_repr p) );
      ( "prevent_cse",
        Pretty_printer.Plain (if cfg.prevent_cse then "True" else "False") );
    ];
  id

let rec remat_rule ~cfg ~differentiated (body : closed_jaxpr) =
  let avals = List.map (fun (v : var) -> v.vaval) body.jaxpr.in_binders in
  let avals = List.filteri (fun i _ -> i >= List.length body.consts) avals in
  let by_nonzero : (bool list, closed_jaxpr * bool list) Hashtbl.t =
    Hashtbl.create 4
  in
  let n = List.length avals in
  let build nonzero =
    let tangent_avals =
      List.concat
        (List.map2 (fun nz a -> if nz then [ a ] else []) nonzero avals)
    in
    let out_nz = ref [] in
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
                Ad.jvp (Jaxpr.eval_closed_jaxpr body) primals full
              in
              out_nz := List.map (fun t -> not (Ad_util.is_symbolic_zero t)) to_;
              po @ List.filter (fun t -> not (Ad_util.is_symbolic_zero t)) to_
          | _ -> failwith "checkpoint: arity")
    in
    (traced, !out_nz)
  in
  Core.Jvp_rule
    {
      primal = Jaxpr.eval_closed_jaxpr body;
      jvp =
        (fun primals tangents ->
          let nonzero =
            List.map (fun t -> not (Ad_util.is_symbolic_zero t)) tangents
          in
          let traced, out_nz =
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
            remat_call ~cfg ~differentiated traced (primals @ nz_tangents)
          in
          let n_out = List.length out_nz in
          let primal_out, tangent_out =
            match Util.split_list outs [ n_out ] with
            | [ a; b ] -> (a, b)
            | _ -> failwith "checkpoint: output arity"
          in
          let rest = ref tangent_out in
          ( primal_out,
            List.map2
              (fun nz p ->
                if nz then begin
                  let t = List.hd !rest in
                  rest := List.tl !rest;
                  t
                end
                else Ad_util.symbolic_zero_like_value p)
              out_nz primal_out ));
    }

and remat_id ?(cfg = default_cfg) ~differentiated body =
  let id = Core.register_custom_rule (remat_rule ~cfg ~differentiated body) in
  Hashtbl.replace remats id { body; differentiated; cfg };
  if differentiated && cfg.prevent_cse then
    Hashtbl.replace Jaxpr.barrier_calls id ();
  remat_view id ~cfg ~differentiated body

and remat_call ~cfg ~differentiated body args =
  Core.bind (Custom_jvp_call (remat_id ~cfg ~differentiated body)) args

let eval_subset ?(post = fun _ _ x -> x) (env : (int, value) Hashtbl.t) eqns
    outs =
  let read = function
    | A_var v -> Hashtbl.find env v.vid
    | A_lit nd -> Concrete nd
    | DropVar _ -> failwith "checkpoint: dropvar read"
  in
  List.iter
    (fun (e : eqn) ->
      let ins = List.map read e.inputs in
      let res = Core.bind e.prim ins in
      List.iter2
        (fun (v : var) r -> Hashtbl.replace env v.vid (post e v r))
        e.outs res)
    eqns;
  List.map read outs

let select flags xs =
  List.concat (List.map2 (fun f x -> if f then [ x ] else []) flags xs)

let last_index p xs =
  let r = ref None in
  List.iteri (fun i x -> if p x then r := Some i) xs;
  !r

let same_var (v : var) = function A_var w -> w.vid = v.vid | _ -> false

let named_like (src : closed_jaxpr) jaxpr consts =
  let c = { jid = Core.fresh_id (); jaxpr; consts } in
  Pretty_printer.register_call_name c.jid (Pretty_printer.call_name src.jid);
  Core.inherit_inline ~from:src.jid c.jid;
  c

let pe_custom_cache :
    ( int * bool list * bool list * bool * bool list * bool list,
      (policy
      * (closed_jaxpr
        * closed_jaxpr
        * bool list
        * bool list
        * var list
        * int option list
        * int option list))
      list )
    Hashtbl.t =
  Hashtbl.create 16

let rec pe_custom ?(forward = true) ?ensure_unk ?ensure_inst policy
    (c : closed_jaxpr) unks_in inst_in =
  let n_out = List.length c.jaxpr.outs in
  let none = List.init n_out (fun _ -> false) in
  let ensure_unk = Option.value ensure_unk ~default:none in
  let ensure_inst = Option.value ensure_inst ~default:none in
  let key = (c.jid, unks_in, inst_in, forward, ensure_unk, ensure_inst) in
  let hits = Option.value (Hashtbl.find_opt pe_custom_cache key) ~default:[] in
  match List.find_opt (fun (p, _) -> p == policy) hits with
  | Some (_, r) -> r
  | None ->
      let r =
        pe_custom_uncached ~forward ~ensure_unk ~ensure_inst policy c unks_in
          inst_in
      in
      Hashtbl.replace pe_custom_cache key ((policy, r) :: hits);
      r

and pe_custom_uncached ~forward ~ensure_unk ~ensure_inst policy
    (c : closed_jaxpr) unks_in inst_in =
  let jx = c.jaxpr in
  let const_binders, binders =
    match Util.split_list jx.in_binders [ List.length c.consts ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "checkpoint: binder arity"
  in
  let env = Hashtbl.create 64 in
  let read = function
    | A_var (v : var) -> Hashtbl.find env v.vid
    | A_lit _ | DropVar _ -> (false, true)
  in
  let write flags (v : var) = Hashtbl.replace env v.vid flags in
  let residuals = ref [] and res_set = Hashtbl.create 16 in
  let add_res (v : var) =
    if not (Hashtbl.mem res_set v.vid) then begin
      Hashtbl.replace res_set v.vid ();
      residuals := v :: !residuals
    end
  in
  let ensure inst = function A_var v when not inst -> add_res v | _ -> () in
  List.iter2 (fun v u -> write u v) binders (List.combine unks_in inst_in);
  List.iter (write (false, true)) const_binders;
  let known = ref [] and staged = ref [] in
  List.iter
    (fun (e : eqn) ->
      let flags = List.map read e.inputs in
      let unks = List.map fst flags and insts = List.map snd flags in
      match custom_rule policy unks insts e with
      | Some (k, s, unks_out, inst_out, new_vars) ->
          known := k :: !known;
          staged := s :: !staged;
          List.iter add_res new_vars;
          List.iter2 write (List.combine unks_out inst_out) e.outs
      | None when List.exists Fun.id unks ->
          List.iter2 ensure insts e.inputs;
          staged := e :: !staged;
          List.iter (write (true, true)) e.outs
      | None ->
          known := e :: !known;
          if
            is_effectful e.prim
            || remat_case policy e.prim (List.map atom_aval e.inputs)
               <> Recompute
          then List.iter (write (false, false)) e.outs
          else begin
            List.iter2 ensure insts e.inputs;
            staged := e :: !staged;
            List.iter (write (false, true)) e.outs
          end)
    jx.eqns;
  let out_flags = List.map read jx.outs in
  let unks_out = List.map fst out_flags and inst_out = List.map snd out_flags in
  List.iter2
    (fun (a, inst) ens -> if ens then ensure inst a)
    (List.combine jx.outs inst_out)
    ensure_inst;
  let unks_out = List.map2 ( || ) unks_out ensure_unk in
  let inst_out = List.map2 ( || ) inst_out ensure_inst in
  let residuals = List.rev !residuals in
  let ins_known = select (List.map not unks_in) binders in
  let outs_known = select (List.map not unks_out) jx.outs in
  let in_fwd =
    List.map
      (fun (r : var) ->
        if forward then last_index (fun (b : var) -> b.vid = r.vid) ins_known
        else None)
      residuals
  in
  let out_fwd =
    List.map
      (fun r -> if forward then last_index (same_var r) outs_known else None)
      residuals
  in
  let keep = List.map2 (fun a b -> a = None && b = None) in_fwd out_fwd in
  let known_c =
    named_like c
      {
        in_binders = const_binders @ ins_known;
        eqns = List.rev !known;
        outs = outs_known @ List.map (fun v -> A_var v) (select keep residuals);
      }
      c.consts
  in
  let staged_c =
    named_like c
      {
        in_binders = const_binders @ residuals @ select inst_in binders;
        eqns = List.rev !staged;
        outs = select inst_out jx.outs;
      }
      c.consts
  in
  (known_c, staged_c, unks_out, inst_out, residuals, in_fwd, out_fwd)

and custom_rule policy unks insts (e : eqn) =
  match e.prim with
  | Xla_call inner -> Some (call_rule policy inner unks insts e)
  | Custom_jvp_call id when Hashtbl.mem remats id ->
      Some (remat_custom_rule policy (Hashtbl.find remats id) unks insts e)
  | Scan { length; reverse; num_consts; num_carry; jaxpr; unroll } ->
      Some
        (scan_custom_rule policy ~length ~reverse ~num_consts ~num_carry ~unroll
           jaxpr unks insts e)
  | _ -> None

and scan_custom_rule policy ~length ~reverse ~num_consts ~num_carry ~unroll
    (jaxpr : closed_jaxpr) unks_in inst_in (e : eqn) =
  let n_out = List.length jaxpr.jaxpr.outs in
  let num_ys = n_out - num_carry in
  let split3 l =
    match Util.split_list l [ num_consts; num_carry ] with
    | [ a; b; c ] -> (a, b, c)
    | _ -> failwith "checkpoint: scan arity"
  in
  let const_uk, carry_uk, xs_uk = split3 unks_in in
  let all_inst = List.map (fun _ -> true) unks_in in
  let rec fix carry_uk k =
    let unks = const_uk @ carry_uk @ xs_uk in
    let ((_, _, unks_out, _, _, _, _) as r) =
      pe_custom ~forward:false
        ~ensure_unk:(carry_uk @ List.init num_ys (fun _ -> false))
        ~ensure_inst:(List.init n_out (fun _ -> true))
        policy jaxpr unks all_inst
    in
    let carry_out = List.filteri (fun i _ -> i < num_carry) unks_out in
    if carry_out = carry_uk then (carry_uk, unks, r)
    else if k = 0 then failwith "Fixpoint not reached"
    else fix (List.map2 ( || ) carry_uk carry_out) (k - 1)
  in
  let carry_uk, unks_in, (known_c, staged_c, unks_out, inst_out, residuals, _, _)
      =
    fix carry_uk (List.length carry_uk)
  in
  let num_res = List.length residuals in
  let n_staged_consts = List.length staged_c.consts in
  let s_consts, s_rest =
    match Util.split_list staged_c.jaxpr.in_binders [ n_staged_consts ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "checkpoint: scan staged binders"
  in
  let s_res, s_ins =
    match Util.split_list s_rest [ num_res ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "checkpoint: scan staged binders"
  in
  let new_inst =
    List.concat
      (List.map2
         (fun a inst -> match a with A_var v when not inst -> [ v ] | _ -> [])
         e.inputs inst_in)
  in
  let count_known l = List.length (List.filter not l) in
  let num_const_known = count_known const_uk in
  let n_known_outs = count_known unks_out in
  let k_consts, k_rest =
    match
      Util.split_list known_c.jaxpr.in_binders [ List.length known_c.consts ]
    with
    | [ a; b ] -> (a, b)
    | _ -> failwith "checkpoint: scan known binders"
  in
  ignore k_consts;
  let k_avals = List.map (fun (v : var) -> v.vaval) k_rest in
  let const_avals, loop_avals =
    match Util.split_list k_avals [ num_const_known ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "checkpoint: scan known avals"
  in
  let holder = ref None in
  let hoist =
    Jaxpr.make_jaxpr const_avals (fun consts ->
        let pvals =
          List.map Partial_eval.partial_val_known consts
          @ List.map Partial_eval.partial_val_unknown loop_avals
        in
        let lin, lin_consts, pvals_out =
          Partial_eval.partial_eval_flat_instantiate
            ~instantiate:(fun i -> i < n_known_outs)
            (Jaxpr.eval_closed_jaxpr known_c)
            pvals
        in
        holder := Some (lin, lin_consts, pvals_out);
        let known_vals =
          List.filter_map (fun (pv : partial_val) -> pv.pv_const) pvals_out
        in
        let kept =
          List.filter
            (function
              | Concrete nd -> Array.length (Ndarray.shape nd) > 0 | _ -> true)
            lin_consts
        in
        known_vals @ kept)
  in
  let lin, lin_consts, pvals_out =
    match !holder with Some h -> h | None -> failwith "checkpoint: scan hoist"
  in
  let lit = Hashtbl.create 4 in
  let lin_cbinders, lin_args =
    match Util.split_list lin.in_binders [ List.length lin_consts ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "checkpoint: scan loop binders"
  in
  let kept_cbinders =
    List.concat
      (List.map2
         (fun (b : var) c ->
           match c with
           | Concrete nd when Array.length (Ndarray.shape nd) = 0 ->
               Hashtbl.replace lit b.vid (A_lit nd);
               []
           | _ -> [ b ])
         lin_cbinders lin_consts)
  in
  let subst = function
    | A_var (v : var) as a -> (
        match Hashtbl.find_opt lit v.vid with Some l -> l | None -> a)
    | a -> a
  in
  let loop_c =
    {
      jid = Core.fresh_id ();
      consts = [];
      jaxpr =
        {
          in_binders = kept_cbinders @ lin_args;
          eqns =
            List.map
              (fun (q : eqn) -> { q with inputs = List.map subst q.inputs })
              lin.eqns;
          outs = List.map subst lin.outs;
        };
    }
  in
  let n_lp_consts = List.length kept_cbinders in
  let loop_dep_res =
    List.filteri (fun i _ -> i >= n_known_outs) pvals_out
    |> List.map Partial_eval.is_unknown
  in
  let num_intensive = count_known loop_dep_res in
  let int_bs = select (List.map not loop_dep_res) s_res in
  let ext_bs = select loop_dep_res s_res in
  let staged_moved =
    {
      staged_c with
      jid = Core.fresh_id ();
      jaxpr =
        { staged_c.jaxpr with in_binders = s_consts @ int_bs @ s_ins @ ext_bs };
    }
  in
  let fresh (a : aval) = { vid = Core.fresh_id (); vaval = a } in
  let res_avals = List.map (fun (v : var) -> v.vaval) residuals in
  let intensive_res =
    List.map fresh (select (List.map not loop_dep_res) res_avals)
  in
  let extensive_res =
    List.map
      (fun (a : aval) ->
        fresh { a with shape = Array.append [| length |] a.shape })
      (select loop_dep_res res_avals)
  in
  let ins_known = select (List.map not unks_in) e.inputs in
  let out_binders_known = select (List.map not unks_out) e.outs in
  let num_carry_known = count_known carry_uk in
  let call_c =
    Jaxpr.make_jaxpr (List.map atom_aval ins_known) (fun ins ->
        let consts_known, ins_lp =
          match Util.split_list ins [ num_const_known ] with
          | [ a; b ] -> (a, b)
          | _ -> failwith "checkpoint: scan known call"
        in
        let out_hoist = Jaxpr.eval_closed_jaxpr hoist consts_known in
        let intensive, consts_lp =
          match Util.split_list out_hoist [ num_intensive ] with
          | [ a; b ] -> (a, b)
          | _ -> failwith "checkpoint: scan hoist outputs"
        in
        let out_loop =
          Core.bind
            (Scan
               {
                 length;
                 reverse;
                 num_consts = n_lp_consts;
                 num_carry = num_carry_known;
                 jaxpr = loop_c;
                 unroll;
               })
            (consts_lp @ ins_lp)
        in
        intensive @ out_loop)
  in
  Pretty_printer.register_call_name call_c.jid "closed_call";
  let mk prim inputs outs =
    { prim; inputs; outs; multiple_results = List.length outs > 1 }
  in
  let eqn_known =
    mk (Xla_call call_c) ins_known
      (intensive_res @ out_binders_known @ extensive_res)
  in
  let eqn_staged =
    mk
      (Scan
         {
           length;
           reverse;
           num_consts = num_intensive + num_consts;
           num_carry;
           jaxpr = staged_moved;
           unroll;
         })
      (List.map (fun v -> A_var v) intensive_res
      @ e.inputs
      @ List.map (fun v -> A_var v) extensive_res)
      (select inst_out e.outs)
  in
  ( eqn_known,
    eqn_staged,
    unks_out,
    inst_out,
    new_inst @ intensive_res @ extensive_res )

and remat_custom_rule policy (r : remat) unks_in inst_in (e : eqn) =
  let known_c, staged_c, unks_out, inst_out, residuals, _, _ =
    pe_custom ~forward:false policy r.body unks_in inst_in
  in
  let ins_known = select (List.map not unks_in) e.inputs in
  let out_binders_known = select (List.map not unks_out) e.outs in
  let ins_staged = select inst_in e.inputs in
  let out_binders_staged = select inst_out e.outs in
  let res_binders =
    List.map
      (fun (v : var) -> { vid = Core.fresh_id (); vaval = v.vaval })
      residuals
  in
  let mk prim inputs outs =
    { prim; inputs; outs; multiple_results = List.length outs > 1 }
  in
  let eqn_known =
    mk
      (Custom_jvp_call
         (remat_id ~cfg:r.cfg ~differentiated:r.differentiated known_c))
      ins_known
      (out_binders_known @ res_binders)
  in
  let eqn_staged =
    mk
      (Custom_jvp_call (remat_id ~cfg:r.cfg ~differentiated:true staged_c))
      (List.map (fun v -> A_var v) res_binders @ ins_staged)
      out_binders_staged
  in
  let new_inst =
    List.concat
      (List.map2
         (fun a inst -> match a with A_var v when not inst -> [ v ] | _ -> [])
         e.inputs inst_in)
  in
  (eqn_known, eqn_staged, unks_out, inst_out, new_inst @ res_binders)

and call_rule policy inner unks_in inst_in (e : eqn) =
  let known_c, staged_c, unks_out, inst_out, residuals, in_fwd, out_fwd =
    pe_custom policy inner unks_in inst_in
  in
  let out_binders_known = select (List.map not unks_out) e.outs in
  let ins_known = select (List.map not unks_in) e.inputs in
  let ins_staged = select inst_in e.inputs in
  let out_binders_staged = select inst_out e.outs in
  let res_binders =
    List.concat
      (List.map2
         (fun (r : var) (fi, fo) ->
           if fi = None && fo = None then
             [ { vid = Core.fresh_id (); vaval = r.vaval } ]
           else [])
         residuals
         (List.combine in_fwd out_fwd))
  in
  let rest = ref res_binders in
  let res_vals =
    List.map2
      (fun fi fo ->
        match (fo, fi) with
        | Some i, _ -> A_var (List.nth out_binders_known i)
        | None, Some i -> List.nth ins_known i
        | None, None ->
            let v = List.hd !rest in
            rest := List.tl !rest;
            A_var v)
      in_fwd out_fwd
  in
  let mk prim inputs outs =
    { prim; inputs; outs; multiple_results = List.length outs > 1 }
  in
  let eqn_known =
    mk (Xla_call known_c) ins_known (out_binders_known @ res_binders)
  in
  let eqn_staged =
    mk (Xla_call staged_c) (res_vals @ ins_staged) out_binders_staged
  in
  let new_inst =
    List.concat
      (List.map2
         (fun a inst -> match a with A_var v when not inst -> [ v ] | _ -> [])
         e.inputs inst_in)
  in
  let new_vars =
    new_inst
    @ List.filter_map (function A_var v -> Some v | _ -> None) res_vals
  in
  (eqn_known, eqn_staged, unks_out, inst_out, new_vars)

type split_eqn = Plain of eqn | Split of eqn * eqn * bool list * var list

let split_calls ~case_of policy (jx : jaxpr) const_binders binders known_flags
    pe_unknown =
  let env = Hashtbl.create 64 in
  let read = function
    | A_var (v : var) -> (
        match Hashtbl.find_opt env v.vid with
        | Some f -> f
        | None -> (false, true))
    | A_lit _ | DropVar _ -> (false, true)
  in
  let write flags (v : var) = Hashtbl.replace env v.vid flags in
  List.iter2 (fun (v : var) k -> write (k = None, true) v) binders known_flags;
  List.iter (write (false, true)) const_binders;
  List.map
    (fun (e : eqn) ->
      let flags = List.map read e.inputs in
      let unks = List.map fst flags and insts = List.map snd flags in
      let agrees (_, _, unks_out, _, _) =
        List.for_all2
          (fun (v : var) u -> u = Hashtbl.mem pe_unknown v.vid)
          e.outs unks_out
      in
      let found = custom_rule policy unks insts e in
      let rejected =
        match found with Some r -> not (agrees r) | None -> false
      in
      match if rejected then None else found with
      | Some (k, s, unks_out, inst_out, new_vars) ->
          List.iter2 write (List.combine unks_out inst_out) e.outs;
          let recomputed =
            List.concat
              (List.map2
                 (fun (v : var) (u, i) ->
                   if (not u) && i then [ v.vid ] else [])
                 e.outs
                 (List.combine unks_out inst_out))
          in
          let saved_known =
            List.map (fun (v : var) -> not (List.mem v.vid recomputed)) k.outs
          in
          Split (k, s, saved_known, new_vars)
      | None when rejected || List.exists Fun.id unks ->
          List.iter
            (fun (v : var) -> write (Hashtbl.mem pe_unknown v.vid, true) v)
            e.outs;
          Plain e
      | None ->
          if is_effectful e.prim || case_of e <> Recompute then
            List.iter (write (false, false)) e.outs
          else List.iter (write (false, true)) e.outs;
          Plain e)
    jx.eqns

let remat_partial_eval trace id (tracers : tracer list) =
  match Hashtbl.find_opt remats id with
  | None -> None
  | Some { body; cfg; _ } ->
      let jx = body.jaxpr in
      let n_const = List.length body.consts in
      let const_binders, binders =
        match Util.split_list jx.in_binders [ n_const ] with
        | [ a; b ] -> (a, b)
        | _ -> failwith "checkpoint: binder arity"
      in
      let known_flags =
        List.map
          (fun (t : tracer) ->
            match t.payload with
            | PE { pval = { pv_const = Some c; _ }; _ } -> Some c
            | _ -> None)
          tracers
      in
      let policy = Option.value cfg.policy ~default:nothing_saveable in
      let unknown_vars = Hashtbl.create 64 and saved = Hashtbl.create 64 in
      List.iter2
        (fun (v : var) k ->
          if k = None then Hashtbl.replace unknown_vars v.vid ())
        binders known_flags;
      let var_of = Hashtbl.create 64 in
      let order = ref [] in
      let note vid =
        if Hashtbl.mem saved vid && not (List.mem vid !order) then
          order := vid :: !order
      in
      let pe_unknown = Hashtbl.create 64 in
      ignore
        (Jaxpr.make_jaxpr
           (List.map (fun (v : var) -> v.vaval) binders)
           (fun args ->
             let pvals =
               List.map2
                 (fun k x ->
                   match k with
                   | Some _ -> Partial_eval.partial_val_known x
                   | None -> Partial_eval.partial_val_unknown (Core.get_aval x))
                 known_flags args
             in
             ignore
               (Partial_eval.partial_eval_flat
                  (fun xs ->
                    let env = Hashtbl.create 64 in
                    List.iter2
                      (fun (v : var) nd ->
                        Hashtbl.replace env v.vid (Concrete nd))
                      const_binders body.consts;
                    List.iter2
                      (fun (v : var) x -> Hashtbl.replace env v.vid x)
                      binders xs;
                    eval_subset
                      ~post:(fun _ v r ->
                        if Partial_eval.is_unknown_value r then
                          Hashtbl.replace pe_unknown v.vid ();
                        r)
                      env jx.eqns jx.outs)
                  pvals);
             []));
      let cases = ref [] in
      let case_of (e : eqn) =
        match List.assq_opt e !cases with
        | Some c -> c
        | None ->
            let c = remat_case policy e.prim (List.map atom_aval e.inputs) in
            cases := (e, c) :: !cases;
            c
      in
      let split =
        split_calls ~case_of policy jx const_binders binders known_flags
          pe_unknown
      in
      let plain (e : eqn) =
        List.iter (fun (v : var) -> Hashtbl.replace var_of v.vid v) e.outs;
        let ins = atom_vars e.inputs in
        if List.exists (fun (v : var) -> Hashtbl.mem pe_unknown v.vid) e.outs
        then begin
          List.iter note ins;
          List.iter
            (fun (v : var) ->
              if Hashtbl.mem pe_unknown v.vid then
                Hashtbl.replace unknown_vars v.vid ())
            e.outs;
          true
        end
        else if is_debug_callback e.prim && case_of e = Recompute then begin
          List.iter note ins;
          true
        end
        else if is_effectful e.prim || case_of e <> Recompute then begin
          List.iter (fun (v : var) -> Hashtbl.replace saved v.vid ()) e.outs;
          false
        end
        else begin
          List.iter note ins;
          true
        end
      in
      let candidates =
        List.concat_map
          (function
            | Plain e -> if plain e then [ e ] else []
            | Split (k, s, saved_known, new_vars) ->
                List.iter
                  (fun (v : var) -> Hashtbl.replace var_of v.vid v)
                  (k.outs @ s.outs);
                List.iter2
                  (fun (v : var) keep ->
                    if keep then Hashtbl.replace saved v.vid ())
                  k.outs saved_known;
                List.iter
                  (fun (v : var) ->
                    if Hashtbl.mem var_of v.vid && not (List.mem v.vid !order)
                    then order := v.vid :: !order)
                  new_vars;
                List.iter
                  (fun (v : var) ->
                    if Hashtbl.mem pe_unknown v.vid then
                      Hashtbl.replace unknown_vars v.vid ())
                  s.outs;
                [ s ])
          split
      in
      let known_list =
        List.map (function Plain e -> e | Split (k, _, _, _) -> k) split
      in
      let out_unknown =
        List.map
          (function
            | A_var v -> Hashtbl.mem unknown_vars v.vid
            | A_lit _ | DropVar _ -> false)
          jx.outs
      in
      let pick flag =
        List.filteri (fun i _ -> List.nth out_unknown i = flag) jx.outs
      in
      let known_outs = pick false and unknown_outs = pick true in
      let staged_eqns = live_eqns (atom_vars unknown_outs) candidates in
      let used = Hashtbl.create 64 in
      List.iter
        (fun vid -> Hashtbl.replace used vid ())
        (atom_vars unknown_outs);
      List.iter
        (fun (e : eqn) ->
          List.iter
            (fun vid -> Hashtbl.replace used vid ())
            (atom_vars e.inputs))
        staged_eqns;
      let produced = Hashtbl.create 64 in
      List.iter (fun (v : var) -> Hashtbl.replace produced v.vid ()) binders;
      List.iter
        (fun (e : eqn) ->
          List.iter (fun (v : var) -> Hashtbl.replace produced v.vid ()) e.outs)
        staged_eqns;
      List.iter
        (fun (e : eqn) -> List.iter note (atom_vars e.inputs))
        staged_eqns;
      List.iter
        (fun vid ->
          if
            (not (Hashtbl.mem produced vid))
            && Hashtbl.mem var_of vid
            && not (List.mem vid !order)
          then order := vid :: !order)
        (atom_vars unknown_outs);
      let residuals =
        List.filter_map
          (fun vid ->
            if Hashtbl.mem used vid then Some (Hashtbl.find var_of vid)
            else None)
          (List.rev !order)
      in
      let used_inputs =
        List.filter
          (fun ((v : var), _) -> Hashtbl.mem used v.vid)
          (List.combine binders tracers)
      in
      let staged =
        Jaxpr.make_jaxpr
          (List.map (fun (v : var) -> v.vaval) residuals
          @ List.map (fun ((v : var), _) -> v.vaval) used_inputs)
          (fun ins ->
            let env = Hashtbl.create 64 in
            List.iter2
              (fun (v : var) nd -> Hashtbl.replace env v.vid (Concrete nd))
              const_binders body.consts;
            let res_in, rest =
              match Util.split_list ins [ List.length residuals ] with
              | [ a; b ] -> (a, b)
              | _ -> failwith "checkpoint: staged arity"
            in
            List.iter2
              (fun (v : var) x -> Hashtbl.replace env v.vid x)
              residuals res_in;
            List.iter2
              (fun ((v : var), _) x -> Hashtbl.replace env v.vid x)
              used_inputs rest;
            eval_subset env staged_eqns unknown_outs)
      in
      let staged, staged_args =
        Stablehlo.Dce.dce_closed staged
          (List.map (fun _ -> true) staged.jaxpr.outs)
          ~instantiate:
            (List.init
               (List.length staged.jaxpr.in_binders - List.length staged.consts)
               (fun _ -> false))
      in
      let keep xs flags =
        List.concat (List.map2 (fun x k -> if k then [ x ] else []) xs flags)
      in
      let res_used, inputs_used =
        match Util.split_list staged_args [ List.length residuals ] with
        | [ a; b ] -> (a, b)
        | _ -> failwith "checkpoint: staged arity"
      in
      let residuals = keep residuals res_used in
      let res_atoms = List.map (fun v -> A_var v) residuals in
      let known_eqns =
        let outs = known_outs @ res_atoms in
        let pruned, _ =
          Stablehlo.Dce.dce_closed
            {
              body with
              jid = Core.fresh_id ();
              jaxpr = { jx with outs; eqns = known_list };
            }
            (List.map (fun _ -> true) outs)
            ~instantiate:(List.map (fun _ -> true) binders)
        in
        pruned.jaxpr.eqns
      in
      let known_used = Hashtbl.create 64 in
      List.iter
        (fun (e : eqn) ->
          List.iter
            (fun vid -> Hashtbl.replace known_used vid ())
            (atom_vars e.inputs))
        known_eqns;
      let precision (a : aval) =
        match a.dtype with
        | Dtype.F16 -> Some (5, 10)
        | Dtype.Bf16 -> Some (8, 7)
        | Dtype.F32 | Dtype.Complex64 -> Some (8, 23)
        | Dtype.F64 | Dtype.Complex128 -> Some (11, 52)
        | d -> (
            match Float_format.of_dtype d with
            | Some f -> Some (f.nexp, f.nmant)
            | None -> None)
      in
      let reduced = Hashtbl.create 8 in
      List.iter
        (fun (v : var) ->
          if Hashtbl.mem known_used v.vid then
            match precision v.vaval with
            | Some p -> Hashtbl.replace reduced v.vid p
            | None -> ())
        residuals;
      let post (e : eqn) (v : var) r =
        match (Hashtbl.find_opt reduced v.vid, e.prim) with
        | None, _ -> r
        | Some p, Reduce_precision { exponent_bits; mantissa_bits }
          when p = (exponent_bits, mantissa_bits) ->
            r
        | Some (exponent_bits, mantissa_bits), _ ->
            Core.bind1 (Reduce_precision { exponent_bits; mantissa_bits }) [ r ]
      in
      let pvals =
        List.map2
          (fun (v : var) k ->
            match k with
            | Some c -> Partial_eval.partial_val_known c
            | None -> Partial_eval.partial_val_unknown v.vaval)
          binders known_flags
      in
      let _, _, known_pvals =
        Partial_eval.partial_eval_flat
          (fun args ->
            let env = Hashtbl.create 64 in
            List.iter2
              (fun (v : var) nd -> Hashtbl.replace env v.vid (Concrete nd))
              const_binders body.consts;
            List.iter2
              (fun (v : var) x -> Hashtbl.replace env v.vid x)
              binders args;
            eval_subset ~post env known_eqns (known_outs @ res_atoms))
          pvals
      in
      let known_all =
        List.map
          (fun (pv : partial_val) ->
            match pv.pv_const with
            | Some c -> c
            | None -> failwith "checkpoint: known output became unknown")
          known_pvals
      in
      let known_values, res_values =
        match Util.split_list known_all [ List.length known_outs ] with
        | [ a; b ] -> (a, b)
        | _ -> failwith "checkpoint: residual arity"
      in
      let used_inputs = keep used_inputs inputs_used in
      let staged_id = remat_id ~cfg ~differentiated:true staged in
      let unknown_values =
        Partial_eval.record_unknown trace (Custom_jvp_call staged_id)
          (List.map (Partial_eval.known_tracer trace) res_values
          @ List.map snd used_inputs)
      in
      let k = ref known_values and u = ref unknown_values in
      let take r =
        let x = List.hd !r in
        r := List.tl !r;
        x
      in
      Some (List.map (fun unk -> if unk then take u else take k) out_unknown)

let remat_transpose id cts (primals : Ad.tval list) =
  match Hashtbl.find_opt remats id with
  | None -> None
  | Some { body; cfg; _ } ->
      let nonlinear =
        List.filter_map
          (function Ad.Prim v -> Some v | Ad.Undef _ -> None)
          primals
      in
      let ct_nz = List.map (fun c -> not (Ad_util.is_symbolic_zero c)) cts in
      let nz_cts =
        List.filter (fun c -> not (Ad_util.is_symbolic_zero c)) cts
      in
      let n_nonlin = List.length nonlinear in
      let in_zero = ref [] in
      let out_avals = List.map atom_aval body.jaxpr.outs in
      let transposed =
        Jaxpr.make_jaxpr
          (List.map Core.get_aval nonlinear @ List.map Core.get_aval nz_cts)
          (fun all ->
            match Util.split_list all [ n_nonlin ] with
            | [ ins; cts_in ] ->
                let rest_ins = ref ins and rest_cts = ref cts_in in
                let take r =
                  let x = List.hd !r in
                  r := List.tl !r;
                  x
                in
                let pvals =
                  List.map
                    (function
                      | Ad.Prim _ ->
                          Partial_eval.partial_val_known (take rest_ins)
                      | Ad.Undef a -> Partial_eval.partial_val_unknown a)
                    primals
                in
                let full_cts =
                  List.map2
                    (fun nz a ->
                      if nz then take rest_cts
                      else Ad_util.symbolic_zero_of_aval a)
                    ct_nz out_avals
                in
                let lin, consts, out_pvals =
                  Partial_eval.partial_eval_flat
                    (Jaxpr.eval_closed_jaxpr body)
                    pvals
                in
                let lin_cts =
                  List.filteri
                    (fun i _ -> Partial_eval.is_unknown (List.nth out_pvals i))
                    full_cts
                in
                let undefs =
                  List.filter_map
                    (function
                      | Ad.Undef a -> Some (Ad.Undef a) | Ad.Prim _ -> None)
                    primals
                in
                let in_cts =
                  Ad.eval_jaxpr_transposed lin
                    (List.map (fun c -> Ad.Prim c) consts @ undefs)
                    lin_cts
                in
                in_zero := List.map Ad_util.is_symbolic_zero in_cts;
                List.filter (fun c -> not (Ad_util.is_symbolic_zero c)) in_cts
            | _ -> failwith "checkpoint: arity")
      in
      let outs =
        remat_call ~cfg ~differentiated:true transposed (nonlinear @ nz_cts)
      in
      let rest = ref outs and zeros = ref !in_zero in
      Some
        (List.map
           (function
             | Ad.Prim _ -> None
             | Ad.Undef _ ->
                 let z = List.hd !zeros in
                 zeros := List.tl !zeros;
                 if z then None
                 else begin
                   let x = List.hd !rest in
                   rest := List.tl !rest;
                   Some x
                 end)
           primals)

let remat_vmap id axis_size vals bdims =
  match Hashtbl.find_opt remats id with
  | None -> None
  | Some { body; differentiated; cfg } ->
      let batched, out_batched =
        Interpreters.Batching.batch_jaxpr_axes axis_size body
          (List.map Core.get_aval vals)
          bdims
      in
      let outs = remat_call ~cfg ~differentiated batched vals in
      Some (outs, List.map (fun b -> if b then Some 0 else None) out_batched)

let () =
  (Jaxpr.remat_body :=
     fun id -> Option.map (fun r -> r.body) (Hashtbl.find_opt remats id));
  (Jaxpr.remat_rebuild :=
     fun id body ->
       let { differentiated; cfg; _ } = Hashtbl.find remats id in
       remat_id ~cfg ~differentiated body);
  (let previous = !Interpreters.Batching.custom_call_batch in
   Interpreters.Batching.custom_call_batch :=
     fun id axis_size vals bdims ->
       match remat_vmap id axis_size vals bdims with
       | Some r -> Some r
       | None -> previous id axis_size vals bdims);
  Partial_eval.custom_call_rule := remat_partial_eval;
  Ad.custom_call_transpose := remat_transpose

let checkpoint ?policy ?(prevent_cse = true) f =
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
        match hoisted with
        | [] when traced.consts = [] ->
            let id =
              remat_id ~cfg:{ policy; prevent_cse } ~differentiated:false traced
            in
            Hashtbl.replace closed_rules avals id;
            Core.bind (Custom_jvp_call id) args
        | [] ->
            let consts = List.map (fun nd -> Concrete nd) traced.consts in
            let id =
              remat_id ~cfg:{ policy; prevent_cse } ~differentiated:false
                { traced with consts = [] }
            in
            Core.bind (Custom_jvp_call id) (consts @ args)
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
  let subst = Hashtbl.create 16 in
  List.iter
    (fun (e : eqn) ->
      match (e.prim, e.outs, e.inputs) with
      | Reduce_precision _, [ o ], [ A_var i ] ->
          Hashtbl.replace subst o.vid i.vid
      | _ -> ())
    jx.eqns;
  let wanted = Hashtbl.create 16 in
  List.iter
    (fun vid ->
      Hashtbl.replace wanted
        (Option.value (Hashtbl.find_opt subst vid) ~default:vid)
        ())
    (atom_vars jx.outs);
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
                 Printf.sprintf "from the argument args[%d]" (i - n_const) );
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
                  | Checkpoint_name name ->
                      Printf.sprintf "named '%s' from " name
                  | Xla_call c when Pretty_printer.jax_prim_name e.prim = "jit"
                    ->
                      Printf.sprintf "output of jitted function '%s' from "
                        (Pretty_printer.call_name c.jid)
                  | prim ->
                      "output of "
                      ^ Pretty_printer.jax_prim_name prim
                      ^ " from " )
            else None)
          e.outs)
      jx.eqns
  in
  literals @ from_binders @ from_eqns

let saved_residuals_text f args =
  String.concat ""
    (List.map
       (fun (a, src) -> Pretty_printer.aval_short a ^ " " ^ src ^ "\n")
       (saved_residuals f args))

let print_saved_residuals f args = print_string (saved_residuals_text f args)
