open Types

let partial_val_known v = { pv_aval = Core.get_aval v; pv_const = Some v }
let partial_val_unknown aval = { pv_aval = aval; pv_const = None }
let is_known pv = match pv.pv_const with Some _ -> true | None -> false
let is_unknown pv = match pv.pv_const with Some _ -> false | None -> true

let new_pe_tracer trace (pval : partial_val) (recipe : recipe option) : tracer =
  {
    id = Core.fresh_id ();
    trace;
    aval = pval.pv_aval;
    payload = PE { pval; recipe };
  }

let new_unknown_arg trace aval =
  new_pe_tracer trace (partial_val_unknown aval) (Some LambdaBinding)

let pe_pval (t : tracer) =
  match t.payload with
  | PE { pval; _ } -> pval
  | _ -> invalid_arg "partial_eval: expected a PartialEval tracer"

let as_tracer = function
  | Tracer t -> t
  | Concrete _ -> invalid_arg "partial_eval: expected a PartialEval tracer"
  | Device _ -> invalid_arg "partial_eval: expected a PartialEval tracer"

let pure trace v = Tracer (new_pe_tracer trace (partial_val_known v) None)

let instantiate_const trace (t : tracer) : tracer =
  let pval = pe_pval t in
  match pval.pv_const with
  | None -> t
  | Some c ->
      new_pe_tracer trace (partial_val_unknown t.aval) (Some (ConstRecipe c))

let record_unknown trace prim (tracers : tracer list) : value list =
  let tracers_in = List.map (instantiate_const trace) tracers in
  let avals_in = List.map (fun (t : tracer) -> t.aval) tracers_in in
  let avals_out = Core.rules.abstract_eval prim avals_in in
  let out_tracers =
    List.map
      (fun a -> new_pe_tracer trace (partial_val_unknown a) None)
      avals_out
  in
  let out_refs = List.map (fun t -> ref (Some t)) out_tracers in
  let recipe =
    EqnRecipe
      {
        er_prim = prim;
        er_inputs = tracers_in;
        er_avals_out = avals_out;
        er_out = out_refs;
      }
  in
  List.iter
    (fun t -> match t.payload with PE p -> p.recipe <- Some recipe | _ -> ())
    out_tracers;
  List.map (fun t -> Tracer t) out_tracers

let scan_rule :
    (trace ->
    length:int ->
    reverse:bool ->
    unroll:int ->
    num_consts:int ->
    num_carry:int ->
    closed_jaxpr ->
    tracer list ->
    value list)
    ref =
  ref (fun _ ~length:_ ~reverse:_ ~unroll:_ ~num_consts:_ ~num_carry:_ _ _ ->
      failwith "partial_eval: scan rule not installed")

let jit_rule : (trace -> closed_jaxpr -> tracer list -> value list) ref =
  ref (fun _ _ _ -> failwith "partial_eval: jit rule not installed")

let process_primitive trace prim (args : value list) : value list =
  let tracers = List.map as_tracer args in
  let pvals = List.map pe_pval tracers in
  if List.for_all is_known pvals then
    let consts =
      List.map
        (fun pv ->
          match pv.pv_const with
          | Some c -> c
          | None -> invalid_arg "partial_eval: known pval without const")
        pvals
    in
    Core.bind prim consts
  else
    match prim with
    | Scan { length; reverse; num_consts; num_carry; jaxpr; unroll } ->
        !scan_rule trace ~length ~reverse ~unroll ~num_consts ~num_carry jaxpr
          tracers
    | Xla_call c -> !jit_rule trace c tracers
    | _ -> record_unknown trace prim tracers

let full_lower_pe v =
  match v with
  | Tracer t -> (
      match t.payload with
      | PE { pval; _ } -> (
          match pval.pv_const with Some c -> Core.full_lower c | None -> v)
      | _ -> v)
  | Concrete _ -> v
  | Device _ -> v

let interpreter : Core.interpreter =
  {
    i_pure = pure;
    i_lift = pure;
    i_full_lower = full_lower_pe;
    i_process_primitive = process_primitive;
    i_process_custom_jvp =
      (fun _ ~primal:_ ~jvp:_ _ ->
        failwith "partial_eval: custom_jvp not supported in M1");
    i_process_custom_vjp =
      (fun _ ~primal:_ ~fwd:_ ~bwd:_ _ ->
        failwith "partial_eval: custom_vjp not supported in M1");
  }

let install () = Core.register_interpreter KPE interpreter
let () = install ()

let tracer_parents (t : tracer) : tracer list =
  match t.payload with
  | PE { recipe = Some (EqnRecipe e); _ } -> e.er_inputs
  | _ -> []

let remove_duplicates (nodes : tracer list) : tracer list =
  let seen : (int, unit) Hashtbl.t = Hashtbl.create 16 in
  List.filter
    (fun (t : tracer) ->
      if Hashtbl.mem seen t.id then false
      else (
        Hashtbl.replace seen t.id ();
        true))
    nodes

let toposort (out_nodes : tracer list) (parents : tracer -> tracer list) :
    tracer list =
  match remove_duplicates out_nodes with
  | [] -> []
  | roots ->
      let child_counts : (int, int) Hashtbl.t = Hashtbl.create 64 in
      let rec count = function
        | [] -> ()
        | node :: rest -> (
            match Hashtbl.find_opt child_counts node.id with
            | Some c ->
                Hashtbl.replace child_counts node.id (c + 1);
                count rest
            | None ->
                Hashtbl.replace child_counts node.id 1;
                count (parents node @ rest))
      in
      count roots;
      List.iter
        (fun (n : tracer) ->
          Hashtbl.replace child_counts n.id (Hashtbl.find child_counts n.id - 1))
        roots;
      let sorted = ref [] in
      let childless =
        ref
          (List.rev
             (List.filter
                (fun (n : tracer) -> Hashtbl.find child_counts n.id = 0)
                roots))
      in
      let rec loop () =
        match !childless with
        | [] -> ()
        | node :: rest ->
            childless := rest;
            sorted := node :: !sorted;
            List.iter
              (fun (p : tracer) ->
                let c = Hashtbl.find child_counts p.id in
                if c = 1 then childless := p :: !childless
                else Hashtbl.replace child_counts p.id (c - 1))
              (parents node);
            loop ()
      in
      loop ();
      !sorted

let tracers_to_jaxpr (tracers_in : tracer list) (tracers_out : tracer list) :
    jaxpr * value list =
  let tracer_to_var : (int, var) Hashtbl.t = Hashtbl.create 64 in
  List.iter
    (fun (t : tracer) ->
      Hashtbl.replace tracer_to_var t.id
        { vid = Core.fresh_id (); vaval = t.aval })
    tracers_in;
  let constvar_to_val : (var * value) list ref = ref [] in
  let constid_to_var : (value * var) list ref = ref [] in
  let processed_eqns : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let eqns = ref [] in
  let recipe_key er_out =
    match er_out with
    | r :: _ -> ( match !r with Some (t : tracer) -> t.id | None -> -1)
    | [] -> -1
  in
  let eqn_order (t : tracer) =
    match t.payload with
    | PE { recipe = Some (EqnRecipe e); _ } -> recipe_key e.er_out
    | _ -> -1
  in
  let recipe_to_eqn prim (inputs : tracer list) (avals_out : aval list) er_out =
    let in_atoms =
      List.map
        (fun (t : tracer) -> A_var (Hashtbl.find tracer_to_var t.id))
        inputs
    in
    let out_binders =
      List.map (fun a -> { vid = Core.fresh_id (); vaval = a }) avals_out
    in
    List.iter2
      (fun tref var ->
        match !tref with
        | Some (t : tracer) -> Hashtbl.replace tracer_to_var t.id var
        | None -> ())
      er_out out_binders;
    {
      prim;
      inputs = in_atoms;
      outs = out_binders;
      multiple_results = List.length out_binders > 1;
    }
  in
  List.iter
    (fun (t : tracer) ->
      match t.payload with
      | PE { recipe = Some LambdaBinding; _ } -> ()
      | PE { recipe = Some (ConstRecipe v); _ } ->
          let var =
            match List.find_opt (fun (v', _) -> v' == v) !constid_to_var with
            | Some (_, var) -> var
            | None ->
                let var = { vid = Core.fresh_id (); vaval = t.aval } in
                constvar_to_val := (var, v) :: !constvar_to_val;
                constid_to_var := (v, var) :: !constid_to_var;
                var
          in
          Hashtbl.replace tracer_to_var t.id var
      | PE { recipe = Some (EqnRecipe e); _ } ->
          let key = recipe_key e.er_out in
          if not (Hashtbl.mem processed_eqns key) then (
            Hashtbl.replace processed_eqns key ();
            eqns :=
              recipe_to_eqn e.er_prim e.er_inputs e.er_avals_out e.er_out
              :: !eqns)
      | PE { recipe = None; _ } ->
          invalid_arg "tracers_to_jaxpr: tracer without recipe"
      | _ -> invalid_arg "tracers_to_jaxpr: expected a PartialEval tracer")
    (List.stable_sort
       (fun (x : tracer) (y : tracer) -> compare (eqn_order x) (eqn_order y))
       (toposort (tracers_in @ tracers_out) tracer_parents));
  let constvals = List.rev !constvar_to_val in
  let constvars = List.map fst constvals in
  let in_binders =
    constvars
    @ List.map (fun (t : tracer) -> Hashtbl.find tracer_to_var t.id) tracers_in
  in
  let out_atoms =
    List.map
      (fun (t : tracer) -> A_var (Hashtbl.find tracer_to_var t.id))
      tracers_out
  in
  let jx = { in_binders; eqns = List.rev !eqns; outs = out_atoms } in
  Jaxpr.typecheck_jaxpr jx;
  (jx, List.map snd constvals)

let partial_eval_flat (f : value list -> value list)
    (pvals_in : partial_val list) : jaxpr * value list * partial_val list =
  Core.with_new_main KPE GNone (fun main ->
      let tracers_in =
        List.map (fun pv -> new_pe_tracer main pv (Some LambdaBinding)) pvals_in
      in
      let outs = f (List.map (fun t -> Tracer t) tracers_in) in
      let tracers_out =
        List.map (fun v -> as_tracer (Core.full_raise main v)) outs
      in
      let pvals_out = List.map pe_pval tracers_out in
      let unk_in = List.filter (fun t -> is_unknown (pe_pval t)) tracers_in in
      let unk_out = List.filter (fun t -> is_unknown (pe_pval t)) tracers_out in
      let jaxpr, consts = tracers_to_jaxpr unk_in unk_out in
      (jaxpr, consts, pvals_out))

let mapped_leading (a : aval) =
  { a with shape = Array.sub a.shape 1 (Array.length a.shape - 1) }

let split_at l n =
  match Util.split_list l [ n ] with
  | [ a; b ] -> (a, b)
  | _ -> failwith "partial_eval: scan arity"

let select flags l = List.filteri (fun i _ -> List.nth flags i) l

let scan_partial_eval trace ~length ~reverse ~unroll ~num_consts ~num_carry
    (body : closed_jaxpr) (tracers : tracer list) : value list =
  let consts, carry, xs =
    match Util.split_list tracers [ num_consts; num_carry ] with
    | [ k; c; x ] -> (k, c, x)
    | _ -> failwith "partial_eval: scan arity"
  in
  let unk (t : tracer) = is_unknown (pe_pval t) in
  let const_unk = List.map unk consts in
  let xs_unk = List.map unk xs in
  let aval_of (t : tracer) = t.aval in
  let const_avals = List.map aval_of consts in
  let carry_avals = List.map aval_of carry in
  let x_avals = List.map (fun (t : tracer) -> mapped_leading t.aval) xs in
  let n_ys = List.length body.jaxpr.outs - num_carry in
  let split_known flags l = (select (List.map not flags) l, select flags l) in
  let partial carry_unk known_args =
    let rest = ref known_args in
    let next aval unknown =
      if unknown then partial_val_unknown aval
      else
        match !rest with
        | v :: tl ->
            rest := tl;
            partial_val_known v
        | [] -> failwith "partial_eval: scan known arity"
    in
    let const_pv = List.map2 next const_avals const_unk in
    let carry_pv = List.map2 next carry_avals carry_unk in
    let x_pv = List.map2 next x_avals xs_unk in
    partial_eval_flat
      (fun args -> Jaxpr.eval_closed_jaxpr body args)
      (const_pv @ carry_pv @ x_pv)
  in
  let known_avals carry_unk =
    fst (split_known const_unk const_avals)
    @ fst (split_known carry_unk carry_avals)
    @ fst (split_known xs_unk x_avals)
  in
  let rec fixpoint carry_unk =
    let observed = ref carry_unk in
    let _ =
      Jaxpr.make_jaxpr (known_avals carry_unk) (fun known_args ->
          let _, _, pvals_out = partial carry_unk known_args in
          let carry_out, _ = split_at pvals_out num_carry in
          observed :=
            List.map2 (fun u p -> u || is_unknown p) carry_unk carry_out;
          [])
    in
    if !observed = carry_unk then carry_unk else fixpoint !observed
  in
  let carry_unk = fixpoint (List.map unk carry) in
  let n_known_consts = List.length (List.filter not const_unk) in
  let n_known_carry = List.length (List.filter not carry_unk) in
  let same_tracer a b =
    match (a, b) with Tracer x, Tracer y -> x.id = y.id | _ -> a == b
  in
  let unknown_jaxpr = ref None in
  let residual_avals = ref [] in
  let residual_fwd = ref [] in
  let known_body =
    Jaxpr.make_jaxpr (known_avals carry_unk) (fun known_args ->
        let jx, residual_consts, pvals_out = partial carry_unk known_args in
        let carry_out, ys_out = split_at pvals_out num_carry in
        let known_value p =
          match p.pv_const with
          | Some c -> c
          | None -> failwith "partial_eval: expected a known scan output"
        in
        let forced =
          List.concat
            (List.map2
               (fun u p -> if u && is_known p then [ known_value p ] else [])
               carry_unk carry_out)
        in
        let known_carry =
          List.filteri (fun i _ -> not (List.nth carry_unk i)) carry_out
        in
        let known_ys = List.filter is_known ys_out in
        let residuals = residual_consts @ forced in
        residual_avals := List.map Core.get_aval residuals;
        let fwd_index r =
          let rec find i = function
            | [] -> None
            | v :: tl ->
                let in_carry =
                  i >= n_known_consts && i < n_known_consts + n_known_carry
                in
                if (not in_carry) && same_tracer v r then Some i
                else find (i + 1) tl
          in
          find 0 known_args
        in
        residual_fwd := List.map fwd_index residuals;
        unknown_jaxpr :=
          Some (jx, List.length residual_consts, carry_out, ys_out);
        let kept =
          List.concat
            (List.map2
               (fun f r -> if f = None then [ r ] else [])
               !residual_fwd residuals)
        in
        List.map known_value known_carry @ List.map known_value known_ys @ kept)
  in
  let jx, n_res_consts, carry_out_pv, ys_out_pv =
    match !unknown_jaxpr with
    | Some r -> r
    | None -> failwith "partial_eval: scan body was not traced"
  in
  let value_of (t : tracer) =
    match (pe_pval t).pv_const with
    | Some c -> c
    | None -> failwith "partial_eval: expected a known scan input"
  in
  let known_consts, unk_consts = split_known const_unk consts in
  let known_carry, unk_carry = split_known carry_unk carry in
  let known_xs, unk_xs = split_known xs_unk xs in
  let known_ins =
    List.map value_of known_consts
    @ List.map value_of known_carry
    @ List.map value_of known_xs
  in
  let known_results =
    Core.bind
      (Scan
         {
           length;
           reverse;
           num_consts = n_known_consts;
           num_carry = n_known_carry;
           jaxpr = known_body;
           unroll;
         })
      known_ins
  in
  let known_carry_out, r1 = split_at known_results n_known_carry in
  let n_known_ys = List.length (List.filter is_known ys_out_pv) in
  let known_ys_out, stacked_res = split_at r1 n_known_ys in
  let residual_vals =
    let rest = ref stacked_res in
    List.map
      (function
        | Some i -> List.nth known_ins i
        | None -> (
            match !rest with
            | v :: tl ->
                rest := tl;
                v
            | [] -> failwith "partial_eval: scan residual arity"))
      !residual_fwd
  in
  let is_intensive =
    List.map
      (function Some i -> i < n_known_consts + n_known_carry | None -> false)
      !residual_fwd
  in
  let int_res, ext_res =
    let i, e = List.partition fst (List.combine is_intensive residual_vals) in
    (List.map snd i, List.map snd e)
  in
  let int_avals, ext_avals =
    let i, e = List.partition fst (List.combine is_intensive !residual_avals) in
    (List.map snd i, List.map snd e)
  in
  let unk_const_avals = List.map aval_of unk_consts in
  let unk_carry_avals = List.map aval_of unk_carry in
  let unk_x_avals =
    List.map (fun (t : tracer) -> mapped_leading t.aval) unk_xs
  in
  let n_int = List.length int_res in
  let n_unk_const = List.length unk_consts in
  let n_unk_carry = List.length unk_carry in
  let n_unk_x = List.length unk_xs in
  let unknown_body =
    Jaxpr.make_jaxpr
      (int_avals @ unk_const_avals @ unk_carry_avals @ unk_x_avals @ ext_avals)
      (fun args ->
        let int_args, r = split_at args n_int in
        let k, r = split_at r n_unk_const in
        let c, r = split_at r n_unk_carry in
        let x, ext_args = split_at r n_unk_x in
        let residuals =
          let ints = ref int_args and exts = ref ext_args in
          List.map
            (fun intensive ->
              let source = if intensive then ints else exts in
              match !source with
              | v :: tl ->
                  source := tl;
                  v
              | [] -> failwith "partial_eval: scan residual arity")
            is_intensive
        in
        let const_res, forced_res = split_at residuals n_res_consts in
        let inner = Jaxpr.eval_jaxpr jx (const_res @ k @ c @ x) in
        let rest = ref inner in
        let forced = ref forced_res in
        let take_inner () =
          match !rest with
          | v :: tl ->
              rest := tl;
              v
          | [] -> failwith "partial_eval: scan unknown output arity"
        in
        let carry_out =
          List.concat
            (List.map2
               (fun u p ->
                 if not u then []
                 else if is_unknown p then [ take_inner () ]
                 else
                   match !forced with
                   | v :: tl ->
                       forced := tl;
                       [ v ]
                   | [] -> failwith "partial_eval: scan forced arity")
               carry_unk carry_out_pv)
        in
        let ys =
          List.filter_map
            (fun p -> if is_unknown p then Some (take_inner ()) else None)
            ys_out_pv
        in
        carry_out @ ys)
  in
  let raise_known v = as_tracer (Core.full_raise trace v) in
  let unknown_results =
    record_unknown trace
      (Scan
         {
           length;
           reverse;
           num_consts = n_int + n_unk_const;
           num_carry = n_unk_carry;
           jaxpr = unknown_body;
           unroll;
         })
      (List.map raise_known int_res
      @ unk_consts @ unk_carry @ unk_xs
      @ List.map raise_known ext_res)
  in
  let unk_carry_out, unk_ys_out = split_at unknown_results n_unk_carry in
  let weave flags known unknown =
    let k = ref known and u = ref unknown in
    List.map
      (fun is_unknown ->
        let source = if is_unknown then u else k in
        match !source with
        | v :: tl ->
            source := tl;
            v
        | [] -> failwith "partial_eval: scan output arity")
      flags
  in
  let carry_results = weave carry_unk known_carry_out unk_carry_out in
  let ys_results =
    weave (List.map is_unknown ys_out_pv) known_ys_out unk_ys_out
  in
  ignore n_ys;
  carry_results @ ys_results

let unknown_zero trace (a : aval) =
  let lit =
    as_tracer
      (Core.full_raise trace
         (Concrete (Ndarray.of_floats a.dtype [||] [| 0.0 |])))
  in
  if Array.length a.shape = 0 then Tracer (instantiate_const trace lit)
  else
    List.hd
      (record_unknown trace
         (Broadcast_in_dim { shape = a.shape; dims = [||] })
         [ lit ])

let () = scan_rule := scan_partial_eval

let same_value a b =
  match (a, b) with Tracer x, Tracer y -> x.id = y.id | _ -> a == b

let jit_partial_eval trace (c : closed_jaxpr) (tracers : tracer list) :
    value list =
  let name = Pretty_printer.call_name c.jid in
  let known_flags = List.map (fun t -> is_known (pe_pval t)) tracers in
  let known_tracers, unknown_tracers =
    let k = List.filter (fun t -> is_known (pe_pval t)) tracers in
    let u = List.filter (fun t -> is_unknown (pe_pval t)) tracers in
    (k, u)
  in
  let known_avals = List.map (fun (t : tracer) -> t.aval) known_tracers in
  let holder = ref None in
  let known_closed =
    Jaxpr.make_jaxpr known_avals (fun known_args ->
        let rest = ref known_args in
        let pvals =
          List.map2
            (fun (t : tracer) k ->
              if k then
                begin match !rest with
                | v :: tl ->
                    rest := tl;
                    partial_val_known v
                | [] -> failwith "partial_eval: jit known arity"
                end
              else partial_val_unknown t.aval)
            tracers known_flags
        in
        let jx, consts, pvals_out =
          partial_eval_flat (fun a -> Jaxpr.eval_closed_jaxpr c a) pvals
        in
        let index_of v =
          let rec go i = function
            | [] -> None
            | a :: tl -> if same_value a v then Some i else go (i + 1) tl
          in
          go 0 known_args
        in
        let plan =
          List.map
            (fun v ->
              match v with
              | Concrete nd when Array.length (Ndarray.shape nd) = 0 ->
                  `Literal nd
              | _ -> (
                  match index_of v with Some i -> `Input i | None -> `Output v))
            consts
        in
        let known_outs = List.filter_map (fun pv -> pv.pv_const) pvals_out in
        let residual_outs =
          List.filter_map
            (function `Output v -> Some v | `Input _ | `Literal _ -> None)
            plan
        in
        let n_binders = List.length plan in
        let const_binders, arg_binders =
          match Util.split_list jx.in_binders [ n_binders ] with
          | [ a; b ] -> (a, b)
          | _ -> failwith "partial_eval: jit binders"
        in
        holder := Some (jx, plan, const_binders, arg_binders, pvals_out);
        known_outs @ residual_outs)
  in
  let jx, plan, const_binders, arg_binders, pvals_out =
    match !holder with
    | Some h -> h
    | None -> failwith "partial_eval: jit was not traced"
  in
  let known_inputs =
    List.map
      (fun t ->
        match (pe_pval t).pv_const with
        | Some v -> v
        | None -> failwith "partial_eval: known jit input")
      known_tracers
  in
  Pretty_printer.register_call_name known_closed.jid name;
  Core.inherit_inline ~from:c.jid known_closed.jid;
  let known_results = Core.bind (Xla_call known_closed) known_inputs in
  let n_known_outs = List.length (List.filter is_known pvals_out) in
  let known_out_vals, residual_vals =
    match Util.split_list known_results [ n_known_outs ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "partial_eval: jit outputs"
  in
  let residual_rest = ref residual_vals in
  let literal_subst : (int, atom) Hashtbl.t = Hashtbl.create 4 in
  let kept =
    List.concat
      (List.map2
         (fun (v : var) entry ->
           match entry with
           | `Literal nd ->
               Hashtbl.replace literal_subst v.vid (A_lit nd);
               []
           | `Input i -> [ (v, List.nth known_inputs i) ]
           | `Output _ -> (
               match !residual_rest with
               | r :: tl ->
                   residual_rest := tl;
                   [ (v, r) ]
               | [] -> failwith "partial_eval: jit residual arity"))
         const_binders plan)
  in
  let subst = function
    | A_var (v : var) as a -> (
        match Hashtbl.find_opt literal_subst v.vid with
        | Some l -> l
        | None -> a)
    | a -> a
  in
  let unknown_jx =
    {
      in_binders = arg_binders @ List.map fst kept;
      eqns =
        List.map
          (fun (e : eqn) -> { e with inputs = List.map subst e.inputs })
          jx.eqns;
      outs = List.map subst jx.outs;
    }
  in
  let unknown_closed =
    { jid = Core.fresh_id (); jaxpr = unknown_jx; consts = [] }
  in
  Pretty_printer.register_call_name unknown_closed.jid name;
  Core.inherit_inline ~from:c.jid unknown_closed.jid;
  let residual_tracers =
    List.map (fun (_, r) -> as_tracer (Core.full_raise trace r)) kept
  in
  let unknown_results =
    record_unknown trace (Xla_call unknown_closed)
      (unknown_tracers @ residual_tracers)
  in
  let k = ref known_out_vals and u = ref unknown_results in
  List.map
    (fun pv ->
      let source = if is_known pv then k else u in
      match !source with
      | v :: tl ->
          source := tl;
          v
      | [] -> failwith "partial_eval: jit merge arity")
    pvals_out

let () = jit_rule := jit_partial_eval
