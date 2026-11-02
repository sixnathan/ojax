open Types

let aval_of_ndarray nd =
  {
    shape = Ndarray.shape nd;
    dtype = Ndarray.dtype nd;
    weak_type = Ndarray.weak nd;
  }

let var_names (jx : jaxpr) =
  let names = Hashtbl.create 16 in
  let counter = ref 0 in
  let bind (v : var) =
    if not (Hashtbl.mem names v.vid) then (
      Hashtbl.replace names v.vid (Pretty_printer.encode_var !counter);
      incr counter)
  in
  let bind_atom = function A_var v -> bind v | A_lit _ | DropVar _ -> () in
  List.iter bind jx.in_binders;
  List.iter
    (fun (e : eqn) ->
      List.iter bind e.outs;
      List.iter bind_atom e.inputs)
    jx.eqns;
  List.iter bind_atom jx.outs;
  names

let jaxpr_type_error names (v : var) message =
  let name =
    Option.value (Hashtbl.find_opt names v.vid) ~default:(string_of_int v.vid)
  in
  raise (Errors.Jaxpr_type_error (Printf.sprintf message name))

let atom_aval names env = function
  | A_var v ->
      if Hashtbl.mem env v.vid then v.vaval
      else jaxpr_type_error names v "Variable '%s' not defined"
  | A_lit nd -> aval_of_ndarray nd
  | DropVar a -> a

let avals_equal (a : aval) (b : aval) =
  a.shape = b.shape && a.dtype = b.dtype && a.weak_type = b.weak_type

let typecheck_jaxpr (jx : jaxpr) : unit =
  let env : (int, unit) Hashtbl.t = Hashtbl.create 16 in
  let names = var_names jx in
  let bind (v : var) =
    if Hashtbl.mem env v.vid then
      jaxpr_type_error names v "Variable '%s' already bound";
    Hashtbl.replace env v.vid ()
  in
  List.iter bind jx.in_binders;
  List.iter
    (fun (e : eqn) ->
      let in_types = List.map (atom_aval names env) e.inputs in
      let out_types = Core.rules.abstract_eval e.prim in_types in
      List.iter2
        (fun (b : var) t ->
          let same =
            match e.prim with
            | Convert_element_type _ ->
                avals_equal { t with weak_type = b.vaval.weak_type } b.vaval
            | _ -> avals_equal t b.vaval
          in
          if not same then
            raise
              (Errors.Jaxpr_type_error
                 (Printf.sprintf
                    "Value for variable '%s' inconsistently typed as %s for \
                     let-binder of type %s"
                    (Option.value
                       (Hashtbl.find_opt names b.vid)
                       ~default:(string_of_int b.vid))
                    (Pretty_printer.aval_short t)
                    (Pretty_printer.aval_short b.vaval))))
        e.outs out_types;
      List.iter bind e.outs)
    jx.eqns;
  List.iter (fun a -> ignore (atom_aval names env a)) jx.outs

let in_eqn_eval = ref false
let eval_transparent = ref false

let eval_jaxpr_body (jx : jaxpr) (args : value list) : value list =
  let env : (int, value) Hashtbl.t = Hashtbl.create 16 in
  let read = function
    | A_var v -> Hashtbl.find env v.vid
    | A_lit nd -> Concrete nd
    | DropVar _ -> invalid_arg "eval_jaxpr: DropVar in read position"
  in
  let write (v : var) x = Hashtbl.replace env v.vid x in
  List.iter2 write jx.in_binders args;
  List.iter
    (fun (e : eqn) ->
      let in_vals = List.map read e.inputs in
      let prev = !in_eqn_eval in
      if not !eval_transparent then in_eqn_eval := true;
      let outs =
        Fun.protect
          ~finally:(fun () -> in_eqn_eval := prev)
          (fun () ->
            match (e.prim, e.outs) with
            | Convert_element_type _, [ v ] when v.vaval.weak_type ->
                let saved = !Core.weak_convert in
                Core.weak_convert := true;
                Fun.protect
                  ~finally:(fun () -> Core.weak_convert := saved)
                  (fun () ->
                    List.map
                      (function
                        | Concrete nd -> Concrete (Ndarray.as_weak nd)
                        | out -> out)
                      (Core.bind e.prim in_vals))
            | _ -> Core.bind e.prim in_vals)
      in
      List.iter2 write e.outs outs)
    jx.eqns;
  List.map read jx.outs

let eval_jaxpr jx args = Key_reuse.suspended (fun () -> eval_jaxpr_body jx args)
let jaxpr_as_fun (jx : jaxpr) args = eval_jaxpr jx args

let eval_closed_jaxpr (cj : closed_jaxpr) (args : value list) : value list =
  eval_jaxpr cj.jaxpr (List.map (fun c -> Concrete c) cj.consts @ args)

let builder_of trace =
  match trace.global_data with
  | GBuilder b -> b
  | _ -> invalid_arg "jaxpr: expected a JaxprBuilder in global_data"

let new_tracer builder trace aval =
  let t = { id = Core.fresh_id (); trace; aval; payload = Jaxpr () } in
  builder.jb_tracers <- t :: builder.jb_tracers;
  t

let add_var builder (t : tracer) =
  let v = { vid = Core.fresh_id (); vaval = t.aval } in
  builder.jb_tracer_to_var <- (t.id, v) :: builder.jb_tracer_to_var;
  v

let getvar builder (t : tracer) =
  match List.assoc_opt t.id builder.jb_tracer_to_var with
  | Some v -> v
  | None -> invalid_arg "jaxpr: tracer has no variable"

let add_const builder (t : tracer) v =
  let var = add_var builder t in
  builder.jb_constvals <- (var, v) :: builder.jb_constvals;
  var

let new_arg builder trace aval =
  let t = new_tracer builder trace aval in
  let v = add_var builder t in
  builder.jb_invars <- builder.jb_invars @ [ v ];
  t

let default_debug =
  { func_src_info = "<unnamed function>"; traced_for = "jit"; arg_names = None }

let pending_debug = ref None

let with_trace_debug debug f =
  pending_debug := Some debug;
  Fun.protect ~finally:(fun () -> pending_debug := None) f

let take_debug () =
  match !pending_debug with
  | Some debug ->
      pending_debug := None;
      debug
  | None -> default_debug

let scalar_constant builder (var : var) =
  List.find_map
    (fun ((c : var), value) ->
      match value with
      | Concrete nd when c.vid = var.vid && Ndarray.shape nd = [||] -> Some nd
      | _ -> None)
    builder.jb_constvals

let is_constant builder (var : var) =
  List.exists (fun ((c : var), _) -> c.vid = var.vid) builder.jb_constvals

let progenitors builder (t : tracer) =
  let var = getvar builder t in
  let active = Hashtbl.create 16 in
  Hashtbl.replace active var.vid ();
  List.iter
    (fun (e : eqn) ->
      let produced =
        List.filter (fun (o : var) -> Hashtbl.mem active o.vid) e.outs
      in
      if produced <> [] then begin
        List.iter (fun (o : var) -> Hashtbl.remove active o.vid) produced;
        List.iter
          (function
            | A_var (v : var) -> Hashtbl.replace active v.vid ()
            | A_lit _ | DropVar _ -> ())
          e.inputs
      end)
    builder.jb_eqns;
  let positions =
    List.concat
      (List.mapi
         (fun i (v : var) -> if Hashtbl.mem active v.vid then [ i ] else [])
         builder.jb_invars)
  in
  let const_eqn (e : eqn) =
    List.exists
      (function
        | A_var (v : var) ->
            scalar_constant builder v <> None
            || (is_constant builder v && Hashtbl.mem active v.vid)
        | A_lit _ -> true
        | DropVar _ -> false)
      e.inputs
  in
  (positions, List.filter const_eqn (List.rev builder.jb_eqns))

let eqn_with_literals builder (e : eqn) =
  {
    e with
    inputs =
      List.map
        (function
          | A_var (v : var) as atom -> (
              match scalar_constant builder v with
              | Some nd -> A_lit nd
              | None -> atom)
          | atom -> atom)
        e.inputs;
  }

let origin_msg v =
  match v with
  | Tracer ({ payload = Jaxpr (); _ } as t) -> (
      let builder = builder_of t.trace in
      let dbg = builder.jb_debug in
      let positions, eqns = progenitors builder t in
      let origin =
        Printf.sprintf
          "The error occurred while tracing the function %s for %s. "
          dbg.func_src_info dbg.traced_for
      in
      let name i =
        match dbg.arg_names with
        | None -> Some "unknown"
        | Some names -> List.nth_opt names i
      in
      let names = List.map name positions in
      if List.mem None names then ""
      else
        let names = List.filter_map Fun.id names in
        let rec take n = function
          | x :: tl when n > 0 -> x :: take (n - 1) tl
          | _ -> []
        in
        match names with
        | _ :: _ ->
            let arg_info =
              match List.rev names with
              | [ only ] -> "the argument " ^ only
              | [ b; a ] -> Printf.sprintf "the arguments %s and %s" a b
              | last :: rest ->
                  Printf.sprintf "the arguments %s, and %s"
                    (String.concat ", " (List.rev rest))
                    last
              | [] -> ""
            in
            "\n" ^ origin
            ^ "This concrete value was not available in Python because it \
               depends on the value"
            ^ (if List.length names > 1 then "s" else "")
            ^ " of " ^ arg_info ^ "."
        | [] when eqns <> [] ->
            let lines =
              List.map
                (fun e ->
                  Printf.sprintf "  operation %s\n    from line %s"
                    (Pretty_printer.eqn_to_jax_string
                       (eqn_with_literals builder e))
                    "<unknown>")
                (take 5 eqns)
            in
            "\n" ^ origin
            ^ "This value became a tracer due to JAX operations on these \
               lines:\n\n" ^ String.concat "\n\n" lines
            ^
            if List.length eqns > 5 then
              "\n\n(Additional originating lines are not shown.)"
            else ""
        | [] -> "\n" ^ origin)
  | _ -> ""

let as_tracer = function
  | Tracer t -> t
  | Concrete _ -> invalid_arg "jaxpr: expected a JaxprTracer"
  | Device _ -> invalid_arg "jaxpr: expected a JaxprTracer"

let plain_const_tracer trace v =
  let builder = builder_of trace in
  let t = new_tracer builder trace (Core.get_aval v) in
  let _ = add_const builder t v in
  Tracer t

let stage_constant v =
  match v with
  | Tracer _ -> v
  | _ ->
      if Core.tracing_dynamically () then
        let main = Core.find_top_trace [] in
        if main.kind = KJaxpr then plain_const_tracer main v else v
      else v

let shared_tracer builder nd =
  List.find_map
    (fun ((var : var), value) ->
      match value with
      | Concrete other when other == nd ->
          List.find_map
            (fun (tid, (v : var)) ->
              if v.vid = var.vid then
                List.find_opt
                  (fun (t : tracer) -> t.id = tid)
                  builder.jb_tracers
              else None)
            builder.jb_tracer_to_var
      | _ -> None)
    builder.jb_constvals

let get_or_make_const_tracer trace v =
  match v with
  | Concrete nd -> (
      match shared_tracer (builder_of trace) nd with
      | Some t -> Tracer t
      | None -> plain_const_tracer trace v)
  | _ -> plain_const_tracer trace v

let trace_custom_call : (primitive -> aval list -> closed_jaxpr option) ref =
  ref (fun _ _ -> None)

let eval_lifted : (closed_jaxpr -> value list -> value list) ref =
  ref (fun _ _ -> invalid_arg "Jaxpr.eval_lifted: not installed")

let hoist_custom_consts prim (c : closed_jaxpr) args =
  match (prim, Core.custom_rule_opt prim) with
  | Custom_jvp_call id, Some (Core.Jvp_rule { jvp; _ }) when c.consts <> [] ->
      let n = List.length c.consts in
      let const_values = List.map (fun nd -> Concrete nd) c.consts in
      let lifted = { c with consts = [] } in
      let drop l = List.filteri (fun i _ -> i >= n) l in
      let rule =
        Core.Jvp_rule
          {
            primal = (fun all -> !eval_lifted lifted all);
            jvp = (fun primals tangents -> jvp (drop primals) (drop tangents));
          }
      in
      let id' = Core.register_custom_rule rule in
      (match Core.custom_names_of id with
      | Some (name, jvp_name) -> Core.register_custom_names id' ~name ~jvp_name
      | None -> ());
      Core.note_custom_num_consts id' n;
      ( Custom_jvp_call id',
        const_values @ args,
        lifted,
        fun avals -> Core.note_custom_call_jaxpr id' avals lifted )
  | _ -> (prim, args, c, fun _ -> ())

let lowering = ref false

let process_primitive trace prim args =
  let builder = builder_of trace in
  let avals_of l = List.map (fun (t : tracer) -> t.aval) l in
  let avals_in = avals_of (List.map as_tracer args) in
  let custom_jaxpr = !trace_custom_call prim avals_in in
  let prim, args, custom_jaxpr =
    match custom_jaxpr with
    | None -> (prim, args, None)
    | Some c ->
        let prim, args, c, note = hoist_custom_consts prim c args in
        let args =
          List.map
            (function
              | Concrete _ as v -> get_or_make_const_tracer trace v | v -> v)
            args
        in
        note (List.map Core.get_aval args);
        (prim, args, Some c)
  in
  let tracers = List.map as_tracer args in
  let avals_in = avals_of tracers in
  match (prim, tracers) with
  | Broadcast_in_dim { shape; _ }, [ t ]
    when t.aval.shape = shape && not !lowering ->
      [ Tracer t ]
  | _ ->
      let avals_out = Core.rules.abstract_eval prim avals_in in
      let avals_out =
        match prim with
        | Convert_element_type _ when !Core.weak_convert ->
            List.map (fun (a : aval) -> { a with weak_type = true }) avals_out
        | _ -> avals_out
      in
      let out_tracers = List.map (new_tracer builder trace) avals_out in
      let inputs = List.map (fun t -> A_var (getvar builder t)) tracers in
      let outvars = List.map (add_var builder) out_tracers in
      (match (custom_jaxpr, outvars) with
      | Some c, (first : var) :: _ -> Core.note_custom_eqn_jaxpr first.vid c
      | _ -> ());
      builder.jb_eqns <-
        {
          prim;
          inputs;
          outs = outvars;
          multiple_results = List.length outvars > 1;
        }
        :: builder.jb_eqns;
      List.map (fun t -> Tracer t) out_tracers

let interpreter : Core.interpreter =
  {
    i_pure = get_or_make_const_tracer;
    i_lift = get_or_make_const_tracer;
    i_full_lower = (fun v -> v);
    i_process_primitive = process_primitive;
    i_process_custom_jvp =
      (fun _ ~primal:_ ~jvp:_ _ ->
        failwith "jaxpr: custom_jvp not supported in M1");
    i_process_custom_vjp =
      (fun _ ~primal:_ ~fwd:_ ~bwd:_ _ ->
        failwith "jaxpr: custom_vjp not supported in M1");
  }

let install () = Core.register_interpreter KJaxpr interpreter
let () = install ()
let is_scalar_value v = Array.length (Core.get_aval v).shape = 0

let inline_literals (jx : jaxpr) (consts : value list) : jaxpr * Ndarray.t list
    =
  let n = List.length consts in
  let const_binders, other_binders =
    match Util.split_list jx.in_binders [ n ] with
    | [ cb; ob ] -> (cb, ob)
    | _ -> invalid_arg "inline_literals: split"
  in
  let scalars = List.map is_scalar_value consts in
  let new_const_binders, lit_binders =
    Util.partition_list scalars const_binders
  in
  let new_consts, lit_vals = Util.partition_list scalars consts in
  let literals : (int, atom) Hashtbl.t = Hashtbl.create 16 in
  List.iter2
    (fun (b : var) v ->
      match v with
      | Concrete nd -> Hashtbl.replace literals b.vid (A_lit nd)
      | Tracer _ -> invalid_arg "inline_literals: non-concrete literal"
      | Device _ -> invalid_arg "inline_literals: non-concrete literal")
    lit_binders lit_vals;
  let subst = function
    | A_var v as a -> (
        match Hashtbl.find_opt literals v.vid with Some l -> l | None -> a)
    | a -> a
  in
  let new_eqns =
    List.map
      (fun (e : eqn) -> { e with inputs = List.map subst e.inputs })
      jx.eqns
  in
  let new_outs = List.map subst jx.outs in
  let new_jaxpr =
    {
      in_binders = new_const_binders @ other_binders;
      eqns = new_eqns;
      outs = new_outs;
    }
  in
  let const_nds =
    List.map
      (function
        | Concrete nd -> nd
        | Tracer _ -> invalid_arg "inline_literals: non-concrete const"
        | Device _ -> invalid_arg "inline_literals: non-concrete const")
      new_consts
  in
  (new_jaxpr, const_nds)

let same_hoisted a b =
  match (a, b) with Tracer x, Tracer y -> x.id = y.id | _ -> a == b

let dedup_hoisted (hoisted : (var * value) list) =
  let kept = ref [] in
  let subst : (int, var) Hashtbl.t = Hashtbl.create 8 in
  List.iter
    (fun ((var : var), v) ->
      match List.find_opt (fun (_, v') -> same_hoisted v v') !kept with
      | Some (first, _) -> Hashtbl.replace subst var.vid first
      | None -> kept := (var, v) :: !kept)
    hoisted;
  (List.rev !kept, subst)

let substitute_vars subst (jx : jaxpr) =
  let on_atom = function
    | A_var v as a -> (
        match Hashtbl.find_opt subst v.vid with
        | Some first -> A_var first
        | None -> a)
    | a -> a
  in
  {
    jx with
    eqns =
      List.map
        (fun (e : eqn) -> { e with inputs = List.map on_atom e.inputs })
        jx.eqns;
    outs = List.map on_atom jx.outs;
  }

let build builder (in_tracers : tracer list) (out_tracers : tracer list) :
    jaxpr * Ndarray.t list * value list =
  let constvals = List.rev builder.jb_constvals in
  let concrete, hoisted =
    List.partition
      (fun (_, v) -> match v with Concrete _ -> true | _ -> false)
      constvals
  in
  let hoisted, subst = dedup_hoisted hoisted in
  let in_binders =
    List.map fst concrete
    @ List.map (getvar builder) in_tracers
    @ List.map fst hoisted
  in
  let outs = List.map (fun t -> A_var (getvar builder t)) out_tracers in
  let eqns = List.rev builder.jb_eqns in
  let jx = substitute_vars subst { in_binders; eqns; outs } in
  typecheck_jaxpr jx;
  let jx, consts = inline_literals jx (List.map snd concrete) in
  typecheck_jaxpr jx;
  (jx, consts, List.map snd hoisted)

let trace_with_refs mode_of (in_avals : aval list)
    (f : value list -> value list) :
    closed_jaxpr * value list * Ref_state.cell list =
  let builder =
    {
      jb_eqns = [];
      jb_tracer_to_var = [];
      jb_const_tracers = [];
      jb_constvals = [];
      jb_tracers = [];
      jb_invars = [];
      jb_debug = take_debug ();
    }
  in
  Core.with_new_main KJaxpr (GBuilder builder) (fun main ->
      Core.new_dynamic main (fun () ->
          let in_tracers =
            List.map (fun a -> new_arg builder main a) in_avals
          in
          let lifted_tracers = ref [] in
          let new_lifted aval =
            let t = new_arg builder main aval in
            lifted_tracers := t :: !lifted_tracers;
            Tracer t
          in
          Ref_state.with_scope (mode_of new_lifted) (fun scope ->
              let outs = f (List.map (fun t -> Tracer t) in_tracers) in
              let cells = Ref_state.lifted scope in
              let out_tracers =
                List.map
                  (fun v -> as_tracer (Core.full_raise main v))
                  (outs @ Ref_state.contents cells)
              in
              let jx, consts, hoisted =
                build builder
                  (in_tracers @ List.rev !lifted_tracers)
                  out_tracers
              in
              ({ jid = Core.fresh_id (); jaxpr = jx; consts }, hoisted, cells))))

let make_jaxpr_hoisted in_avals f =
  let closed, hoisted, _ =
    trace_with_refs (fun _ -> Ref_state.Bare) in_avals f
  in
  (closed, hoisted)

let make_jaxpr_stateful in_avals f =
  trace_with_refs (fun lift -> Ref_state.Lift lift) in_avals f

let make_jaxpr_forbidding_refs ~message in_avals f =
  let closed, hoisted, _ =
    trace_with_refs (fun _ -> Ref_state.Forbid message) in_avals f
  in
  (closed, hoisted)

let make_jaxpr (in_avals : aval list) (f : value list -> value list) :
    closed_jaxpr =
  match make_jaxpr_hoisted in_avals f with
  | closed, [] -> closed
  | _ -> invalid_arg "make_jaxpr: function closed over a tracer"

let make_jaxpr_for_lowering in_avals f =
  let saved = !lowering in
  lowering := true;
  Fun.protect
    ~finally:(fun () -> lowering := saved)
    (fun () -> make_jaxpr in_avals f)

let lift_while ~cond_nconsts ~body_nconsts (cond : closed_jaxpr)
    (body : closed_jaxpr) (avals : aval list) =
  let split args =
    match Util.split_list args [ cond_nconsts; body_nconsts ] with
    | [ c; b; r ] -> (c, b, r)
    | _ -> invalid_arg "lift_while: arity"
  in
  let pred_of_cond args =
    let c, _, r = split args in
    match eval_closed_jaxpr cond (c @ r) with
    | [ p ] -> p
    | _ -> invalid_arg "lift_while: predicate arity"
  in
  let pred_rank =
    let cond_out =
      match cond.jaxpr.outs with
      | [ A_var v ] -> v.vaval
      | [ DropVar a ] -> a
      | [ A_lit nd ] ->
          {
            shape = Ndarray.shape nd;
            dtype = Ndarray.dtype nd;
            weak_type = false;
          }
      | _ -> invalid_arg "lift_while: predicate arity"
    in
    Array.length cond_out.shape
  in
  let cond' =
    make_jaxpr_for_lowering avals (fun args ->
        let p = pred_of_cond args in
        if pred_rank = 0 then [ p ]
        else Core.bind (Reduce_or (Array.init pred_rank Fun.id)) [ p ])
  in
  let body' =
    make_jaxpr_for_lowering avals (fun args ->
        let c, b, r = split args in
        let next = eval_closed_jaxpr body (b @ r) in
        if pred_rank = 0 then c @ b @ next
        else
          let p = pred_of_cond args in
          let select n o =
            let shape = (Core.get_aval n).shape in
            let spread =
              Core.bind1
                (Broadcast_in_dim { shape; dims = Array.init pred_rank Fun.id })
                [ p ]
            in
            Core.bind1 Select_n [ spread; o; n ]
          in
          c @ b @ List.map2 select next r)
  in
  (cond', body')

let custom_call_jaxpr id avals =
  match Core.find_custom_call_jaxpr id avals with
  | Some c -> Some c
  | None -> (
      match Core.custom_rule id with
      | Core.Jvp_rule { primal; _ } | Core.Vjp_rule { primal; _ } ->
          let c = make_jaxpr avals primal in
          Core.note_custom_call_jaxpr id avals c;
          Some c
      | Core.Linear_rule _ | Core.Callback_rule _ -> None)

let () =
  trace_custom_call :=
    fun prim avals ->
      match prim with
      | Custom_jvp_call id | Custom_vjp_call id -> (
          match Core.find_custom_call_jaxpr id avals with
          | None -> custom_call_jaxpr id avals
          | Some _ -> (
              match Core.custom_rule id with
              | Core.Jvp_rule { primal; _ } | Core.Vjp_rule { primal; _ } ->
                  Some (make_jaxpr avals primal)
              | Core.Linear_rule _ | Core.Callback_rule _ -> None))
      | _ -> None

let () = eval_lifted := eval_closed_jaxpr
let () = Core.call_inline := fun c args -> eval_closed_jaxpr c args
let () = Core.call_direct := fun c args -> eval_closed_jaxpr c args

let () =
  Pretty_printer.custom_num_consts :=
    fun (e : eqn) ->
      match e.prim with
      | Custom_jvp_call id -> Core.custom_num_consts_of id
      | _ -> None

let () =
  Pretty_printer.custom_call :=
    fun id vid avals ->
      match Core.custom_names_of id with
      | Some (name, jvp_name) -> (
          match Option.bind vid Core.find_custom_eqn_jaxpr with
          | Some c -> Some (name, jvp_name, c)
          | None ->
              Option.map
                (fun c -> (name, jvp_name, c))
                (custom_call_jaxpr id avals))
      | None -> None

let has_side_effect = function
  | Ffi_call { ffi_has_side_effect; _ } -> ffi_has_side_effect
  | Dce_sink _ | Create_token | After_all -> true
  | _ -> false

let rec eqn_has_effect (e : eqn) =
  match e.prim with
  | Xla_call c -> List.exists eqn_has_effect c.jaxpr.eqns
  | p -> has_side_effect p

let dce_cache : (int * bool list, closed_jaxpr * bool list) Hashtbl.t =
  Hashtbl.create 64

let rec dce_closed (c : closed_jaxpr) used_outs =
  let key = (c.jid, used_outs) in
  match Hashtbl.find_opt dce_cache key with
  | Some r -> r
  | None ->
      let r = dce_closed_uncached c used_outs in
      Hashtbl.replace dce_cache key r;
      r

and dce_closed_uncached (c : closed_jaxpr) used_outs =
  let jx = c.jaxpr in
  let used = Hashtbl.create 64 in
  let note = function
    | A_var (v : var) -> Hashtbl.replace used v.vid ()
    | A_lit _ | DropVar _ -> ()
  in
  List.iter2 (fun a keep -> if keep then note a) jx.outs used_outs;
  let is_used (v : var) = Hashtbl.mem used v.vid in
  let kept =
    List.fold_left
      (fun acc (e : eqn) ->
        match e.prim with
        | Custom_jvp_call id when e.outs <> [] && List.for_all is_used e.outs ->
            let first = List.hd e.outs in
            let body =
              match Core.find_custom_eqn_jaxpr first.vid with
              | Some body -> Some body
              | None ->
                  custom_call_jaxpr id (List.map Pretty_printer.atom_type e.inputs)
            in
            Option.iter
              (fun (body : closed_jaxpr) ->
                let pruned, _ =
                  dce_closed body (List.map (fun _ -> true) body.jaxpr.outs)
                in
                let n_consts = List.length body.consts in
                let consts =
                  List.filteri (fun i _ -> i < n_consts) pruned.jaxpr.in_binders
                in
                let args =
                  List.filteri (fun i _ -> i >= n_consts) body.jaxpr.in_binders
                in
                Core.note_custom_eqn_jaxpr first.vid
                  {
                    pruned with
                    jaxpr = { pruned.jaxpr with in_binders = consts @ args };
                  })
              body;
            List.iter note e.inputs;
            e :: acc
        | Xla_call inner when not (eqn_has_effect e) ->
            let outs_used = List.map is_used e.outs in
            if not (List.exists Fun.id outs_used) then acc
            else begin
              let pruned, ins_used = dce_closed inner outs_used in
              let inputs =
                List.concat
                  (List.map2
                     (fun a keep -> if keep then [ a ] else [])
                     e.inputs ins_used)
              in
              List.iter note inputs;
              {
                prim = Xla_call pruned;
                inputs;
                outs = List.filter is_used e.outs;
                multiple_results = List.length (List.filter is_used e.outs) > 1;
              }
              :: acc
            end
        | _ ->
            if eqn_has_effect e || e.outs = [] || List.exists is_used e.outs
            then begin
              List.iter note e.inputs;
              e :: acc
            end
            else acc)
      [] (List.rev jx.eqns)
  in
  let n_consts = List.length c.consts in
  let const_binders, arg_binders =
    match Util.split_list jx.in_binders [ n_consts ] with
    | [ cb; ab ] -> (cb, ab)
    | _ -> invalid_arg "Stablehlo.Emit: binder split"
  in
  let kept_args = List.filter is_used arg_binders in
  let outs =
    List.concat
      (List.map2 (fun a keep -> if keep then [ a ] else []) jx.outs used_outs)
  in
  let pruned =
    {
      c with
      jid = Core.fresh_id ();
      jaxpr = { in_binders = const_binders @ kept_args; eqns = kept; outs };
    }
  in
  Pretty_printer.register_call_name pruned.jid (Pretty_printer.call_name c.jid);
  Core.inherit_inline ~from:c.jid pruned.jid;
  (pruned, List.map is_used arg_binders)

let dce_keep_args (c : closed_jaxpr) =
  let n = List.length c.consts in
  let args = List.filteri (fun i _ -> i >= n) c.jaxpr.in_binders in
  let pruned, _ = dce_closed c (List.map (fun _ -> true) c.jaxpr.outs) in
  let consts_binders =
    List.filteri (fun i _ -> i < n) pruned.jaxpr.in_binders
  in
  {
    pruned with
    jaxpr = { pruned.jaxpr with in_binders = consts_binders @ args };
  }

let call_cache : (string, closed_jaxpr) Hashtbl.t = Hashtbl.create 64

let call_key name key (avals : aval list) =
  String.concat "|"
    (name :: key
    :: string_of_bool (Config.value Config.enable_x64)
    :: List.map
         (fun (a : aval) ->
           Dtype.short_name a.dtype
           ^ Pretty_printer.aval_short a
           ^ string_of_bool a.weak_type)
         avals)

let eager_jit :
    (string -> string -> (value list -> value list) -> value list -> value list)
    ref =
  ref (fun _ _ f args -> f args)

let known_copies : (int, closed_jaxpr) Hashtbl.t = Hashtbl.create 16

let jaxpr_forwarding (c : closed_jaxpr) =
  let fwds : (int, int) Hashtbl.t = Hashtbl.create 16 in
  let n_consts = List.length c.consts in
  List.iteri
    (fun i (v : var) -> if i >= n_consts then Hashtbl.replace fwds v.vid (i - n_consts))
    c.jaxpr.in_binders;
  let origin = function
    | A_var (v : var) -> Hashtbl.find_opt fwds v.vid
    | A_lit _ | DropVar _ -> None
  in
  List.iter
    (fun (e : eqn) ->
      match (e.prim, e.inputs, e.outs) with
      | Convert_element_type dt, [ (A_var (i : var) as a) ], [ o ]
        when i.vaval.dtype = dt && i.vaval.weak_type = o.vaval.weak_type ->
          Option.iter (Hashtbl.replace fwds o.vid) (origin a)
      | Broadcast_in_dim { shape; _ }, [ (A_var (i : var) as a) ], [ o ]
        when i.vaval.shape = shape ->
          Option.iter (Hashtbl.replace fwds o.vid) (origin a)
      | _ -> ())
    c.jaxpr.eqns;
  List.map origin c.jaxpr.outs

let rec known_copy (c : closed_jaxpr) =
  match Hashtbl.find_opt known_copies c.jid with
  | Some copy -> copy
  | None ->
      let vars = Hashtbl.create 16 in
      let fresh (v : var) =
        match Hashtbl.find_opt vars v.vid with
        | Some f -> f
        | None ->
            let f = { v with vid = Core.fresh_id () } in
            Hashtbl.replace vars v.vid f;
            f
      in
      let subst : (int, atom) Hashtbl.t = Hashtbl.create 4 in
      let atom = function
        | A_var v as a -> (
            match Hashtbl.find_opt subst v.vid with
            | Some forwarded -> forwarded
            | None -> ( match a with A_var v -> A_var (fresh v) | a -> a))
        | a -> a
      in
      let in_binders = List.map fresh c.jaxpr.in_binders in
      let eqns =
        List.map
          (fun (e : eqn) ->
            let inputs = List.map atom e.inputs in
            match e.prim with
            | Xla_call inner when not (eqn_has_effect e) ->
                let inner, in_fwd = known_call inner in
                let outs =
                  List.concat
                    (List.map2
                       (fun (o : var) f ->
                         match f with
                         | Some i ->
                             Hashtbl.replace subst o.vid (List.nth inputs i);
                             []
                         | None -> [ fresh o ])
                       e.outs in_fwd)
                in
                {
                  prim = Xla_call inner;
                  inputs;
                  outs;
                  multiple_results = List.length outs > 1;
                }
            | Xla_call inner ->
                {
                  e with
                  prim = Xla_call (known_copy inner);
                  inputs;
                  outs = List.map fresh e.outs;
                }
            | _ -> { e with inputs; outs = List.map fresh e.outs })
          c.jaxpr.eqns
      in
      let copy =
        {
          c with
          jid = Core.fresh_id ();
          jaxpr = { in_binders; eqns; outs = List.map atom c.jaxpr.outs };
        }
      in
      Pretty_printer.register_call_name copy.jid
        (Pretty_printer.call_name c.jid);
      Core.inherit_inline ~from:c.jid copy.jid;
      Hashtbl.replace known_copies c.jid copy;
      copy

and known_call (c : closed_jaxpr) =
  let copy = known_copy c in
  let in_fwd = jaxpr_forwarding copy in
  if List.for_all Option.is_none in_fwd then (copy, in_fwd)
  else
    let key = -copy.jid in
    let pruned =
      match Hashtbl.find_opt known_copies key with
      | Some pruned -> pruned
      | None ->
          let outs =
            List.concat
              (List.map2
                 (fun a f -> if f = None then [ a ] else [])
                 copy.jaxpr.outs in_fwd)
          in
          let pruned =
            {
              copy with
              jid = Core.fresh_id ();
              jaxpr = { copy.jaxpr with outs };
            }
          in
          Pretty_printer.register_call_name pruned.jid
            (Pretty_printer.call_name c.jid);
          Core.inherit_inline ~from:c.jid pruned.jid;
          Hashtbl.replace known_copies key pruned;
          pruned
    in
    (pruned, in_fwd)

let in_linearize_scope args =
  match !Core.linearize_scope with
  | Some scope ->
      !Core.dynamic_trace == scope && (Core.find_top_trace args).kind <> KPE
  | None -> false

let named_jit ~name ?(key = "") f args =
  if not (Core.top_is_staging args) then !eager_jit name key f args
  else
    let avals = List.map Core.get_aval args in
    let ck = call_key name key avals in
    let closed =
      match Hashtbl.find_opt call_cache ck with
      | Some c -> Some c
      | None -> (
          match make_jaxpr_hoisted avals f with
          | c, [] ->
              Pretty_printer.register_call_name c.jid name;
              Hashtbl.replace call_cache ck c;
              Some c
          | _ -> None)
    in
    match closed with
    | Some c when in_linearize_scope args ->
        let known, in_fwd = known_call c in
        let results = ref (Core.bind (Xla_call known) args) in
        List.map
          (function
            | Some i -> List.nth args i
            | None -> (
                match !results with
                | v :: tl ->
                    results := tl;
                    v
                | [] -> invalid_arg "Jaxpr.named_jit: forwarded outputs"))
          in_fwd
    | Some c -> Core.bind (Xla_call c) args
    | None -> f args

let copy_call_name ~(source : closed_jaxpr) (target : closed_jaxpr) =
  if Pretty_printer.has_call_name source.jid then
    Pretty_printer.register_call_name target.jid
      (Pretty_printer.call_name source.jid)
