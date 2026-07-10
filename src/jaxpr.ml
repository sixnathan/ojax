open Types

let aval_of_ndarray nd =
  { shape = Ndarray.shape nd; dtype = Ndarray.dtype nd; weak_type = false }

let atom_aval env = function
  | A_var v ->
      if Hashtbl.mem env v.vid then v.vaval
      else invalid_arg "typecheck: unbound variable"
  | A_lit nd -> aval_of_ndarray nd
  | DropVar a -> a

let avals_equal (a : aval) (b : aval) =
  a.shape = b.shape && a.dtype = b.dtype && a.weak_type = b.weak_type

let typecheck_jaxpr (jx : jaxpr) : unit =
  let env : (int, unit) Hashtbl.t = Hashtbl.create 16 in
  let bind (v : var) =
    if Hashtbl.mem env v.vid then invalid_arg "typecheck: variable bound twice";
    Hashtbl.replace env v.vid ()
  in
  List.iter bind jx.in_binders;
  List.iter
    (fun (e : eqn) ->
      let in_types = List.map (atom_aval env) e.inputs in
      let out_types = Core.rules.abstract_eval e.prim in_types in
      List.iter2
        (fun (b : var) t ->
          if not (avals_equal t b.vaval) then
            invalid_arg "typecheck: output type mismatch")
        e.outs out_types;
      List.iter bind e.outs)
    jx.eqns;
  List.iter (fun a -> ignore (atom_aval env a)) jx.outs

let eval_jaxpr (jx : jaxpr) (args : value list) : value list =
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
      let outs = Core.bind e.prim in_vals in
      List.iter2 write e.outs outs)
    jx.eqns;
  List.map read jx.outs

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
  let _ = add_var builder t in
  t

let as_tracer = function
  | Tracer t -> t
  | Concrete _ -> invalid_arg "jaxpr: expected a JaxprTracer"
  | Device _ -> invalid_arg "jaxpr: expected a JaxprTracer"

let get_or_make_const_tracer trace v =
  let builder = builder_of trace in
  let t = new_tracer builder trace (Core.get_aval v) in
  let _ = add_const builder t v in
  Tracer t

let process_primitive trace prim args =
  let builder = builder_of trace in
  let tracers = List.map as_tracer args in
  let avals_in = List.map (fun (t : tracer) -> t.aval) tracers in
  let avals_out = Core.rules.abstract_eval prim avals_in in
  let out_tracers = List.map (new_tracer builder trace) avals_out in
  let inputs = List.map (fun t -> A_var (getvar builder t)) tracers in
  let outvars = List.map (add_var builder) out_tracers in
  builder.jb_eqns <-
    { prim; inputs; outs = outvars; multiple_results = List.length outvars > 1 }
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

let build builder (in_tracers : tracer list) (out_tracers : tracer list) :
    jaxpr * Ndarray.t list =
  let constvals = List.rev builder.jb_constvals in
  let constvars = List.map fst constvals in
  let in_binders = constvars @ List.map (getvar builder) in_tracers in
  let outs = List.map (fun t -> A_var (getvar builder t)) out_tracers in
  let eqns = List.rev builder.jb_eqns in
  let jx = { in_binders; eqns; outs } in
  typecheck_jaxpr jx;
  let jx, consts = inline_literals jx (List.map snd constvals) in
  typecheck_jaxpr jx;
  (jx, consts)

let make_jaxpr (in_avals : aval list) (f : value list -> value list) :
    closed_jaxpr =
  let builder =
    {
      jb_eqns = [];
      jb_tracer_to_var = [];
      jb_const_tracers = [];
      jb_constvals = [];
      jb_tracers = [];
    }
  in
  Core.with_new_main KJaxpr (GBuilder builder) (fun main ->
      Core.new_dynamic main (fun () ->
          let in_tracers =
            List.map (fun a -> new_arg builder main a) in_avals
          in
          let outs = f (List.map (fun t -> Tracer t) in_tracers) in
          let out_tracers =
            List.map (fun v -> as_tracer (Core.full_raise main v)) outs
          in
          let jx, consts = build builder in_tracers out_tracers in
          { jid = Core.fresh_id (); jaxpr = jx; consts }))
