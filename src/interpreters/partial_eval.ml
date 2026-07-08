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

let pe_pval (t : tracer) =
  match t.payload with
  | PE { pval; _ } -> pval
  | _ -> invalid_arg "partial_eval: expected a PartialEval tracer"

let as_tracer = function
  | Tracer t -> t
  | Concrete _ -> invalid_arg "partial_eval: expected a PartialEval tracer"

let pure trace v = Tracer (new_pe_tracer trace (partial_val_known v) None)

let instantiate_const trace (t : tracer) : tracer =
  let pval = pe_pval t in
  match pval.pv_const with
  | None -> t
  | Some c ->
      new_pe_tracer trace (partial_val_unknown t.aval) (Some (ConstRecipe c))

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
      (fun t ->
        match t.payload with PE p -> p.recipe <- Some recipe | _ -> ())
      out_tracers;
    List.map (fun t -> Tracer t) out_tracers

let full_lower_pe v =
  match v with
  | Tracer t -> (
      match t.payload with
      | PE { pval; _ } -> (
          match pval.pv_const with Some c -> Core.full_lower c | None -> v)
      | _ -> v)
  | Concrete _ -> v

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
          (List.filter
             (fun (n : tracer) -> Hashtbl.find child_counts n.id = 0)
             roots)
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
  let processed_eqns : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let eqns = ref [] in
  let recipe_key er_out =
    match er_out with
    | r :: _ -> ( match !r with Some (t : tracer) -> t.id | None -> -1)
    | [] -> -1
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
          let var = { vid = Core.fresh_id (); vaval = t.aval } in
          constvar_to_val := (var, v) :: !constvar_to_val;
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
    (toposort tracers_out tracer_parents);
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
