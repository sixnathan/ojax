open Types

type policy_kind = Save_all | Selective

type policy = {
  kind : policy_kind;
  allows : name:string option -> primitive -> bool;
}

let everything_saveable = { kind = Save_all; allows = (fun ~name:_ _ -> true) }
let nothing_saveable = { kind = Selective; allows = (fun ~name:_ _ -> false) }

let checkpoint_dots =
  {
    kind = Selective;
    allows = (fun ~name:_ -> function Dot_general _ -> true | _ -> false);
  }

let dots_with_no_batch_dims_saveable =
  {
    kind = Selective;
    allows =
      (fun ~name:_ -> function
        | Dot_general d ->
            Array.length d.lhs_batch = 0 && Array.length d.rhs_batch = 0
        | _ -> false);
  }

let save_only_these_names names =
  {
    kind = Selective;
    allows =
      (fun ~name prim ->
        match (name, prim) with
        | Some name, _ -> List.mem name names
        | None, _ -> false);
  }

let policy_allows policy ?name prim = policy.allows ~name prim
let checkpoint_name x _name = x

let rec contains_partial_eval = function
  | Tracer { payload = PE _; _ } -> true
  | Tracer { payload = JVP { primal; tangent }; _ } ->
      contains_partial_eval primal || contains_partial_eval tangent
  | Tracer { payload = Batch { v; _ }; _ } -> contains_partial_eval v
  | Concrete _ | Device _ | Tracer _ -> false

let is_reverse_linearization args =
  let top = Core.find_top_trace args in
  top.kind = KJVP && List.exists contains_partial_eval args

let rematerialized_vjp fun_ args =
  let num_args = List.length args in
  let primal xs = fun_ xs in
  let fwd xs = fun_ xs @ xs in
  let bwd inputs =
    let residuals, cotangents =
      match Util.split_list inputs [ num_args ] with
      | [ residuals; cotangents ] -> (residuals, cotangents)
      | _ -> failwith "checkpoint: backward argument count mismatch"
    in
    let _, pullback = Interpreters.Ad.vjp fun_ residuals in
    pullback cotangents
  in
  Custom_derivatives.custom_vjp ~primal ~fwd ~bwd args

let checkpoint ?(prevent_cse = true) ?(policy = nothing_saveable) fun_ args =
  let _ = prevent_cse in
  match policy.kind with
  | Save_all -> fun_ args
  | Selective ->
      if is_reverse_linearization args then rematerialized_vjp fun_ args
      else fun_ args

let remat = checkpoint

let saved_residuals fun_ primals =
  let pvals_in =
    List.map Interpreters.Partial_eval.partial_val_known primals
    @ List.map
        (fun x ->
          Interpreters.Partial_eval.partial_val_unknown (Core.get_aval x))
        primals
  in
  let f_jvp inputs =
    let px, tx = Util.split_half inputs in
    let po, to_ = Interpreters.Ad.jvp fun_ px tx in
    po @ to_
  in
  let _, consts, _ =
    Interpreters.Partial_eval.partial_eval_flat f_jvp pvals_in
  in
  List.mapi
    (fun i value ->
      ( Core.get_aval value,
        Printf.sprintf "saved residual at flattened index %d" i ))
    consts
