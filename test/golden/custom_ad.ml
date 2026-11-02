module U = Yojson.Safe.Util
module TU = Ojax.Tree_util
module UF = Ojax.Numpy.Ufuncs
module JU = Ojax.Numpy.Jnp_util
module AC = Ojax.Numpy.Array_creation
module A = Ojax.Api
module CD = Ojax.Custom_derivatives
module CP = Ojax.Checkpoint

let name = "custom_ad"
let arrays = List.map (fun v -> TU.Leaf v)
let values = List.map Compose_fuzz.as_leaf
let scale factor vs = List.map (fun v -> UF.multiply v factor) vs

let policy_of = function
  | "everything_saveable" -> CP.everything_saveable
  | "nothing_saveable" -> CP.nothing_saveable
  | "dots_saveable" -> CP.dots_saveable
  | "dots_with_no_batch_dims_saveable" -> CP.dots_with_no_batch_dims_saveable
  | "save_anything_except_these_names" ->
      CP.save_anything_except_these_names [ "n" ]
  | "save_any_names_but_these" -> CP.save_any_names_but_these [ "n" ]
  | "save_only_these_names" -> CP.save_only_these_names [ "n" ]
  | "save_from_both_policies" ->
      CP.save_from_both_policies CP.dots_saveable
        (CP.save_only_these_names [ "n" ])
  | "offload_dot_with_no_batch_dims" ->
      CP.offload_dot_with_no_batch_dims ~offload_src:"device"
        ~offload_dst:"pinned_host"
  | "save_and_offload_only_these_names" ->
      CP.save_and_offload_only_these_names ~names_which_can_be_saved:[]
        ~names_which_can_be_offloaded:[ "n" ] ~offload_src:"device"
        ~offload_dst:"pinned_host"
  | other -> failwith ("custom_ad: policy " ^ other)

let vjp_of f primals cts =
  let y, pull = A.vjp f (arrays primals) in
  let shaped = TU.tree_unflatten (TU.tree_structure y) cts in
  List.concat_map TU.tree_leaves (pull shaped)

let extend layer f =
  let flat, shape = Compose_fuzz.flat_call f in
  let wrap call args =
    let out = call (values args) in
    TU.tree_unflatten !shape out
  in
  let k () = JU.py_float (U.to_number (U.member "k" layer)) in
  match U.to_string (U.member "t" layer) with
  | "custom_vjp" ->
      let fwd a = (flat a, a) in
      let bwd res cts = scale (JU.py_float 1.5) (vjp_of f res cts) in
      Some (wrap (CD.custom_vjp flat ~fwd ~bwd))
  | "custom_vjp_nondiff" ->
      let primal a = flat (List.tl a) in
      let fwd a = (flat (List.tl a), List.tl a) in
      let bwd res cts =
        match res with
        | k :: res -> scale k (vjp_of f res cts)
        | [] -> failwith "custom_ad: nondiff"
      in
      let call = CD.custom_vjp ~nondiff_argnums:[ 0 ] primal ~fwd ~bwd in
      Some (wrap (fun a -> call (k () :: a)))
  | "custom_jvp_nondiff" ->
      let primal a = flat (List.tl a) in
      let rule primals tangents =
        match primals with
        | k :: primals ->
            let y, dy = A.jvp f (arrays primals) (arrays tangents) in
            shape := TU.tree_structure y;
            (TU.tree_leaves y, scale k (TU.tree_leaves dy))
        | [] -> failwith "custom_ad: nondiff"
      in
      let call =
        CD.custom_jvp ~nondiff_argnums:[ 0 ] ~name:"<lambda>" ~jvp_name:"rule"
          primal ~jvp:rule
      in
      Some (wrap (fun a -> call (k () :: a)))
  | "custom_jvp_symbolic" ->
      let rule primals tangents =
        let tangents =
          List.map
            (function
              | CD.Symbolic_zero { aval } ->
                  AC.zeros ~dtype:aval.Ojax.Types.dtype aval.shape
              | CD.Tangent t -> t)
            tangents
        in
        let y, dy = A.jvp f (arrays primals) (arrays tangents) in
        shape := TU.tree_structure y;
        ( TU.tree_leaves y,
          List.map
            (fun v -> CD.Tangent v)
            (scale (JU.py_float 1.5) (TU.tree_leaves dy)) )
      in
      Some (wrap (CD.custom_jvp_symbolic flat ~jvp:rule))
  | "custom_gradient" ->
      let g a =
        let y, pull = A.vjp f (arrays a) in
        shape := TU.tree_structure y;
        ( TU.tree_leaves y,
          fun cts ->
            let shaped = TU.tree_unflatten (TU.tree_structure y) cts in
            scale (JU.py_float 1.5)
              (List.concat_map TU.tree_leaves (pull shaped)) )
      in
      Some (wrap (CD.custom_gradient g))
  | "closure_convert" ->
      Some
        (fun args ->
          let a = values args in
          let converted, consts =
            CD.closure_convert flat (List.map Ojax.Core.get_aval a)
          in
          let out = converted (a @ consts) in
          TU.tree_unflatten !shape out)
  | "checkpoint_policy" ->
      let policy = policy_of (U.to_string (U.member "policy" layer)) in
      let prevent_cse = U.to_bool (U.member "prevent_cse" layer) in
      Some (wrap (A.checkpoint ~policy ~prevent_cse flat))
  | "checkpoint_name" ->
      Some
        (fun args -> TU.tree_map (fun v -> CP.checkpoint_name v "n") (f args))
  | _ -> None

let suites ~goldens_root ~nd_of_npz ~string_of_dtype ~concrete =
  Compose_fuzz.suites ~name ~extend ~goldens_root ~nd_of_npz ~string_of_dtype
    ~concrete ()
