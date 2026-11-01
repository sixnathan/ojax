module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module RD = Numpy.Reductions
module AC = Numpy.Array_creation
module IX = Numpy.Indexing
open Dist_util

let invert_permutation i =
  let size = Array.fold_left ( * ) 1 (shape i) in
  let base = AC.zeros_like i in
  let vals = NL.arange ~dtype:(dtype i) (float_of_int size) in
  IX.put base i vals

let sem ?(axis = 0) ?(ddof = 1) a =
  let b = match promote_complex [ a ] with [ b ] -> b | _ -> assert false in
  let size = (shape b).(axis) in
  let denom = sc b (Float.sqrt (float_of_int size)) in
  UF.divide (RD.std ~axis:[| axis |] ~ddof b) denom

let sem ?axis ?ddof a =
  Eager_jit.one ~jit:"sem" "scipy.stats.sem" (axis, ddof) [] [ a ]
    (fun _ vals ->
      match vals with [ a ] -> sem ?axis ?ddof a | _ -> assert false)

let lane_mode x =
  let module L = Lax_api in
  let n = (shape x).(0) in
  let sorted = List.hd (L.sort [ x ]) in
  let different =
    L.ne (L.slice_in_dim sorted 1 n) (L.slice_in_dim sorted 0 (n - 1))
  in
  let first = L.full [| 1 |] 1.0 Dtype.Bool in
  let starts = L.concatenate [ first; different ] 0 in
  let idt = Dtypes.default_int_dtype () in
  let groups =
    L.sub (L.cumsum (NL.astype starts idt)) (L.full [| n |] 1.0 idt)
  in
  let ones = L.full [| n |] 1.0 idt in
  let indices = L.reshape groups [| n; 1 |] in
  let counts =
    L.scatter_add (L.full [| n |] 0.0 idt) indices ones
      {
        Types.update_window_dims = [||];
        inserted_window_dims = [| 0 |];
        scatter_dims_to_operand_dims = [| 0 |];
        s_operand_batching_dims = [||];
        s_scatter_indices_batching_dims = [||];
      }
  in
  let best = L.argmax counts ~axis:0 ~index_dtype:idt in
  let matches = L.eq groups (L.broadcast_in_dim best [| n |] [||]) in
  let position = L.argmax matches ~axis:0 ~index_dtype:idt in
  let value = L.dynamic_index_in_dim ~keepdims:false sorted position in
  let count = L.reduce_max counts [| 0 |] in
  (value, count)

let mode ?(axis = Some 0) ?(nan_policy = "propagate") ?(keepdims = false) a =
  if nan_policy = "omit" then
    invalid_arg "scipy.stats.mode: nan_policy='omit' is not supported by JAX.";
  let x = NL.atleast_1d a in
  let sh = shape x in
  let rank = Array.length sh in
  let axis = Option.map (fun ax -> if ax < 0 then ax + rank else ax) axis in
  let output_shape =
    match (axis, keepdims) with
    | None, true -> Array.make rank 1
    | None, false -> [||]
    | Some ax, true -> Array.mapi (fun i s -> if i = ax then 1 else s) sh
    | Some ax, false ->
        Array.of_list (List.filteri (fun i _ -> i <> ax) (Array.to_list sh))
  in
  let moved =
    match axis with
    | Some ax -> NL.moveaxis [| ax |] [| 0 |] x
    | None -> NL.reshape x [| Array.fold_left ( * ) 1 sh |]
  in
  let msh = shape moved in
  let lanes = Array.fold_left ( * ) 1 msh / max msh.(0) 1 in
  let rows = NL.reshape moved [| msh.(0); lanes |] in
  let results =
    Interpreters.Batching.vmap
      (fun args ->
        let v, c = lane_mode (List.hd args) in
        [ v; c ])
      [ Some 1 ] [ rows ]
  in
  match results with
  | [ v; c ] -> (NL.reshape v output_shape, NL.reshape c output_shape)
  | _ -> failwith "scipy.stats.mode: arity"

let mode ?axis ?nan_policy ?keepdims a =
  Eager_jit.pair ~jit:"mode" "scipy.stats.mode" (axis, nan_policy, keepdims)
    [] [ a ] (fun _ vals ->
      match vals with
      | [ a ] -> mode ?axis ?nan_policy ?keepdims a
      | _ -> assert false)

let rec rankdata ?(method_ = "average") ?axis ?(nan_policy = "propagate") a =
  let module L = Lax_api in
  if nan_policy = "omit" then
    invalid_arg
      "scipy.stats.rankdata: nan_policy='omit' is not supported by JAX.";
  if not (List.mem method_ [ "average"; "min"; "max"; "dense"; "ordinal" ]) then
    invalid_arg ("unknown method '" ^ method_ ^ "'");
  match axis with
  | Some ax ->
      Numpy.Extras.apply_along_axis (fun lane -> rankdata ~method_ lane) ax a
  | None -> (
      let flat = NL.ravel a in
      let n = (shape flat).(0) in
      let out_dt = Dtypes.default_float_dtype () in
      let idt = Dtypes.default_int_dtype () in
      let iota = NL.arange ~dtype:idt (float_of_int n) in
      let arr, sorter = L.sort_key_val flat iota in
      let inv = NL.argsort_values sorter in
      let one = L.full [| n |] 1.0 idt in
      if method_ = "ordinal" then NL.astype (L.add inv one) out_dt
      else
        let different =
          L.ne (L.slice_in_dim arr 1 n) (L.slice_in_dim arr 0 (n - 1))
        in
        let obs =
          L.concatenate [ L.full [| 1 |] 1.0 Dtype.Bool; different ] 0
        in
        let dense =
          Numpy.Indexing.take_index ~axis:0 (L.cumsum (NL.astype obs idt)) inv
        in
        if method_ = "dense" then NL.astype dense out_dt
        else
          let positions = Numpy.Extras.flatnonzero ~size:n ~fill_value:n obs in
          let count =
            L.concatenate
              [ NL.astype positions idt; L.full [| 1 |] (float_of_int n) idt ]
              0
          in
          let upper = Numpy.Indexing.take_index ~axis:0 count dense in
          let lower =
            L.add
              (Numpy.Indexing.take_index ~axis:0 count (L.sub dense one))
              one
          in
          match method_ with
          | "max" -> NL.astype upper out_dt
          | "min" -> NL.astype lower out_dt
          | _ ->
              let sum =
                L.add (NL.astype upper out_dt) (NL.astype lower out_dt)
              in
              L.mul sum (L.full_like sum 0.5))

let rankdata ?method_ ?axis ?nan_policy a =
  Eager_jit.one ~jit:"rankdata" "scipy.stats.rankdata"
    (method_, axis, nan_policy) [] [ a ] (fun _ vals ->
      match vals with
      | [ a ] -> rankdata ?method_ ?axis ?nan_policy a
      | _ -> assert false)
