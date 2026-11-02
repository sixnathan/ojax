module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module RD = Numpy.Reductions
module AC = Numpy.Array_creation
module IX = Numpy.Indexing
module T = Types
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
  let n = Array.fold_left ( * ) 1 (shape x) in
  let vals, counts =
    match Numpy.Extras.unique ~return_counts:true ~size:n x with
    | [ vals; counts ] -> (vals, counts)
    | _ -> failwith "scipy.stats.mode: unique arity"
  in
  (IX.dynamic_at vals (NL.argmax counts), RD.max counts)

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

let invert_permutation_at i =
  let size = Array.fold_left ( * ) 1 (shape i) in
  let base = AC.empty_like i in
  Numpy.Nd_index.set base [ Numpy.Nd_index.Arr i ]
    (NL.arange ~dtype:(dtype i) (float_of_int size))

let rankdata_body method_ a =
  let module L = Lax_api in
  let module NI = Numpy.Nd_index in
  let n = (shape a).(0) in
  let out_dt = Dtypes.default_float_dtype () in
  let lit = Numpy.Jnp_util.py_int in
  let arr, sorter =
    L.sort_key_val a
      (NL.arange ~dtype:(Dtypes.default_int_dtype ()) (float_of_int n))
  in
  let inv = invert_permutation_at sorter in
  if method_ = "ordinal" then NL.astype (UF.add inv (lit 1)) out_dt
  else
    let first = T.Concrete (Ndarray.of_floats Dtype.Bool [| 1 |] [| 1.0 |]) in
    let different =
      L.ne
        (NI.take arr [ NI.Range { start = Some 1; stop = None } ])
        (NI.take arr [ NI.Range { start = None; stop = Some (-1) } ])
    in
    let obs = NL.concatenate ~axis:0 [ first; different ] in
    let dense = NI.take (Numpy.Array_methods.cumsum obs) [ NI.Arr inv ] in
    if method_ = "dense" then NL.astype dense out_dt
    else
      let count =
        NL.astype
          (List.hd (Numpy.Extras.nonzero ~size:(n + 1) ~fill_value:n obs))
          out_dt
      in
      match method_ with
      | "max" -> NI.take count [ NI.Arr dense ]
      | "min" ->
          UF.add
            (NI.take count [ NI.Arr (UF.subtract dense (lit 1)) ])
            (py_float 1.0)
      | _ ->
          let sum =
            UF.add
              (UF.add
                 (NI.take count [ NI.Arr dense ])
                 (NI.take count [ NI.Arr (UF.subtract dense (lit 1)) ]))
              (py_float 1.0)
          in
          UF.multiply (py_float 0.5) sum

let rec rankdata_impl ?(method_ = "average") ?axis ?(nan_policy = "propagate") a
    =
  let module L = Lax_api in
  if nan_policy = "omit" then
    invalid_arg
      "scipy.stats.rankdata: nan_policy='omit' is not supported by JAX.";
  if not (List.mem method_ [ "average"; "min"; "max"; "dense"; "ordinal" ]) then
    invalid_arg ("unknown method '" ^ method_ ^ "'");
  match axis with
  | Some ax ->
      Numpy.Extras.apply_along_axis
        (fun lane -> rankdata_jit ~method_ lane)
        ax a
  | None ->
      let flat = NL.ravel a in
      let out_dt = Dtypes.default_float_dtype () in
      List.hd
        (L.cond
           (RD.any (UF.isnan flat))
           (fun args ->
             [ AC.full_like ~dtype:out_dt (List.hd args) Float_format.nan ])
           (fun args -> [ rankdata_body method_ (List.hd args) ])
           [ flat ])

and rankdata_jit ?method_ ?axis ?nan_policy a =
  Eager_jit.one ~jit:"rankdata" "scipy.stats.rankdata"
    (method_, axis, nan_policy) [] [ a ] (fun _ vals ->
      match vals with
      | [ a ] -> rankdata_impl ?method_ ?axis ?nan_policy a
      | _ -> assert false)

let rankdata = rankdata_jit
