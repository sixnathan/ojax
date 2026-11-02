module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let weak v = (get_aval v).T.weak_type
let ndim v = Array.length (shape v)
let size v = Array.fold_left ( * ) 1 (shape v)
let bind1 = C.bind1
let const_full dt sh x = Core.splat dt sh x
let zeros_like v = const_full (dtype v) (shape v) 0.0

let canonicalize_axis ax n =
  let a = if ax < 0 then ax + n else ax in
  if a < 0 || a >= n then
    invalid_arg
      (Printf.sprintf "axis %d out of bounds for array of ndim %d" ax n);
  a

let convert v dt =
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let reshape v sh = bind1 (T.Reshape sh) [ v ]
let ravel v = if ndim v = 1 then v else reshape v [| size v |]
let broadcast_to = Jnp_util.broadcast_to

let is_inexact = function
  | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn | D.F8_e3m4
  | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz | D.F8_e5m2
  | D.F8_e5m2fnuz | D.F8_e8m0fnu | D.F32 | D.F64 | D.Complex64 | D.Complex128 ->
      true
  | _ -> false

let to_inexact_dtype = function
  | D.F32 -> D.F32
  | D.F64 -> D.F64
  | ( D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn | D.F8_e3m4
    | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz | D.F8_e5m2
    | D.F8_e5m2fnuz | D.F8_e8m0fnu ) as narrow ->
      narrow
  | D.I32 | D.Bool | D.Uint32 | D.I8 | D.I16 | D.I4 | D.Uint8 | D.Uint16
  | D.Uint4 ->
      D.F32
  | (D.I2 | D.Uint2) as narrow -> narrow
  | D.Float0 -> invalid_arg "float0 has no inexact dtype"
  | D.I64 | D.Uint64 -> D.F64
  | (D.Complex64 | D.Complex128) as c -> c

let to_inexact v = convert v (to_inexact_dtype (dtype v))
let to_bool v = if dtype v = D.Bool then v else convert v D.Bool

let promote_integer_dtype dt =
  let di = Dtypes.default_int_dtype () in
  match dt with
  | D.Bool -> di
  | D.I32 -> if di = D.I64 then D.I64 else D.I32
  | D.I64 -> D.I64
  | D.I8 | D.I16 | D.I2 | D.I4 -> if di = D.I64 then D.I64 else D.I32
  | D.Float0 -> invalid_arg "float0 has no reduction dtype"
  | D.Uint8 | D.Uint16 | D.Uint2 | D.Uint4 | D.Uint32 ->
      if di = D.I64 then D.Uint64 else D.Uint32
  | D.Uint64 -> D.Uint64
  | ( D.F32 | D.F64 | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn
    | D.F8_e3m4 | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz
    | D.F8_e5m2 | D.F8_e5m2fnuz | D.F8_e8m0fnu | D.Complex64 | D.Complex128 ) as
    f ->
      f

let all_axes v = Array.init (ndim v) (fun i -> i)

let resolve_axes v = function
  | None -> all_axes v
  | Some ax -> Array.map (fun i -> canonicalize_axis i (ndim v)) ax

let mem_axis axes i = Array.exists (fun a -> a = i) axes

let keepdims_shape orig axes =
  Array.mapi (fun i d -> if mem_axis axes i then 1 else d) orig

let expand_dims v orig axes =
  if Array.length axes = 0 then v
  else
    let dims =
      List.filter
        (fun i -> not (mem_axis axes i))
        (List.init (Array.length orig) Fun.id)
    in
    bind1
      (T.Broadcast_in_dim
         { shape = keepdims_shape orig axes; dims = Array.of_list dims })
      [ v ]

let reduce_axes prim ?(keepdims = false) axes v =
  let orig = shape v in
  let r =
    match (prim axes, dtype v) with
    | (T.Reduce_sum _ as reducer), ((D.F16 | D.Bf16) as narrow) ->
        convert (bind1 reducer [ convert v D.F32 ]) narrow
    | reducer, _ -> bind1 reducer [ v ]
  in
  if keepdims then expand_dims r orig axes else r

let axis_count v axes =
  let s = shape v in
  Array.fold_left (fun acc i -> acc * s.(i)) 1 axes

let isnan v = bind1 T.Ne [ v; v ]
let where = Jnp_util.where

let check_where name = function
  | None -> None
  | Some w ->
      if dtype w <> D.Bool then
        invalid_arg
          (Printf.sprintf "jnp.%s: where must be None or a boolean array." name);
      Some w

let upcast_f16 = function D.F16 | D.Bf16 -> D.F32 | d -> d

let init_value dt v =
  match dt with
  | D.Bool -> if v = 0.0 then 0.0 else 1.0
  | (D.I8 | D.I16 | D.I32 | D.I64 | D.Uint8 | D.Uint16 | D.Uint32 | D.Uint64)
    when Float.abs v = Float.infinity ->
      let f = Dtypes.iinfo dt in
      if v < 0.0 then f.imin else f.imax
  | _ -> v

let init_array dt sh v =
  match (dt, v) with
  | D.I64, v when v = Float.neg_infinity ->
      T.Concrete
        (Nd.of_int64s dt sh
           (Array.make (Array.fold_left ( * ) 1 sh) Int64.min_int))
  | D.I64, v when v = Float.infinity ->
      T.Concrete
        (Nd.of_int64s dt sh
           (Array.make (Array.fold_left ( * ) 1 sh) Int64.max_int))
  | D.Uint64, v when v = Float.infinity ->
      T.Concrete
        (Nd.of_int64s dt sh (Array.make (Array.fold_left ( * ) 1 sh) (-1L)))
  | D.Uint64, v when v = Float.neg_infinity -> const_full dt sh 0.0
  | _ -> const_full dt sh (init_value dt v)

let reduction ~name ~prim ~bool_prim ~combine ~bool_combine ~init ~has_identity
    ~upcast ~promote_integers ?dtype:dt ?axis ?(keepdims = false) ?initial
    ?where:mask a =
  let mask = check_where name mask in
  (match (initial, has_identity, mask) with
  | None, false, Some _ ->
      invalid_arg
        (Printf.sprintf
           "reduction operation %s does not have an identity, so to use a \
            where mask one has to specify 'initial'"
           name)
  | _ -> ());
  let axes = resolve_axes a axis in
  if
    initial = None && (not has_identity)
    && Array.exists (fun i -> (shape a).(i) < 1) axes
  then
    invalid_arg
      (Printf.sprintf
         "zero-size array to reduction operation %s which has no identity" name);
  let result_dtype =
    match dt with
    | Some d -> Dtypes.canonicalize_dtype d
    | None ->
        if promote_integers then promote_integer_dtype (dtype a) else dtype a
  in
  let computation =
    if upcast && dt = None then upcast_f16 result_dtype else result_dtype
  in
  let a = convert a computation in
  let filled =
    match mask with
    | None -> a
    | Some m ->
        let m = broadcast_to m (shape a) in
        where m a (init_array computation (shape a) init)
  in
  let is_bool = computation = D.Bool in
  let chosen = if is_bool then bool_prim else prim in
  let out = bind1 (chosen axes) [ filled ] in
  let out =
    match initial with
    | None -> out
    | Some i ->
        if shape i <> [||] then
          invalid_arg
            (Printf.sprintf
               "initial value must be a scalar. Got array of rank %d" (ndim i));
        let i = convert i computation in
        bind1 (if is_bool then bool_combine else combine) [ i; out ]
  in
  let out = if keepdims then expand_dims out (shape a) axes else out in
  convert out result_dtype

let cast_to_numeric a = if dtype a = D.Bool then convert a D.I32 else a

let sum ?dtype:dt ?axis ?keepdims ?initial ?where:mask
    ?(promote_integers = true) a =
  reduction ~name:"sum"
    ~prim:(fun ax -> T.Reduce_sum ax)
    ~bool_prim:(fun ax -> T.Reduce_or ax)
    ~combine:T.Add ~bool_combine:T.Or ~init:0.0 ~has_identity:true ~upcast:true
    ~promote_integers ?dtype:dt ?axis ?keepdims ?initial ?where:mask
    (cast_to_numeric a)

let prod ?dtype:dt ?axis ?keepdims ?initial ?where:mask
    ?(promote_integers = true) a =
  reduction ~name:"prod"
    ~prim:(fun ax -> T.Reduce_prod ax)
    ~bool_prim:(fun ax -> T.Reduce_and ax)
    ~combine:T.Mul ~bool_combine:T.And ~init:1.0 ~has_identity:true ~upcast:true
    ~promote_integers ?dtype:dt ?axis ?keepdims ?initial ?where:mask
    (cast_to_numeric a)

let max ?axis ?keepdims ?initial ?where:mask a =
  reduction ~name:"max"
    ~prim:(fun ax -> T.Reduce_max ax)
    ~bool_prim:(fun ax -> T.Reduce_max ax)
    ~combine:T.Max ~bool_combine:T.Or ~init:Float.neg_infinity
    ~has_identity:false ~upcast:false ~promote_integers:false ?axis ?keepdims
    ?initial ?where:mask a

let min ?axis ?keepdims ?initial ?where:mask a =
  reduction ~name:"min"
    ~prim:(fun ax -> T.Reduce_min ax)
    ~bool_prim:(fun ax -> T.Reduce_min ax)
    ~combine:T.Min ~bool_combine:T.And ~init:Float.infinity ~has_identity:false
    ~upcast:false ~promote_integers:false ?axis ?keepdims ?initial ?where:mask a

let amax = max
let amin = min

let all ?axis ?keepdims ?where:mask a =
  reduction ~name:"all"
    ~prim:(fun ax -> T.Reduce_and ax)
    ~bool_prim:(fun ax -> T.Reduce_and ax)
    ~combine:T.And ~bool_combine:T.And ~init:1.0 ~has_identity:true
    ~upcast:false ~promote_integers:false ?axis ?keepdims ?where:mask
    (to_bool a)

let any ?axis ?keepdims ?where:mask a =
  reduction ~name:"any"
    ~prim:(fun ax -> T.Reduce_or ax)
    ~bool_prim:(fun ax -> T.Reduce_or ax)
    ~combine:T.Or ~bool_combine:T.Or ~init:0.0 ~has_identity:true ~upcast:false
    ~promote_integers:false ?axis ?keepdims ?where:mask (to_bool a)

let count_over ?axis ?(keepdims = false) ?where:mask a cdt =
  match mask with
  | Some m -> sum ?axis ~keepdims ~dtype:cdt (broadcast_to m (shape a))
  | None ->
      let axes = resolve_axes a axis in
      let n = float_of_int (axis_count a axes) in
      ignore keepdims;
      const_full cdt [||] n

let mean ?dtype:dt ?axis ?keepdims ?where:mask a =
  let mask = check_where "mean" mask in
  let result_dtype =
    match dt with
    | Some d -> Dtypes.canonicalize_dtype d
    | None -> to_inexact_dtype (dtype a)
  in
  let computation = upcast_f16 result_dtype in
  let normalizer = count_over ?axis ?keepdims ?where:mask a computation in
  let total = sum ?axis ?keepdims ?where:mask ~dtype:computation a in
  convert (bind1 T.Div [ total; normalizer ]) result_dtype

let is_complex_dtype = function
  | D.Complex64 | D.Complex128 -> true
  | _ -> false

let real_of_dtype = function
  | D.Complex128 -> D.F64
  | D.Complex64 -> D.F32
  | d -> d

let squared_magnitude cdt centered =
  if is_complex_dtype cdt then
    bind1 T.Real [ bind1 T.Mul [ centered; bind1 T.Conj [ centered ] ] ]
  else bind1 T.Square [ centered ]

let var_promote a_dtype dt =
  match dt with
  | Some d ->
      if is_complex_dtype a_dtype && not (is_complex_dtype d) then
        invalid_arg
          "jax.numpy.var does not yet support real dtype parameters when \
           computing the variance of an array of complex values.";
      (upcast_f16 d, d)
  | None ->
      if not (is_inexact a_dtype) then
        let d = to_inexact_dtype a_dtype in
        (upcast_f16 d, d)
      else (upcast_f16 a_dtype, real_of_dtype a_dtype)

let correction_of ?ddof_value ddof correction =
  let integer c =
    let dt = Dtypes.default_int_dtype () in
    T.Concrete (Nd.of_floats dt [||] [| c |])
  in
  match (correction, ddof) with
  | None, _ when ddof_value <> None -> Option.get ddof_value
  | None, d -> integer (float_of_int (Option.value d ~default:0))
  | Some c, (None | Some 0) ->
      T.Concrete (Nd.of_floats (Dtypes.default_float_dtype ()) [||] [| c |])
  | Some _, Some _ ->
      invalid_arg "ddof and correction can't be provided simultaneously."

let axis_key = function
  | None -> "none"
  | Some ax -> String.concat "," (Array.to_list (Array.map string_of_int ax))

let statics_key dt axis keepdims =
  Printf.sprintf "%s %s %b"
    (match dt with Some d -> Dtype.short_name d | None -> "none")
    (axis_key axis)
    (Option.value keepdims ~default:false)

let split_optional mask a_mean rest =
  let mask, rest =
    match (mask, rest) with Some _, m :: tl -> (Some m, tl) | _ -> (None, rest)
  in
  let a_mean =
    match (a_mean, rest) with Some _, m :: _ -> Some m | _ -> None
  in
  (mask, a_mean)

let optional_args mask a_mean = Option.to_list mask @ Option.to_list a_mean

let var_body dt axis keepdims mask a_mean correction a =
  let computation, result_dtype = var_promote (dtype a) dt in
  let af = convert a computation in
  let m =
    match a_mean with
    | Some m -> convert m computation
    | None -> mean ?axis ~dtype:computation ~keepdims:true ?where:mask af
  in
  let centered = squared_magnitude computation (bind1 T.Sub [ af; m ]) in
  let rdt = real_of_dtype computation in
  let normalizer = count_over ?axis ?keepdims ?where:mask af rdt in
  let normalizer = bind1 T.Sub [ normalizer; convert correction rdt ] in
  let total = sum ?axis ~dtype:rdt ?keepdims ?where:mask centered in
  let result = convert (bind1 T.Div [ total; normalizer ]) result_dtype in
  where
    (bind1 T.Gt [ normalizer; const_full rdt [||] 0.0 ])
    result
    (Jnp_util.py_float Float_format.nan)

let var_value ?dtype:dt ?axis ?keepdims ?where:mask ?mean:a_mean correction a =
  let mask = check_where "var" mask in
  let dt = Option.map Dtypes.canonicalize_dtype dt in
  Jnp_util.jit1 "_var"
    ~key:
      (statics_key dt axis keepdims
      ^ string_of_bool (mask <> None)
      ^ string_of_bool (a_mean <> None))
    (function
      | a :: correction :: rest ->
          let mask, a_mean = split_optional mask a_mean rest in
          var_body dt axis keepdims mask a_mean correction a
      | _ -> invalid_arg "var: arity")
    (a :: correction :: optional_args mask a_mean)

let var ?dtype:dt ?axis ?keepdims ?ddof ?where:mask ?mean:a_mean ?correction a =
  var_value ?dtype:dt ?axis ?keepdims ?where:mask ?mean:a_mean
    (correction_of ddof correction)
    a

let std ?dtype:dt ?axis ?keepdims ?ddof ?ddof_value ?where:mask ?mean:a_mean
    ?correction a =
  let dt = Option.map Dtypes.canonicalize_dtype dt in
  (match dt with
  | Some d when not (is_inexact d) ->
      invalid_arg "dtype argument to jnp.std must be inexact"
  | _ -> ());
  Jnp_util.jit1 "_std"
    ~key:
      (statics_key dt axis keepdims
      ^ string_of_bool (mask <> None)
      ^ string_of_bool (a_mean <> None))
    (function
      | a :: correction :: rest ->
          let mask, a_mean = split_optional mask a_mean rest in
          bind1 T.Sqrt
            [
              var_value ?dtype:dt ?axis ?keepdims ?where:mask ?mean:a_mean
                correction a;
            ]
      | _ -> invalid_arg "std: arity")
    (a :: correction_of ?ddof_value ddof correction :: optional_args mask a_mean)

let ptp ?axis ?(keepdims = false) a =
  bind1 T.Sub [ amax ?axis ~keepdims a; amin ?axis ~keepdims a ]

let count_nonzero ?axis ?(keepdims = false) a =
  sum ?axis ~keepdims (bind1 T.Ne [ a; const_full (dtype a) [||] 0.0 ])

let count_nonzero ?axis ?keepdims p0 =
  Jnp_util.jit1 "count_nonzero"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string (axis, keepdims) [])))
    (function
      | [ p0 ] -> count_nonzero ?axis ?keepdims p0
      | _ -> invalid_arg "count_nonzero: arity")
    [ p0 ]

let average_with_sum ?axis ?(keepdims = false) ?weights a =
  match weights with
  | None ->
      let a = to_inexact a in
      let avg = mean ?axis ~keepdims a in
      let axes = resolve_axes a axis in
      (avg, const_full (dtype avg) [||] (float_of_int (axis_count a axes)))
  | Some w ->
      let rt, _ = Dtypes.result_type [ (dtype a, weak a); (dtype w, weak w) ] in
      let cdt = to_inexact_dtype rt in
      let axes = resolve_axes a axis in
      let a2 = convert a cdt and w2 = convert w cdt in
      let w2 =
        if shape w2 = shape a2 then w2
        else begin
          if axis = None then
            invalid_arg
              "Axis must be specified when shapes of a and weights differ.";
          if shape w2 <> Array.map (fun i -> (shape a2).(i)) axes then
            invalid_arg
              "Shape of weights must be consistent with shape of a along \
               specified axis.";
          let new_shape =
            Array.mapi (fun i d -> if Array.mem i axes then d else 1) (shape a2)
          in
          let order =
            Array.init (Array.length axes) Fun.id
            |> Array.to_list
            |> List.stable_sort (fun x y -> compare axes.(x) axes.(y))
            |> Array.of_list
          in
          let moved =
            if Array.for_all Fun.id (Array.mapi (fun i o -> i = o) order) then
              w2
            else bind1 (T.Transpose order) [ w2 ]
          in
          reshape moved new_shape
        end
      in
      let total = sum ?axis ~keepdims w2 in
      let product = bind1 T.Mul [ a2; w2 ] in
      (bind1 T.Div [ sum ?axis ~keepdims product; total ], total)

let average ?axis ?keepdims ?weights a =
  fst (average_with_sum ?axis ?keepdims ?weights a)

let average_returned ?axis ?keepdims ?weights a =
  let avg, total = average_with_sum ?axis ?keepdims ?weights a in
  (avg, broadcast_to total (shape avg))

let nan_reduce ~reduce ~init ~nan_if_all_nan ?axis ?keepdims a =
  if not (is_inexact (dtype a)) then reduce ?axis ?keepdims a
  else begin
    let filled = where (isnan a) (init_array (dtype a) [||] init) a in
    let out = reduce ?axis ?keepdims filled in
    if nan_if_all_nan then
      let allnan = all ?axis ?keepdims (isnan a) in
      where allnan (const_full (dtype a) [||] Float_format.nan) out
    else out
  end

let nansum ?dtype:dt ?axis ?keepdims ?initial ?where:mask a =
  nan_reduce
    ~reduce:(fun ?axis ?keepdims v ->
      sum ?dtype:dt ?axis ?keepdims ?initial ?where:mask v)
    ~init:0.0 ~nan_if_all_nan:false ?axis ?keepdims a

let nansum ?dtype ?axis ?keepdims ?initial ?where p0 =
  let initial_given = initial <> None in
  let where_given = where <> None in
  if initial_given || where_given then
    nansum ?dtype ?axis ?keepdims ?initial ?where p0
  else
    Jnp_util.jit1 "nansum"
      ~key:
        (Digest.to_hex
           (Digest.string (Marshal.to_string (dtype, axis, keepdims) [])))
      (function
        | [ p0 ] -> nansum ?dtype ?axis ?keepdims ?initial ?where p0
        | _ -> invalid_arg "nansum: arity")
      [ p0 ]

let nanprod ?dtype:dt ?axis ?keepdims ?initial ?where:mask a =
  nan_reduce
    ~reduce:(fun ?axis ?keepdims v ->
      prod ?dtype:dt ?axis ?keepdims ?initial ?where:mask v)
    ~init:1.0 ~nan_if_all_nan:false ?axis ?keepdims a

let nanprod ?dtype ?axis ?keepdims ?initial ?where p0 =
  let initial_given = initial <> None in
  let where_given = where <> None in
  if initial_given || where_given then
    nanprod ?dtype ?axis ?keepdims ?initial ?where p0
  else
    Jnp_util.jit1 "nanprod"
      ~key:
        (Digest.to_hex
           (Digest.string (Marshal.to_string (dtype, axis, keepdims) [])))
      (function
        | [ p0 ] -> nanprod ?dtype ?axis ?keepdims ?initial ?where p0
        | _ -> invalid_arg "nanprod: arity")
      [ p0 ]

let nanmax ?axis ?keepdims ?initial ?where:mask a =
  nan_reduce
    ~reduce:(fun ?axis ?keepdims v ->
      amax ?axis ?keepdims ?initial ?where:mask v)
    ~init:Float.neg_infinity ~nan_if_all_nan:true ?axis ?keepdims a

let nanmax ?axis ?keepdims ?initial ?where p0 =
  let initial_given = initial <> None in
  let where_given = where <> None in
  if initial_given || where_given then nanmax ?axis ?keepdims ?initial ?where p0
  else
    Jnp_util.jit1 "nanmax"
      ~key:
        (Digest.to_hex (Digest.string (Marshal.to_string (axis, keepdims) [])))
      (function
        | [ p0 ] -> nanmax ?axis ?keepdims ?initial ?where p0
        | _ -> invalid_arg "nanmax: arity")
      [ p0 ]

let nanmin ?axis ?keepdims ?initial ?where:mask a =
  nan_reduce
    ~reduce:(fun ?axis ?keepdims v ->
      amin ?axis ?keepdims ?initial ?where:mask v)
    ~init:Float.infinity ~nan_if_all_nan:true ?axis ?keepdims a

let nanmin ?axis ?keepdims ?initial ?where p0 =
  let initial_given = initial <> None in
  let where_given = where <> None in
  if initial_given || where_given then nanmin ?axis ?keepdims ?initial ?where p0
  else
    Jnp_util.jit1 "nanmin"
      ~key:
        (Digest.to_hex (Digest.string (Marshal.to_string (axis, keepdims) [])))
      (function
        | [ p0 ] -> nanmin ?axis ?keepdims ?initial ?where p0
        | _ -> invalid_arg "nanmin: arity")
      [ p0 ]

let nanmean ?dtype:dt ?axis ?keepdims ?where:mask a =
  if not (is_inexact (dtype a)) then
    mean ?dtype:dt ?axis ?keepdims ?where:mask a
  else begin
    let cdt =
      match dt with
      | Some d -> Dtypes.canonicalize_dtype d
      | None -> to_inexact_dtype (dtype a)
    in
    let notnan = bind1 T.Not [ isnan a ] in
    let normalizer = sum ~dtype:cdt ?axis ?keepdims ?where:mask notnan in
    bind1 T.Div [ nansum ~dtype:cdt ?axis ?keepdims ?where:mask a; normalizer ]
  end

let nanmean ?dtype ?axis ?keepdims ?where p0 =
  let where_given = where <> None in
  if where_given then nanmean ?dtype ?axis ?keepdims ?where p0
  else
    Jnp_util.jit1 "nanmean"
      ~key:
        (Digest.to_hex
           (Digest.string (Marshal.to_string (dtype, axis, keepdims) [])))
      (function
        | [ p0 ] -> nanmean ?dtype ?axis ?keepdims ?where p0
        | _ -> invalid_arg "nanmean: arity")
      [ p0 ]

let ddof_argument ddof ddof_value =
  match ddof_value with
  | Some _ -> ddof_value
  | None -> Option.map Jnp_util.py_int ddof

let nanvar ?dtype:dt ?axis ?keepdims ?ddof ?where:mask ?mean:a_mean a =
  let mask = check_where "nanvar" mask in
  let dt = Option.map Dtypes.canonicalize_dtype dt in
  let computation, result_dtype = var_promote (dtype a) dt in
  let af = convert a computation in
  let rdt = real_of_dtype computation in
  let m =
    match a_mean with
    | Some m -> convert m computation
    | None -> nanmean ~dtype:computation ?axis ~keepdims:true ?where:mask af
  in
  let nan_a = isnan af in
  let centered = where nan_a (Jnp_util.py_int 0) (bind1 T.Sub [ af; m ]) in
  let centered = squared_magnitude computation centered in
  let notnan = bind1 T.Not [ isnan af ] in
  let count = sum ?axis ?keepdims ?where:mask notnan in
  let nz =
    Ufuncs.subtract count (Option.value ddof ~default:(Jnp_util.py_int 0))
  in
  let bad = bind1 T.Le [ nz; const_full (dtype nz) [||] 0.0 ] in
  let result = sum ?axis ?keepdims ?where:mask centered in
  let result = where bad (Jnp_util.py_float Float_format.nan) result in
  let divisor = where bad (Jnp_util.py_int 1) nz in
  ignore rdt;
  convert (bind1 T.Div [ result; convert divisor (dtype result) ]) result_dtype

let jit_with_ddof name ?dtype ?axis ?keepdims ddof body p0 =
  Jnp_util.jit1 name
    ~key:
      (Digest.to_hex
         (Digest.string
            (Marshal.to_string (dtype, axis, keepdims, ddof <> None) [])))
    (function
      | [ p0 ] -> body None p0
      | [ p0; d ] -> body (Some d) p0
      | _ -> invalid_arg (name ^ ": arity"))
    (p0 :: Option.to_list ddof)

let nanvar ?dtype ?axis ?keepdims ?ddof ?where ?mean p0 =
  if where <> None || mean <> None then
    nanvar ?dtype ?axis ?keepdims ?ddof ?where ?mean p0
  else
    jit_with_ddof "nanvar" ?dtype ?axis ?keepdims ddof
      (fun ddof p0 -> nanvar ?dtype ?axis ?keepdims ?ddof p0)
      p0

let nanstd ?dtype:dt ?axis ?keepdims ?ddof ?where:mask ?mean:a_mean a =
  bind1 T.Sqrt
    [ nanvar ?dtype:dt ?axis ?keepdims ?ddof ?where:mask ?mean:a_mean a ]

let nanstd ?dtype ?axis ?keepdims ?ddof ?where ?mean p0 =
  if where <> None || mean <> None then
    nanstd ?dtype ?axis ?keepdims ?ddof ?where ?mean p0
  else
    jit_with_ddof "nanstd" ?dtype ?axis ?keepdims ddof
      (fun ddof p0 -> nanstd ?dtype ?axis ?keepdims ?ddof p0)
      p0

let cumulative ~fill ~promote ~make ?dtype:dt ?axis a =
  let a, ax =
    match axis with
    | None -> ((if ndim a = 1 then a else ravel a), 0)
    | Some ax -> (a, canonicalize_axis ax (ndim a))
  in
  let a =
    match fill with
    | Some fv when is_inexact (dtype a) ->
        where (isnan a) (const_full (dtype a) (shape a) fv) a
    | _ -> a
  in
  let a, result, computation =
    match dt with
    | None ->
        let r =
          if promote || dtype a = D.Bool then promote_integer_dtype (dtype a)
          else dtype a
        in
        (a, r, r)
    | Some D.Bool ->
        let a =
          if dtype a = D.Bool then a else bind1 T.Ne [ a; zeros_like a ]
        in
        (a, D.Bool, promote_integer_dtype D.Bool)
    | Some d ->
        let d = Dtypes.canonicalize_dtype d in
        (a, d, d)
  in
  convert (bind1 (make ax) [ convert a computation ]) result

let cumsum_make ax = T.Cumsum { axis = ax; reverse = false }
let cumprod_make ax = T.Cumprod { axis = ax; reverse = false }

let cumsum ?dtype:dt ?axis a =
  cumulative ~fill:None ~promote:false ~make:cumsum_make ?dtype:dt ?axis a

let cumsum_keyed ~by_keyword ?dtype ?axis p0 =
  Jnp_util.jit1 "cumsum"
    ~key:
      (Digest.to_hex
         (Digest.string
            (if by_keyword then Marshal.to_string (dtype, axis, true) []
             else Marshal.to_string (dtype, axis) [])))
    (function
      | [ p0 ] -> cumsum ?dtype ?axis p0 | _ -> invalid_arg "cumsum: arity")
    [ p0 ]

let cumsum ?dtype ?axis p0 = cumsum_keyed ~by_keyword:false ?dtype ?axis p0

let cumprod ?dtype:dt ?axis a =
  cumulative ~fill:None ~promote:false ~make:cumprod_make ?dtype:dt ?axis a

let cumprod ?dtype ?axis p0 =
  Jnp_util.jit1 "cumprod"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string (dtype, axis) [])))
    (function
      | [ p0 ] -> cumprod ?dtype ?axis p0 | _ -> invalid_arg "cumprod: arity")
    [ p0 ]

let nancumsum ?dtype:dt ?axis a =
  cumulative ~fill:(Some 0.0) ~promote:false ~make:cumsum_make ?dtype:dt ?axis a

let nancumsum ?dtype ?axis p0 =
  Jnp_util.jit1 "nancumsum"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string (dtype, axis) [])))
    (function
      | [ p0 ] -> nancumsum ?dtype ?axis p0
      | _ -> invalid_arg "nancumsum: arity")
    [ p0 ]

let nancumprod ?dtype:dt ?axis a =
  cumulative ~fill:(Some 1.0) ~promote:false ~make:cumprod_make ?dtype:dt ?axis
    a

let nancumprod ?dtype ?axis p0 =
  Jnp_util.jit1 "nancumprod"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string (dtype, axis) [])))
    (function
      | [ p0 ] -> nancumprod ?dtype ?axis p0
      | _ -> invalid_arg "nancumprod: arity")
    [ p0 ]

let cum_api_axis kind x axis =
  if ndim x = 0 then
    invalid_arg
      (Printf.sprintf
         "The input must be non-scalar to take a cumulative %s, however a \
          scalar value or scalar array was given."
         kind);
  match axis with
  | None ->
      if ndim x > 1 then
        invalid_arg
          (Printf.sprintf
             "The input array has rank %d, however axis was not set to an \
              explicit value. The axis argument is only optional for \
              one-dimensional arrays."
             (ndim x));
      0
  | Some ax -> canonicalize_axis ax (ndim x)

let cumulative_sum ?dtype:dt ?axis ?(include_initial = false) x =
  let ax = cum_api_axis "sum" x axis in
  let out =
    cumulative ~fill:None ~promote:true ~make:cumsum_make ?dtype:dt ~axis:ax x
  in
  let rt = dtype out in
  if include_initial then begin
    let zsh = Array.copy (shape x) in
    zsh.(ax) <- 1;
    bind1 (T.Concatenate ax) [ const_full rt zsh 0.0; out ]
  end
  else out

let cumulative_prod ?dtype:dt ?axis ?(include_initial = false) x =
  let ax = cum_api_axis "product" x axis in
  let out =
    cumulative ~fill:None ~promote:false ~make:cumprod_make ?dtype:dt ~axis:ax x
  in
  let rt = dtype out in
  if include_initial then begin
    let osh = Array.copy (shape x) in
    osh.(ax) <- 1;
    bind1 (T.Concatenate ax) [ const_full rt osh 1.0; out ]
  end
  else out

let transpose_perm v perm = bind1 (T.Transpose perm) [ v ]

let move_to_last v ax =
  let nd = ndim v in
  if ax = nd - 1 then v
  else begin
    let perm = Array.make nd 0 in
    let k = ref 0 in
    for i = 0 to nd - 1 do
      if i <> ax then begin
        perm.(!k) <- i;
        incr k
      end
    done;
    perm.(nd - 1) <- ax;
    transpose_perm v perm
  end

let reduce_last_sum v = bind1 (T.Reduce_sum [| ndim v - 1 |]) [ v ]
let reduce_last_or v = bind1 (T.Reduce_or [| ndim v - 1 |]) [ v ]

let quantile_nans a q ~axis ~method_ ~keepdims ~squash_nans =
  let a = to_inexact a in
  let cdt = dtype a in
  let nd0 = ndim a in
  let a, ax, keepdim_shape =
    match axis with
    | None ->
        let ks = if keepdims then Some (Array.make nd0 1) else None in
        (ravel a, 0, ks)
    | Some ax0 ->
        let ax = canonicalize_axis ax0 nd0 in
        let ks =
          if keepdims then begin
            let s = Array.copy (shape a) in
            s.(ax) <- 1;
            Some s
          end
          else None
        in
        (a, ax, ks)
  in
  let ndc = ndim a in
  let am = move_to_last a ax in
  let amsh = shape am in
  let n = amsh.(ndc - 1) in
  let bsh = Array.sub amsh 0 (ndc - 1) in
  let q = convert q cdt in
  let qsh = shape q in
  let qnd = Array.length qsh in
  let target = Array.append qsh bsh in
  let counts =
    if squash_nans then reduce_last_sum (convert (bind1 T.Not [ isnan am ]) cdt)
    else const_full cdt bsh (float_of_int n)
  in
  let am =
    if squash_nans then where (isnan am) (const_full cdt (shape am) infinity) am
    else am
  in
  let am_sorted =
    bind1
      (T.Sort { dimension = ndc - 1; is_stable = true; num_keys = 1 })
      [ am ]
  in
  let q_r = reshape q (Array.append qsh (Array.make (ndc - 1) 1)) in
  let counts_r = reshape counts (Array.append (Array.make qnd 1) bsh) in
  let q_full = broadcast_to q_r target in
  let counts_full = broadcast_to counts_r target in
  let cm1 = bind1 T.Sub [ counts_full; const_full cdt target 1.0 ] in
  let q_scaled = bind1 T.Mul [ q_full; cm1 ] in
  let low = bind1 T.Floor [ q_scaled ] in
  let high = bind1 T.Ceil [ q_scaled ] in
  let high_weight = bind1 T.Sub [ q_scaled; low ] in
  let low_weight = bind1 T.Sub [ const_full cdt target 1.0; high_weight ] in
  let clamp v =
    bind1 T.Max [ const_full cdt target 0.0; bind1 T.Min [ v; cm1 ] ]
  in
  let low_c = clamp low in
  let high_c = clamp high in
  let gshape = Array.append target [| n |] in
  let ng = Array.length gshape in
  let iota =
    bind1 (T.Iota { dtype = cdt; shape = gshape; dimension = ng - 1 }) []
  in
  let am_b = broadcast_to am_sorted gshape in
  let take idx =
    let idx_e = reshape idx (Array.append target [| 1 |]) in
    let idx_b = broadcast_to idx_e gshape in
    let onehot = bind1 T.Eq [ iota; idx_b ] in
    let masked = where onehot am_b (const_full cdt gshape 0.0) in
    reduce_last_sum masked
  in
  let low_value = take low_c in
  let high_value = take high_c in
  let result =
    match method_ with
    | "linear" ->
        bind1 T.Add
          [
            bind1 T.Mul [ low_value; low_weight ];
            bind1 T.Mul [ high_value; high_weight ];
          ]
    | "lower" -> low_value
    | "higher" -> high_value
    | "nearest" ->
        let pred = bind1 T.Le [ high_weight; const_full cdt target 0.5 ] in
        where pred low_value high_value
    | "midpoint" ->
        bind1 T.Mul
          [ bind1 T.Add [ low_value; high_value ]; const_full cdt target 0.5 ]
    | m -> invalid_arg ("jnp.quantile: unsupported method " ^ m)
  in
  let result =
    if squash_nans then result
    else begin
      let anynan = reduce_last_or (isnan am) in
      let anynan_r = reshape anynan (Array.append (Array.make qnd 1) bsh) in
      let anynan_full = broadcast_to anynan_r target in
      where anynan_full (const_full cdt target Float_format.nan) result
    end
  in
  match keepdim_shape with
  | None -> result
  | Some ks -> reshape result (Array.append qsh ks)

let quantile_plain a q ~axis ~method_ ~keepdims =
  let a = to_inexact a in
  if match dtype a with D.Complex64 | D.Complex128 -> true | _ -> false then
    invalid_arg
      "quantile does not support complex input, as the operation is poorly \
       defined.";
  let q = to_inexact q in
  let keepdim =
    if keepdims && axis = None then Array.make (ndim a) 1 else [||]
  in
  let a, ax =
    match axis with
    | None -> (ravel a, 0)
    | Some ax -> (a, canonicalize_axis ax (ndim a))
  in
  let q_ndim = ndim q in
  if q_ndim > 1 then
    invalid_arg (Printf.sprintf "q must be have rank <= 1, got rank %d" q_ndim);
  let a_shape = shape a in
  let nan_mask = any ~axis:[| ax |] ~keepdims:true (isnan a) in
  let a = where nan_mask (Jnp_util.py_float Float_format.nan) a in
  let a =
    bind1 (T.Sort { dimension = ax; is_stable = true; num_keys = 1 }) [ a ]
  in
  let qdt = dtype q in
  let lit dt v = T.Concrete (Nd.of_floats dt [||] [| v |]) in
  let n () = lit qdt (float_of_int a_shape.(ax)) in
  let n_minus_1 () = bind1 T.Sub [ n (); lit qdt 1.0 ] in
  let q = bind1 T.Mul [ q; n_minus_1 () ] in
  let low = bind1 T.Floor [ q ] in
  let high = bind1 T.Ceil [ q ] in
  let high_weight = bind1 T.Sub [ q; low ] in
  let low_weight = bind1 T.Sub [ lit qdt 1.0; high_weight ] in
  let clamped v =
    let top = n_minus_1 () in
    bind1 T.Clamp [ lit qdt 0.0; v; top ]
  in
  let low = clamped low in
  let high = clamped high in
  let idt = Dtypes.default_int_dtype () in
  let low = bind1 (T.Convert_element_type idt) [ low ] in
  let high = bind1 (T.Convert_element_type idt) [ high ] in
  let slice_sizes = Array.copy a_shape in
  slice_sizes.(ax) <- 1;
  let rank = Array.length a_shape in
  let offset_stop = if keepdims then rank + q_ndim else rank + q_ndim - 1 in
  let dimension_numbers =
    {
      T.offset_dims = Array.init (offset_stop - q_ndim) (fun i -> i + q_ndim);
      collapsed_slice_dims = (if keepdims then [||] else [| ax |]);
      start_index_map = [| ax |];
      g_operand_batching_dims = [||];
      g_start_indices_batching_dims = [||];
      g_sorted_unique = false;
    }
  in
  let gather_at idx =
    let idx_shape = shape idx in
    let wide =
      bind1
        (T.Broadcast_in_dim
           {
             shape = Array.append idx_shape [| 1 |];
             dims = Array.init (Array.length idx_shape) Fun.id;
           })
        [ idx ]
    in
    bind1 (T.Gather { dimension_numbers; slice_sizes }) [ a; wide ]
  in
  let low_value = gather_at low in
  let high_value = gather_at high in
  let widen w like =
    if q_ndim = 1 then
      bind1 (T.Broadcast_in_dim { shape = shape like; dims = [| 0 |] }) [ w ]
    else w
  in
  let low_weight = widen low_weight low_value in
  let high_weight = widen high_weight high_value in
  let result =
    match method_ with
    | "linear" ->
        let product v w = bind1 T.Mul [ convert v qdt; w ] in
        let left = product low_value low_weight in
        let right = product high_value high_weight in
        bind1 T.Add [ left; right ]
    | "lower" -> low_value
    | "higher" -> high_value
    | "nearest" ->
        let pred = bind1 T.Le [ high_weight; lit qdt 0.5 ] in
        bind1 T.Select_n [ pred; high_value; low_value ]
    | "midpoint" ->
        let total = bind1 T.Add [ low_value; high_value ] in
        bind1 T.Mul [ total; lit (dtype low_value) 0.5 ]
    | m ->
        invalid_arg
          (Printf.sprintf
             "method can only be 'linear', 'lower', 'higher', 'midpoint', \
              'nearest' or 'inverted_cdf', got %s"
             m)
  in
  let result =
    if keepdims && Array.length keepdim > 0 then
      reshape result
        (if q_ndim > 0 then Array.append [| (shape q).(0) |] keepdim
         else keepdim)
    else result
  in
  convert result (dtype a)

let quantile_core a q ~axis ~method_ ~keepdims ~squash_nans =
  if squash_nans then quantile_nans a q ~axis ~method_ ~keepdims ~squash_nans
  else quantile_plain a q ~axis ~method_ ~keepdims

let quantile ?axis ?(keepdims = false) ?(method_ = "linear") a q =
  quantile_core a q ~axis ~method_ ~keepdims ~squash_nans:false

let quantile ?axis ?keepdims ?method_ p0 p1 =
  Jnp_util.jit1 "quantile"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (axis, keepdims, method_) [])))
    (function
      | [ p0; p1 ] -> quantile ?axis ?keepdims ?method_ p0 p1
      | _ -> invalid_arg "quantile: arity")
    [ p0; p1 ]

let nanquantile ?axis ?(keepdims = false) ?(method_ = "linear") a q =
  quantile_core a q ~axis ~method_ ~keepdims ~squash_nans:true

let nanquantile ?axis ?keepdims ?method_ p0 p1 =
  Jnp_util.jit1 "nanquantile"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (axis, keepdims, method_) [])))
    (function
      | [ p0; p1 ] -> nanquantile ?axis ?keepdims ?method_ p0 p1
      | _ -> invalid_arg "nanquantile: arity")
    [ p0; p1 ]

let percentile ?axis ?(keepdims = false) ?(method_ = "linear") a q =
  let q = to_inexact q in
  let hundred = T.Concrete (Nd.of_floats (dtype q) [||] [| 100.0 |]) in
  quantile ?axis ~keepdims ~method_ a (bind1 T.Div [ q; hundred ])

let percentile ?axis ?keepdims ?method_ p0 p1 =
  Jnp_util.jit1 "percentile"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (axis, keepdims, method_) [])))
    (function
      | [ p0; p1 ] -> percentile ?axis ?keepdims ?method_ p0 p1
      | _ -> invalid_arg "percentile: arity")
    [ p0; p1 ]

let nanpercentile ?axis ?(keepdims = false) ?(method_ = "linear") a q =
  let cdt = to_inexact_dtype (dtype a) in
  let q100 = bind1 T.Div [ convert q cdt; const_full cdt (shape q) 100.0 ] in
  nanquantile ?axis ~keepdims ~method_ a q100

let nanpercentile ?axis ?keepdims ?method_ p0 p1 =
  Jnp_util.jit1 "nanpercentile"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (axis, keepdims, method_) [])))
    (function
      | [ p0; p1 ] -> nanpercentile ?axis ?keepdims ?method_ p0 p1
      | _ -> invalid_arg "nanpercentile: arity")
    [ p0; p1 ]

let median ?axis ?(keepdims = false) a =
  quantile ?axis ~keepdims ~method_:"midpoint" a (Jnp_util.py_float 0.5)

let median ?axis ?keepdims p0 =
  Jnp_util.jit1 "median"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string (axis, keepdims) [])))
    (function
      | [ p0 ] -> median ?axis ?keepdims p0 | _ -> invalid_arg "median: arity")
    [ p0 ]

let nanmedian ?axis ?(keepdims = false) a =
  let q = const_full (to_inexact_dtype (dtype a)) [||] 0.5 in
  nanquantile ?axis ~keepdims ~method_:"midpoint" a q

let nanmedian ?axis ?keepdims p0 =
  Jnp_util.jit1 "nanmedian"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string (axis, keepdims) [])))
    (function
      | [ p0 ] -> nanmedian ?axis ?keepdims p0
      | _ -> invalid_arg "nanmedian: arity")
    [ p0 ]

let count_nonzero ?axis ?keepdims a0 =
  Eager_jit.one "numpy.reductions.count_nonzero" (axis, keepdims) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> count_nonzero ?axis ?keepdims a0
      | _ -> assert false)

let nanmin ?axis ?keepdims ?initial ?where a0 =
  Eager_jit.one "numpy.reductions.nanmin" (axis, keepdims) [ initial; where ]
    [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ initial; where ], [ a0 ] -> nanmin ?axis ?keepdims ?initial ?where a0
      | _ -> assert false)

let nanmax ?axis ?keepdims ?initial ?where a0 =
  Eager_jit.one "numpy.reductions.nanmax" (axis, keepdims) [ initial; where ]
    [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ initial; where ], [ a0 ] -> nanmax ?axis ?keepdims ?initial ?where a0
      | _ -> assert false)

let nansum ?dtype ?axis ?keepdims ?initial ?where a0 =
  Eager_jit.one "numpy.reductions.nansum" (dtype, axis, keepdims)
    [ initial; where ] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ initial; where ], [ a0 ] ->
          nansum ?dtype ?axis ?keepdims ?initial ?where a0
      | _ -> assert false)

let nanprod ?dtype ?axis ?keepdims ?initial ?where a0 =
  Eager_jit.one "numpy.reductions.nanprod" (dtype, axis, keepdims)
    [ initial; where ] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ initial; where ], [ a0 ] ->
          nanprod ?dtype ?axis ?keepdims ?initial ?where a0
      | _ -> assert false)

let nanmean ?dtype ?axis ?keepdims ?where a0 =
  Eager_jit.one "numpy.reductions.nanmean" (dtype, axis, keepdims) [ where ]
    [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ where ], [ a0 ] -> nanmean ?dtype ?axis ?keepdims ?where a0
      | _ -> assert false)

let nanvar ?dtype ?axis ?keepdims ?ddof ?ddof_value ?where ?mean a0 =
  Eager_jit.one "numpy.reductions.nanvar" (dtype, axis, keepdims)
    [ where; mean; ddof_argument ddof ddof_value ]
    [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ where; mean; ddof ], [ a0 ] ->
          nanvar ?dtype ?axis ?keepdims ?ddof ?where ?mean a0
      | _ -> assert false)

let nanstd ?dtype ?axis ?keepdims ?ddof ?ddof_value ?where ?mean a0 =
  Eager_jit.one "numpy.reductions.nanstd" (dtype, axis, keepdims)
    [ where; mean; ddof_argument ddof ddof_value ]
    [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ where; mean; ddof ], [ a0 ] ->
          nanstd ?dtype ?axis ?keepdims ?ddof ?where ?mean a0
      | _ -> assert false)

let cumsum_by_keyword ?axis a0 =
  Eager_jit.one "numpy.reductions.cumsum_by_keyword" axis [] [ a0 ]
    (fun _ vals ->
      match vals with
      | [ a0 ] -> cumsum_keyed ~by_keyword:true ?axis a0
      | _ -> invalid_arg "cumsum: arity")

let cumsum ?dtype ?axis a0 =
  Eager_jit.one "numpy.reductions.cumsum" (dtype, axis) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> cumsum ?dtype ?axis a0
      | _ -> assert false)

let cumprod ?dtype ?axis a0 =
  Eager_jit.one "numpy.reductions.cumprod" (dtype, axis) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> cumprod ?dtype ?axis a0
      | _ -> assert false)

let nancumsum ?dtype ?axis a0 =
  Eager_jit.one "numpy.reductions.nancumsum" (dtype, axis) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> nancumsum ?dtype ?axis a0
      | _ -> assert false)

let nancumprod ?dtype ?axis a0 =
  Eager_jit.one "numpy.reductions.nancumprod" (dtype, axis) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> nancumprod ?dtype ?axis a0
      | _ -> assert false)

let quantile ?axis ?keepdims ?method_ a0 a1 =
  Eager_jit.one "numpy.reductions.quantile" (axis, keepdims, method_) []
    [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> quantile ?axis ?keepdims ?method_ a0 a1
      | _ -> assert false)

let nanquantile ?axis ?keepdims ?method_ a0 a1 =
  Eager_jit.one "numpy.reductions.nanquantile" (axis, keepdims, method_) []
    [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> nanquantile ?axis ?keepdims ?method_ a0 a1
      | _ -> assert false)

let nanpercentile ?axis ?keepdims ?method_ a0 a1 =
  Eager_jit.one "numpy.reductions.nanpercentile" (axis, keepdims, method_) []
    [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> nanpercentile ?axis ?keepdims ?method_ a0 a1
      | _ -> assert false)

let median ?axis ?keepdims a0 =
  Eager_jit.one "numpy.reductions.median" (axis, keepdims) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> median ?axis ?keepdims a0
      | _ -> assert false)

let nanmedian ?axis ?keepdims a0 =
  Eager_jit.one "numpy.reductions.nanmedian" (axis, keepdims) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> nanmedian ?axis ?keepdims a0
      | _ -> assert false)

let sum ?dtype ?axis ?keepdims ?initial ?where ?promote_integers a0 =
  Eager_jit.one "numpy.reductions.sum" (dtype, axis, keepdims, promote_integers)
    [ initial; where ] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ initial; where ], [ a0 ] ->
          sum ?dtype ?axis ?keepdims ?initial ?where ?promote_integers a0
      | _ -> assert false)

let prod ?dtype ?axis ?keepdims ?initial ?where ?promote_integers a0 =
  Eager_jit.one "numpy.reductions.prod"
    (dtype, axis, keepdims, promote_integers) [ initial; where ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ initial; where ], [ a0 ] ->
          prod ?dtype ?axis ?keepdims ?initial ?where ?promote_integers a0
      | _ -> assert false)

let max ?axis ?keepdims ?initial ?where a0 =
  Eager_jit.one "numpy.reductions.max" (axis, keepdims) [ initial; where ]
    [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ initial; where ], [ a0 ] -> max ?axis ?keepdims ?initial ?where a0
      | _ -> assert false)

let min ?axis ?keepdims ?initial ?where a0 =
  Eager_jit.one "numpy.reductions.min" (axis, keepdims) [ initial; where ]
    [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ initial; where ], [ a0 ] -> min ?axis ?keepdims ?initial ?where a0
      | _ -> assert false)

let amax ?axis ?keepdims ?initial ?where a0 =
  Eager_jit.one "numpy.reductions.amax" (axis, keepdims) [ initial; where ]
    [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ initial; where ], [ a0 ] -> amax ?axis ?keepdims ?initial ?where a0
      | _ -> assert false)

let amin ?axis ?keepdims ?initial ?where a0 =
  Eager_jit.one "numpy.reductions.amin" (axis, keepdims) [ initial; where ]
    [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ initial; where ], [ a0 ] -> amin ?axis ?keepdims ?initial ?where a0
      | _ -> assert false)

let all ?axis ?keepdims ?where a0 =
  Eager_jit.one "numpy.reductions.all" (axis, keepdims) [ where ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ where ], [ a0 ] -> all ?axis ?keepdims ?where a0
      | _ -> assert false)

let any ?axis ?keepdims ?where a0 =
  Eager_jit.one "numpy.reductions.any" (axis, keepdims) [ where ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ where ], [ a0 ] -> any ?axis ?keepdims ?where a0
      | _ -> assert false)

let mean ?dtype ?axis ?keepdims ?where a0 =
  Eager_jit.one "numpy.reductions.mean" (dtype, axis, keepdims) [ where ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ where ], [ a0 ] -> mean ?dtype ?axis ?keepdims ?where a0
      | _ -> assert false)

let ptp ?axis ?keepdims a0 =
  Eager_jit.one "numpy.reductions.ptp" (axis, keepdims) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> ptp ?axis ?keepdims a0
      | _ -> assert false)

let average ?axis ?keepdims ?weights a0 =
  Eager_jit.one "numpy.reductions.average" (axis, keepdims) [ weights ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ weights ], [ a0 ] -> average ?axis ?keepdims ?weights a0
      | _ -> assert false)

let percentile ?axis ?keepdims ?method_ a0 a1 =
  Eager_jit.one "numpy.reductions.percentile" (axis, keepdims, method_) []
    [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> percentile ?axis ?keepdims ?method_ a0 a1
      | _ -> assert false)
