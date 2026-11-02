module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Numpy.Lax_numpy

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let bind1 = C.bind1
let const_full dt sh x = Core.splat dt sh x
let num_segments_context = "segment_sum() `num_segments` argument."
let scalar_int nd = int_of_float (Nd.fold (fun _ x -> x) 0.0 nd)

let rec concrete_dim v =
  match v with
  | T.Concrete nd -> scalar_int nd
  | T.Device b -> scalar_int (Pjrt.Buffer.to_host b)
  | T.Tracer { payload = T.JVP { primal; _ }; _ } -> concrete_dim primal
  | T.Tracer _ ->
      let aval = get_aval v in
      raise
        (Errors.concretization_type_error ~context:num_segments_context
           ~error_repr:
             (Printf.sprintf "traced array with shape %s[%s]"
                (D.numpy_name aval.T.dtype)
                (String.concat ","
                   (Array.to_list (Array.map string_of_int aval.T.shape))))
           ~origin_msg:(Jaxpr.origin_msg v) ())

let identity_sum = 0.0
let identity_prod = 1.0

let identity_max dt =
  match dt with
  | D.F32 | D.F64 | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn
  | D.F8_e3m4 | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz
  | D.F8_e5m2 | D.F8_e5m2fnuz | D.F8_e8m0fnu ->
      Dtypes.max_identity dt
  | D.I2 -> -2.0
  | D.I4 -> -8.0
  | D.Float0 -> invalid_arg "segment: float0 has no max identity"
  | D.I8 -> -128.0
  | D.I16 -> -32768.0
  | D.Uint8 | D.Uint16 | D.Uint2 | D.Uint4 | D.Uint64 -> 0.0
  | D.I32 -> -2147483648.0
  | D.I64 -> -9223372036854775808.0
  | D.Uint32 -> 0.0
  | D.Bool -> 0.0
  | D.Complex64 | D.Complex128 ->
      invalid_arg "segment: complex has no max identity"

let identity_min dt =
  match dt with
  | D.F32 | D.F64 | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn
  | D.F8_e3m4 | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz
  | D.F8_e5m2 | D.F8_e5m2fnuz | D.F8_e8m0fnu ->
      Dtypes.min_identity dt
  | D.I2 -> 1.0
  | D.I4 -> 7.0
  | D.Uint2 -> 3.0
  | D.Uint4 -> 15.0
  | D.Float0 -> invalid_arg "segment: float0 has no min identity"
  | D.I8 -> 127.0
  | D.I16 -> 32767.0
  | D.Uint8 -> 255.0
  | D.Uint16 -> 65535.0
  | D.Uint64 -> 18446744073709551615.0
  | D.I32 -> 2147483647.0
  | D.I64 -> 9223372036854775807.0
  | D.Uint32 -> 4294967295.0
  | D.Bool -> 1.0
  | D.Complex64 | D.Complex128 ->
      invalid_arg "segment: complex has no min identity"

let segment_dnums rank =
  {
    T.update_window_dims = Array.init (rank - 1) (fun i -> i + 1);
    inserted_window_dims = [| 0 |];
    scatter_dims_to_operand_dims = [| 0 |];
    s_operand_batching_dims = [||];
    s_scatter_indices_batching_dims = [||];
    s_sorted_unique = false;
    s_clip = false;
  }

let scatter_flagged prim ~indices_are_sorted ~unique_indices =
  if indices_are_sorted || unique_indices then
    Np_ints.mark_scatter_flags ~sorted:indices_are_sorted ~unique:unique_indices
      prim
  else prim

let segment_update prim identity reducer ?num_segments
    ?(indices_are_sorted = false) ?(unique_indices = false) ?bucket_size ?mode
    data segment_ids =
  let dt = dtype data in
  let dsh = shape data in
  let n = if Array.length dsh = 0 then 0 else dsh.(0) in
  let nseg =
    match num_segments with
    | Some k -> k
    | None ->
        concrete_dim
          (Numpy.Ufuncs.add
             (Numpy.Reductions.max segment_ids)
             (Numpy.Jnp_util.py_int 1))
  in
  if nseg < 0 then invalid_arg "num_segments must be non-negative.";
  let mode = Option.map Lax_config.gather_scatter_mode_of_string mode in
  let clip = mode = Some Lax_config.Clip in
  let narrow ids =
    if dtype ids = D.I64 then bind1 (T.Convert_element_type D.I32) [ ids ]
    else ids
  in
  let finish scatter_prim args =
    let prim =
      scatter_flagged ~indices_are_sorted ~unique_indices scatter_prim
    in
    let prim =
      match mode with
      | Some Lax_config.Promise_in_bounds -> Np_ints.mark_prim prim
      | Some Lax_config.One_hot -> Np_ints.mark_scatter_one_hot prim
      | _ -> prim
    in
    bind1 prim args
  in
  match bucket_size with
  | None ->
      let out_shape = Array.copy dsh in
      if Array.length out_shape > 0 then out_shape.(0) <- nseg;
      let out = const_full dt out_shape identity in
      let scatter_indices =
        bind1
          (T.Broadcast_in_dim { shape = [| n; 1 |]; dims = [| 0 |] })
          [ narrow segment_ids ]
      in
      let dnums = { (segment_dnums (Array.length dsh)) with T.s_clip = clip } in
      finish (prim dnums) [ out; scatter_indices; data ]
  | Some size ->
      let num_buckets = (n + size - 1) / size in
      let tail = Array.sub dsh 1 (Array.length dsh - 1) in
      let out =
        const_full dt (Array.append [| num_buckets; nseg |] tail) identity
      in
      let idt = Dtypes.default_int_dtype () in
      let iota =
        bind1 (T.Iota { dtype = idt; shape = [| n |]; dimension = 0 }) []
      in
      let buckets =
        Numpy.Ufuncs.floor_divide iota (Numpy.Jnp_util.py_int size)
      in
      let broadcast v shape dims =
        bind1 (T.Broadcast_in_dim { shape; dims }) [ v ]
      in
      let ids_row = broadcast segment_ids [| 1; n |] [| 1 |] in
      let bucket_row = narrow (broadcast buckets [| 1; n |] [| 1 |]) in
      let ids_row = narrow ids_row in
      let bucket_col = broadcast bucket_row [| 1; n; 1 |] [| 0; 1 |] in
      let ids_col = broadcast ids_row [| 1; n; 1 |] [| 0; 1 |] in
      let stacked = bind1 (T.Concatenate 2) [ bucket_col; ids_col ] in
      let rank = Array.length dsh in
      let dnums =
        {
          T.update_window_dims = Array.init (rank - 1) (fun i -> i + 2);
          inserted_window_dims = [| 0; 1 |];
          scatter_dims_to_operand_dims = [| 0; 1 |];
          s_operand_batching_dims = [||];
          s_scatter_indices_batching_dims = [||];
          s_sorted_unique = false;
          s_clip = clip;
        }
      in
      let updates =
        broadcast data (Array.append [| 1 |] dsh)
          (Array.init (Array.length dsh) (fun i -> i + 1))
      in
      let scattered = finish (prim dnums) [ out; stacked; updates ] in
      NL.astype (reducer scattered) dt

let segment_sum ?num_segments ?indices_are_sorted ?unique_indices ?bucket_size
    ?mode data segment_ids =
  segment_update
    (fun dnums -> T.Scatter_add { dimension_numbers = dnums })
    identity_sum
    (Numpy.Reductions.sum ~axis:[| 0 |])
    ?num_segments ?indices_are_sorted ?unique_indices ?bucket_size ?mode data
    segment_ids

let segment_prod ?num_segments ?indices_are_sorted ?unique_indices ?bucket_size
    ?mode data segment_ids =
  segment_update
    (fun dnums ->
      T.Scatter_mul { dimension_numbers = dnums; unique_indices = false })
    identity_prod
    (Numpy.Reductions.prod ~axis:[| 0 |])
    ?num_segments ?indices_are_sorted ?unique_indices ?bucket_size ?mode data
    segment_ids

let segment_max ?num_segments ?indices_are_sorted ?unique_indices ?bucket_size
    ?mode data segment_ids =
  segment_update
    (fun dnums -> T.Scatter_max { dimension_numbers = dnums })
    (identity_max (dtype data))
    (Numpy.Reductions.max ~axis:[| 0 |])
    ?num_segments ?indices_are_sorted ?unique_indices ?bucket_size ?mode data
    segment_ids

let segment_min ?num_segments ?indices_are_sorted ?unique_indices ?bucket_size
    ?mode data segment_ids =
  segment_update
    (fun dnums -> T.Scatter_min { dimension_numbers = dnums })
    (identity_min (dtype data))
    (Numpy.Reductions.min ~axis:[| 0 |])
    ?num_segments ?indices_are_sorted ?unique_indices ?bucket_size ?mode data
    segment_ids
