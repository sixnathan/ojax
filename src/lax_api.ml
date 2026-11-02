open Types

type padding = Same | Same_lower | Valid | Explicit of (int * int) array

let b1 = Core.bind1
let aval v = Core.get_aval v
let shape v = (aval v).shape
let dtype_of v = (aval v).dtype
let ndim v = Array.length (shape v)
let canon_axis n a = if a < 0 then a + n else a
let product a = Array.fold_left ( * ) 1 a
let filled dt sh x = Core.splat dt sh x

let broadcast_scalar v sh =
  if shape v = sh then v
  else if ndim v = 0 then
    b1 (Broadcast_in_dim { shape = sh; dims = [||] }) [ v ]
  else
    invalid_arg
      (Printf.sprintf
         "lax operands must have equal shapes or be scalars, got ranks %d and \
          %d"
         (ndim v) (Array.length sh))

let unify a b =
  if shape a = shape b || ndim a = 0 || ndim a = ndim b then (a, b)
  else (a, broadcast_scalar b (if ndim b = 0 then shape b else shape a))

let lift2 prim a b =
  let x, y = unify a b in
  b1 prim [ x; y ]

let abs x = b1 Abs [ x ]
let acos x = b1 Acos [ x ]
let acosh x = b1 Acosh [ x ]
let add = lift2 Add
let asin x = b1 Asin [ x ]
let asinh x = b1 Asinh [ x ]
let atan x = b1 Atan [ x ]
let atan2 = lift2 Atan2
let atanh x = b1 Atanh [ x ]
let bitwise_and = lift2 And
let bitwise_not x = b1 Not [ x ]
let bitwise_or = lift2 Or
let bitwise_xor = lift2 Xor

let cbrt ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Cbrt [ x ]

let ceil x = b1 Ceil [ x ]
let clz x = b1 Clz [ x ]
let complex = lift2 Complex
let conj x = b1 Conj [ x ]

let cos ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Cos [ x ]

let cosh x = b1 Cosh [ x ]
let div = lift2 Div
let eq = lift2 Eq

let exp ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Exp [ x ]

let exp2 ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Exp2 [ x ]

let expm1 ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Expm1 [ x ]

let floor x = b1 Floor [ x ]
let ge = lift2 Ge
let gt = lift2 Gt
let imag x = b1 Imag [ x ]
let integer_pow x y = b1 (Integer_pow y) [ x ]
let is_finite x = b1 Is_finite [ x ]
let le = lift2 Le

let log ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Log [ x ]

let log1p ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Log1p [ x ]

let logistic ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Logistic [ x ]

let lt = lift2 Lt
let max = lift2 Max
let min = lift2 Min
let mul = lift2 Mul
let mulhi = lift2 Mulhi
let ne = lift2 Ne
let neg x = b1 Neg [ x ]
let nextafter = lift2 Nextafter
let population_count x = b1 Population_count [ x ]
let pow = lift2 Pow
let real x = b1 Real [ x ]
let reciprocal x = b1 (Integer_pow (-1)) [ x ]
let rem = lift2 Rem

type rounding_method = Lax_config.rounding_method =
  | Away_from_zero
  | To_nearest_even

let round ?(rounding_method = Away_from_zero) x =
  match rounding_method with
  | Away_from_zero -> b1 Round [ x ]
  | To_nearest_even -> b1 Round_nearest_even [ x ]

let stage x = Core.full_raise (Core.find_top_trace [ x ]) x

let rsqrt ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Rsqrt [ x ]

let shift_left = lift2 Shift_left
let shift_right_arithmetic = lift2 Shift_right_arithmetic
let shift_right_logical = lift2 Shift_right_logical
let sign x = b1 Sign [ x ]

let sin ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Sin [ x ]

let sinh x = b1 Sinh [ x ]

let sqrt ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Sqrt [ x ]

let square x = b1 Square [ x ]
let sub = lift2 Sub

let tan ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Tan [ x ]

let tanh ?accuracy x =
  ignore (accuracy : Lax_config.accuracy option);
  b1 Tanh [ x ]

let betainc a b x = b1 Regularized_incomplete_beta [ a; b; x ]
let stop_gradient = Custom_derivatives.stop_gradient
let optimization_barrier xs = Core.bind Optimization_barrier xs
let convert_element_type x dt = b1 (Convert_element_type dt) [ x ]
let bitcast_convert_type x dt = b1 (Bitcast_convert_type dt) [ x ]

let reduce_precision x ~exponent_bits ~mantissa_bits =
  b1 (Reduce_precision { exponent_bits; mantissa_bits }) [ x ]

let clamp lo x hi = b1 Clamp [ lo; x; hi ]
let select pred on_true on_false = b1 Select_n [ pred; on_false; on_true ]

let select_n pred cases =
  let sh =
    match cases with c :: _ -> shape c | [] -> invalid_arg "select_n"
  in
  b1 Select_n (broadcast_scalar pred sh :: cases)

let full sh fill_value dt = filled dt sh fill_value

let full_like x fill_value =
  let a = aval x in
  if not a.weak_type then filled a.dtype a.shape fill_value
  else if Array.length a.shape = 0 then Core.py_scalar a.dtype fill_value
  else if Core.staging () then
    b1
      (Broadcast_in_dim { shape = a.shape; dims = [||] })
      [ Core.py_scalar a.dtype fill_value ]
  else
    match filled a.dtype a.shape fill_value with
    | Concrete nd -> Concrete (Ndarray.as_weak nd)
    | other -> other

let empty dt = filled dt [| 0 |] 0.0

let iota dt size =
  b1 (Iota { dtype = dt; shape = [| size |]; dimension = 0 }) []

let broadcasted_iota dt sh dimension =
  b1 (Iota { dtype = dt; shape = sh; dimension }) []

let reshape ?dimensions x new_sizes =
  let v =
    match dimensions with Some perm -> b1 (Transpose perm) [ x ] | None -> x
  in
  if
    Array.length new_sizes > 0
    && dimensions = None
    && new_sizes = (Core.get_aval v).shape
  then v
  else b1 (Reshape new_sizes) [ v ]

let transpose x perm = b1 (Transpose perm) [ x ]
let rev x dims = b1 (Rev dims) [ x ]
let tile x reps = b1 (Tile reps) [ x ]

let squeeze x dims =
  let n = ndim x in
  if Array.length dims = 0 then x
  else b1 (Squeeze (Array.map (canon_axis n) dims)) [ x ]

let expand_dims x dims =
  let n = ndim x + Array.length dims in
  let dims = Array.map (canon_axis n) dims in
  let sh = shape x in
  let rest = ref (Array.to_list sh) in
  let out =
    Array.init n (fun i ->
        if Array.mem i dims then 1
        else
          match !rest with
          | h :: t ->
              rest := t;
              h
          | [] -> invalid_arg "expand_dims")
  in
  let kept =
    List.filter (fun i -> not (Array.mem i dims)) (List.init n Fun.id)
  in
  if Array.length dims = 0 then x
  else b1 (Broadcast_in_dim { shape = out; dims = Array.of_list kept }) [ x ]

let broadcast_in_dim x sh dims =
  b1 (Broadcast_in_dim { shape = sh; dims }) [ x ]

let broadcast x sizes =
  let k = Array.length sizes in
  let sh = Array.append sizes (shape x) in
  broadcast_in_dim x sh (Array.init (ndim x) (fun i -> i + k))

let broadcast_to_rank x rank =
  let missing = rank - ndim x in
  if missing <= 0 then x else expand_dims x (Array.init missing Fun.id)

let broadcast_shapes shapes =
  let rank =
    List.fold_left (fun m s -> Stdlib.max m (Array.length s)) 0 shapes
  in
  Array.init rank (fun i ->
      List.fold_left
        (fun acc s ->
          let k = i - (rank - Array.length s) in
          let d = if k < 0 then 1 else s.(k) in
          if d = 1 then acc
          else if acc = 1 || acc = d then d
          else invalid_arg "Incompatible shapes for broadcasting")
        1 shapes)

let collapse_from x start =
  let sh = shape x in
  let n = Array.length sh in
  let start = if start < 0 then start + n else start in
  let size = product (Array.sub sh start (n - start)) in
  b1 (Reshape (Array.append (Array.sub sh 0 start) [| size |])) [ x ]

let collapse x start stop =
  let sh = shape x in
  let n = Array.length sh in
  let stop = if stop < 0 then stop + n else stop in
  let size = product (Array.sub sh start (stop - start)) in
  let out =
    Array.concat
      [ Array.sub sh 0 start; [| size |]; Array.sub sh stop (n - stop) ]
  in
  b1 (Reshape out) [ x ]

let concatenate xs dimension = b1 (Concatenate dimension) xs
let stack xs axis = b1 (Stack axis) xs
let unstack x axis = Core.bind (Unstack axis) [ x ]
let split x sizes axis = Core.bind (Split { sizes; axis }) [ x ]
let pad x value config = b1 (Pad config) [ x; value ]

let slice ?strides x start_indices limit_indices =
  b1 (Slice { start_indices; limit_indices; strides }) [ x ]

let slice_in_dim ?(stride = 1) ?(axis = 0) x start stop =
  let sh = shape x in
  let n = Array.length sh in
  let axis = canon_axis n axis in
  let bound v =
    let v = if v < 0 then v + sh.(axis) else v in
    Stdlib.min (Stdlib.max v 0) sh.(axis)
  in
  let lo = Array.make n 0 and hi = Array.copy sh and st = Array.make n 1 in
  lo.(axis) <- bound start;
  hi.(axis) <- bound stop;
  st.(axis) <- stride;
  slice ~strides:st x lo hi

let index_in_dim ?(axis = 0) ?(keepdims = true) x index =
  let axis = canon_axis (ndim x) axis in
  let size = (shape x).(axis) in
  let index = if index < 0 then index + size else index in
  if index < 0 || index >= size then invalid_arg "index out of bounds for axis";
  let v = slice_in_dim ~axis x index (index + 1) in
  if keepdims then v else b1 (Squeeze [| axis |]) [ v ]

let index_scalar i =
  Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int i |])

let normalized_start start size =
  let aval = Core.get_aval start in
  let dt = aval.dtype in
  let literal v = Concrete (Ndarray.of_floats dt [||] [| v |]) in
  if Dtypes.issubdtype dt `Unsigned then start
  else
    match start with
    | Concrete nd when Array.length (Ndarray.shape nd) = 0 ->
        let v = Ndarray.get_f nd [||] in
        if v < 0.0 then literal (v +. float_of_int size) else start
    | _ ->
        let negative = b1 Lt [ start; literal 0.0 ] in
        let widened =
          if aval.weak_type then b1 (Convert_element_type dt) [ start ]
          else start
        in
        let wrapped = b1 Add [ widened; literal (float_of_int size) ] in
        b1 Select_n [ negative; start; wrapped ]

let dynamic_slice_normalized x starts slice_sizes =
  b1 (Dynamic_slice { slice_sizes }) (x :: starts)

let dynamic_slice ?(allow_negative_indices = true) x starts slice_sizes =
  let starts =
    if allow_negative_indices then
      List.mapi (fun i start -> normalized_start start (shape x).(i)) starts
    else starts
  in
  dynamic_slice_normalized x starts slice_sizes

let dynamic_update_slice ?(allow_negative_indices = false) x update starts =
  let starts =
    if allow_negative_indices then
      List.mapi (fun i start -> normalized_start start (shape x).(i)) starts
    else starts
  in
  b1 Dynamic_update_slice (x :: update :: starts)

let starts_in_dim ?(allow_negative_indices = true) x axis start =
  let dt = (Core.get_aval start).dtype in
  let zero = Concrete (Ndarray.of_floats dt [||] [| 0.0 |]) in
  let sh = shape x in
  let start =
    if allow_negative_indices then normalized_start start sh.(axis) else start
  in
  List.init (Array.length sh) (fun d -> if d = axis then start else zero)

let dynamic_slice_in_dim ?(axis = 0) ?allow_negative_indices x start size =
  let n = ndim x in
  let axis = canon_axis n axis in
  let sizes = Array.copy (shape x) in
  sizes.(axis) <- size;
  dynamic_slice_normalized x
    (starts_in_dim ?allow_negative_indices x axis start)
    sizes

let dynamic_index_in_dim ?(axis = 0) ?(keepdims = true) x index =
  let axis = canon_axis (ndim x) axis in
  match
    Jaxpr.named_jit ~name:"dynamic_index_in_dim"
      ~key:(Printf.sprintf "%d %b" axis keepdims)
      (function
        | [ x; index ] ->
            let v = dynamic_slice_in_dim ~axis x index 1 in
            [ (if keepdims then v else b1 (Squeeze [| axis |]) [ v ]) ]
        | _ -> invalid_arg "dynamic_index_in_dim: arity")
      [ x; index ]
  with
  | [ v ] -> v
  | _ -> invalid_arg "dynamic_index_in_dim: arity"

let dynamic_update_slice_in_dim ?(axis = 0) ?allow_negative_indices x update
    start =
  let n = ndim x in
  let axis = canon_axis n axis in
  dynamic_update_slice x update
    (starts_in_dim ?allow_negative_indices x axis start)

let dynamic_update_index_in_dim ?(axis = 0) x update index =
  let axis = canon_axis (ndim x) axis in
  let update =
    if ndim update = ndim x then update else expand_dims update [| axis |]
  in
  dynamic_update_slice_in_dim ~axis x update index

let index_upper_bound indices dims window =
  let idx_shape = shape indices in
  let bound =
    Concrete
      (Ndarray.of_floats (dtype_of indices)
         [| Array.length dims |]
         (Array.map (fun d -> float_of_int window.(d)) dims))
  in
  b1
    (Broadcast_in_dim
       { shape = idx_shape; dims = [| Array.length idx_shape - 1 |] })
    [ bound ]

let clamp_indices indices dims window =
  b1 Clamp
    [
      filled (dtype_of indices) (shape indices) 0.0;
      indices;
      index_upper_bound indices dims window;
    ]

let gather_default_fill dt =
  if Dtypes.issubdtype dt `Inexact then Float.nan
  else if dt = Dtype.Bool then 1.0
  else if Dtypes.issubdtype dt `Signed then (Dtypes.iinfo dt).imin
  else (Dtypes.iinfo dt).imax

let gather ?(mode = Lax_config.Promise_in_bounds) ?fill_value
    ?(indices_are_sorted = false) ?(unique_indices = false) x indices
    dimension_numbers slice_sizes =
  let flag prim =
    if indices_are_sorted || unique_indices then
      Np_ints.mark_gather_flags ~sorted:indices_are_sorted
        ~unique:unique_indices prim
    else prim
  in
  let prim = Gather { dimension_numbers; slice_sizes } in
  match mode with
  | Lax_config.Promise_in_bounds -> b1 (flag prim) [ x; indices ]
  | Lax_config.Clip ->
      b1 (Np_ints.mark_gather_clip ~sorted:false (flag prim)) [ x; indices ]
  | Lax_config.One_hot ->
      invalid_arg "gather: GatherScatterMode.ONE_HOT is not supported"
  | Lax_config.Fill_or_drop ->
      Lax.Slicing.gather_fill ~flag ?fill_value x indices dimension_numbers
        slice_sizes

let scatter_indices mode x indices updates (dn : scatter_dims) =
  match mode with
  | Some Lax_config.Clip ->
      let pos = ref 0 in
      let window =
        Array.mapi
          (fun d size ->
            if
              Array.mem d dn.inserted_window_dims
              || Array.mem d dn.s_operand_batching_dims
            then size - 1
            else begin
              let extent = (shape updates).(dn.update_window_dims.(!pos)) in
              incr pos;
              size - extent
            end)
          (shape x)
      in
      clamp_indices indices dn.scatter_dims_to_operand_dims window
  | _ -> indices

let flagged_scatter prim ~indices_are_sorted ~unique_indices =
  if indices_are_sorted || unique_indices then
    Np_ints.mark_scatter_flags ~sorted:indices_are_sorted ~unique:unique_indices
      prim
  else prim

let scatter ?mode ?(indices_are_sorted = false) ?(unique_indices = false) x
    indices updates dimension_numbers =
  let indices = scatter_indices mode x indices updates dimension_numbers in
  b1
    (flagged_scatter ~indices_are_sorted ~unique_indices:false
       (Scatter { dimension_numbers; unique_indices }))
    [ x; indices; updates ]

let scatter_add ?mode ?(indices_are_sorted = false) ?(unique_indices = false) x
    indices updates dimension_numbers =
  let indices = scatter_indices mode x indices updates dimension_numbers in
  b1
    (flagged_scatter ~indices_are_sorted ~unique_indices
       (Scatter_add { dimension_numbers }))
    [ x; indices; updates ]

let scatter_sub ?mode ?(indices_are_sorted = false) ?(unique_indices = false) x
    indices updates dimension_numbers =
  let indices = scatter_indices mode x indices updates dimension_numbers in
  b1
    (flagged_scatter ~indices_are_sorted ~unique_indices
       (Scatter_sub { dimension_numbers }))
    [ x; indices; updates ]

let scatter_mul ?mode ?(indices_are_sorted = false) ?(unique_indices = false) x
    indices updates dimension_numbers =
  let indices = scatter_indices mode x indices updates dimension_numbers in
  b1
    (flagged_scatter ~indices_are_sorted ~unique_indices:false
       (Scatter_mul { dimension_numbers; unique_indices }))
    [ x; indices; updates ]

let scatter_min ?mode ?(indices_are_sorted = false) ?(unique_indices = false) x
    indices updates dimension_numbers =
  let indices = scatter_indices mode x indices updates dimension_numbers in
  b1
    (flagged_scatter ~indices_are_sorted ~unique_indices
       (Scatter_min { dimension_numbers }))
    [ x; indices; updates ]

let scatter_max ?mode ?(indices_are_sorted = false) ?(unique_indices = false) x
    indices updates dimension_numbers =
  let indices = scatter_indices mode x indices updates dimension_numbers in
  b1
    (flagged_scatter ~indices_are_sorted ~unique_indices
       (Scatter_max { dimension_numbers }))
    [ x; indices; updates ]

let reduce_sum x axes = b1 (Reduce_sum axes) [ x ]
let reduce_prod x axes = b1 (Reduce_prod axes) [ x ]
let reduce_max x axes = b1 (Reduce_max axes) [ x ]
let reduce_min x axes = b1 (Reduce_min axes) [ x ]
let reduce_and x axes = b1 (Reduce_and axes) [ x ]
let reduce_or x axes = b1 (Reduce_or axes) [ x ]
let reduce_xor x axes = b1 (Reduce_xor axes) [ x ]
let scalar_aval dt = { shape = [||]; dtype = dt; weak_type = false }

let reducer_jaxpr dt computation =
  let a = scalar_aval dt in
  Jaxpr.make_jaxpr [ a; a ] (function
    | [ x; y ] -> [ computation x y ]
    | _ -> invalid_arg "reduce: computation takes two arguments")

let reduce x init computation dimensions =
  let jaxpr = reducer_jaxpr (dtype_of x) computation in
  b1 (Reduce { jaxpr; dimensions }) [ x; init ]

let get_max_identity dt =
  match dt with
  | Dtype.Bool -> 0.0
  | _ when Dtypes.issubdtype dt `Inexact -> Float.neg_infinity
  | _ -> (Dtypes.iinfo dt).imin

let get_min_identity dt =
  match dt with
  | Dtype.Bool -> 1.0
  | _ when Dtypes.issubdtype dt `Inexact -> Float.infinity
  | _ -> (Dtypes.iinfo dt).imax

let concrete_scalar v =
  let read nd =
    match Ndarray.dtype nd with
    | Dtype.Complex64 | Dtype.Complex128 ->
        let z = Ndarray.get_c nd [||] in
        if z.Complex.im = 0.0 then Some z.Complex.re else Some Float.nan
    | _ -> Some (Ndarray.get_f nd [||])
  in
  match v with
  | Tracer _ -> None
  | _ when ndim v <> 0 -> None
  | Concrete nd -> read nd
  | Device d -> read (Pjrt.Buffer.to_host d)

let reduce_window x init computation ~window_dimensions ~window_strides
    ?base_dilation ?window_dilation padding_config =
  let n = ndim x in
  let ones = Array.make n 1 in
  let window =
    {
      window_dimensions;
      window_strides;
      w_padding = padding_config;
      base_dilation = Option.value base_dilation ~default:ones;
      window_dilation = Option.value window_dilation ~default:ones;
    }
  in
  let generic () =
    let reducer = reducer_jaxpr (dtype_of x) computation in
    b1 (Reduce_window { reducer; window }) [ x; init ]
  in
  match concrete_scalar init with
  | Some v when computation == add && v = 0.0 ->
      b1 (Reduce_window_sum window) [ x ]
  | Some v when computation == max && v = get_max_identity (dtype_of init) ->
      b1 (Reduce_window_max window) [ x ]
  | Some v when computation == min && v = get_min_identity (dtype_of init) ->
      b1 (Reduce_window_min window) [ x ]
  | _ -> generic ()

let argmax x ~axis ~index_dtype = b1 (Argmax { axis; index_dtype }) [ x ]
let argmin x ~axis ~index_dtype = b1 (Argmin { axis; index_dtype }) [ x ]

let sort ?(dimension = -1) ?(is_stable = true) ?(num_keys = 1) xs =
  match xs with
  | [] -> invalid_arg "sort requires at least one operand"
  | first :: _ ->
      let dimension = canon_axis (ndim first) dimension in
      Core.bind (Sort { dimension; is_stable; num_keys }) xs

let sort_key_val ?dimension ?is_stable keys values =
  match sort ?dimension ?is_stable ~num_keys:1 [ keys; values ] with
  | [ k; v ] -> (k, v)
  | _ -> failwith "sort_key_val: arity"

let top_k ?(axis = -1) x k =
  let axis = canon_axis (ndim x) axis in
  match Core.bind (Top_k { k; axis }) [ x ] with
  | [ values; indices ] -> (values, indices)
  | _ -> failwith "top_k: arity"

let dtype_bits = Dtypes.itemsize_bits

let convert_to dt v =
  if dtype_of v = dt then v else b1 (Convert_element_type dt) [ v ]

let preset_supported_on_cpu = function
  | Lax_config.Preset_default | Lax_config.F16_f16_f16 | Lax_config.F32_f32_f32
  | Lax_config.F64_f64_f64 | Lax_config.Bf16_bf16_f32
  | Lax_config.Bf16_bf16_f32_x3 | Lax_config.Bf16_bf16_f32_x6 ->
      true
  | _ -> false

let preset_input_types = function
  | Lax_config.Preset_default -> None
  | Lax_config.F16_f16_f16 -> Some [ Dtype.F16 ]
  | Lax_config.Bf16_bf16_f32 -> Some [ Dtype.Bf16; Dtype.F32 ]
  | Lax_config.F64_f64_f64 -> Some [ Dtype.F64 ]
  | _ -> Some [ Dtype.F32 ]

let preset_output_types preset lhs rhs =
  match preset with
  | Lax_config.Preset_default -> None
  | Lax_config.F16_f16_f16 -> Some [ Dtype.F16 ]
  | Lax_config.Bf16_bf16_f32 ->
      if Dtypes.promote_types lhs rhs = Dtype.Bf16 then
        Some [ Dtype.F32; Dtype.Bf16 ]
      else Some [ Dtype.F32 ]
  | Lax_config.F64_f64_f64 -> Some [ Dtype.F64 ]
  | _ -> Some [ Dtype.F32 ]

let pick_supported dt = function
  | None -> dt
  | Some targets -> if List.mem dt targets then dt else List.hd targets

let precision_rank dt =
  let float_rank complex component =
    let info = Dtypes.finfo component in
    let _, overflow_exponent = Float.frexp info.fmax in
    (complex, overflow_exponent, info.nmant, info.bits)
  in
  match dt with
  | Dtype.Bool -> (0, -1, -1, 8)
  | Dtype.Complex64 -> float_rank 1 Dtype.F32
  | Dtype.Complex128 -> float_rank 1 Dtype.F64
  | _ when Dtypes.issubdtype dt `Floating -> float_rank 0 dt
  | _ when Dtypes.issubdtype dt `Signed -> (0, -1, -1, (2 * dtype_bits dt) + 1)
  | _ -> (0, -1, -1, 2 * dtype_bits dt)

let dot_accumulate prim x y acc =
  let dx = dtype_of x in
  if dx <> dtype_of y || precision_rank acc >= precision_rank dx then
    b1 prim [ convert_to acc x; convert_to acc y ]
  else convert_to acc (b1 prim [ x; y ])

let dot_with ?precision ?preferred_element_type (dn : dot_dims) x y =
  let level_name = function
    | Lax_config.Precision_default -> "DEFAULT"
    | Lax_config.Precision_high -> "HIGH"
    | Lax_config.Precision_highest -> "HIGHEST"
  in
  let dot_precision =
    match precision with
    | Some (Lax_config.Precision (l, r)) -> Some (level_name l, level_name r)
    | Some _ -> None
    | None -> (
        match Config.matmul_preset () with
        | Some _ -> None
        | None -> Option.map (fun l -> (l, l)) (Config.matmul_level ()))
  in
  let prim =
    Dot_general
      { dn with dot_precision; untyped_dot = preferred_element_type = None }
  in
  let lhs = dtype_of x and rhs = dtype_of y in
  let result = if lhs = rhs then lhs else Dtypes.promote_types lhs rhs in
  let has_algorithm =
    match precision with
    | Some (Lax_config.Algorithm _ | Lax_config.Preset _) -> true
    | _ -> false
  in
  (match preferred_element_type with
  | Some p
    when (not has_algorithm)
         && (not (Dtypes.issubdtype result `Floating))
         && dtype_bits p < dtype_bits result ->
      invalid_arg
        (Printf.sprintf
           "`preferred_element_type` must not be narrower than the original \
            type, got preferred_element_type of %s for result type of %s."
           (Dtype.short_name p) (Dtype.short_name result))
  | _ -> ());
  let out = Option.value preferred_element_type ~default:result in
  match precision with
  | Some (Lax_config.Algorithm _) ->
      invalid_arg
        "The precision DotAlgorithm is not supported by dot_general on CPU"
  | Some (Lax_config.Preset preset) when not (preset_supported_on_cpu preset) ->
      invalid_arg
        (Printf.sprintf
           "The precision '%s' is not supported by dot_general on CPU"
           (Lax_config.dot_algorithm_preset_name preset))
  | Some (Lax_config.Preset preset) ->
      let inputs = preset_input_types preset in
      let lhs_t = pick_supported lhs inputs in
      let rhs_t = pick_supported rhs inputs in
      let acc = pick_supported out (preset_output_types preset lhs_t rhs_t) in
      convert_to out
        (dot_accumulate prim (convert_to lhs_t x) (convert_to rhs_t y) acc)
  | _ ->
      if lhs <> rhs then b1 prim [ convert_to out x; convert_to out y ]
      else if out = lhs then b1 prim [ x; y ]
      else dot_accumulate prim x y out

let dot ?precision ?preferred_element_type ?dimension_numbers x y =
  let nx = ndim x and ny = ndim y in
  match dimension_numbers with
  | Some ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)) ->
      dot_with ?precision ?preferred_element_type
        {
          lhs_contract;
          rhs_contract;
          lhs_batch;
          rhs_batch;
          untyped_dot = false;
          dot_precision = None;
        }
        x y
  | None ->
      if nx = 0 || ny = 0 then mul x y
      else
        dot_with ?precision ?preferred_element_type
          {
            lhs_contract = [| nx - 1 |];
            rhs_contract = [| (if ny = 1 then 0 else ny - 2) |];
            lhs_batch = [||];
            rhs_batch = [||];
            untyped_dot = false;
            dot_precision = None;
          }
          x y

let dot_general ?precision ?preferred_element_type x y ~lhs_contract
    ~rhs_contract ~lhs_batch ~rhs_batch =
  dot_with ?precision ?preferred_element_type
    {
      lhs_contract;
      rhs_contract;
      lhs_batch;
      rhs_batch;
      dot_precision = None;
      untyped_dot = false;
    }
    x y

let batch_matmul ?precision ?preferred_element_type x y =
  let n = ndim x in
  if ndim y <> n || n < 2 then
    invalid_arg "batch_matmul requires equal ranks of at least two";
  let batch = Array.init (n - 2) Fun.id in
  dot_general ?precision ?preferred_element_type x y
    ~lhs_contract:[| n - 1 |]
    ~rhs_contract:[| n - 2 |]
    ~lhs_batch:batch ~rhs_batch:batch

let batch_matmul_typed x y =
  batch_matmul
    ~preferred_element_type:(Dtypes.promote_types (dtype_of x) (dtype_of y))
    x y

let dot_typed x y =
  dot
    ~preferred_element_type:(Dtypes.promote_types (dtype_of x) (dtype_of y))
    x y

let padtype_to_pads in_shape window_shape window_strides padding =
  match padding with
  | Explicit p -> p
  | Valid -> Array.map (fun _ -> (0, 0)) in_shape
  | Same | Same_lower ->
      Array.mapi
        (fun i size ->
          let stride = window_strides.(i) in
          let out = (size + stride - 1) / stride in
          let total =
            Stdlib.max (((out - 1) * stride) + window_shape.(i) - size) 0
          in
          let small = total / 2 and large = total - (total / 2) in
          if padding = Same then (small, large) else (large, small))
        in_shape

let index_of s c =
  match String.index_opt s c with
  | Some i -> i
  | None -> invalid_arg ("missing dimension " ^ String.make 1 c ^ " in " ^ s)

let conv_general_permutations (lhs, rhs, out) =
  let spatial s a b =
    String.to_seq s |> List.of_seq |> List.filter (fun c -> c <> a && c <> b)
  in
  let lhs_spatial = spatial lhs 'N' 'C' in
  let rhs_spatial = spatial rhs 'I' 'O' in
  let out_spatial = spatial out 'N' 'C' in
  if
    List.sort compare lhs_spatial <> List.sort compare rhs_spatial
    || List.sort compare lhs_spatial <> List.sort compare out_spatial
  then invalid_arg "conv dimension specs must share spatial characters";
  let spec s a b =
    Array.of_list
      (index_of s a :: index_of s b
      :: List.map (fun c -> index_of s c) rhs_spatial)
  in
  {
    lhs_spec = spec lhs 'N' 'C';
    rhs_spec = spec rhs 'O' 'I';
    out_spec = spec out 'N' 'C';
  }

let conv_dimension_numbers specs = conv_general_permutations specs

let default_conv_dims rank =
  let id = Array.init rank Fun.id in
  { lhs_spec = id; rhs_spec = id; out_spec = id }

let conv_general_dilated ?conv_precision ?lhs_dilation ?rhs_dilation
    ?dimension_numbers ?(feature_group_count = 1) ?(batch_group_count = 1) lhs
    rhs ~window_strides padding =
  let rank = ndim lhs in
  let sdims = rank - 2 in
  let ones = Array.make sdims 1 in
  let dn =
    match dimension_numbers with Some d -> d | None -> default_conv_dims rank
  in
  let lhs_dilation = Option.value lhs_dilation ~default:ones in
  let rhs_dilation = Option.value rhs_dilation ~default:ones in
  let spatial spec sh = Array.init sdims (fun i -> sh.(spec.(i + 2))) in
  let lhs_sp = spatial dn.lhs_spec (shape lhs) in
  let rhs_sp = spatial dn.rhs_spec (shape rhs) in
  let dilate sh d =
    Array.mapi (fun i s -> if s = 0 then 0 else 1 + (d.(i) * (s - 1))) sh
  in
  let pads =
    padtype_to_pads
      (dilate lhs_sp lhs_dilation)
      (dilate rhs_sp rhs_dilation)
      window_strides padding
  in
  b1
    (Conv_general_dilated
       {
         window_strides;
         padding = pads;
         lhs_dilation;
         rhs_dilation;
         dimension_numbers = dn;
         feature_group_count;
         batch_group_count;
         conv_precision;
       })
    [ lhs; rhs ]

let conv lhs rhs ~window_strides padding =
  conv_general_dilated lhs rhs ~window_strides padding

let conv_with_general_padding ?lhs_dilation ?rhs_dilation lhs rhs
    ~window_strides padding =
  conv_general_dilated ?lhs_dilation ?rhs_dilation lhs rhs ~window_strides
    (Explicit padding)

let conv_transpose_pad k s padding =
  match padding with
  | Same ->
      let pad_len = k + s - 2 in
      let pad_a = if s > k - 1 then k - 1 else (pad_len + 1) / 2 in
      (pad_a, pad_len - pad_a)
  | Valid ->
      let pad_len = k + s - 2 + Stdlib.max (k - s) 0 in
      let pad_a = k - 1 in
      (pad_a, pad_len - pad_a)
  | Same_lower | Explicit _ ->
      invalid_arg "conv_transpose padding must be SAME or VALID"

let conv_transpose ?rhs_dilation ?dimension_numbers ?(transpose_kernel = false)
    lhs rhs ~strides padding =
  let rank = ndim lhs in
  let sdims = rank - 2 in
  let dn =
    match dimension_numbers with Some d -> d | None -> default_conv_dims rank
  in
  let rhs_shape = shape rhs in
  let dilation = Option.value rhs_dilation ~default:(Array.make sdims 1) in
  let pads =
    match padding with
    | Explicit p -> p
    | _ ->
        Array.init sdims (fun i ->
            let k = rhs_shape.(dn.rhs_spec.(i + 2)) in
            let effective = ((k - 1) * dilation.(i)) + 1 in
            conv_transpose_pad effective strides.(i) padding)
  in
  let rhs =
    if transpose_kernel then begin
      let flipped = rev rhs (Array.sub dn.rhs_spec 2 sdims) in
      let perm = Array.init rank Fun.id in
      perm.(dn.rhs_spec.(0)) <- dn.rhs_spec.(1);
      perm.(dn.rhs_spec.(1)) <- dn.rhs_spec.(0);
      transpose flipped perm
    end
    else rhs
  in
  conv_general_dilated ~lhs_dilation:strides ~rhs_dilation:dilation
    ~dimension_numbers:dn lhs rhs ~window_strides:(Array.make sdims 1)
    (Explicit pads)

let cond = Lax.cond
let scan = Lax.scan
let while_loop = Lax.while_loop
let cumsum = Lax.cumsum
let cumprod = Lax.cumprod
let cummax = Lax.cummax
let cummin = Lax.cummin
let cumlogsumexp = Lax.cumlogsumexp
let custom_linear_solve = Lax.custom_linear_solve
let map_scan f xs = Lax.scan f [] xs
let switch = Lax.switch

let map ?batch_size f xs =
  match batch_size with
  | None -> map_scan f xs
  | Some size when size <= 0 -> invalid_arg "map: batch_size must be positive"
  | Some size ->
      let total =
        match xs with
        | [] -> invalid_arg "map: no input"
        | x :: _ -> (shape x).(0)
      in
      let rec chunks start =
        if start >= total then []
        else
          let stop = Stdlib.min total (start + size) in
          let slices = List.map (fun x -> slice_in_dim x start stop) xs in
          Interpreters.Batching.vmap f (List.map (fun _ -> Some 0) xs) slices
          :: chunks stop
      in
      let parts = chunks 0 in
      List.mapi
        (fun i _ ->
          let column = List.map (fun part -> List.nth part i) parts in
          match column with [ only ] -> only | _ -> concatenate column 0)
        (List.hd parts)

let fori_loop ?unroll lower upper body init =
  match (lower, upper) with
  | Concrete lo, Concrete hi
    when Array.length (Ndarray.shape lo) = 0
         && Array.length (Ndarray.shape hi) = 0 ->
      let lower_ = int_of_float (Ndarray.get_f lo [||]) in
      let upper_ = int_of_float (Ndarray.get_f hi [||]) in
      let length = Stdlib.max (upper_ - lower_) 0 in
      let dt = Ndarray.dtype lo in
      let start =
        Concrete
          (Ndarray.as_weak
             (Ndarray.of_floats dt [||] [| float_of_int lower_ |]))
      in
      let step = function
        | i :: rest ->
            let next = add i (Core.py_scalar dt 1.0) in
            next :: body i rest
        | [] -> failwith "fori_loop: arity"
      in
      List.tl (Lax.scan ~length ?unroll step (start :: init) [])
  | _ ->
      let lower =
        if dtype_of lower = dtype_of upper then lower
        else convert_to (dtype_of upper) lower
      in
      List.tl
        (List.tl
           (Lax.while_loop_keyed ~cond_key:"fori"
              (function
                | i :: upper :: _ -> lt i upper
                | _ -> failwith "fori_loop: arity")
              (function
                | i :: upper :: rest ->
                    add i (filled (dtype_of i) [||] 1.0) :: upper :: body i rest
                | _ -> failwith "fori_loop: arity")
              (lower :: upper :: init)))

let custom_root f initial_guess solve tangent_solve =
  let avals = List.map aval initial_guess in
  let n = List.length initial_guess in
  let closed_f, f_consts = Custom_derivatives.closure_convert f avals in
  let closed_solve, solve_consts =
    Custom_derivatives.closure_convert (solve f) avals
  in
  let linearize_and_solve args =
    match Util.split_list args [ n ] with
    | [ x; b ] ->
        let matvec v = snd (Interpreters.Ad.jvp f x v) in
        tangent_solve matvec b
    | _ -> failwith "custom_root: arity"
  in
  let closed_ls, ls_consts =
    Custom_derivatives.closure_convert linearize_and_solve (avals @ avals)
  in
  let nf = List.length f_consts and ns = List.length solve_consts in
  let nl = List.length ls_consts in
  let split all =
    match Util.split_list all [ nf; ns; nl ] with
    | [ pf; ps; pl; x ] -> (pf, ps, pl, x)
    | _ -> failwith "custom_root: arity"
  in
  let primal all =
    let _, ps, _, x0 = split all in
    closed_solve (x0 @ ps)
  in
  let jvp primals tangents =
    let pf, _, pl, _ = split primals in
    let solution = primal primals in
    let tf, _, _, _ = split tangents in
    let _, rhs =
      Interpreters.Ad.jvp (fun hs -> closed_f (solution @ hs)) pf tf
    in
    let answer = List.map neg (closed_ls (solution @ rhs @ pl)) in
    (solution, answer)
  in
  Custom_derivatives.custom_jvp ~name:"_custom_root" ~jvp_name:"_root_jvp"
    primal ~jvp
    (f_consts @ solve_consts @ ls_consts @ initial_guess)

let platform_dependent ?default ?cuda ~cpu () =
  match (Config.current_platform (), cuda, default) with
  | "cpu", _, _ -> cpu ()
  | "cuda", Some branch, _ -> branch ()
  | _, _, Some branch -> branch ()
  | platform, _, None ->
      invalid_arg
        (Printf.sprintf
           "platform_dependent: no branch for platform %s and no default"
           platform)

let associative_scan ?(axis = 0) ?(reverse = false) fn xs =
  let axis = canon_axis (ndim xs) axis in
  match
    Interpreters.Ad.associative_scan_flat
      (fun a b -> List.map2 fn a b)
      ~axis ~reverse [ xs ]
  with
  | [ out ] -> out
  | _ -> invalid_arg "associative_scan: arity"

let shape_as_value x =
  let sh = shape x in
  Concrete
    (Ndarray.of_floats Dtype.I32
       [| Array.length sh |]
       (Array.map float_of_int sh))

let index_take x indices axes =
  let n_axes = Array.length axes in
  let rows = (shape (List.hd indices)).(0) in
  let picked =
    List.init rows (fun r ->
        let starts = Array.make (ndim x) (index_scalar 0) in
        List.iteri
          (fun i idx -> starts.(axes.(i)) <- index_in_dim ~keepdims:false idx r)
          indices;
        let sizes =
          Array.mapi (fun d s -> if Array.mem d axes then 1 else s) (shape x)
        in
        let v = dynamic_slice x (Array.to_list starts) sizes in
        b1 (Squeeze axes) [ v ])
  in
  ignore n_axes;
  stack picked 0

let fft x ~fft_type ~fft_lengths =
  let dt = dtype_of x in
  let is_complex = Dtypes.issubdtype dt `Complex in
  let target =
    match fft_type with
    | Lax_config.Fft_real ->
        if is_complex then
          invalid_arg "only real valued inputs supported for rfft";
        Dtypes.to_inexact_dtype dt
    | _ ->
        if is_complex then dt
        else if Dtypes.to_inexact_dtype dt = Dtype.F64 then Dtype.Complex128
        else Dtype.Complex64
  in
  let x = convert_to target x in
  if Array.length fft_lengths = 0 then x
  else b1 (Fft { fft_type; fft_lengths }) [ x ]

let dilate_dim d rate = if d = 0 then 0 else 1 + (rate * (d - 1))

let reduce_window_shape_tuple operand_shape window_dimensions window_strides
    padding ?base_dilation ?window_dilation () =
  let n = Array.length operand_shape in
  let ones = Array.make n 1 in
  let base = Option.value base_dilation ~default:ones in
  let wd = Option.value window_dilation ~default:ones in
  Array.init n (fun i ->
      let lo, hi = padding.(i) in
      let padded = dilate_dim operand_shape.(i) base.(i) + lo + hi in
      let window = dilate_dim window_dimensions.(i) wd.(i) in
      if padded < window then 0
      else ((padded - window) / window_strides.(i)) + 1)

let conv_shape_tuple lhs_shape rhs_shape strides pads =
  let n = Array.length lhs_shape in
  Array.init n (fun i ->
      if i = 0 then lhs_shape.(0)
      else if i = 1 then rhs_shape.(0)
      else
        let lo, hi = pads.(i - 2) in
        let padded = lhs_shape.(i) + lo + hi in
        let k = rhs_shape.(i) in
        if padded < k then 0 else ((padded - k) / strides.(i - 2)) + 1)

let permute_shape shape spec = Array.map (fun d -> shape.(d)) spec

let conv_general_shape_tuple lhs_shape rhs_shape window_strides padding
    dimension_numbers =
  let canon_lhs = permute_shape lhs_shape dimension_numbers.lhs_spec in
  let canon_rhs = permute_shape rhs_shape dimension_numbers.rhs_spec in
  let spatial a = Array.sub a 2 (Array.length a - 2) in
  let pads =
    padtype_to_pads (spatial canon_lhs) (spatial canon_rhs) window_strides
      padding
  in
  let canonical = conv_shape_tuple canon_lhs canon_rhs window_strides pads in
  let out = Array.make (Array.length canonical) 0 in
  Array.iteri (fun i d -> out.(d) <- canonical.(i)) dimension_numbers.out_spec;
  out

let conv_transpose_shape_tuple lhs_shape rhs_shape window_strides padding
    dimension_numbers =
  let canon_lhs = permute_shape lhs_shape dimension_numbers.lhs_spec in
  let canon_rhs = permute_shape rhs_shape dimension_numbers.rhs_spec in
  let spatial a = Array.sub a 2 (Array.length a - 2) in
  let k = spatial canon_rhs in
  let pads =
    match padding with
    | Explicit p -> p
    | _ ->
        Array.mapi
          (fun i size -> conv_transpose_pad size window_strides.(i) padding)
          k
  in
  let canonical =
    Array.init (Array.length canon_lhs) (fun i ->
        if i = 0 then canon_lhs.(0)
        else if i = 1 then canon_rhs.(0)
        else
          let lo, hi = pads.(i - 2) in
          let dilated = dilate_dim canon_lhs.(i) window_strides.(i - 2) in
          dilated + lo + hi - canon_rhs.(i) + 1)
  in
  let out = Array.make (Array.length canonical) 0 in
  Array.iteri (fun i d -> out.(d) <- canonical.(i)) dimension_numbers.out_spec;
  out

let conv_general_dilated_patches ?window_strides ?(padding = Valid)
    ?lhs_dilation ?rhs_dilation ?dimension_numbers ?precision lhs ~filter_shape
    =
  let rank = ndim lhs in
  let k = Array.length filter_shape in
  let dn =
    match dimension_numbers with Some d -> d | None -> default_conv_dims rank
  in
  let in_features = (shape lhs).(dn.lhs_spec.(1)) in
  let window = product filter_shape in
  let eye = Interpreters.Ad.lax_eye (dtype_of lhs) window window in
  let spread =
    b1
      (Broadcast_in_dim
         { shape = [| in_features; window; window |]; dims = [| 1; 2 |] })
      [ eye ]
  in
  let kernel =
    b1
      (Reshape (Array.append [| in_features * window; 1 |] filter_shape))
      [ spread ]
  in
  let kernel =
    let rest =
      List.filter
        (fun d -> d <> dn.rhs_spec.(0) && d <> dn.rhs_spec.(1))
        (List.init (k + 2) Fun.id)
    in
    let order = Array.make (k + 2) 0 in
    order.(dn.rhs_spec.(0)) <- 0;
    order.(dn.rhs_spec.(1)) <- 1;
    List.iteri (fun i d -> order.(d) <- i + 2) rest;
    if order = Array.init (k + 2) Fun.id then kernel
    else b1 (Transpose order) [ kernel ]
  in
  let strides = Option.value window_strides ~default:(Array.make k 1) in
  conv_general_dilated ?lhs_dilation ?rhs_dilation ~dimension_numbers:dn
    ?conv_precision:precision ~feature_group_count:in_features lhs kernel
    ~window_strides:strides padding

let scatter_apply x indices func dimension_numbers =
  let updates_shape = shape indices in
  let n_updates = updates_shape.(0) in
  let unused = filled (dtype_of x) [| n_updates |] 0.0 in
  let touched =
    scatter
      (filled Dtype.Bool (shape x) 0.0)
      indices
      (filled Dtype.Bool [| n_updates |] 1.0)
      dimension_numbers
  in
  ignore unused;
  select touched (func x) x

let with_sharding_constraint x _sharding = x
let pcast x ~dtype:_ = x

let composite decomposition ~name:_ ~attrs:_ ~version:_ args =
  decomposition args

let random_gamma_grad_hook : (value -> value -> value) ref =
  ref (fun _ _ -> failwith "lax: random_gamma_grad lowering is not linked")

let random_gamma_grad alpha sample = !random_gamma_grad_hook alpha sample
let rng_state = lazy (Stdlib.Random.State.make_self_init ())

let rng_uniform lo hi out_shape =
  let host = function
    | [ a; c ] ->
        let read v =
          match v with
          | Concrete nd -> Ndarray.get_f nd [||]
          | Device d -> Ndarray.get_f (Pjrt.Buffer.to_host d) [||]
          | Tracer _ -> invalid_arg "rng_uniform: expected concrete bounds"
        in
        let lo_v = read a and hi_v = read c in
        let state = Lazy.force rng_state in
        [
          Concrete
            (Ndarray.of_floats (dtype_of a) out_shape
               (Array.init (product out_shape) (fun _ ->
                    lo_v
                    +. (Stdlib.Random.State.float state 1.0 *. (hi_v -. lo_v)))));
        ]
    | _ -> invalid_arg "rng_uniform: arity"
  in
  List.hd
    (Custom_derivatives.pure_callback host
       ~result_avals:
         [ { shape = out_shape; dtype = dtype_of lo; weak_type = false } ]
       [ lo; hi ])
