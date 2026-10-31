open Types

type padding = Same | Same_lower | Valid | Explicit of (int * int) array

let b1 = Core.bind1
let aval v = Core.get_aval v
let shape v = (aval v).shape
let dtype_of v = (aval v).dtype
let ndim v = Array.length (shape v)
let canon_axis n a = if a < 0 then a + n else a
let product a = Array.fold_left ( * ) 1 a

let filled dt sh x =
  let n = product sh in
  match dt with
  | Dtype.Complex64 | Dtype.Complex128 ->
      Concrete
        (Ndarray.of_complex dt sh (Array.make n { Complex.re = x; im = 0.0 }))
  | _ -> Concrete (Ndarray.of_floats dt sh (Array.make n x))

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
  if shape a = shape b then (a, b)
  else if ndim a = 0 then (broadcast_scalar a (shape b), b)
  else (a, broadcast_scalar b (shape a))

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
let cbrt x = b1 Cbrt [ x ]
let ceil x = b1 Ceil [ x ]
let clz x = b1 Clz [ x ]
let complex = lift2 Complex
let conj x = b1 Conj [ x ]
let cos x = b1 Cos [ x ]
let cosh x = b1 Cosh [ x ]
let div = lift2 Div
let eq = lift2 Eq
let exp x = b1 Exp [ x ]
let exp2 x = b1 Exp2 [ x ]
let expm1 x = b1 Expm1 [ x ]
let floor x = b1 Floor [ x ]
let ge = lift2 Ge
let gt = lift2 Gt
let imag x = b1 Imag [ x ]
let integer_pow x y = b1 (Integer_pow y) [ x ]
let is_finite x = b1 Is_finite [ x ]
let le = lift2 Le
let log x = b1 Log [ x ]
let log1p x = b1 Log1p [ x ]
let logistic x = b1 Logistic [ x ]
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

type rounding_method = Away_from_zero | To_nearest_even

let round ?(rounding_method = Away_from_zero) x =
  match rounding_method with
  | Away_from_zero -> b1 Round [ x ]
  | To_nearest_even -> b1 Round_nearest_even [ x ]

let rsqrt x = b1 Rsqrt [ x ]
let shift_left = lift2 Shift_left
let shift_right_arithmetic = lift2 Shift_right_arithmetic
let shift_right_logical = lift2 Shift_right_logical
let sign x = b1 Sign [ x ]
let sin x = b1 Sin [ x ]
let sinh x = b1 Sinh [ x ]
let sqrt x = b1 Sqrt [ x ]
let square x = b1 Square [ x ]
let sub = lift2 Sub
let tan x = b1 Tan [ x ]
let tanh x = b1 Tanh [ x ]
let betainc a b x = b1 Regularized_incomplete_beta [ a; b; x ]
let stop_gradient = Custom_derivatives.stop_gradient
let optimization_barrier xs = Core.bind Optimization_barrier xs
let convert_element_type x dt = b1 (Convert_element_type dt) [ x ]
let bitcast_convert_type x dt = b1 (Bitcast_convert_type dt) [ x ]

let reduce_precision x ~exponent_bits ~mantissa_bits =
  b1 (Reduce_precision { exponent_bits; mantissa_bits }) [ x ]

let clamp lo x hi =
  let sh = shape x in
  b1 Clamp [ broadcast_scalar lo sh; x; broadcast_scalar hi sh ]

let select pred on_true on_false =
  let sh = shape on_true in
  b1 Select_n [ broadcast_scalar pred sh; on_false; on_true ]

let select_n pred cases =
  let sh =
    match cases with c :: _ -> shape c | [] -> invalid_arg "select_n"
  in
  b1 Select_n (broadcast_scalar pred sh :: cases)

let full sh fill_value dt = filled dt sh fill_value
let full_like x fill_value = filled (dtype_of x) (shape x) fill_value
let empty dt = filled dt [| 0 |] 0.0

let iota dt size =
  b1 (Iota { dtype = dt; shape = [| size |]; dimension = 0 }) []

let broadcasted_iota dt sh dimension =
  b1 (Iota { dtype = dt; shape = sh; dimension }) []

let reshape ?dimensions x new_sizes =
  let v =
    match dimensions with Some perm -> b1 (Transpose perm) [ x ] | None -> x
  in
  b1 (Reshape new_sizes) [ v ]

let transpose x perm = b1 (Transpose perm) [ x ]
let rev x dims = b1 (Rev dims) [ x ]
let tile x reps = b1 (Tile reps) [ x ]

let squeeze x dims =
  let n = ndim x in
  b1 (Squeeze (Array.map (canon_axis n) dims)) [ x ]

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
  b1 (Reshape out) [ x ]

let broadcast_in_dim x sh dims =
  b1 (Broadcast_in_dim { shape = sh; dims }) [ x ]

let broadcast x sizes =
  let k = Array.length sizes in
  let sh = Array.append sizes (shape x) in
  broadcast_in_dim x sh (Array.init (ndim x) (fun i -> i + k))

let broadcast_to_rank x rank =
  let missing = rank - ndim x in
  if missing <= 0 then x
  else b1 (Reshape (Array.append (Array.make missing 1) (shape x))) [ x ]

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

let dynamic_slice x starts slice_sizes =
  b1 (Dynamic_slice { slice_sizes }) (x :: starts)

let dynamic_update_slice x update starts =
  b1 Dynamic_update_slice (x :: update :: starts)

let starts_in_dim n axis start =
  List.init n (fun d -> if d = axis then start else index_scalar 0)

let dynamic_slice_in_dim ?(axis = 0) x start size =
  let n = ndim x in
  let axis = canon_axis n axis in
  let sizes = Array.copy (shape x) in
  sizes.(axis) <- size;
  dynamic_slice x (starts_in_dim n axis start) sizes

let dynamic_index_in_dim ?(axis = 0) ?(keepdims = true) x index =
  let axis = canon_axis (ndim x) axis in
  let v = dynamic_slice_in_dim ~axis x index 1 in
  if keepdims then v else b1 (Squeeze [| axis |]) [ v ]

let dynamic_update_slice_in_dim ?(axis = 0) x update start =
  let n = ndim x in
  let axis = canon_axis n axis in
  dynamic_update_slice x update (starts_in_dim n axis start)

let dynamic_update_index_in_dim ?(axis = 0) x update index =
  let axis = canon_axis (ndim x) axis in
  let update =
    if ndim update = ndim x then update else expand_dims update [| axis |]
  in
  dynamic_update_slice_in_dim ~axis x update index

let gather x indices dimension_numbers slice_sizes =
  b1 (Gather { dimension_numbers; slice_sizes }) [ x; indices ]

let scatter ?(unique_indices = false) x indices updates dimension_numbers =
  b1 (Scatter { dimension_numbers; unique_indices }) [ x; indices; updates ]

let scatter_add x indices updates dimension_numbers =
  b1 (Scatter_add { dimension_numbers }) [ x; indices; updates ]

let scatter_sub x indices updates dimension_numbers =
  b1 (Scatter_sub { dimension_numbers }) [ x; indices; updates ]

let scatter_mul ?(unique_indices = false) x indices updates dimension_numbers =
  b1 (Scatter_mul { dimension_numbers; unique_indices }) [ x; indices; updates ]

let scatter_min x indices updates dimension_numbers =
  b1 (Scatter_min { dimension_numbers }) [ x; indices; updates ]

let scatter_max x indices updates dimension_numbers =
  b1 (Scatter_max { dimension_numbers }) [ x; indices; updates ]

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
  if computation == add then b1 (Reduce_window_sum window) [ x ]
  else if computation == max then b1 (Reduce_window_max window) [ x ]
  else if computation == min then b1 (Reduce_window_min window) [ x ]
  else
    let reducer = reducer_jaxpr (dtype_of x) computation in
    b1 (Reduce_window { reducer; window }) [ x; init ]

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

let dot x y =
  let nx = ndim x and ny = ndim y in
  if nx = 0 || ny = 0 then mul x y
  else
    b1
      (Dot_general
         {
           lhs_contract = [| nx - 1 |];
           rhs_contract = [| (if ny = 1 then 0 else ny - 2) |];
           lhs_batch = [||];
           rhs_batch = [||];
         })
      [ x; y ]

let dot_general x y ~lhs_contract ~rhs_contract ~lhs_batch ~rhs_batch =
  b1 (Dot_general { lhs_contract; rhs_contract; lhs_batch; rhs_batch }) [ x; y ]

let batch_matmul x y =
  let n = ndim x in
  if ndim y <> n || n < 2 then
    invalid_arg "batch_matmul requires equal ranks of at least two";
  let batch = Array.init (n - 2) Fun.id in
  dot_general x y
    ~lhs_contract:[| n - 1 |]
    ~rhs_contract:[| n - 2 |]
    ~lhs_batch:batch ~rhs_batch:batch

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

let conv_general_dilated ?lhs_dilation ?rhs_dilation ?dimension_numbers
    ?(feature_group_count = 1) ?(batch_group_count = 1) lhs rhs ~window_strides
    padding =
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
  Lax.complex_bilinear
    (fun x y ->
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
           })
        [ x; y ])
    lhs rhs

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

let map f xs =
  Lax.scan
    (fun args ->
      let n_x = List.length xs in
      ignore n_x;
      match args with
      | _ :: slices -> index_scalar 0 :: f slices
      | [] -> failwith "map: arity")
    [ index_scalar 0 ]
    xs
  |> List.tl

let rec switch index branches operands =
  match branches with
  | [] -> invalid_arg "Empty branch sequence"
  | [ only ] -> only operands
  | first :: rest ->
      let last = List.length branches - 1 in
      let i =
        clamp (index_scalar 0)
          (convert_element_type index Dtype.I32)
          (index_scalar last)
      in
      Lax.cond
        (lt i (index_scalar 1))
        first
        (fun ops -> switch (sub i (index_scalar 1)) rest ops)
        operands

let fori_loop lower upper body init =
  let n_init = List.length init in
  let outs =
    Lax.while_loop
      (function i :: _ -> lt i upper | [] -> failwith "fori_loop: arity")
      (function
        | i :: rest -> add i (filled (dtype_of i) [||] 1.0) :: body i rest
        | [] -> failwith "fori_loop: arity")
      (lower :: init)
  in
  ignore n_init;
  List.tl outs

let custom_root f initial_guess solve tangent_solve =
  let avals = List.map aval initial_guess in
  let closed, hoisted = Custom_derivatives.closure_convert f avals in
  let n = List.length initial_guess in
  let split all =
    match Util.split_list all [ n ] with
    | [ x; h ] -> (x, h)
    | _ -> failwith "custom_root: arity"
  in
  let f_of h x = closed (x @ h) in
  let primal all =
    let x0, h = split all in
    solve (f_of h) x0
  in
  let jvp primals tangents =
    let x0, h = split primals in
    let _, th = split tangents in
    let solution = solve (f_of h) x0 in
    let _, rhs = Interpreters.Ad.jvp (fun hs -> f_of hs solution) h th in
    let matvec v = snd (Interpreters.Ad.jvp (f_of h) solution v) in
    let answer = tangent_solve matvec (List.map neg rhs) in
    (solution, answer)
  in
  Custom_derivatives.custom_jvp primal ~jvp (initial_guess @ hoisted)

let platform_dependent ?default ~cpu () =
  ignore default;
  cpu ()

let associative_scan ?(axis = 0) fn xs =
  let n = ndim xs in
  let axis = canon_axis n axis in
  let pieces = unstack xs axis in
  match pieces with
  | [] -> xs
  | first :: rest ->
      let _, reversed =
        List.fold_left
          (fun (acc, out) x ->
            let v = fn acc x in
            (v, v :: out))
          (first, [ first ]) rest
      in
      stack (List.rev reversed) axis

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

let fft x ~fft_type ~fft_lengths = b1 (Fft { fft_type; fft_lengths }) [ x ]
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
    ?lhs_dilation ?rhs_dilation ?dimension_numbers lhs ~filter_shape =
  let rank = ndim lhs in
  let k = Array.length filter_shape in
  let dn =
    match dimension_numbers with Some d -> d | None -> default_conv_dims rank
  in
  let in_features = (shape lhs).(dn.lhs_spec.(1)) in
  let window = product filter_shape in
  let out_features = in_features * window in
  let cell = window * in_features in
  let kernel_shape = Array.append [| out_features; 1 |] filter_shape in
  let kernel =
    Concrete
      (Ndarray.of_floats (dtype_of lhs) kernel_shape
         (Array.init (out_features * window) (fun i ->
              let row = i / window and col = i mod window in
              if row mod window = col then 1.0 else 0.0)))
  in
  ignore cell;
  let strides = Option.value window_strides ~default:(Array.make k 1) in
  let kernel_spec = Array.init (k + 2) Fun.id in
  conv_general_dilated ?lhs_dilation ?rhs_dilation
    ~dimension_numbers:{ dn with rhs_spec = kernel_spec }
    ~feature_group_count:in_features lhs kernel ~window_strides:strides padding

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

let random_gamma_grad alpha sample =
  let one = full_like alpha 1.0 in
  let log_pdf =
    sub (sub (mul (sub alpha one) (log sample)) sample) (b1 Lgamma [ alpha ])
  in
  neg (div (b1 Igamma_grad_a [ alpha; sample ]) (exp log_pdf))

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
