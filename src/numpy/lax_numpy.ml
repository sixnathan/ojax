module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let ndim v = Array.length (shape v)
let size v = Array.fold_left ( * ) 1 (shape v)

let canonicalize_axis ax n =
  let a = if ax < 0 then ax + n else ax in
  if a < 0 || a >= n then
    invalid_arg
      (Printf.sprintf "axis %d out of bounds for array of ndim %d" ax n);
  a

let const_full dt sh x = Core.splat dt sh x
let zeros_like v = const_full (dtype v) (shape v) 0.0
let bind1 = C.bind1

let is_inexact = function
  | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn | D.F8_e3m4
  | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz | D.F8_e5m2
  | D.F8_e5m2fnuz | D.F8_e8m0fnu | D.F32 | D.F64 | D.Complex64 | D.Complex128 ->
      true
  | _ -> false

let is_integral = function
  | D.I8 | D.I16 | D.I2 | D.I4 | D.I32 | D.I64 | D.Uint8 | D.Uint16 | D.Uint2
  | D.Uint4 | D.Uint32 | D.Uint64 ->
      true
  | _ -> false

let is_bool = function D.Bool -> true | _ -> false
let to_inexact_dtype = Dtypes.to_inexact_dtype

let convert v dt =
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let broadcast_to = Jnp_util.broadcast_to

let broadcast_shapes a b =
  let na = Array.length a and nb = Array.length b in
  let n = max na nb in
  Array.init n (fun i ->
      let da = if i < n - na then 1 else a.(i - (n - na)) in
      let db = if i < n - nb then 1 else b.(i - (n - nb)) in
      if da = db then da
      else if da = 1 then db
      else if db = 1 then da
      else invalid_arg "broadcast: incompatible shapes")

let promote2 x y =
  let dt, _ =
    Dtypes.result_type
      [
        ((get_aval x).T.dtype, (get_aval x).T.weak_type);
        ((get_aval y).T.dtype, (get_aval y).T.weak_type);
      ]
  in
  let sh = broadcast_shapes (shape x) (shape y) in
  (broadcast_to (convert x dt) sh, broadcast_to (convert y dt) sh, dt, sh)

let transpose ?axes v =
  let n = ndim v in
  let perm =
    match axes with
    | None -> Array.init n (fun i -> n - 1 - i)
    | Some ax -> Array.map (fun i -> canonicalize_axis i n) ax
  in
  bind1 (T.Transpose perm) [ v ]

let permute_dims v axes = transpose ~axes v

let matrix_transpose v =
  let n = ndim v in
  if n < 2 then
    invalid_arg (Printf.sprintf "matrix_transpose requires ndim >= 2; got %d" n);
  let perm = Array.init n (fun i -> i) in
  perm.(n - 1) <- n - 2;
  perm.(n - 2) <- n - 1;
  bind1 (T.Transpose perm) [ v ]

let rev_dims v dims = bind1 (T.Rev dims) [ v ]
let int_key a = String.concat "," (Array.to_list (Array.map string_of_int a))

let flip ?axis v =
  let n = ndim v in
  let dims =
    match axis with
    | None -> Array.init n (fun i -> i)
    | Some ax -> Array.map (fun i -> canonicalize_axis i n) ax
  in
  Jnp_util.jit1 "_flip" ~key:(int_key dims)
    (function [ v ] -> rev_dims v dims | _ -> invalid_arg "flip: arity")
    [ v ]

let fliplr v = rev_dims v [| 1 |]
let flipud v = rev_dims v [| 0 |]

let reshape v sh =
  let total = size v in
  let known =
    Array.fold_left (fun acc d -> if d = -1 then acc else acc * d) 1 sh
  in
  let resolved = Array.map (fun d -> if d = -1 then total / known else d) sh in
  if Array.length resolved > 0 && resolved = shape v then v
  else bind1 (T.Reshape resolved) [ v ]

let reverse_axes v = Array.init (ndim v) (fun i -> ndim v - 1 - i)

let reshape ?(order = "C") v sh =
  match order with
  | "C" -> reshape v sh
  | "F" ->
      let flipped = Array.of_list (List.rev (Array.to_list sh)) in
      let c_order = bind1 (T.Transpose (reverse_axes v)) [ v ] in
      let out = reshape c_order flipped in
      bind1 (T.Transpose (reverse_axes out)) [ out ]
  | "A" -> failwith "The 'A' order option is not supported by jnp.reshape."
  | o -> invalid_arg ("Unrecognized order " ^ o)

let ravel ?(order = "C") v =
  match order with
  | "C" -> reshape v [| size v |]
  | "F" -> reshape (bind1 (T.Transpose (reverse_axes v)) [ v ]) [| size v |]
  | "A" -> failwith "The 'A' order option is not supported by jnp.ravel."
  | o -> invalid_arg ("Unrecognized order " ^ o)

let rot90 ?(k = 1) ?(axes = (0, 1)) v =
  let n = ndim v in
  if n < 2 then
    invalid_arg (Printf.sprintf "rot90 requires ndim >= 2; got %d" n);
  let ax1 = canonicalize_axis (fst axes) n in
  let ax2 = canonicalize_axis (snd axes) n in
  if ax1 = ax2 then invalid_arg "rot90: axes must be different";
  let k = ((k mod 4) + 4) mod 4 in
  if k = 0 then v
  else if k = 2 then flip ~axis:[| ax2 |] (flip ~axis:[| ax1 |] v)
  else begin
    let perm = Array.init n (fun i -> i) in
    perm.(ax1) <- ax2;
    perm.(ax2) <- ax1;
    if k = 1 then transpose ~axes:perm (flip ~axis:[| ax2 |] v)
    else flip ~axis:[| ax2 |] (transpose ~axes:perm v)
  end

let rot90 ?k ?axes p0 =
  Jnp_util.jit1 "rot90"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string (k, axes) [])))
    (function [ p0 ] -> rot90 ?k ?axes p0 | _ -> invalid_arg "rot90: arity")
    [ p0 ]

let trunc v =
  let dt = dtype v in
  if is_integral dt || is_bool dt then v
  else
    let below = bind1 T.Lt [ v; zeros_like v ] in
    bind1 T.Select_n [ below; bind1 T.Floor [ v ]; bind1 T.Ceil [ v ] ]

let trunc p0 =
  Jnp_util.jit1 "trunc"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> trunc p0 | _ -> invalid_arg "trunc: arity")
    [ p0 ]

let isnan v = bind1 T.Ne [ v; v ]

let fmin x y =
  let x, y, _, _ = promote2 x y in
  let mask = bind1 T.Or [ bind1 T.Lt [ x; y ]; isnan y ] in
  bind1 T.Select_n [ mask; y; x ]

let fmin p0 p1 =
  Jnp_util.jit1 "fmin"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> fmin p0 p1 | _ -> invalid_arg "fmin: arity")
    [ p0; p1 ]

let fmax x y =
  let x, y, _, _ = promote2 x y in
  let mask = bind1 T.Or [ bind1 T.Gt [ x; y ]; isnan y ] in
  bind1 T.Select_n [ mask; y; x ]

let fmax p0 p1 =
  Jnp_util.jit1 "fmax"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> fmax p0 p1 | _ -> invalid_arg "fmax: arity")
    [ p0; p1 ]

let slice_axis v axis lo hi =
  let sh = shape v in
  let n = Array.length sh in
  let start = Array.make n 0 in
  let limit = Array.copy sh in
  start.(axis) <- lo;
  limit.(axis) <- hi;
  bind1
    (T.Slice { start_indices = start; limit_indices = limit; strides = None })
    [ v ]

let diff ?(n = 1) ?(axis = -1) ?prepend ?append v =
  if n = 0 then v
  else begin
    if n < 0 then invalid_arg "diff: order must be non-negative";
    let nd = ndim v in
    if nd = 0 then invalid_arg "diff requires at least one dimension";
    let axis = canonicalize_axis axis nd in
    let edge w =
      if ndim w = 0 then begin
        let sh = Array.copy (shape v) in
        sh.(axis) <- 1;
        bind1 (T.Broadcast_in_dim { shape = sh; dims = [||] }) [ w ]
      end
      else w
    in
    let parts =
      Option.to_list (Option.map edge prepend)
      @ [ v ]
      @ Option.to_list (Option.map edge append)
    in
    let v =
      match parts with
      | [ only ] -> only
      | _ ->
          let dt =
            fst
              (Dtypes.result_type (List.map (fun w -> (dtype w, false)) parts))
          in
          bind1 (T.Concatenate axis) (List.map (fun w -> convert w dt) parts)
    in
    let bool_dtype = is_bool (dtype v) in
    let step acc =
      let d = (shape acc).(axis) in
      let hi = slice_axis acc axis 1 d in
      let lo = slice_axis acc axis 0 (d - 1) in
      if bool_dtype then bind1 T.Ne [ hi; lo ] else bind1 T.Sub [ hi; lo ]
    in
    let rec loop acc i = if i = 0 then acc else loop (step acc) (i - 1) in
    loop v n
  end

let diff ?n ?axis ?prepend ?append p0 =
  let prepend_given = prepend <> None in
  let append_given = append <> None in
  if prepend_given || append_given then diff ?n ?axis ?prepend ?append p0
  else
    Jnp_util.jit1 "diff"
      ~key:(Digest.to_hex (Digest.string (Marshal.to_string (n, axis) [])))
      (function
        | [ p0 ] -> diff ?n ?axis ?prepend ?append p0
        | _ -> invalid_arg "diff: arity")
      [ p0 ]

let ediff1d ?to_end ?to_begin v =
  let a = ravel v in
  let d = (shape a).(0) in
  let body = bind1 T.Sub [ slice_axis a 0 1 d; slice_axis a 0 0 (d - 1) ] in
  let extra w = ravel (convert w (dtype a)) in
  let parts =
    Option.to_list (Option.map extra to_begin)
    @ [ body ]
    @ Option.to_list (Option.map extra to_end)
  in
  match parts with [ only ] -> only | _ -> bind1 (T.Concatenate 0) parts

let ediff1d ?to_end ?to_begin p0 =
  let to_end_given = to_end <> None in
  let to_begin_given = to_begin <> None in
  if to_end_given || to_begin_given then ediff1d ?to_end ?to_begin p0
  else
    Jnp_util.jit1 "ediff1d"
      ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
      (function
        | [ p0 ] -> ediff1d ?to_end ?to_begin p0
        | _ -> invalid_arg "ediff1d: arity")
      [ p0 ]

let is_complex_dtype = function
  | D.Complex64 | D.Complex128 -> true
  | _ -> false

let real v = if is_complex_dtype (dtype v) then bind1 T.Real [ v ] else v

let imag v =
  if is_complex_dtype (dtype v) then bind1 T.Imag [ v ] else zeros_like v

let degrees v =
  let dt = dtype v in
  const_full dt (shape v) (180.0 /. Float.pi) |> fun c -> bind1 T.Mul [ v; c ]

let angle ?(deg = false) v =
  let re = real v in
  let im = imag v in
  let dt = dtype re in
  let re, im =
    if (not (is_inexact dt)) || (is_inexact (dtype v) && ndim v = 0) then
      let fd = Dtypes.default_float_dtype () in
      (convert re fd, convert im fd)
    else (re, im)
  in
  let result = bind1 T.Atan2 [ im; re ] in
  if deg then degrees result else result

let angle ?deg p0 =
  Jnp_util.jit1 "angle"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string deg [])))
    (function [ p0 ] -> angle ?deg p0 | _ -> invalid_arg "angle: arity")
    [ p0 ]

let iscomplex v =
  let i = imag v in
  bind1 T.Ne [ i; zeros_like i ]

let iscomplex p0 =
  Jnp_util.jit1 "iscomplex"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> iscomplex p0 | _ -> invalid_arg "iscomplex: arity")
    [ p0 ]

let isreal v =
  let i = imag v in
  bind1 T.Eq [ i; zeros_like i ]

let isreal p0 =
  Jnp_util.jit1 "isreal"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> isreal p0 | _ -> invalid_arg "isreal: arity")
    [ p0 ]

let iscomplexobj v = is_complex_dtype (dtype v)
let isrealobj v = not (iscomplexobj v)
let isscalar v = ndim v = 0

let result_type vs =
  let pairs =
    List.map (fun v -> ((get_aval v).T.dtype, (get_aval v).T.weak_type)) vs
  in
  fst (Dtypes.result_type pairs)

type dtype_class =
  | Cdtype of D.t
  | Signedinteger
  | Unsignedinteger
  | Complexfloating
  | Extended
  | Prng_key_class
  | Integer
  | Floating
  | Inexact
  | Number
  | Generic
  | Cbool

let class_chain = function
  | Cdtype D.Bool | Cbool -> [ Cbool; Generic ]
  | Cdtype D.Float0 -> [ Generic ]
  | Cdtype d when Dtypes.issubdtype d `Signed ->
      [ Signedinteger; Integer; Number; Generic ]
  | Cdtype d when Dtypes.issubdtype d `Unsigned ->
      [ Unsignedinteger; Integer; Number; Generic ]
  | Cdtype d when Dtypes.issubdtype d `Floating ->
      [ Floating; Inexact; Number; Generic ]
  | Cdtype _ -> [ Complexfloating; Inexact; Number; Generic ]
  | Signedinteger -> [ Signedinteger; Integer; Number; Generic ]
  | Unsignedinteger -> [ Unsignedinteger; Integer; Number; Generic ]
  | Floating -> [ Floating; Inexact; Number; Generic ]
  | Complexfloating -> [ Complexfloating; Inexact; Number; Generic ]
  | Integer -> [ Integer; Number; Generic ]
  | Inexact -> [ Inexact; Number; Generic ]
  | Number -> [ Number; Generic ]
  | Generic -> [ Generic ]
  | Extended -> [ Extended; Generic ]
  | Prng_key_class -> [ Prng_key_class; Extended; Generic ]

let complexfloating = Complexfloating
let unsignedinteger = Unsignedinteger
let extended = Extended
let issubdtype a b = a = b || List.mem b (class_chain a)

let flip1 y =
  Jnp_util.jit1 "_flip" ~key:"0"
    (function [ y ] -> rev_dims y [| 0 |] | _ -> invalid_arg "flip: arity")
    [ y ]

let conv1d_body ~mode ~correlating a v =
  let flip_kernel = not correlating in
  let dt, _ =
    Dtypes.result_type
      [
        (to_inexact_dtype (dtype a), false); (to_inexact_dtype (dtype v), false);
      ]
  in
  let a = convert a dt and v = convert v dt in
  let la = (shape a).(0) and lv = (shape v).(0) in
  if la = 0 || lv = 0 then invalid_arg "conv: inputs cannot be empty";
  let v =
    match dtype v with
    | (D.Complex64 | D.Complex128) when correlating -> bind1 T.Conj [ v ]
    | _ -> v
  in
  let swapped = la < lv in
  let x, y = if swapped then (v, a) else (a, v) in
  let y = if flip_kernel then flip1 y else y in
  let ly = (shape y).(0) in
  let padding =
    match mode with
    | "valid" -> [| (0, 0) |]
    | "same" -> [| (ly / 2, ly - (ly / 2) - 1) |]
    | "full" -> [| (ly - 1, ly - 1) |]
    | _ -> invalid_arg "conv: mode must be one of full, same, valid"
  in
  let lx = (shape x).(0) in
  let lifted v n =
    bind1 (T.Broadcast_in_dim { shape = [| 1; 1; n |]; dims = [| 2 |] }) [ v ]
  in
  let lhs = lifted x lx in
  let rhs = lifted y ly in
  let out =
    bind1
      (T.Conv_general_dilated
         {
           window_strides = [| 1 |];
           padding;
           lhs_dilation = [| 1 |];
           rhs_dilation = [| 1 |];
           dimension_numbers =
             {
               lhs_spec = [| 0; 1; 2 |];
               rhs_spec = [| 0; 1; 2 |];
               out_spec = [| 0; 1; 2 |];
             };
           feature_group_count = 1;
           batch_group_count = 1;
         })
      [ lhs; rhs ]
  in
  let out = if correlating && swapped then rev_dims out [| 2 |] else out in
  bind1 (T.Squeeze [| 0; 1 |]) [ out ]

let conv1d ~mode ~correlating a v =
  Jnp_util.jit1 "_conv"
    ~key:(mode ^ string_of_bool correlating)
    (function
      | [ a; v ] -> conv1d_body ~mode ~correlating a v
      | _ -> invalid_arg "conv: arity")
    [ a; v ]

let convolve ?(mode = "full") a v = conv1d ~mode ~correlating:false a v

let convolve ?mode p0 p1 =
  Jnp_util.jit1 "convolve"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string mode [])))
    (function
      | [ p0; p1 ] -> convolve ?mode p0 p1 | _ -> invalid_arg "convolve: arity")
    [ p0; p1 ]

let correlate ?(mode = "valid") a v = conv1d ~mode ~correlating:true a v

let correlate ?mode p0 p1 =
  Jnp_util.jit1 "correlate"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string mode [])))
    (function
      | [ p0; p1 ] -> correlate ?mode p0 p1
      | _ -> invalid_arg "correlate: arity")
    [ p0; p1 ]

let bind = C.bind
let scalar dt x = const_full dt [||] x
let full dt sh x = const_full dt sh x

let promote_scalar_kept x y =
  let dt, is_weak =
    Dtypes.result_type
      [
        ((get_aval x).T.dtype, (get_aval x).T.weak_type);
        ((get_aval y).T.dtype, (get_aval y).T.weak_type);
      ]
  in
  let x = C.convert_like ~weak:is_weak x dt in
  let y = C.convert_like ~weak:is_weak y dt in
  if ndim x = ndim y || ndim x = 0 || ndim y = 0 then (x, y)
  else
    let sh = broadcast_shapes (shape x) (shape y) in
    (broadcast_to x sh, broadcast_to y sh)

let maximum a b =
  let a, b = promote_scalar_kept a b in
  bind1 T.Max [ a; b ]

let minimum a b =
  let a, b = promote_scalar_kept a b in
  bind1 T.Min [ a; b ]

let where_ = Jnp_util.where

let finfo_max = function
  | D.F64 | D.Complex128 -> Float.max_float
  | D.F16 -> 65504.0
  | D.Bf16 -> 3.3895313892515355e+38
  | _ -> 3.4028234663852886e+38

let isposinf v = bind1 T.Eq [ v; full (dtype v) (shape v) Float.infinity ]
let isneginf v = bind1 T.Eq [ v; full (dtype v) (shape v) Float.neg_infinity ]

let nan_to_num ?(nan = 0.0) ?posinf ?neginf x =
  let dt = dtype x in
  if not (is_inexact dt) then x
  else begin
    let maxv = finfo_max dt in
    let pv = Option.value posinf ~default:maxv in
    let nv = Option.value neginf ~default:(-.maxv) in
    let out = where_ (isnan x) (scalar dt nan) x in
    let out = where_ (isposinf out) (scalar dt pv) out in
    where_ (isneginf out) (scalar dt nv) out
  end

let nan_to_num ?nan ?posinf ?neginf p0 =
  Jnp_util.jit1 "nan_to_num"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (nan, posinf, neginf) [])))
    (function
      | [ p0 ] -> nan_to_num ?nan ?posinf ?neginf p0
      | _ -> invalid_arg "nan_to_num: arity")
    [ p0 ]

let real_dtype_of = function
  | D.Complex64 -> D.F32
  | D.Complex128 -> D.F64
  | d -> d

let isclose ?(rtol = 1e-5) ?(atol = 1e-8) ?(equal_nan = false) a b =
  let dt0, _ =
    Dtypes.result_type
      [
        ((get_aval a).T.dtype, (get_aval a).T.weak_type);
        ((get_aval b).T.dtype, (get_aval b).T.weak_type);
      ]
  in
  let dt = to_inexact_dtype dt0 in
  let sh = broadcast_shapes (shape a) (shape b) in
  let a = broadcast_to (convert a dt) sh in
  let b = broadcast_to (convert b dt) sh in
  let rdt = real_dtype_of dt in
  let rt = full rdt sh rtol and at = full rdt sh atol in
  let finite v =
    if rdt = dt then bind1 T.Is_finite [ v ]
    else
      bind1 T.And
        [
          bind1 T.Is_finite [ bind1 T.Real [ v ] ];
          bind1 T.Is_finite [ bind1 T.Imag [ v ] ];
        ]
  in
  let in_range =
    bind1 T.Le
      [
        bind1 T.Abs [ bind1 T.Sub [ a; b ] ];
        bind1 T.Add [ at; bind1 T.Mul [ rt; bind1 T.Abs [ b ] ] ];
      ]
  in
  let out =
    bind1 T.Or [ bind1 T.Eq [ a; b ]; bind1 T.And [ finite b; in_range ] ]
  in
  if equal_nan then bind1 T.Or [ out; bind1 T.And [ isnan a; isnan b ] ]
  else out

let isclose ?rtol ?atol ?equal_nan p0 p1 =
  Jnp_util.jit1 "isclose"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (rtol, atol, equal_nan) [])))
    (function
      | [ p0; p1 ] -> isclose ?rtol ?atol ?equal_nan p0 p1
      | _ -> invalid_arg "isclose: arity")
    [ p0; p1 ]

let allclose ?rtol ?atol ?equal_nan a b =
  let c = isclose ?rtol ?atol ?equal_nan a b in
  let n = ndim c in
  bind1 (T.Reduce_and (Array.init n (fun i -> i))) [ c ]

let allclose ?rtol ?atol ?equal_nan p0 p1 =
  Jnp_util.jit1 "allclose"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (rtol, atol, equal_nan) [])))
    (function
      | [ p0; p1 ] -> allclose ?rtol ?atol ?equal_nan p0 p1
      | _ -> invalid_arg "allclose: arity")
    [ p0; p1 ]

let clip_values ?min ?max arr =
  let present = Option.to_list min @ Option.to_list max in
  Jnp_util.jit1 "clip"
    ~key:(string_of_bool (min <> None) ^ string_of_bool (max <> None))
    (function
      | arr :: bounds -> (
          let rest = ref bounds in
          let next given =
            match given with
            | None -> None
            | Some _ -> (
                match !rest with
                | v :: tl ->
                    rest := tl;
                    Some v
                | [] -> invalid_arg "clip: arity")
          in
          let low = next min in
          let high = next max in
          let target =
            let dts = List.filter_map Fun.id [ low; high ] in
            fst
              (Dtypes.result_type
                 ((dtype arr, false) :: List.map (fun v -> (dtype v, true)) dts))
          in
          let arr = convert arr target in
          let arr =
            match low with
            | Some v -> bind1 T.Max [ convert v target; arr ]
            | None -> arr
          in
          match high with
          | Some v -> bind1 T.Min [ convert v target; arr ]
          | None -> arr)
      | [] -> invalid_arg "clip: arity")
    (arr :: present)

let clip_strong_max maximum_value arr =
  Jnp_util.jit1 "clip" ~key:"falsetrue"
    (function [ arr; hi ] -> minimum hi arr | _ -> invalid_arg "clip: arity")
    [ arr; maximum_value ]

let clip ?min ?max arr =
  (match dtype arr with
  | D.Complex64 | D.Complex128 ->
      invalid_arg
        "Clip received a complex value either through the input or the min/max \
         keywords. Complex values have no ordering and cannot be clipped."
  | _ -> ());
  let scalar_arg m =
    if Float.is_integer m && (is_integral (dtype arr) || is_bool (dtype arr))
    then Jnp_util.py_int (int_of_float m)
    else Jnp_util.py_float m
  in
  let lo = Option.map scalar_arg min and hi = Option.map scalar_arg max in
  let extras = List.filter_map Fun.id [ lo; hi ] in
  Jnp_util.jit1 "clip"
    ~key:(string_of_bool (min <> None) ^ string_of_bool (max <> None))
    (function
      | arr :: rest -> (
          let lo, hi =
            match (min, max, rest) with
            | Some _, Some _, [ l; h ] -> (Some l, Some h)
            | Some _, None, [ l ] -> (Some l, None)
            | None, Some _, [ h ] -> (None, Some h)
            | _ -> (None, None)
          in
          let arr = match lo with None -> arr | Some m -> maximum m arr in
          match hi with None -> arr | Some m -> minimum m arr)
      | [] -> invalid_arg "clip: arity")
    (arr :: extras)

let rec round ?(decimals = 0) v =
  let dt = dtype v in
  if is_bool dt then
    invalid_arg "jnp.round: data type bool not compatible with finfo";
  if is_integral dt then v
  else if dt = D.Complex64 || dt = D.Complex128 then
    bind1 T.Complex
      [
        round ~decimals (bind1 T.Real [ v ]);
        round ~decimals (bind1 T.Imag [ v ]);
      ]
  else if decimals = 0 then bind1 T.Round_nearest_even [ v ]
  else begin
    let x =
      if dt = D.F16 then bind1 (T.Convert_element_type D.F32) [ v ] else v
    in
    let factor = scalar (dtype x) (10.0 ** float_of_int decimals) in
    let out =
      bind1 T.Div
        [ bind1 T.Round_nearest_even [ bind1 T.Mul [ x; factor ] ]; factor ]
    in
    if dt = D.F16 then bind1 (T.Convert_element_type D.F16) [ out ] else out
  end

let round ?decimals p0 =
  Jnp_util.jit1 "round"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string decimals [])))
    (function [ p0 ] -> round ?decimals p0 | _ -> invalid_arg "round: arity")
    [ p0 ]

let around ?decimals v = round ?decimals v

let around ?decimals p0 =
  Jnp_util.jit1 "around"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string decimals [])))
    (function
      | [ p0 ] -> around ?decimals p0 | _ -> invalid_arg "around: arity")
    [ p0 ]

let expand_dims v axes =
  let nd = ndim v in
  let result_ndim = nd + Array.length axes in
  let axes = Array.map (fun a -> canonicalize_axis a result_ndim) axes in
  let is_new = Array.make result_ndim false in
  Array.iter (fun a -> is_new.(a) <- true) axes;
  let sh = shape v in
  let out_shape = Array.make result_ndim 1 in
  let dims = Array.make nd 0 in
  let j = ref 0 in
  for i = 0 to result_ndim - 1 do
    if not is_new.(i) then begin
      out_shape.(i) <- sh.(!j);
      dims.(!j) <- i;
      incr j
    end
  done;
  bind1 (T.Broadcast_in_dim { shape = out_shape; dims }) [ v ]

let squeeze ?axis v =
  let sh = shape v in
  let nd = Array.length sh in
  let dims =
    match axis with
    | None ->
        Array.of_list
          (List.filter (fun i -> sh.(i) = 1) (List.init nd (fun i -> i)))
    | Some ax ->
        let d = Array.map (fun a -> canonicalize_axis a nd) ax in
        Array.sort compare d;
        d
  in
  if Array.length dims = 0 then v else bind1 (T.Squeeze dims) [ v ]

let swapaxes axis1 axis2 v =
  let n = ndim v in
  let perm = Array.init n (fun i -> i) in
  let a1 = canonicalize_axis axis1 n and a2 = canonicalize_axis axis2 n in
  perm.(a1) <- a2;
  perm.(a2) <- a1;
  bind1 (T.Transpose perm) [ v ]

let moveaxis source destination v =
  let n = ndim v in
  let src = Array.map (fun a -> canonicalize_axis a n) source in
  let dst = Array.map (fun a -> canonicalize_axis a n) destination in
  if Array.length src <> Array.length dst then
    invalid_arg "moveaxis: inconsistent number of elements";
  let src_list = Array.to_list src in
  let perm =
    ref (List.filter (fun i -> not (List.mem i src_list)) (List.init n Fun.id))
  in
  let pairs =
    List.sort compare (List.combine (Array.to_list dst) (Array.to_list src))
  in
  let rec insert l i x =
    if i = 0 then x :: l
    else match l with [] -> [ x ] | h :: t -> h :: insert t (i - 1) x
  in
  List.iter (fun (d, s) -> perm := insert !perm d s) pairs;
  bind1 (T.Transpose (Array.of_list !perm)) [ v ]

let broadcast_shapes_n shapes = List.fold_left broadcast_shapes [||] shapes

let broadcast_arrays vs =
  let sh =
    List.fold_left (fun acc v -> broadcast_shapes acc (shape v)) [||] vs
  in
  List.map (fun v -> broadcast_to v sh) vs

let resize v new_shape =
  let arr = ravel v in
  let sz = (shape arr).(0) in
  let new_size = Array.fold_left ( * ) 1 new_shape in
  let repeats = (new_size + sz - 1) / sz in
  let tiled = bind1 (T.Tile [| repeats |]) [ arr ] in
  let sliced = slice_axis tiled 0 0 new_size in
  reshape sliced new_shape

let resize p0 s1 =
  Jnp_util.jit1 "resize"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string s1 [])))
    (function [ p0 ] -> resize p0 s1 | _ -> invalid_arg "resize: arity")
    [ p0 ]

let mod_floor a b =
  let zero = full (dtype a) (shape a) 0.0 in
  let trunc_mod = bind1 T.Rem [ a; b ] in
  let nz = bind1 T.Ne [ trunc_mod; zero ] in
  let do_plus =
    bind1 T.And
      [
        bind1 T.Ne [ bind1 T.Lt [ trunc_mod; zero ]; bind1 T.Lt [ b; zero ] ];
        nz;
      ]
  in
  bind1 T.Select_n [ do_plus; trunc_mod; bind1 T.Add [ trunc_mod; b ] ]

let unravel_index indices dims =
  let n = Array.length dims in
  let dt = dtype indices in
  let ish = shape indices in
  let out = Array.make n indices in
  let cur = ref indices in
  for i = n - 1 downto 0 do
    let s = full dt ish (float_of_int dims.(i)) in
    let r = mod_floor !cur s in
    out.(i) <- r;
    cur := bind1 T.Div [ bind1 T.Sub [ !cur; r ]; s ]
  done;
  let zero = full dt ish 0.0 in
  let negone = full dt ish (-1.0) in
  let oob_pos = bind1 T.Gt [ !cur; zero ] in
  let oob_neg = bind1 T.Lt [ !cur; negone ] in
  List.init n (fun i ->
      let smax = full dt ish (float_of_int (dims.(i) - 1)) in
      where_ oob_pos smax (where_ oob_neg zero out.(i)))

let unwrap ?discont ?(axis = -1) ?(period = 2.0 *. Float.pi) p =
  let nd = ndim p in
  let axis = canonicalize_axis axis nd in
  let dt =
    if is_bool (dtype p) then Dtypes.default_float_dtype ()
    else to_inexact_dtype (dtype p)
  in
  let p = convert p dt in
  let size = (shape p).(axis) in
  if size = 0 then p
  else begin
    let interval = period /. 2.0 in
    let discont = Option.value discont ~default:(period /. 2.0) in
    let dd = diff ~axis p in
    let sh = shape dd in
    let per = full dt sh period in
    let iv = full dt sh interval in
    let ddmod = bind1 T.Sub [ mod_floor (bind1 T.Add [ dd; iv ]) per; iv ] in
    let cond1 =
      bind1 T.And
        [
          bind1 T.Eq [ ddmod; full dt sh (-.interval) ];
          bind1 T.Gt [ dd; full dt sh 0.0 ];
        ]
    in
    let ddmod = where_ cond1 iv ddmod in
    let ph_correct =
      where_
        (bind1 T.Lt [ bind1 T.Abs [ dd ]; full dt sh discont ])
        (full dt sh 0.0)
        (bind1 T.Sub [ ddmod; dd ])
    in
    let first = slice_axis p axis 0 1 in
    let rest = slice_axis p axis 1 size in
    let cs = bind1 (T.Cumsum { axis; reverse = false }) [ ph_correct ] in
    bind1 (T.Concatenate axis) [ first; bind1 T.Add [ rest; cs ] ]
  end

let unwrap ?discont ?axis ?period p0 =
  Jnp_util.jit1 "unwrap"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (discont, axis, period) [])))
    (function
      | [ p0 ] -> unwrap ?discont ?axis ?period p0
      | _ -> invalid_arg "unwrap: arity")
    [ p0 ]

type sections = Count of int | Indices of int array

let do_split ~array_split ?(axis = 0) v s =
  let nd = ndim v in
  let axis = canonicalize_axis axis nd in
  let size = (shape v).(axis) in
  let sizes =
    match s with
    | Indices idx ->
        let bounds = Array.concat [ [| 0 |]; idx; [| size |] ] in
        Array.init
          (Array.length bounds - 1)
          (fun i -> bounds.(i + 1) - bounds.(i))
    | Count nsec ->
        let part = size / nsec and r = size mod nsec in
        if r = 0 then Array.make nsec part
        else if array_split then
          Array.init nsec (fun i -> if i < r then part + 1 else part)
        else invalid_arg "array split does not result in an equal division"
  in
  bind (T.Split { sizes = Np_ints.mark sizes; axis }) [ v ]

let split ?axis v s = do_split ~array_split:false ?axis v s
let array_split ?axis v s = do_split ~array_split:true ?axis v s
let vsplit v s = do_split ~array_split:false ~axis:0 v s

let hsplit v s =
  do_split ~array_split:false ~axis:(if ndim v = 1 then 0 else 1) v s

let dsplit v s = do_split ~array_split:false ~axis:2 v s

let select ?default condlist choicelist =
  let k = List.length condlist in
  if k = 0 then invalid_arg "select: condlist must be non-empty";
  if List.length choicelist <> k then
    invalid_arg "select: condlist and choicelist must have the same length";
  let false0 = scalar D.Bool 0.0 in
  let conds_b = broadcast_arrays (false0 :: condlist) in
  let conds_i = List.map (fun c -> convert c D.I32) conds_b in
  let stacked = bind1 (T.Stack 0) conds_i in
  let idx =
    bind1
      (T.Argmax { axis = 0; index_dtype = Dtypes.default_int_dtype () })
      [ stacked ]
  in
  let default_v =
    match default with
    | Some d -> d
    | None -> scalar (result_type choicelist) 0.0
  in
  let dt =
    match default with
    | Some d -> result_type (d :: choicelist)
    | None -> result_type choicelist
  in
  let cases = List.map (fun v -> convert v dt) (default_v :: choicelist) in
  bind1 T.Select_n (broadcast_arrays (idx :: cases))

let all_reduce v =
  let n = ndim v in
  if n = 0 then v else bind1 (T.Reduce_and (Array.init n Fun.id)) [ v ]

let promote_values vs =
  let dt = result_type vs in
  List.map (fun v -> convert v dt) vs

let astype v dt =
  match (dtype v, dt) with
  | (D.Complex64 | D.Complex128), D.Bool -> bind1 T.Ne [ v; zeros_like v ]
  | _ -> convert v dt

let copy v = bind1 T.Copy [ v ]

let atleast_1d v =
  Jnp_util.jit1 "atleast_1d"
    (function
      | [ v ] -> if ndim v >= 1 then v else expand_dims v [| 0 |]
      | _ -> invalid_arg "atleast_1d: arity")
    [ v ]

let atleast_2d v =
  match ndim v with
  | 0 -> expand_dims v [| 0; 1 |]
  | 1 -> expand_dims v [| 0 |]
  | _ -> v

let atleast_2d p0 =
  Jnp_util.jit1 "atleast_2d"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> atleast_2d p0 | _ -> invalid_arg "atleast_2d: arity")
    [ p0 ]

let atleast_3d v =
  match ndim v with
  | 0 -> expand_dims v [| 0; 1; 2 |]
  | 1 -> expand_dims v [| 0; 2 |]
  | 2 -> expand_dims v [| 2 |]
  | _ -> v

let atleast_3d p0 =
  Jnp_util.jit1 "atleast_3d"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> atleast_3d p0 | _ -> invalid_arg "atleast_3d: arity")
    [ p0 ]

let with_dtype dt arrays =
  match dt with
  | None -> promote_values arrays
  | Some d ->
      let d = Dtypes.canonicalize_dtype d in
      List.map (fun v -> convert v d) arrays

let concatenate ?(axis = 0) ?dtype:dt arrays =
  match arrays with
  | [] -> invalid_arg "concatenate: need at least one array"
  | first :: _ ->
      let ax = canonicalize_axis axis (ndim first) in
      bind1 (T.Concatenate ax) (with_dtype dt arrays)

let concat ?(axis = 0) arrays = concatenate ~axis arrays

let stack ?(axis = 0) ?dtype:dt arrays =
  match arrays with
  | [] -> invalid_arg "stack: need at least one array"
  | first :: _ ->
      let ax = canonicalize_axis axis (ndim first + 1) in
      bind1 (T.Stack ax) (with_dtype dt arrays)

let unstack ?(axis = 0) v =
  bind (T.Unstack (canonicalize_axis axis (ndim v))) [ v ]

let vstack ?dtype:dt arrays =
  concatenate ~axis:0 ?dtype:dt (List.map atleast_2d arrays)

let hstack ?dtype:dt arrays =
  let arrs = List.map atleast_1d arrays in
  let ax = if ndim (List.hd arrs) = 1 then 0 else 1 in
  concatenate ~axis:ax ?dtype:dt arrs

let dstack ?dtype:dt arrays =
  concatenate ~axis:2 ?dtype:dt (List.map atleast_3d arrays)

let column_stack arrays =
  let prep v = if ndim v < 2 then transpose (atleast_2d v) else v in
  concatenate ~axis:1 (List.map prep arrays)

let tile v reps =
  let nd = ndim v in
  let nr = Array.length reps in
  let reps =
    if nr < nd then Array.append (Array.make (nd - nr) 1) reps else reps
  in
  let nr2 = Array.length reps in
  let v =
    if nr2 > nd then expand_dims v (Array.init (nr2 - nd) Fun.id) else v
  in
  bind1 (T.Tile reps) [ v ]

let rec repeat ?axis ?total_repeat_length a repeats =
  match total_repeat_length with
  | Some total -> repeat_to_total ?axis a repeats total
  | None -> repeat_scalar ?axis a repeats

and repeat_to_total ?axis a repeats total =
  let a, ax =
    match axis with
    | None -> (ravel a, 0)
    | Some x -> (a, canonicalize_axis x (ndim a))
  in
  let full = repeat_scalar ~axis:ax a repeats in
  let n = (shape full).(ax) in
  if total = 0 then begin
    let sh = Array.copy (shape a) in
    sh.(ax) <- 0;
    const_full (dtype a) sh 0.0
  end
  else if (shape a).(ax) = 0 then a
  else if total <= n then slice_axis full ax 0 total
  else begin
    let last = slice_axis full ax (n - 1) n in
    let tail = repeat_scalar ~axis:ax last (total - n) in
    bind1 (T.Concatenate ax) [ full; tail ]
  end

and repeat_scalar ?axis a repeats =
  let a, ax =
    match axis with
    | None -> (ravel a, 0)
    | Some x -> (a, canonicalize_axis x (ndim a))
  in
  let ish = shape a in
  let nd = Array.length ish in
  let aux_axis = ax + 1 in
  let aux_shape =
    Array.init (nd + 1) (fun i ->
        if i < aux_axis then ish.(i)
        else if i = aux_axis then repeats
        else ish.(i - 1))
  in
  let dims = Array.init nd (fun i -> if i < aux_axis then i else i + 1) in
  let bc = bind1 (T.Broadcast_in_dim { shape = aux_shape; dims }) [ a ] in
  let result_shape = Array.copy ish in
  result_shape.(ax) <- ish.(ax) * repeats;
  reshape bc result_shape

let pad_widths_config pad_width =
  Np_ints.mark_triples_where
    (fun _ j -> j < 2)
    (Array.map (fun (lo, hi) -> (lo, hi, 0)) pad_width)

let pad_constant v pad_width cval =
  let cfg = pad_widths_config pad_width in
  bind1 (T.Pad cfg) [ v; scalar (dtype v) cval ]

let no_padding_on_empty (lo, hi) mode =
  if lo > 0 || hi > 0 then
    invalid_arg (Printf.sprintf "Cannot apply '%s' padding to empty axis" mode)

let pad_wrap v pad_width =
  let out = ref v in
  Array.iteri
    (fun i (lo, hi) ->
      let size_i = (shape !out).(i) in
      if size_i = 0 then no_padding_on_empty (lo, hi) "wrap"
      else begin
        let array = !out in
        let left_repeats = lo / size_i and left_rem = lo mod size_i in
        let right_repeats = hi / size_i and right_rem = hi mod size_i in
        let total = left_repeats + right_repeats + 1 in
        let parts =
          (if left_rem > 0 then
             [ slice_axis array i (size_i - left_rem) size_i ]
           else [])
          @ List.init total (fun _ -> array)
          @ if right_rem > 0 then [ slice_axis array i 0 right_rem ] else []
        in
        out := bind1 (T.Concatenate i) parts
      end)
    pad_width;
  !out

let pad_symmetric_or_reflect v pad_width mode reflect_type =
  let two_times_minus edge x = bind1 T.Sub [ bind1 T.Add [ edge; edge ]; x ] in
  let out = ref v in
  Array.iteri
    (fun i (lo, hi) ->
      let axis_size = (shape !out).(i) in
      if axis_size = 0 then no_padding_on_empty (lo, hi) mode
      else if lo <> 0 || hi <> 0 then begin
        let offset = if mode = "reflect" && axis_size > 1 then 1 else 0 in
        let build array padding before =
          let array = ref array in
          let current = ref padding in
          let d () = (shape !array).(i) in
          let edge =
            ref
              (if before then slice_axis !array i 0 1
               else slice_axis !array i (d () - 1) (d ()))
          in
          while !current > 0 do
            let curr_pad = min !current (axis_size - offset) in
            current := !current - curr_pad;
            let n = d () in
            let x =
              if before then slice_axis !array i offset (offset + curr_pad)
              else
                let stop =
                  if mode = "symmetric" || axis_size = 1 then n else n - 1
                in
                slice_axis !array i (n - (curr_pad + offset)) stop
            in
            let x = bind1 (T.Rev [| i |]) [ x ] in
            let x =
              if reflect_type = "odd" then begin
                let broadcast_edge =
                  bind1
                    (T.Broadcast_in_dim
                       { shape = shape x; dims = Array.init (ndim x) Fun.id })
                    [ !edge ]
                in
                let y = two_times_minus broadcast_edge x in
                if axis_size > 1 then begin
                  let m = (shape y).(i) in
                  edge :=
                    if before then slice_axis y i 0 1
                    else slice_axis y i (m - 1) m
                end;
                y
              end
              else x
            in
            array :=
              bind1 (T.Concatenate i)
                (if before then [ x; !array ] else [ !array; x ])
          done;
          !array
        in
        out := build (build !out lo true) hi false
      end)
    pad_width;
  !out

let pad_edge v pad_width =
  let out = ref v in
  Array.iteri
    (fun i (lo, hi) ->
      let n = (shape !out).(i) in
      if n = 0 then no_padding_on_empty (lo, hi) "edge"
      else begin
        let before = repeat ~axis:i (slice_axis !out i 0 1) lo in
        let after = repeat ~axis:i (slice_axis !out i (n - 1) n) hi in
        out := bind1 (T.Concatenate i) [ before; !out; after ]
      end)
    pad_width;
  !out

let pad_linear_ramp v pad_width end_values =
  let out = ref v in
  Array.iteri
    (fun i (lo, hi) ->
      let array = !out in
      let dt = dtype array in
      let n = (shape array).(i) in
      let start_lo, start_hi = end_values.(i) in
      let ramp edge start count =
        let sh = Array.copy (shape array) in
        sh.(i) <- count;
        let fractions =
          Array.init count (fun k ->
              float_of_int k /. float_of_int (Stdlib.max count 1))
        in
        let compute_dt =
          if is_integral dt || is_bool dt then Dtypes.default_float_dtype ()
          else dt
        in
        let frac =
          bind1
            (T.Broadcast_in_dim { shape = sh; dims = [| i |] })
            [ T.Concrete (Nd.of_floats compute_dt [| count |] fractions) ]
        in
        let e =
          bind1
            (T.Broadcast_in_dim
               { shape = sh; dims = Array.init (ndim array) Fun.id })
            [ convert edge compute_dt ]
        in
        let s = full compute_dt sh start in
        let value =
          bind1 T.Add [ s; bind1 T.Mul [ bind1 T.Sub [ e; s ]; frac ] ]
        in
        convert value dt
      in
      let before = ramp (slice_axis array i 0 1) start_lo lo in
      let after =
        bind1 (T.Rev [| i |])
          [ ramp (slice_axis array i (n - 1) n) start_hi hi ]
      in
      out := bind1 (T.Concatenate i) [ before; array; after ])
    pad_width;
  !out

let pad_stats v pad_width stat_length stat_func =
  let out = ref v in
  Array.iteri
    (fun i (lo, hi) ->
      let array = !out in
      let stat_before, stat_after =
        match stat_length with
        | None ->
            let s = stat_func array i in
            (s, s)
        | Some lengths ->
            let n = (shape array).(i) in
            let lb, la = lengths.(i) in
            if lb = 0 || la = 0 then
              invalid_arg "stat_length of 0 yields no value for padding";
            let lb = min lb n and la = min la n in
            ( stat_func (slice_axis array i 0 lb) i,
              stat_func (slice_axis array i (n - la) n) i )
      in
      let fit s =
        let s =
          if is_integral (dtype array) && not (is_integral (dtype s)) then
            bind1 T.Round_nearest_even [ s ]
          else s
        in
        convert s (dtype array)
      in
      let before = repeat ~axis:i (fit stat_before) lo in
      let after = repeat ~axis:i (fit stat_after) hi in
      out := bind1 (T.Concatenate i) [ before; array; after ])
    pad_width;
  !out

let pad_body ?(mode = "constant") ?stat_length ?end_values
    ?(reflect_type = "even") v pad_width constant_values =
  if ndim v = 0 then v
  else begin
    if Array.length pad_width <> ndim v then
      invalid_arg
        (Printf.sprintf "Expected pad_width to have %d pairs; got %d" (ndim v)
           (Array.length pad_width));
    Array.iter
      (fun (lo, hi) ->
        if lo < 0 || hi < 0 then
          invalid_arg "index can't contain negative values")
      pad_width;
    let keep_stat f a axis = f a axis in
    match mode with
    | "constant" -> pad_constant v pad_width constant_values
    | "wrap" -> pad_wrap v pad_width
    | "symmetric" | "reflect" ->
        pad_symmetric_or_reflect v pad_width mode reflect_type
    | "edge" -> pad_edge v pad_width
    | "linear_ramp" ->
        let ends =
          match end_values with
          | Some e -> e
          | None -> Array.make (ndim v) (0.0, 0.0)
        in
        pad_linear_ramp v pad_width ends
    | "maximum" ->
        pad_stats v pad_width stat_length
          (keep_stat (fun a axis ->
               Reductions.amax ~axis:[| axis |] ~keepdims:true a))
    | "minimum" ->
        pad_stats v pad_width stat_length
          (keep_stat (fun a axis ->
               Reductions.amin ~axis:[| axis |] ~keepdims:true a))
    | "mean" ->
        pad_stats v pad_width stat_length
          (keep_stat (fun a axis ->
               Reductions.mean ~axis:[| axis |] ~keepdims:true a))
    | "median" ->
        pad_stats v pad_width stat_length
          (keep_stat (fun a axis -> Reductions.median ~axis ~keepdims:true a))
    | "empty" -> pad_constant v pad_width 0.0
    | m ->
        failwith
          (Printf.sprintf "Unimplemented padding mode '%s' for np.pad." m)
  end

let pad_key mode pad_width stat_length end_values reflect_type =
  Digest.to_hex
    (Digest.string
       (Marshal.to_string
          (mode, pad_width, stat_length, end_values, reflect_type)
          []))

let pad ?(mode = "constant") ?stat_length ?end_values ?(reflect_type = "even")
    ?(float_literal = false) v pad_width constant_values =
  let key = pad_key mode pad_width stat_length end_values reflect_type in
  let body v =
    pad_body ~mode ?stat_length ?end_values ~reflect_type v pad_width 0.0
  in
  if ndim v = 0 then v
  else if mode = "constant" then begin
    let literal =
      if Float.is_integer constant_values && not float_literal then
        Jnp_util.py_int (int_of_float constant_values)
      else Jnp_util.py_float constant_values
    in
    Jnp_util.jit1 "_pad" ~key
      (function
        | [ v; cval ] ->
            let cfg = pad_widths_config pad_width in
            let cval =
              C.convert_like ~weak:(C.get_aval v).T.weak_type cval (dtype v)
            in
            bind1 (Np_ints.mark_prim (T.Pad cfg)) [ v; cval ]
        | _ -> invalid_arg "pad: arity")
      [ v; literal ]
  end
  else
    Jnp_util.jit1 "_pad" ~key
      (function [ v ] -> body v | _ -> invalid_arg "pad: arity")
      [ v ]

let i0 v =
  if is_complex_dtype (dtype v) then
    invalid_arg "Unsupported input type to jax.numpy.i0: complex";
  let v = convert v (to_inexact_dtype (dtype v)) in
  let ax = bind1 T.Abs [ v ] in
  bind1 T.Mul [ bind1 T.Exp [ ax ]; bind1 T.Bessel_i0e [ ax ] ]

let i0 p0 =
  Jnp_util.jit1 "i0"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> i0 p0 | _ -> invalid_arg "i0: arity")
    [ p0 ]

let array_equal ?(equal_nan = false) a b =
  if shape a <> shape b then scalar D.Bool 0.0
  else begin
    let a, b, _, _ = promote2 a b in
    let eq = bind1 T.Eq [ a; b ] in
    let eq =
      if equal_nan then bind1 T.Or [ eq; bind1 T.And [ isnan a; isnan b ] ]
      else eq
    in
    all_reduce eq
  end

let array_equiv a b =
  match promote2 a b with
  | exception Invalid_argument _ -> scalar D.Bool 0.0
  | a, b, _, _ -> all_reduce (bind1 T.Eq [ a; b ])

let arange ?(start = 0.0) ?step ~dtype stop =
  match step with
  | Some step ->
      let working =
        match dtype with
        | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn | D.F8_e3m4
        | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz | D.F8_e5m2
        | D.F8_e5m2fnuz | D.F8_e8m0fnu ->
            D.F32
        | _ -> dtype
      in
      let size = max 0 (int_of_float (ceil ((stop -. start) /. step))) in
      let idx =
        bind1 (T.Iota { dtype = working; shape = [| size |]; dimension = 0 }) []
      in
      convert
        (bind1 T.Add
           [ scalar working start; bind1 T.Mul [ scalar working step; idx ] ])
        dtype
  | None ->
      let size = max 0 (int_of_float (ceil (stop -. start))) in
      let idx =
        bind1 (T.Iota { dtype; shape = [| size |]; dimension = 0 }) []
      in
      if start = 0.0 then idx else bind1 T.Add [ scalar dtype start; idx ]

let eye ?m ?(k = 0) ~dtype n =
  let m = Option.value m ~default:n in
  let idt = Dtypes.default_int_dtype () in
  let i =
    bind1 (T.Iota { dtype = idt; shape = [| n; m |]; dimension = 0 }) []
  in
  let j =
    bind1 (T.Iota { dtype = idt; shape = [| n; m |]; dimension = 1 }) []
  in
  let ik = bind1 T.Add [ i; C.splat idt [||] (float_of_int k) ] in
  convert (bind1 T.Eq [ ik; j ]) dtype

let identity ~dtype n = eye ~dtype n

let indices ~dtype dims =
  let n = Array.length dims in
  let outs =
    List.init n (fun i ->
        let idx =
          bind1 (T.Iota { dtype; shape = [| dims.(i) |]; dimension = 0 }) []
        in
        bind1 (T.Broadcast_in_dim { shape = dims; dims = [| i |] }) [ idx ])
  in
  stack ~axis:0 outs

let meshgrid ?(indexing = "xy") ?(sparse = false) xs =
  let args = Array.of_list xs in
  let n = Array.length args in
  let swap = indexing = "xy" && n >= 2 in
  if swap then begin
    let t = args.(0) in
    args.(0) <- args.(1);
    args.(1) <- t
  end;
  let base = Array.map (fun a -> (shape a).(0)) args in
  let out =
    Array.mapi
      (fun i a ->
        let sh =
          if sparse then
            Array.init n (fun j -> if j = i then (shape a).(0) else 1)
          else base
        in
        bind1 (T.Broadcast_in_dim { shape = sh; dims = [| i |] }) [ a ])
      args
  in
  if swap then begin
    let t = out.(0) in
    out.(0) <- out.(1);
    out.(1) <- t
  end;
  Array.to_list out

let ix_ xs =
  let n = List.length xs in
  List.mapi
    (fun i a ->
      let sh = Array.init n (fun j -> if j = i then (shape a).(0) else 1) in
      bind1 (T.Broadcast_in_dim { shape = sh; dims = [| i |] }) [ a ])
    xs

let binop prim x y =
  let dt = result_type [ x; y ] in
  let x, y = Jnp_util.promote_ranks (convert x dt) (convert y dt) in
  bind1 prim [ x; y ]

let mul2 = binop T.Mul
let sub2 = binop T.Sub
let add2 = binop T.Add
let mul_scalar c v = bind1 T.Mul [ v; full (dtype v) (shape v) c ]
let squeeze_last v = bind1 (T.Squeeze [| ndim v - 1 |]) [ v ]

let last_index v i =
  let n = ndim v in
  squeeze_last (slice_axis v (n - 1) i (i + 1))

let append ?axis arr values =
  match axis with
  | None -> concatenate ~axis:0 [ ravel arr; ravel values ]
  | Some ax -> concatenate ~axis:ax [ arr; values ]

let append ?axis p0 p1 =
  Jnp_util.jit1 "append"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string axis [])))
    (function
      | [ p0; p1 ] -> append ?axis p0 p1 | _ -> invalid_arg "append: arity")
    [ p0; p1 ]

let argmax ?axis ?(keepdims = false) a =
  (match dtype a with
  | D.Complex64 | D.Complex128 ->
      raise (Errors.jax_type_error "argmax does not accept complex input.")
  | _ -> ());
  let orig_nd = ndim a in
  let a2, ax, dims =
    match axis with
    | None -> (ravel a, 0, Array.init orig_nd Fun.id)
    | Some x ->
        let c = canonicalize_axis x orig_nd in
        (a, c, [| c |])
  in
  let r =
    bind1
      (T.Argmax { axis = ax; index_dtype = Dtypes.default_int_dtype () })
      [ a2 ]
  in
  if keepdims then expand_dims r dims else r

let cross ?(axisa = -1) ?(axisb = -1) ?(axisc = -1) ?axis a b =
  let axisa, axisb, axisc =
    match axis with Some ax -> (ax, ax, ax) | None -> (axisa, axisb, axisc)
  in
  let a = moveaxis [| axisa |] [| ndim a - 1 |] a in
  let b = moveaxis [| axisb |] [| ndim b - 1 |] b in
  let na = (shape a).(ndim a - 1) and nb = (shape b).(ndim b - 1) in
  if not ((na = 2 || na = 3) && (nb = 2 || nb = 3)) then
    invalid_arg "Dimension must be either 2 or 3 for cross product";
  if na = 2 && nb = 2 then
    let a0 = last_index a 0 and a1 = last_index a 1 in
    let b0 = last_index b 0 and b1 = last_index b 1 in
    sub2 (mul2 a0 b1) (mul2 a1 b0)
  else begin
    let a0 = last_index a 0 and a1 = last_index a 1 in
    let a2 = if na = 3 then last_index a 2 else zeros_like a0 in
    let b0 = last_index b 0 and b1 = last_index b 1 in
    let b2 = if nb = 3 then last_index b 2 else zeros_like b0 in
    let c0 = sub2 (mul2 a1 b2) (mul2 a2 b1) in
    let c1 = sub2 (mul2 a2 b0) (mul2 a0 b2) in
    let c2 = sub2 (mul2 a0 b1) (mul2 a1 b0) in
    let lift v =
      let s = shape v in
      bind1
        (T.Broadcast_in_dim
           {
             shape = Array.append [| 1 |] s;
             dims = Array.init (Array.length s) (fun i -> i + 1);
           })
        [ v ]
    in
    let c = bind1 (T.Concatenate 0) (List.map lift [ c0; c1; c2 ]) in
    moveaxis [| 0 |] [| axisc |] c
  end

let cross ?axisa ?axisb ?axisc ?axis p0 p1 =
  Jnp_util.jit1 "cross"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (axisa, axisb, axisc, axis) [])))
    (function
      | [ p0; p1 ] -> cross ?axisa ?axisb ?axisc ?axis p0 p1
      | _ -> invalid_arg "cross: arity")
    [ p0; p1 ]

let diag_indices ?(ndim = 2) n =
  let dt = Dtypes.default_int_dtype () in
  let one = bind1 (T.Iota { dtype = dt; shape = [| n |]; dimension = 0 }) [] in
  List.init ndim (fun _ -> one)

let diag_indices_from arr =
  let s = shape arr in
  let nd = Array.length s in
  if nd < 2 then invalid_arg "input array must be at least 2-d";
  Array.iter
    (fun d ->
      if d <> s.(0) then
        invalid_arg "All dimensions of input must be of equal length")
    s;
  diag_indices ~ndim:nd s.(0)

let tri ?m ?(k = 0) ~dtype n =
  let m = Option.value m ~default:n in
  convert (Lax.Linalg.tri_mask n m k) dtype

let batch_mask mask s =
  let nd = Array.length s in
  if nd = 2 then mask
  else
    bind1
      (T.Broadcast_in_dim { shape = s; dims = [| nd - 2; nd - 1 |] })
      [ mask ]

let tril ?(k = 0) m =
  let s = shape m in
  let nd = Array.length s in
  if nd < 2 then invalid_arg "Argument to jax.numpy.tril must be at least 2D";
  Jnp_util.jit1 "tril" ~key:(string_of_int k)
    (function
      | [ m ] ->
          let mask = tri ~m:s.(nd - 1) ~k ~dtype:D.Bool s.(nd - 2) in
          bind1 T.Select_n [ batch_mask mask s; zeros_like m; m ]
      | _ -> invalid_arg "tril: arity")
    [ m ]

let triu ?(k = 0) m =
  let s = shape m in
  let nd = Array.length s in
  if nd < 2 then invalid_arg "Argument to jax.numpy.triu must be at least 2D";
  Jnp_util.jit1 "triu" ~key:(string_of_int k)
    (function
      | [ m ] ->
          let mask = tri ~m:s.(nd - 1) ~k:(k - 1) ~dtype:D.Bool s.(nd - 2) in
          bind1 T.Select_n [ batch_mask mask s; m; zeros_like m ]
      | _ -> invalid_arg "triu: arity")
    [ m ]

let reduce_over v axis =
  if is_bool (dtype v) then bind1 (T.Reduce_or [| axis |]) [ v ]
  else bind1 (T.Reduce_sum [| axis |]) [ v ]

let trace ?(offset = 0) ?(axis1 = 0) ?(axis2 = 1) ?dtype a =
  let s = shape a in
  let nd = Array.length s in
  let a1 = canonicalize_axis axis1 nd and a2 = canonicalize_axis axis2 nd in
  if a1 = a2 then invalid_arg "axis1 and axis2 can not be same";
  let na = s.(a1) and ma = s.(a2) in
  let am = moveaxis [| axis1; axis2 |] [| nd - 2; nd - 1 |] a in
  let mask = eye ~m:ma ~k:offset ~dtype:D.Bool na in
  let masked = where_ mask am (zeros_like am) in
  Reductions.sum ?dtype ~axis:[| nd - 2; nd - 1 |] masked

let trace ?offset ?axis1 ?axis2 ?dtype a0 =
  Eager_jit.one ~jit:"trace" "numpy.lax_numpy.trace"
    (offset, axis1, axis2, dtype) [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> trace ?offset ?axis1 ?axis2 ?dtype a0
      | _ -> assert false)

let normalize_index i n =
  let negative = bind1 T.Lt [ i; scalar (dtype i) 0.0 ] in
  let shifted = bind1 T.Add [ i; scalar (dtype i) (float_of_int n) ] in
  bind1 T.Select_n [ negative; i; shifted ]

let expand_last i =
  let sh = shape i in
  let n = Array.length sh in
  bind1
    (T.Broadcast_in_dim
       { shape = Array.append sh [| 1 |]; dims = Array.init n Fun.id })
    [ i ]

let take_trailing a indices =
  let s = shape a in
  let nd = Array.length s in
  let k = List.length indices in
  let wrapped =
    List.mapi (fun d i -> normalize_index i s.(nd - k + d)) indices
  in
  let wrapped = List.map (fun w -> convert w D.I32) wrapped in
  let start =
    match wrapped with
    | [ w ] -> expand_last w
    | _ -> bind1 (T.Concatenate 1) (List.map expand_last wrapped)
  in
  let batch = Array.length (shape (List.hd indices)) in
  let dnums =
    {
      T.offset_dims = Array.init (nd - k) Fun.id;
      collapsed_slice_dims = Array.init k (fun d -> nd - k + d);
      start_index_map = Array.init k (fun d -> nd - k + d);
      g_operand_batching_dims = [||];
      g_start_indices_batching_dims = [||];
    }
  in
  ignore batch;
  bind1
    (T.Gather
       {
         dimension_numbers = dnums;
         slice_sizes = Array.mapi (fun d e -> if d >= nd - k then 1 else e) s;
       })
    [ a; start ]

let index_at a i =
  let s = shape a in
  let i = if i < 0 then i + s.(0) else i in
  let n = Array.length s in
  let start = Array.init n (fun d -> if d = 0 then i else 0) in
  let limit = Array.init n (fun d -> if d = 0 then i + 1 else s.(d)) in
  let sliced =
    bind1
      (T.Slice { start_indices = start; limit_indices = limit; strides = None })
      [ a ]
  in
  bind1 (T.Squeeze [| 0 |]) [ sliced ]

let default_diag ~offset ~axis1 ~axis2 a =
  let s = shape a in
  let nd = Array.length s in
  let a1 = canonicalize_axis axis1 nd and a2 = canonicalize_axis axis2 nd in
  let am = moveaxis [| axis1; axis2 |] [| nd - 2; nd - 1 |] a in
  let diag_size = max 0 (min (s.(a1) + min offset 0) (s.(a2) - max offset 0)) in
  let iota_from start =
    let base =
      bind1
        (T.Iota
           {
             dtype = Dtypes.default_int_dtype ();
             shape = [| diag_size |];
             dimension = 0;
           })
        []
    in
    if start = 0 then base
    else
      bind1 T.Add
        [ base; scalar (Dtypes.default_int_dtype ()) (float_of_int start) ]
  in
  let i = iota_from 0 in
  let j = iota_from (abs offset) in
  if offset >= 0 then take_trailing am [ i; j ] else take_trailing am [ j; i ]

let mosaic_diag a_eye a =
  let product = bind1 T.Mul [ a_eye; a ] in
  if is_bool (dtype product) then bind1 (T.Reduce_or [| 0 |]) [ product ]
  else bind1 (T.Reduce_sum [| 0 |]) [ product ]

let diagonal ?(offset = 0) ?(axis1 = 0) ?(axis2 = 1) a =
  let s = shape a in
  let nd = Array.length s in
  if nd < 2 then
    invalid_arg "diagonal requires an array of at least two dimensions.";
  if s.(0) <> s.(1) || nd <> 2 || offset <> 0 || is_bool (dtype a) then
    default_diag ~offset ~axis1 ~axis2 a
  else begin
    let a_eye = eye ~dtype:(dtype a) s.(0) in
    let platforms = [| Some [| "mosaic" |]; None |] in
    let index = bind1 (T.Platform_index platforms) [] in
    let branch_avals = [ get_aval a_eye; get_aval a ] in
    let t =
      Jaxpr.make_jaxpr branch_avals (function
        | [ _; v ] -> [ default_diag ~offset ~axis1 ~axis2 v ]
        | _ -> invalid_arg "diagonal: arity")
    in
    let f =
      Jaxpr.make_jaxpr branch_avals (function
        | [ e; v ] -> [ mosaic_diag e v ]
        | _ -> invalid_arg "diagonal: arity")
    in
    List.hd
      (C.bind
         (T.Cond { branches = [ f; t ]; platforms = Some platforms })
         [ index; a_eye; a ])
  end

let diagonal ?offset ?axis1 ?axis2 p0 =
  Jnp_util.jit1 "diagonal"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (offset, axis1, axis2) [])))
    (function
      | [ p0 ] -> diagonal ?offset ?axis1 ?axis2 p0
      | _ -> invalid_arg "diagonal: arity")
    [ p0 ]

let diag ?(k = 0) v =
  let s = shape v in
  match Array.length s with
  | 1 ->
      let zero () = scalar (dtype v) 0.0 in
      let n = s.(0) + abs k in
      let padded =
        bind1 (T.Pad [| (max 0 (-k), max 0 k, 0) |]) [ v; zero () ]
      in
      let mask = eye ~k ~dtype:D.Bool n in
      let col =
        bind1
          (T.Broadcast_in_dim { shape = [| n; 1 |]; dims = [| 0 |] })
          [ padded ]
      in
      where_ mask col (zero ())
  | 2 -> diagonal ~offset:k v
  | _ -> invalid_arg "diag input must be 1d or 2d"

let diag ?k a0 =
  Eager_jit.one ~jit:"_diag" "numpy.lax_numpy.diag" k [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> diag ?k a0 | _ -> assert false)

let diagflat ?(k = 0) v = diag ~k (ravel v)

let kron a b =
  let dt = result_type [ a; b ] in
  let a = convert a dt and b = convert b dt in
  let nda = ndim a and ndb = ndim b in
  let a =
    if nda < ndb then expand_dims a (Array.init (ndb - nda) Fun.id) else a
  in
  let b =
    if ndb < nda then expand_dims b (Array.init (nda - ndb) Fun.id) else b
  in
  let nd = ndim a in
  let a_r = expand_dims a (Array.init nd (fun i -> 1 + (2 * i))) in
  let b_r = expand_dims b (Array.init nd (fun i -> 2 * i)) in
  let sa = shape a and sb = shape b in
  let out_shape = Array.init nd (fun i -> sa.(i) * sb.(i)) in
  reshape (mul2 a_r b_r) out_shape

let kron p0 p1 =
  Jnp_util.jit1 "kron"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> kron p0 p1 | _ -> invalid_arg "kron: arity")
    [ p0; p1 ]

let vander ?n ?(increasing = false) x =
  if ndim x <> 1 then invalid_arg "x must be a one-dimensional array";
  let len = (shape x).(0) in
  let nn = Option.value n ~default:len in
  if nn < 0 then invalid_arg "N must be nonnegative";
  let dt = dtype x in
  let io = bind1 (T.Iota { dtype = dt; shape = [| nn |]; dimension = 0 }) [] in
  let io =
    if increasing then io
    else bind1 T.Sub [ scalar dt (float_of_int (nn - 1)); io ]
  in
  let lift v sh dims = bind1 (T.Broadcast_in_dim { shape = sh; dims }) [ v ] in
  let xcol = lift x [| len; 1 |] [| 0 |] in
  let expo = lift io [| 1; nn |] [| 1 |] in
  Ufuncs.power xcol expo

let vander ?n ?increasing p0 =
  Jnp_util.jit1 "vander"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string (n, increasing) [])))
    (function
      | [ p0 ] -> vander ?n ?increasing p0 | _ -> invalid_arg "vander: arity")
    [ p0 ]

let trapezoid ?x ?(dx = 1.0) ?(axis = -1) y =
  let y = convert y (to_inexact_dtype (dtype y)) in
  let nd = ndim y in
  let ax = canonicalize_axis axis nd in
  let dxd =
    match x with
    | None -> None
    | Some xv ->
        let xv = convert xv (to_inexact_dtype (dtype xv)) in
        if ndim xv = 1 then Some (diff xv)
        else invalid_arg "trapezoid: only 1-dimensional x is supported"
  in
  let y = moveaxis [| ax |] [| nd - 1 |] y in
  let ny = shape y in
  let l = ny.(nd - 1) in
  match dxd with
  | None ->
      let s = Reductions.sum ~axis:[| nd - 1 |] y in
      let y0 = last_index y 0 in
      let yl = last_index y (l - 1) in
      let total = add2 y0 yl in
      let ends = bind1 T.Mul [ scalar (dtype total) 0.5; total ] in
      let out = sub2 s ends in
      bind1 T.Mul [ scalar (dtype out) dx; out ]
  | Some dxa ->
      let y1 = slice_axis y (nd - 1) 1 l in
      let y0 = slice_axis y (nd - 1) 0 (l - 1) in
      let prod = mul2 dxa (add2 y1 y0) in
      mul_scalar 0.5 (bind1 (T.Reduce_sum [| nd - 1 |]) [ prod ])

let trapezoid ?x ?dx ?axis p0 =
  let x_given = x <> None in
  if x_given then trapezoid ?x ?dx ?axis p0
  else
    Jnp_util.jit1 "trapezoid"
      ~key:(Digest.to_hex (Digest.string (Marshal.to_string (dx, axis) [])))
      (function
        | [ p0 ] -> trapezoid ?x ?dx ?axis p0
        | _ -> invalid_arg "trapezoid: arity")
      [ p0 ]

let any_reduce v =
  let n = ndim v in
  if n = 0 then v else bind1 (T.Reduce_or (Array.init n Fun.id)) [ v ]

let all_over ?axis ?(keepdims = false) v =
  match axis with
  | None ->
      let r = all_reduce v in
      if keepdims then reshape r (Array.make (ndim v) 1) else r
  | Some ax ->
      let axc = canonicalize_axis ax (ndim v) in
      let r = bind1 (T.Reduce_and [| axc |]) [ v ] in
      if keepdims then expand_dims r [| axc |] else r

let scalar_at v i = squeeze (slice_axis v 0 i (i + 1))

let argmin ?axis ?(keepdims = false) a =
  (match dtype a with
  | D.Complex64 | D.Complex128 ->
      raise (Errors.jax_type_error "argmin does not accept complex input.")
  | _ -> ());
  let orig_nd = ndim a in
  let a2, ax, dims =
    match axis with
    | None -> (ravel a, 0, Array.init orig_nd Fun.id)
    | Some x ->
        let c = canonicalize_axis x orig_nd in
        (a, c, [| c |])
  in
  let r =
    bind1
      (T.Argmin { axis = ax; index_dtype = Dtypes.default_int_dtype () })
      [ a2 ]
  in
  if keepdims then expand_dims r dims else r

let nanargmax ?axis ?(keepdims = false) a =
  (match dtype a with
  | D.Complex64 | D.Complex128 ->
      raise (Errors.jax_type_error "nanargmax does not accept complex input.")
  | _ -> ());
  if not (is_inexact (dtype a)) then argmax ?axis ~keepdims a
  else begin
    let nan_mask = isnan a in
    let a' = where_ nan_mask (const_full (dtype a) (shape a) neg_infinity) a in
    let res = argmax ?axis ~keepdims a' in
    let all_nan = all_over ?axis ~keepdims nan_mask in
    where_ all_nan (scalar (dtype res) (-1.0)) res
  end

let nanargmin ?axis ?(keepdims = false) a =
  (match dtype a with
  | D.Complex64 | D.Complex128 ->
      raise (Errors.jax_type_error "nanargmin does not accept complex input.")
  | _ -> ());
  if not (is_inexact (dtype a)) then argmin ?axis ~keepdims a
  else begin
    let nan_mask = isnan a in
    let a' = where_ nan_mask (const_full (dtype a) (shape a) infinity) a in
    let res = argmin ?axis ~keepdims a' in
    let all_nan = all_over ?axis ~keepdims nan_mask in
    where_ all_nan (scalar (dtype res) (-1.0)) res
  end

let roll_static a shift axis =
  let n = max (Array.length shift) (Array.length axis) in
  let get arr i = arr.(if Array.length arr = 1 then 0 else i) in
  Jnp_util.jit1 "_roll_static"
    ~key:(int_key shift ^ "/" ^ int_key axis)
    (function
      | [ a ] ->
          let acc = ref a in
          for k = 0 to n - 1 do
            let ax = get axis k and s = get shift k in
            let sz = (shape !acc).(ax) in
            if sz <> 0 then begin
              let i = ((-s mod sz) + sz) mod sz in
              let slice lo hi =
                let cur = shape !acc in
                let nd = Array.length cur in
                bind1
                  (T.Slice
                     {
                       start_indices =
                         Array.init nd (fun d -> if d = ax then lo else 0);
                       limit_indices =
                         Array.mapi (fun d e -> if d = ax then hi else e) cur;
                       strides = Some (Array.make nd 1);
                     })
                  [ !acc ]
              in
              let high = slice i sz in
              let low = slice 0 i in
              acc := bind1 (T.Concatenate ax) [ high; low ]
            end
          done;
          !acc
      | _ -> invalid_arg "roll: arity")
    [ a ]

let roll ?axis a shift =
  match axis with
  | None -> reshape (roll_static (ravel a) shift [| 0 |]) (shape a)
  | Some ax ->
      let axc = Array.map (fun x -> canonicalize_axis x (ndim a)) ax in
      roll_static a shift axc

let rollaxis ?(start = 0) axis a =
  let nd = ndim a in
  let axis = canonicalize_axis axis nd in
  if not (-nd <= start && start <= nd) then
    invalid_arg
      (Printf.sprintf "start=%d must satisfy %d<=start<=%d" start (-nd) nd);
  let start = if start < 0 then start + nd else start in
  let start = if start > axis then start - 1 else start in
  moveaxis [| axis |] [| start |] a

let rollaxis ?start s0 p1 =
  Jnp_util.jit1 "rollaxis"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string (start, s0) [])))
    (function
      | [ p1 ] -> rollaxis ?start s0 p1 | _ -> invalid_arg "rollaxis: arity")
    [ p1 ]

let abs_of v =
  match dtype v with
  | D.Bool | D.Uint8 | D.Uint16 | D.Uint32 | D.Uint64 -> v
  | _ -> bind1 T.Abs [ v ]

let gcd x1 x2 =
  let x1, x2 =
    match promote_values [ x1; x2 ] with
    | [ a; b ] -> (a, b)
    | _ -> assert false
  in
  if not (is_integral (dtype x1)) then
    invalid_arg "Arguments to jax.numpy.gcd must be integers.";
  let x1, x2 =
    match broadcast_arrays [ x1; x2 ] with
    | [ a; b ] -> (a, b)
    | _ -> assert false
  in
  let cond carry =
    match carry with
    | [ _; b ] -> any_reduce (bind1 T.Ne [ b; scalar (dtype b) 0.0 ])
    | _ -> assert false
  in
  let body carry =
    match carry with
    | [ a; b ] ->
        let ne0 () = bind1 T.Ne [ b; scalar (dtype b) 0.0 ] in
        let t1 = where_ (ne0 ()) b a in
        let nz = ne0 () in
        let t2 = where_ nz (bind1 T.Rem [ a; b ]) (scalar (dtype b) 0.0) in
        let lt1 () = bind1 T.Lt [ t1; t2 ] in
        let u1 = where_ (lt1 ()) t2 t1 in
        let u2 = where_ (lt1 ()) t1 t2 in
        [ u1; u2 ]
    | _ -> assert false
  in
  let init = [ abs_of x1; abs_of x2 ] in
  List.hd (Lax.while_loop cond body init)

let gcd p0 p1 =
  Jnp_util.jit1 "gcd"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> gcd p0 p1 | _ -> invalid_arg "gcd: arity")
    [ p0; p1 ]

let lcm x1 x2 =
  let x1, x2 =
    match promote_values [ x1; x2 ] with
    | [ a; b ] -> (a, b)
    | _ -> assert false
  in
  let x1 = abs_of x1 and x2 = abs_of x2 in
  if not (is_integral (dtype x1)) then
    invalid_arg "Arguments to jax.numpy.lcm must be integers.";
  let d = gcd x1 x2 in
  let is_zero = bind1 T.Eq [ d; scalar (dtype d) 0.0 ] in
  let prod = Ufuncs.multiply x1 (Ufuncs.floor_divide x2 d) in
  where_ is_zero (scalar (dtype d) 0.0) prod

let lcm p0 p1 =
  Jnp_util.jit1 "lcm"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> lcm p0 p1 | _ -> invalid_arg "lcm: arity")
    [ p0; p1 ]

let check_search_method = function
  | "scan" | "scan_unrolled" | "sort" | "compare_all" -> ()
  | m ->
      invalid_arg
        ("method must be one of ['scan', 'scan_unrolled', 'sort', \
          'compare_all']; got " ^ m)

let gather_1d a indices =
  let sh = shape indices in
  let dnums =
    {
      T.offset_dims = [||];
      collapsed_slice_dims = [| 0 |];
      start_index_map = [| 0 |];
      g_operand_batching_dims = [||];
      g_start_indices_batching_dims = [||];
    }
  in
  bind1
    (T.Gather { dimension_numbers = dnums; slice_sizes = [| 1 |] })
    [ a; reshape indices (Array.append sh [| 1 |]) ]

let canonicalize_for_sort x =
  let dt = dtype x in
  let result =
    bind1 T.Select_n
      [ bind1 T.Eq [ x; scalar dt 0.0 ]; x; const_full dt (shape x) 0.0 ]
  in
  bind1 T.Select_n
    [ bind1 T.Ne [ x; x ]; result; const_full dt (shape x) Float.nan ]

let sort_compare_keys x =
  let dt = dtype x in
  if is_complex_dtype dt then
    [
      canonicalize_for_sort (bind1 T.Real [ x ]);
      canonicalize_for_sort (bind1 T.Imag [ x ]);
    ]
  else if is_inexact dt then [ canonicalize_for_sort x ]
  else [ x ]

let sort_comparator strict x y =
  let xk = sort_compare_keys x in
  let yk = sort_compare_keys y in
  let last = if strict then T.Lt_to else T.Le_to in
  List.fold_left2
    (fun acc a b ->
      match acc with
      | None -> Some (bind1 last [ a; b ])
      | Some p ->
          Some
            (bind1 T.Or
               [
                 bind1 T.Lt_to [ a; b ];
                 bind1 T.And [ bind1 T.Eq_to [ a; b ]; p ];
               ]))
    None (List.rev xk) (List.rev yk)
  |> Option.get

let scan_levels n =
  let rec go k = if 1 lsl k >= n + 1 then k else go (k + 1) in
  go 0

let dynamic_item a i =
  let sliced = bind1 (T.Dynamic_slice { slice_sizes = [| 1 |] }) [ a; i ] in
  bind1 (T.Squeeze [| 0 |]) [ sliced ]

let rec searchsorted_scan_impl ~side ~dt ~unrolled sorted_arr query =
  Jnp_util.jit1 "_searchsorted_scan_impl"
    ~key:(Printf.sprintf "%s,%s,%b" side (D.short_name dt) unrolled)
    (function
      | [ sorted_arr; query ] ->
          let n = (shape sorted_arr).(0) in
          if n = 0 then const_full dt (shape query) 0.0
          else if Array.length (shape query) > 0 then
            List.hd
              (Interpreters.Batching.vmap
                 (function
                   | [ s; q ] ->
                       [ searchsorted_scan_impl ~side ~dt ~unrolled s q ]
                   | _ -> invalid_arg "searchsorted: arity")
                 [ None; Some 0 ] [ sorted_arr; query ])
          else
            let unsigned = if dt = D.I64 then D.Uint64 else D.Uint32 in
            let n_levels = scan_levels n in
            let init =
              [
                T.Concrete (Nd.of_floats unsigned [||] [| 0.0 |]);
                T.Concrete (Nd.of_floats unsigned [||] [| float_of_int n |]);
              ]
            in
            let body = function
              | [ low; high ] ->
                  let half =
                    Ufuncs.floor_divide
                      (bind1 T.Sub [ high; low ])
                      (Jnp_util.py_int 2)
                  in
                  let mid = bind1 T.Add [ low; half ] in
                  let item = dynamic_item sorted_arr mid in
                  let go_left =
                    if side = "left" then sort_comparator false query item
                    else sort_comparator true query item
                  in
                  [
                    bind1 T.Select_n [ go_left; mid; low ];
                    bind1 T.Select_n [ go_left; high; mid ];
                  ]
              | _ -> invalid_arg "searchsorted: state"
            in
            let out =
              Lax.scan ~length:n_levels
                ~unroll:(if unrolled then n_levels else 1)
                body init []
            in
            convert (List.nth out 1) dt
      | _ -> invalid_arg "searchsorted: arity")
    [ sorted_arr; query ]

let searchsorted ?(side = "left") ?sorter ?(method_ = "scan") a v =
  if ndim a <> 1 then invalid_arg "a should be 1-dimensional";
  check_search_method method_;
  let args = [ a; v ] @ Option.to_list sorter in
  Jnp_util.jit1 "searchsorted"
    ~key:(side ^ "," ^ method_)
    (function
      | a :: v :: rest -> (
          let a, v =
            match promote_values [ a; v ] with
            | [ a; v ] -> (a, v)
            | _ -> assert false
          in
          let a = match rest with [ s ] -> take_trailing a [ s ] | _ -> a in
          let dt = D.I32 in
          match method_ with
          | "scan" | "scan_unrolled" ->
              searchsorted_scan_impl ~side ~dt
                ~unrolled:(method_ = "scan_unrolled")
                a v
          | _ ->
              let na = (shape a).(0) in
              let vsh = shape v in
              let out_shape = Array.append [| na |] vsh in
              let a_b =
                bind1
                  (T.Broadcast_in_dim { shape = out_shape; dims = [| 0 |] })
                  [ a ]
              in
              let v_dims = Array.init (Array.length vsh) (fun i -> i + 1) in
              let v_b =
                bind1
                  (T.Broadcast_in_dim { shape = out_shape; dims = v_dims })
                  [ v ]
              in
              let cmp =
                if side = "right" then bind1 T.Le [ a_b; v_b ]
                else bind1 T.Lt [ a_b; v_b ]
              in
              bind1 (T.Reduce_sum [| 0 |]) [ convert cmp dt ])
      | _ -> invalid_arg "searchsorted: arity")
    args

let digitize ?(right = false) ?(method_ = "scan") x bins =
  check_search_method method_;
  if ndim bins <> 1 then
    invalid_arg "digitize: bins must be a 1-dimensional array";
  let nb = (shape bins).(0) in
  if nb = 0 then const_full D.I32 (shape x) 0.0
  else begin
    let side = if right then "left" else "right" in
    let inc = bind1 T.Ge [ index_at bins (-1); index_at bins 0 ] in
    let asc = searchsorted ~side ~method_ bins x in
    let desc_search = searchsorted ~side ~method_ (rev_dims bins [| 0 |]) x in
    let nb_lit =
      T.Concrete
        (Nd.as_weak
           (Nd.of_floats (dtype desc_search) [||] [| float_of_int nb |]))
    in
    let desc = bind1 T.Sub [ nb_lit; desc_search ] in
    where_ inc asc desc
  end

let digitize ?right ?method_ a0 a1 =
  Eager_jit.one ~jit:"digitize" "numpy.lax_numpy.digitize" (right, method_) []
    [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> digitize ?right ?method_ a0 a1
      | _ -> assert false)

let dtype_of = dtype

let cov ?y ?(rowvar = true) ?(bias = false) ?ddof ?fweights ?aweights ?dtype m =
  let dt_common =
    match y with
    | None -> dtype_of m
    | Some yv ->
        fst (Dtypes.result_type [ (dtype_of m, false); (dtype_of yv, false) ])
  in
  let dtin = to_inexact_dtype dt_common in
  let m = convert m dtin in
  let y = Option.map (fun v -> convert v dtin) y in
  let x0 = atleast_2d m in
  let x0 = if (not rowvar) && ndim m <> 1 then matrix_transpose x0 else x0 in
  let x =
    match y with
    | None -> x0
    | Some yv ->
        let y2 = atleast_2d yv in
        let y2 =
          if (not rowvar) && (shape y2).(0) <> 1 then matrix_transpose y2
          else y2
        in
        concatenate ~axis:0 [ x0; y2 ]
  in
  let x = match dtype with Some d -> convert x d | None -> x in
  let dt = dtype_of x in
  let nvars = (shape x).(0) and nobs = (shape x).(1) in
  let ddof = match ddof with Some d -> d | None -> if bias then 0 else 1 in
  let check_weights name w =
    if ndim w > 1 then
      failwith (Printf.sprintf "cannot handle multidimensional %s" name);
    if (shape w).(0) <> nobs then
      failwith (Printf.sprintf "incompatible numbers of samples and %s" name)
  in
  Option.iter (check_weights "fweights") fweights;
  Option.iter (check_weights "aweights") aweights;
  (match fweights with
  | Some f when not (is_integral (dtype_of f)) ->
      raise (Errors.jax_type_error "fweights must be integer.")
  | _ -> ());
  let magnitude w =
    match dtype_of w with
    | D.Bool | D.Uint8 | D.Uint16 | D.Uint32 | D.Uint64 -> w
    | _ -> bind1 T.Abs [ w ]
  in
  let f_abs = Option.map magnitude fweights in
  let a_abs = Option.map magnitude aweights in
  let as_dt w = convert w dt in
  let w =
    match (f_abs, a_abs) with
    | None, None -> None
    | Some f, None -> Some (as_dt f)
    | None, Some a -> Some (as_dt a)
    | Some f, Some a -> Some (bind1 T.Mul [ as_dt f; as_dt a ])
  in
  let sum_all v = bind1 (T.Reduce_sum (Array.init (ndim v) Fun.id)) [ v ] in
  let avg, w_sum =
    match w with
    | None ->
        ( Reductions.mean ~axis:[| 1 |] x,
          bind1
            (T.Broadcast_in_dim { shape = [| nvars |]; dims = [||] })
            [ scalar dt (float_of_int nobs) ] )
    | Some wv ->
        let lifted =
          bind1
            (T.Broadcast_in_dim { shape = [| 1; nobs |]; dims = [| 1 |] })
            [ wv ]
        in
        let total = bind1 (T.Reduce_sum [| 1 |]) [ lifted ] in
        let product = bind1 T.Mul [ x; lifted ] in
        let summed = bind1 (T.Reduce_sum [| 1 |]) [ product ] in
        let mean = bind1 T.Div [ summed; total ] in
        (mean, Jnp_util.broadcast_to total (shape mean))
  in
  let first =
    bind1
      (T.Slice
         { start_indices = [| 0 |]; limit_indices = [| 1 |]; strides = None })
      [ w_sum ]
  in
  let first = bind1 (T.Squeeze [| 0 |]) [ first ] in
  let f =
    match (w, a_abs) with
    | None, _ -> scalar dt (float_of_int (nobs - ddof))
    | Some _, _ when ddof = 0 -> first
    | Some _, None -> bind1 T.Sub [ first; scalar dt (float_of_int ddof) ]
    | Some wv, Some a ->
        let scaled =
          bind1 T.Mul
            [
              scalar dt (float_of_int ddof);
              sum_all (bind1 T.Mul [ wv; as_dt a ]);
            ]
        in
        bind1 T.Sub [ first; bind1 T.Div [ scaled; first ] ]
  in
  let lifted_avg =
    bind1
      (T.Broadcast_in_dim { shape = [| nvars; 1 |]; dims = [| 0 |] })
      [ avg ]
  in
  let centered = bind1 T.Sub [ x; lifted_avg ] in
  let weighted =
    match w with
    | None -> centered
    | Some wv ->
        let lifted =
          bind1
            (T.Broadcast_in_dim { shape = [| 1; nobs |]; dims = [| 1 |] })
            [ wv ]
        in
        bind1 T.Mul [ centered; lifted ]
  in
  let x_t = bind1 (T.Transpose [| 1; 0 |]) [ weighted ] in
  let x_t =
    match dt with
    | D.Complex64 | D.Complex128 -> bind1 T.Conj [ x_t ]
    | _ -> x_t
  in
  let prod =
    bind1
      (T.Dot_general
         {
           lhs_contract = [| 1 |];
           rhs_contract = [| 0 |];
           lhs_batch = [||];
           rhs_batch = [||];
           dot_precision = None;
         })
      [ centered; x_t ]
  in
  squeeze (bind1 T.Div [ prod; f ])

let cov ?y ?rowvar ?bias ?ddof ?fweights ?aweights ?dtype p0 =
  let y_given = y <> None in
  let fweights_given = fweights <> None in
  let aweights_given = aweights <> None in
  if y_given || fweights_given then
    cov ?y ?rowvar ?bias ?ddof ?fweights ?aweights ?dtype p0
  else
    Jnp_util.jit1 "cov"
      ~key:
        (Digest.to_hex
           (Digest.string
              (Marshal.to_string (rowvar, bias, ddof, dtype, aweights_given) [])))
      (function
        | [ p0 ] -> cov ?y ?rowvar ?bias ?ddof ?dtype p0
        | [ p0; aweights ] -> cov ?y ?rowvar ?bias ?ddof ~aweights ?dtype p0
        | _ -> invalid_arg "cov: arity")
      (p0 :: Option.to_list aweights)

let corrcoef ?y ?(rowvar = true) ?dtype x =
  let c = cov ?y ~rowvar ?dtype x in
  if ndim c = 0 then bind1 T.Div [ c; c ]
  else begin
    let complex =
      match dtype_of c with D.Complex64 | D.Complex128 -> true | _ -> false
    in
    let d = diag c in
    let d = if complex then bind1 T.Real [ d ] else d in
    let stddev = convert (bind1 T.Sqrt [ d ]) (dtype_of c) in
    let n = (shape stddev).(0) in
    let columns =
      bind1
        (T.Broadcast_in_dim { shape = [| n; 1 |]; dims = [| 0 |] })
        [ stddev ]
    in
    let c = bind1 T.Div [ c; columns ] in
    let rows =
      bind1
        (T.Broadcast_in_dim { shape = [| 1; n |]; dims = [| 1 |] })
        [ stddev ]
    in
    let c = bind1 T.Div [ c; rows ] in
    let limit x =
      T.Concrete (Ndarray.of_floats (Dtypes.default_int_dtype ()) [||] [| x |])
    in
    let bound v = clip_values ~min:(limit (-1.0)) ~max:(limit 1.0) v in
    if complex then begin
      let re = bound (bind1 T.Real [ c ]) in
      let im = bound (bind1 T.Imag [ c ]) in
      bind1 T.Complex [ re; im ]
    end
    else bound (if complex then bind1 T.Real [ c ] else c)
  end

let corrcoef ?y ?rowvar ?dtype p0 =
  let y_given = y <> None in
  if y_given then corrcoef ?y ?rowvar ?dtype p0
  else
    Jnp_util.jit1 "corrcoef"
      ~key:
        (Digest.to_hex (Digest.string (Marshal.to_string (rowvar, dtype) [])))
      (function
        | [ p0 ] -> corrcoef ?y ?rowvar ?dtype p0
        | _ -> invalid_arg "corrcoef: arity")
      [ p0 ]

let argsort_values v =
  let n = (shape v).(0) in
  let dt = dtype v in
  let iota = bind1 (T.Iota { dtype = dt; shape = [| n |]; dimension = 0 }) [] in
  match
    C.bind
      (T.Sort { dimension = 0; is_stable = true; num_keys = 1 })
      [ v; iota ]
  with
  | [ _; order ] -> order
  | _ -> failwith "argsort_values: arity"

let from_dlpack ?device ?copy x = Dlpack.from_dlpack ?device ?copy x

let fmin a0 a1 =
  Eager_jit.one "numpy.lax_numpy.fmin" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with [], [ a0; a1 ] -> fmin a0 a1 | _ -> assert false)

let fmax a0 a1 =
  Eager_jit.one "numpy.lax_numpy.fmax" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with [], [ a0; a1 ] -> fmax a0 a1 | _ -> assert false)

let trunc a0 =
  Eager_jit.one "numpy.lax_numpy.trunc" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> trunc a0 | _ -> assert false)

let convolve ?mode a0 a1 =
  Eager_jit.one "numpy.lax_numpy.convolve" mode [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> convolve ?mode a0 a1
      | _ -> assert false)

let correlate ?mode a0 a1 =
  Eager_jit.one "numpy.lax_numpy.correlate" mode [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> correlate ?mode a0 a1
      | _ -> assert false)

let rot90 ?k ?axes a0 =
  Eager_jit.one "numpy.lax_numpy.rot90" (k, axes) [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> rot90 ?k ?axes a0
      | _ -> assert false)

let iscomplex a0 =
  Eager_jit.one "numpy.lax_numpy.iscomplex" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> iscomplex a0 | _ -> assert false)

let isreal a0 =
  Eager_jit.one "numpy.lax_numpy.isreal" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> isreal a0 | _ -> assert false)

let angle ?deg a0 =
  Eager_jit.one "numpy.lax_numpy.angle" deg [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> angle ?deg a0 | _ -> assert false)

let diff ?n ?axis ?prepend ?append a0 =
  Eager_jit.one "numpy.lax_numpy.diff" (n, axis) [ prepend; append ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ prepend; append ], [ a0 ] -> diff ?n ?axis ?prepend ?append a0
      | _ -> assert false)

let ediff1d ?to_end ?to_begin a0 =
  Eager_jit.one "numpy.lax_numpy.ediff1d" () [ to_end; to_begin ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ to_end; to_begin ], [ a0 ] -> ediff1d ?to_end ?to_begin a0
      | _ -> assert false)

let ravel ?order a0 =
  Eager_jit.one "numpy.lax_numpy.ravel" order [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> ravel ?order a0
      | _ -> assert false)

let resize a0 a1 =
  Eager_jit.one "numpy.lax_numpy.resize" a1 [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> resize a0 a1 | _ -> assert false)

let swapaxes a0 a1 a2 =
  Eager_jit.one "numpy.lax_numpy.swapaxes" (a0, a1) [] [ a2 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a2 ] -> swapaxes a0 a1 a2
      | _ -> assert false)

let isclose ?rtol ?atol ?equal_nan a0 a1 =
  Eager_jit.one "numpy.lax_numpy.isclose" (rtol, atol, equal_nan) [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> isclose ?rtol ?atol ?equal_nan a0 a1
      | _ -> assert false)

let clip ?min ?max a0 =
  Eager_jit.one "numpy.lax_numpy.clip" (min, max) [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> clip ?min ?max a0
      | _ -> assert false)

let round ?decimals a0 =
  Eager_jit.one "numpy.lax_numpy.round" decimals [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> round ?decimals a0
      | _ -> assert false)

let around ?decimals a0 =
  Eager_jit.one "numpy.lax_numpy.around" decimals [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> around ?decimals a0
      | _ -> assert false)

let nan_to_num ?nan ?posinf ?neginf a0 =
  Eager_jit.one "numpy.lax_numpy.nan_to_num" (nan, posinf, neginf) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> nan_to_num ?nan ?posinf ?neginf a0
      | _ -> assert false)

let allclose ?rtol ?atol ?equal_nan a0 a1 =
  Eager_jit.one "numpy.lax_numpy.allclose" (rtol, atol, equal_nan) [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> allclose ?rtol ?atol ?equal_nan a0 a1
      | _ -> assert false)

let unwrap ?discont ?axis ?period a0 =
  Eager_jit.one "numpy.lax_numpy.unwrap" (discont, axis, period) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> unwrap ?discont ?axis ?period a0
      | _ -> assert false)

let unstack ?axis a0 =
  Eager_jit.many "numpy.lax_numpy.unstack" axis [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> unstack ?axis a0
      | _ -> assert false)

let atleast_1d a0 =
  Eager_jit.one "numpy.lax_numpy.atleast_1d" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> atleast_1d a0 | _ -> assert false)

let atleast_2d a0 =
  Eager_jit.one "numpy.lax_numpy.atleast_2d" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> atleast_2d a0 | _ -> assert false)

let atleast_3d a0 =
  Eager_jit.one "numpy.lax_numpy.atleast_3d" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> atleast_3d a0 | _ -> assert false)

let i0 a0 =
  Eager_jit.one "numpy.lax_numpy.i0" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> i0 a0 | _ -> assert false)

let trapezoid ?x ?dx ?axis a0 =
  Eager_jit.one "numpy.lax_numpy.trapezoid" (dx, axis) [ x ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ x ], [ a0 ] -> trapezoid ?x ?dx ?axis a0
      | _ -> assert false)

let tril ?k a0 =
  Eager_jit.one "numpy.lax_numpy.tril" k [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> tril ?k a0 | _ -> assert false)

let triu ?k a0 =
  Eager_jit.one "numpy.lax_numpy.triu" k [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> triu ?k a0 | _ -> assert false)

let diagonal ?offset ?axis1 ?axis2 a0 =
  Eager_jit.one "numpy.lax_numpy.diagonal" (offset, axis1, axis2) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> diagonal ?offset ?axis1 ?axis2 a0
      | _ -> assert false)

let append ?axis a0 a1 =
  Eager_jit.one "numpy.lax_numpy.append" axis [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> append ?axis a0 a1
      | _ -> assert false)

let cross ?axisa ?axisb ?axisc ?axis a0 a1 =
  Eager_jit.one "numpy.lax_numpy.cross" (axisa, axisb, axisc, axis)
    [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> cross ?axisa ?axisb ?axisc ?axis a0 a1
      | _ -> assert false)

let kron a0 a1 =
  Eager_jit.one "numpy.lax_numpy.kron" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with [], [ a0; a1 ] -> kron a0 a1 | _ -> assert false)

let vander ?n ?increasing a0 =
  Eager_jit.one "numpy.lax_numpy.vander" (n, increasing) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> vander ?n ?increasing a0
      | _ -> assert false)

let rollaxis ?start a0 a1 =
  Eager_jit.one "numpy.lax_numpy.rollaxis" (start, a0) [] [ a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a1 ] -> rollaxis ?start a0 a1
      | _ -> assert false)

let gcd a0 a1 =
  Eager_jit.one "numpy.lax_numpy.gcd" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with [], [ a0; a1 ] -> gcd a0 a1 | _ -> assert false)

let lcm a0 a1 =
  Eager_jit.one "numpy.lax_numpy.lcm" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with [], [ a0; a1 ] -> lcm a0 a1 | _ -> assert false)

let cov ?y ?rowvar ?bias ?ddof ?fweights ?aweights ?dtype a0 =
  Eager_jit.one "numpy.lax_numpy.cov" (rowvar, bias, ddof, dtype)
    [ y; fweights; aweights ] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ y; fweights; aweights ], [ a0 ] ->
          cov ?y ?rowvar ?bias ?ddof ?fweights ?aweights ?dtype a0
      | _ -> assert false)

let corrcoef ?y ?rowvar ?dtype a0 =
  Eager_jit.one "numpy.lax_numpy.corrcoef" (rowvar, dtype) [ y ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ y ], [ a0 ] -> corrcoef ?y ?rowvar ?dtype a0
      | _ -> assert false)

let searchsorted ?side ?sorter ?method_ a0 a1 =
  Eager_jit.one "numpy.lax_numpy.searchsorted" (side, method_) [ sorter ]
    [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [ sorter ], [ a0; a1 ] -> searchsorted ?side ?sorter ?method_ a0 a1
      | _ -> assert false)

let flip ?axis a0 =
  Eager_jit.one "numpy.lax_numpy.flip" axis [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> flip ?axis a0 | _ -> assert false)

let squeeze ?axis a0 =
  Eager_jit.one "numpy.lax_numpy.squeeze" axis [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> squeeze ?axis a0
      | _ -> assert false)

let moveaxis a0 a1 a2 =
  Eager_jit.one "numpy.lax_numpy.moveaxis" (a0, a1) [] [ a2 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a2 ] -> moveaxis a0 a1 a2
      | _ -> assert false)

let argmax ?axis ?keepdims a0 =
  Eager_jit.one "numpy.lax_numpy.argmax" (axis, keepdims) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> argmax ?axis ?keepdims a0
      | _ -> assert false)

let argmin ?axis ?keepdims a0 =
  Eager_jit.one "numpy.lax_numpy.argmin" (axis, keepdims) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> argmin ?axis ?keepdims a0
      | _ -> assert false)

let nanargmax ?axis ?keepdims a0 =
  Eager_jit.one ~jit:"_nanargmax" "numpy.lax_numpy.nanargmax" (axis, keepdims)
    [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> nanargmax ?axis ?keepdims a0
      | _ -> assert false)

let nanargmin ?axis ?keepdims a0 =
  Eager_jit.one ~jit:"_nanargmin" "numpy.lax_numpy.nanargmin" (axis, keepdims)
    [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> nanargmin ?axis ?keepdims a0
      | _ -> assert false)

let roll ?axis a0 a1 =
  Eager_jit.one "numpy.lax_numpy.roll" (axis, a1) [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> roll ?axis a0 a1
      | _ -> assert false)
