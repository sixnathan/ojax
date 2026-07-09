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

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (Array.fold_left ( * ) 1 sh) x))

let zeros_like v = const_full (dtype v) (shape v) 0.0
let bind1 = C.bind1
let is_inexact = function D.F32 | D.F64 -> true | _ -> false
let is_integral = function D.I32 | D.I64 -> true | _ -> false
let is_bool = function D.Bool -> true | _ -> false

let to_inexact_dtype dt =
  if is_inexact dt then dt else Dtypes.default_float_dtype ()

let convert v dt =
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let broadcast_to v sh =
  let s = shape v in
  if s = sh then v
  else begin
    let nd_in = Array.length s and nd_out = Array.length sh in
    let dims = Array.init nd_in (fun i -> i + (nd_out - nd_in)) in
    Array.iteri
      (fun k i ->
        if s.(k) <> sh.(i) && s.(k) <> 1 then
          invalid_arg "broadcast_to: incompatible shapes")
      dims;
    bind1 (T.Broadcast_in_dim { shape = sh; dims }) [ v ]
  end

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

let flip ?axis v =
  let n = ndim v in
  match axis with
  | None -> rev_dims v (Array.init n (fun i -> i))
  | Some ax -> rev_dims v (Array.map (fun i -> canonicalize_axis i n) ax)

let fliplr v = rev_dims v [| 1 |]
let flipud v = rev_dims v [| 0 |]

let reshape v sh =
  let total = size v in
  let known =
    Array.fold_left (fun acc d -> if d = -1 then acc else acc * d) 1 sh
  in
  let resolved = Array.map (fun d -> if d = -1 then total / known else d) sh in
  bind1 (T.Reshape resolved) [ v ]

let ravel v = reshape v [| size v |]

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

let trunc v =
  let dt = dtype v in
  if is_integral dt || is_bool dt then v
  else
    let below = bind1 T.Lt [ v; zeros_like v ] in
    bind1 T.Select_n [ below; bind1 T.Floor [ v ]; bind1 T.Ceil [ v ] ]

let isnan v = bind1 T.Ne [ v; v ]

let fmin x y =
  let x, y, _, _ = promote2 x y in
  let mask = bind1 T.Or [ bind1 T.Lt [ x; y ]; isnan y ] in
  bind1 T.Select_n [ mask; y; x ]

let fmax x y =
  let x, y, _, _ = promote2 x y in
  let mask = bind1 T.Or [ bind1 T.Gt [ x; y ]; isnan y ] in
  bind1 T.Select_n [ mask; y; x ]

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

let diff ?(n = 1) ?(axis = -1) v =
  if n = 0 then v
  else begin
    if n < 0 then invalid_arg "diff: order must be non-negative";
    let nd = ndim v in
    if nd = 0 then invalid_arg "diff requires at least one dimension";
    let axis = canonicalize_axis axis nd in
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

let ediff1d v =
  let a = ravel v in
  let d = (shape a).(0) in
  bind1 T.Sub [ slice_axis a 0 1 d; slice_axis a 0 0 (d - 1) ]

let real v = v
let imag v = zeros_like v

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

let iscomplex v =
  let i = imag v in
  bind1 T.Ne [ i; zeros_like i ]

let isreal v =
  let i = imag v in
  bind1 T.Eq [ i; zeros_like i ]

let iscomplexobj _ = false
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
  | Integer
  | Floating
  | Inexact
  | Number
  | Generic
  | Cbool

let class_chain = function
  | Cdtype D.F32 | Cdtype D.F64 | Floating ->
      [ Floating; Inexact; Number; Generic ]
  | Cdtype D.I32 | Cdtype D.I64 | Signedinteger ->
      [ Signedinteger; Integer; Number; Generic ]
  | Cdtype D.Bool | Cbool -> [ Cbool; Generic ]
  | Integer -> [ Integer; Number; Generic ]
  | Inexact -> [ Inexact; Number; Generic ]
  | Number -> [ Number; Generic ]
  | Generic -> [ Generic ]

let issubdtype a b = a = b || List.mem b (class_chain a)

let conv1d ~mode ~flip_kernel ~reverse_out a v =
  let dt, _ =
    Dtypes.result_type
      [
        (to_inexact_dtype (dtype a), false); (to_inexact_dtype (dtype v), false);
      ]
  in
  let a = convert a dt and v = convert v dt in
  let la = (shape a).(0) and lv = (shape v).(0) in
  if la = 0 || lv = 0 then invalid_arg "conv: inputs cannot be empty";
  let x, y = if la < lv then (v, a) else (a, v) in
  let y = if flip_kernel then rev_dims y [| 0 |] else y in
  let ly = (shape y).(0) in
  let padding =
    match mode with
    | "valid" -> [| (0, 0) |]
    | "same" -> [| (ly / 2, ly - (ly / 2) - 1) |]
    | "full" -> [| (ly - 1, ly - 1) |]
    | _ -> invalid_arg "conv: mode must be one of full, same, valid"
  in
  let lx = (shape x).(0) in
  let lhs = reshape x [| 1; 1; lx |] in
  let rhs = reshape y [| 1; 1; ly |] in
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
  let m = (shape out).(2) in
  let flat = reshape out [| m |] in
  if reverse_out then rev_dims flat [| 0 |] else flat

let convolve ?(mode = "full") a v =
  conv1d ~mode ~flip_kernel:true ~reverse_out:false a v

let correlate ?(mode = "valid") a v =
  let la = (shape a).(0) and lv = (shape v).(0) in
  conv1d ~mode ~flip_kernel:false ~reverse_out:(la < lv) a v
