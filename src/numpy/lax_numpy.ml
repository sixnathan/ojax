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

let bind = C.bind
let scalar dt x = const_full dt [||] x
let full dt sh x = const_full dt sh x

let maximum a b =
  let a, b, _, _ = promote2 a b in
  bind1 T.Max [ a; b ]

let minimum a b =
  let a, b, _, _ = promote2 a b in
  bind1 T.Min [ a; b ]

let where_ c x y =
  let dt, _ =
    Dtypes.result_type
      [
        ((get_aval x).T.dtype, (get_aval x).T.weak_type);
        ((get_aval y).T.dtype, (get_aval y).T.weak_type);
      ]
  in
  let sh = broadcast_shapes (broadcast_shapes (shape c) (shape x)) (shape y) in
  let cb = broadcast_to c sh in
  let xb = broadcast_to (convert x dt) sh in
  let yb = broadcast_to (convert y dt) sh in
  bind1 T.Select_n [ cb; yb; xb ]

let finfo_max = function
  | D.F64 -> Float.max_float
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
  let rt = full dt sh rtol and at = full dt sh atol in
  let in_range =
    bind1 T.Le
      [
        bind1 T.Abs [ bind1 T.Sub [ a; b ] ];
        bind1 T.Add [ at; bind1 T.Mul [ rt; bind1 T.Abs [ b ] ] ];
      ]
  in
  let out =
    bind1 T.Or
      [ bind1 T.Eq [ a; b ]; bind1 T.And [ bind1 T.Is_finite [ b ]; in_range ] ]
  in
  if equal_nan then bind1 T.Or [ out; bind1 T.And [ isnan a; isnan b ] ]
  else out

let allclose ?rtol ?atol ?equal_nan a b =
  let c = isclose ?rtol ?atol ?equal_nan a b in
  let n = ndim c in
  bind1 (T.Reduce_and (Array.init n (fun i -> i))) [ c ]

let clip ?min ?max arr =
  let dt = dtype arr in
  let arr =
    match min with None -> arr | Some m -> maximum (scalar dt m) arr
  in
  match max with None -> arr | Some m -> minimum (scalar dt m) arr

let round ?(decimals = 0) v =
  let dt = dtype v in
  if is_integral dt || is_bool dt then v
  else if decimals = 0 then bind1 T.Round [ v ]
  else begin
    let factor = full dt (shape v) (10.0 ** float_of_int decimals) in
    bind1 T.Div [ bind1 T.Round [ bind1 T.Mul [ v; factor ] ]; factor ]
  end

let around ?decimals v = round ?decimals v

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
  bind1 (T.Squeeze dims) [ v ]

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

let unravel_index indices dims =
  let n = Array.length dims in
  let dt = dtype indices in
  let ish = shape indices in
  let out = Array.make n indices in
  let cur = ref indices in
  for i = n - 1 downto 0 do
    let s = full dt ish (float_of_int dims.(i)) in
    out.(i) <- bind1 T.Rem [ !cur; s ];
    cur := bind1 T.Div [ !cur; s ]
  done;
  let zero = full dt ish 0.0 in
  let negone = full dt ish (-1.0) in
  let oob_pos = bind1 T.Gt [ !cur; zero ] in
  let oob_neg = bind1 T.Lt [ !cur; negone ] in
  List.init n (fun i ->
      let smax = full dt ish (float_of_int (dims.(i) - 1)) in
      where_ oob_pos smax (where_ oob_neg zero out.(i)))

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

let unwrap ?discont ?(axis = -1) ?(period = 2.0 *. Float.pi) p =
  let nd = ndim p in
  let axis = canonicalize_axis axis nd in
  let dt = to_inexact_dtype (dtype p) in
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
  bind (T.Split { sizes; axis }) [ v ]

let split ?axis v s = do_split ~array_split:false ?axis v s
let array_split ?axis v s = do_split ~array_split:true ?axis v s
let vsplit v s = do_split ~array_split:false ~axis:0 v s

let hsplit v s =
  do_split ~array_split:false ~axis:(if ndim v = 1 then 0 else 1) v s

let dsplit v s = do_split ~array_split:false ~axis:2 v s

let select condlist choicelist =
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
  let dt = result_type choicelist in
  let default_v = scalar dt 0.0 in
  let cases = List.map (fun v -> convert v dt) (default_v :: choicelist) in
  bind1 T.Select_n (broadcast_arrays (idx :: cases))

let all_reduce v =
  let n = ndim v in
  if n = 0 then v else bind1 (T.Reduce_and (Array.init n Fun.id)) [ v ]

let promote_values vs =
  let dt = result_type vs in
  List.map (fun v -> convert v dt) vs

let astype v dt = convert v dt
let copy v = bind1 T.Copy [ v ]
let atleast_1d v = if ndim v >= 1 then v else expand_dims v [| 0 |]

let atleast_2d v =
  match ndim v with
  | 0 -> expand_dims v [| 0; 1 |]
  | 1 -> expand_dims v [| 0 |]
  | _ -> v

let atleast_3d v =
  match ndim v with
  | 0 -> expand_dims v [| 0; 1; 2 |]
  | 1 -> expand_dims v [| 0; 2 |]
  | 2 -> expand_dims v [| 2 |]
  | _ -> v

let concatenate ?(axis = 0) arrays =
  match arrays with
  | [] -> invalid_arg "concatenate: need at least one array"
  | first :: _ ->
      let ax = canonicalize_axis axis (ndim first) in
      bind1 (T.Concatenate ax) (promote_values arrays)

let concat ?(axis = 0) arrays = concatenate ~axis arrays

let stack ?(axis = 0) arrays =
  match arrays with
  | [] -> invalid_arg "stack: need at least one array"
  | first :: _ ->
      let ax = canonicalize_axis axis (ndim first + 1) in
      bind1 (T.Stack ax) (promote_values arrays)

let unstack ?(axis = 0) v =
  bind (T.Unstack (canonicalize_axis axis (ndim v))) [ v ]

let vstack arrays = concatenate ~axis:0 (List.map atleast_2d arrays)

let hstack arrays =
  let arrs = List.map atleast_1d arrays in
  let ax = if ndim (List.hd arrs) = 1 then 0 else 1 in
  concatenate ~axis:ax arrs

let dstack arrays = concatenate ~axis:2 (List.map atleast_3d arrays)

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

let pad v pad_width cval =
  let cfg = Array.map (fun (lo, hi) -> (lo, hi, 0)) pad_width in
  bind1 (T.Pad cfg) [ v; scalar (dtype v) cval ]

let i0 v =
  let v = convert v (to_inexact_dtype (dtype v)) in
  let ax = bind1 T.Abs [ v ] in
  bind1 T.Mul [ bind1 T.Exp [ ax ]; bind1 T.Bessel_i0e [ ax ] ]

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

let arange ?(start = 0.0) ?(step = 1.0) ~dtype stop =
  let size = max 0 (int_of_float (ceil ((stop -. start) /. step))) in
  let idx = bind1 (T.Iota { dtype; shape = [| size |]; dimension = 0 }) [] in
  if start = 0.0 && step = 1.0 then idx
  else
    bind1 T.Add
      [
        full dtype [| size |] start;
        bind1 T.Mul [ full dtype [| size |] step; idx ];
      ]

let eye ?m ?(k = 0) ~dtype n =
  let m = Option.value m ~default:n in
  let i =
    bind1 (T.Iota { dtype = D.I32; shape = [| n; m |]; dimension = 0 }) []
  in
  let j =
    bind1 (T.Iota { dtype = D.I32; shape = [| n; m |]; dimension = 1 }) []
  in
  let ik =
    if k = 0 then i
    else bind1 T.Add [ i; full D.I32 [| n; m |] (float_of_int k) ]
  in
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
  let x, y, _, _ = promote2 x y in
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

let argmax ?axis ?(keepdims = false) a =
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
    let c = stack ~axis:0 [ c0; c1; c2 ] in
    moveaxis [| 0 |] [| axisc |] c
  end

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
  let i =
    bind1 (T.Iota { dtype = D.I32; shape = [| n; m |]; dimension = 0 }) []
  in
  let j =
    bind1 (T.Iota { dtype = D.I32; shape = [| n; m |]; dimension = 1 }) []
  in
  let ik =
    if k = 0 then i
    else bind1 T.Add [ i; full D.I32 [| n; m |] (float_of_int k) ]
  in
  convert (bind1 T.Ge [ ik; j ]) dtype

let tril ?(k = 0) m =
  let s = shape m in
  let nd = Array.length s in
  if nd < 2 then invalid_arg "Argument to jax.numpy.tril must be at least 2D";
  let mask = tri ~m:s.(nd - 1) ~k ~dtype:D.Bool s.(nd - 2) in
  where_ mask m (zeros_like m)

let triu ?(k = 0) m =
  let s = shape m in
  let nd = Array.length s in
  if nd < 2 then invalid_arg "Argument to jax.numpy.triu must be at least 2D";
  let mask = tri ~m:s.(nd - 1) ~k:(k - 1) ~dtype:D.Bool s.(nd - 2) in
  where_ mask (zeros_like m) m

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
  let acc =
    let dt_in = (get_aval masked).T.dtype in
    if is_inexact dt_in then masked
    else
      convert masked
        (fst
           (Dtypes.result_type
              [ (dt_in, false); (Dtypes.default_int_dtype (), false) ]))
  in
  let res = bind1 (T.Reduce_sum [| nd - 2; nd - 1 |]) [ acc ] in
  match dtype with Some d -> convert res d | None -> res

let diagonal ?(offset = 0) ?(axis1 = 0) ?(axis2 = 1) a =
  let s = shape a in
  let nd = Array.length s in
  if nd < 2 then
    invalid_arg "diagonal requires an array of at least two dimensions.";
  let a1 = canonicalize_axis axis1 nd and a2 = canonicalize_axis axis2 nd in
  let na = s.(a1) and ma = s.(a2) in
  let am = moveaxis [| axis1; axis2 |] [| nd - 2; nd - 1 |] a in
  let diag_size = max 0 (min (na + min offset 0) (ma - max offset 0)) in
  let mask = eye ~m:ma ~k:offset ~dtype:D.Bool na in
  let masked = where_ mask am (zeros_like am) in
  if offset >= 0 then
    let reduced = reduce_over masked (nd - 2) in
    slice_axis reduced (nd - 2) offset (offset + diag_size)
  else
    let reduced = reduce_over masked (nd - 1) in
    slice_axis reduced (nd - 2) (-offset) (-offset + diag_size)

let diag ?(k = 0) v =
  let s = shape v in
  match Array.length s with
  | 1 ->
      let n = s.(0) + abs k in
      let padded = pad v [| (max 0 (-k), max 0 k) |] 0.0 in
      let mask = eye ~k ~dtype:D.Bool n in
      let col =
        bind1
          (T.Broadcast_in_dim { shape = [| n; 1 |]; dims = [| 0 |] })
          [ padded ]
      in
      where_ mask col (const_full (dtype v) [| n; n |] 0.0)
  | 2 -> diagonal ~offset:k v
  | _ -> invalid_arg "diag input must be 1d or 2d"

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

let vander ?n ?(increasing = false) x =
  if ndim x <> 1 then invalid_arg "x must be a one-dimensional array";
  let len = (shape x).(0) in
  let nn = Option.value n ~default:len in
  if nn < 0 then invalid_arg "N must be nonnegative";
  let dt = dtype x in
  let io = bind1 (T.Iota { dtype = dt; shape = [| nn |]; dimension = 0 }) [] in
  let io =
    if increasing then io
    else bind1 T.Sub [ full dt [| nn |] (float_of_int (nn - 1)); io ]
  in
  let xcol =
    bind1 (T.Broadcast_in_dim { shape = [| len; nn |]; dims = [| 0 |] }) [ x ]
  in
  let expo =
    bind1 (T.Broadcast_in_dim { shape = [| len; nn |]; dims = [| 1 |] }) [ io ]
  in
  bind1 T.Pow [ xcol; expo ]

let repeat ?axis a repeats =
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
      let s = bind1 (T.Reduce_sum [| nd - 1 |]) [ y ] in
      let y0 = last_index y 0 and yl = last_index y (l - 1) in
      let ends = mul_scalar 0.5 (add2 y0 yl) in
      mul_scalar dx (sub2 s ends)
  | Some dxa ->
      let y1 = slice_axis y (nd - 1) 1 l in
      let y0 = slice_axis y (nd - 1) 0 (l - 1) in
      let prod = mul2 dxa (add2 y1 y0) in
      mul_scalar 0.5 (bind1 (T.Reduce_sum [| nd - 1 |]) [ prod ])

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
  if not (is_inexact (dtype a)) then argmax ?axis ~keepdims a
  else begin
    let nan_mask = isnan a in
    let a' = where_ nan_mask (const_full (dtype a) (shape a) neg_infinity) a in
    let res = argmax ?axis ~keepdims a' in
    let all_nan = all_over ?axis ~keepdims nan_mask in
    where_ all_nan (scalar (dtype res) (-1.0)) res
  end

let nanargmin ?axis ?(keepdims = false) a =
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
  let acc = ref a in
  for k = 0 to n - 1 do
    let ax = get axis k and s = get shift k in
    let sz = (shape !acc).(ax) in
    if sz <> 0 then begin
      let i = ((-s mod sz) + sz) mod sz in
      if i <> 0 then
        acc :=
          concatenate ~axis:ax
            [ slice_axis !acc ax i sz; slice_axis !acc ax 0 i ]
    end
  done;
  !acc

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
    | [ _; b ] -> any_reduce (bind1 T.Ne [ b; zeros_like b ])
    | _ -> assert false
  in
  let body carry =
    match carry with
    | [ a; b ] ->
        let nz = bind1 T.Ne [ b; zeros_like b ] in
        let bsafe = where_ nz b (const_full (dtype b) (shape b) 1.0) in
        let t1 = where_ nz b a in
        let t2 = where_ nz (bind1 T.Rem [ a; bsafe ]) (zeros_like b) in
        let lt = bind1 T.Lt [ t1; t2 ] in
        [ where_ lt t2 t1; where_ lt t1 t2 ]
    | _ -> assert false
  in
  let init = [ bind1 T.Abs [ x1 ]; bind1 T.Abs [ x2 ] ] in
  List.hd (Lax.while_loop cond body init)

let lcm x1 x2 =
  let x1, x2 =
    match promote_values [ x1; x2 ] with
    | [ a; b ] -> (a, b)
    | _ -> assert false
  in
  let x1 = bind1 T.Abs [ x1 ] and x2 = bind1 T.Abs [ x2 ] in
  if not (is_integral (dtype x1)) then
    invalid_arg "Arguments to jax.numpy.lcm must be integers.";
  let d = gcd x1 x2 in
  let is_zero = bind1 T.Eq [ d; zeros_like d ] in
  let dsafe = where_ is_zero (const_full (dtype d) (shape d) 1.0) d in
  let prod = mul2 x1 (binop T.Div x2 dsafe) in
  where_ is_zero (zeros_like prod) prod

let searchsorted ?(side = "left") a v =
  if ndim a <> 1 then invalid_arg "a should be 1-dimensional";
  let a, v =
    match promote_values [ a; v ] with [ a; v ] -> (a, v) | _ -> assert false
  in
  let na = (shape a).(0) in
  let vsh = shape v in
  let out_shape = Array.append [| na |] vsh in
  let a_b =
    bind1 (T.Broadcast_in_dim { shape = out_shape; dims = [| 0 |] }) [ a ]
  in
  let v_dims = Array.init (Array.length vsh) (fun i -> i + 1) in
  let v_b =
    bind1 (T.Broadcast_in_dim { shape = out_shape; dims = v_dims }) [ v ]
  in
  let cmp =
    if side = "right" then bind1 T.Le [ a_b; v_b ] else bind1 T.Lt [ a_b; v_b ]
  in
  bind1 (T.Reduce_sum [| 0 |]) [ convert cmp D.I32 ]

let digitize ?(right = false) x bins =
  if ndim bins <> 1 then
    invalid_arg "digitize: bins must be a 1-dimensional array";
  let nb = (shape bins).(0) in
  if nb = 0 then const_full D.I32 (shape x) 0.0
  else begin
    let side = if right then "left" else "right" in
    let inc = bind1 T.Ge [ scalar_at bins (nb - 1); scalar_at bins 0 ] in
    let asc = searchsorted ~side bins x in
    let desc =
      bind1 T.Sub
        [
          const_full D.I32 (shape x) (float_of_int nb);
          searchsorted ~side (flip bins) x;
        ]
    in
    where_ inc asc desc
  end

let dtype_of = dtype

let cov ?y ?(rowvar = true) ?(bias = false) ?ddof ?dtype m =
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
  let sum1 = bind1 (T.Reduce_sum [| 1 |]) [ x ] in
  let avg = bind1 T.Div [ sum1; full dt [| nvars |] (float_of_int nobs) ] in
  let xc = sub2 x (expand_dims avg [| 1 |]) in
  let prod =
    bind1
      (T.Dot_general
         {
           lhs_contract = [| 1 |];
           rhs_contract = [| 1 |];
           lhs_batch = [||];
           rhs_batch = [||];
         })
      [ xc; xc ]
  in
  let res =
    bind1 T.Div [ prod; full dt (shape prod) (float_of_int (nobs - ddof)) ]
  in
  squeeze res

let corrcoef ?y ?(rowvar = true) ?dtype x =
  let c = cov ?y ~rowvar ?dtype x in
  if ndim c = 0 then bind1 T.Div [ c; c ]
  else begin
    let d = diagonal c in
    let stddev = convert (bind1 T.Sqrt [ d ]) (dtype_of c) in
    let c = binop T.Div c (expand_dims stddev [| 1 |]) in
    let c = binop T.Div c (expand_dims stddev [| 0 |]) in
    clip ~min:(-1.0) ~max:1.0 c
  end
