open Types
module L = Lax_api
module NF = Numpy.Fft

let b1 = Core.bind1
let shape v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype
let rank v = Array.length (shape v)
let canon n a = if a < 0 then a + n else a

let real_of = function
  | Dtype.Complex128 | Dtype.F64 -> Dtype.F64
  | _ -> Dtype.F32

let complex_of = function
  | Dtype.F64 | Dtype.Complex128 -> Dtype.Complex128
  | _ -> Dtype.Complex64

let pi = 4.0 *. Float.atan 1.0

let expanded_along ndim axis values make =
  let sh = Array.make ndim 1 in
  sh.(axis) <- Array.length values;
  make sh values

let twiddle dt ndim axis n =
  let cdt = complex_of dt in
  expanded_along ndim axis
    (Array.init n (fun k ->
         let angle = -.pi *. float_of_int k /. float_of_int (2 * n) in
         { Complex.re = cos angle; im = sin angle }))
    (fun sh vs -> Concrete (Ndarray.of_complex cdt sh vs))

let broadcast_to v sh =
  L.broadcast_in_dim v sh (Array.init (Array.length sh) Fun.id)

let ortho_norm out axis =
  let ndim = rank out and n = (shape out).(axis) in
  let factor =
    expanded_along ndim axis
      (Array.init n (fun i ->
           sqrt ((if i = 0 then 4.0 else 2.0) *. float_of_int n)))
      (fun sh vs -> Concrete (Ndarray.of_floats (dtype_of out) sh vs))
  in
  b1 Div [ out; broadcast_to factor (shape out) ]

let pad_axis x axis n =
  match n with
  | None -> x
  | Some target ->
      let cfg =
        Array.mapi
          (fun a s -> if a = axis then (0, target - s, 0) else (0, 0, 0))
          (shape x)
      in
      b1 (Pad cfg) [ x; L.full [||] 0.0 (dtype_of x) ]

let check_norm name norm =
  match norm with
  | None | Some "backward" | Some "ortho" -> ()
  | Some other ->
      invalid_arg
        (Printf.sprintf "jax.scipy.fft.%s: norm=%s is not implemented" name
           other)

let check_type t = if t <> 2 then invalid_arg "Only DCT type 2 is implemented."

let is_complex v =
  match dtype_of v with
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let inexact x =
  match dtype_of x with
  | Dtype.F32 | Dtype.F64 | Dtype.Complex64 | Dtype.Complex128 -> x
  | Dtype.I64 -> b1 (Convert_element_type Dtype.F64) [ x ]
  | _ -> b1 (Convert_element_type Dtype.F32) [ x ]

let rec dct ?(type_ = 2) ?n ?(axis = -1) ?norm x =
  check_type type_;
  check_norm "dct" norm;
  let x = inexact x in
  if is_complex x then
    b1 Complex
      [
        dct ~type_ ?n ~axis ?norm (b1 Real [ x ]);
        dct ~type_ ?n ~axis ?norm (b1 Imag [ x ]);
      ]
  else
    let axis = canon (rank x) axis in
    let x = pad_axis x axis n in
    let size = (shape x).(axis) in
    let evens = L.slice_in_dim ~stride:2 ~axis x 0 size in
    let odds = L.rev (L.slice_in_dim ~stride:2 ~axis x 1 size) [| axis |] in
    let v = L.concatenate [ evens; odds ] axis in
    let spectrum = NF.fft ~axis v in
    let w = twiddle (dtype_of x) (rank x) axis size in
    let product = b1 Mul [ spectrum; broadcast_to w (shape spectrum) ] in
    let doubled = L.mul (b1 Real [ product ]) (L.full [||] 2.0 (dtype_of x)) in
    if norm = Some "ortho" then ortho_norm doubled axis else doubled

let rec idct ?(type_ = 2) ?n ?(axis = -1) ?norm x =
  check_type type_;
  check_norm "idct" norm;
  let x = inexact x in
  if is_complex x then
    b1 Complex
      [
        idct ~type_ ?n ~axis ?norm (b1 Real [ x ]);
        idct ~type_ ?n ~axis ?norm (b1 Imag [ x ]);
      ]
  else
    let axis = canon (rank x) axis in
    let x = pad_axis x axis n in
    let size = (shape x).(axis) in
    let dt = dtype_of x in
    let x = if norm = Some "ortho" then x else ortho_norm x axis in
    let x = ortho_norm x axis in
    let w = twiddle dt (rank x) axis size in
    let as_c = b1 (Convert_element_type (complex_of dt)) [ x ] in
    let divided = b1 Div [ as_c; broadcast_to w (shape as_c) ] in
    let scaled =
      L.mul divided (L.full [||] (float_of_int (2 * size)) (dtype_of divided))
    in
    let inverse = b1 Real [ NF.ifft ~axis scaled ] in
    let half = (size + 1) / 2 in
    let v0 = L.slice_in_dim ~axis inverse 0 half in
    let v1 = L.rev (L.slice_in_dim ~axis inverse half size) [| axis |] in
    let pieces0 = L.unstack v0 axis and pieces1 = L.unstack v1 axis in
    let rec weave a b =
      match (a, b) with
      | x :: xs, y :: ys -> x :: y :: weave xs ys
      | rest, [] -> rest
      | [], rest -> rest
    in
    L.stack (weave pieces0 pieces1) axis

let resolve_axes ?s ?axes x =
  let n = rank x in
  match (axes, s) with
  | Some a, _ -> List.map (canon n) a
  | None, Some sizes ->
      let k = List.length sizes in
      List.init k (fun i -> n - k + i)
  | None, None -> List.init n Fun.id

let along_axes f ?s ?axes x =
  let ax = resolve_axes ?s ?axes x in
  let sizes =
    match s with
    | Some l -> List.map Option.some l
    | None -> List.map (fun _ -> None) ax
  in
  List.fold_left2 (fun v a n -> f ?n ~axis:a v) x ax sizes

let dctn ?(type_ = 2) ?s ?axes ?norm x =
  check_type type_;
  along_axes (fun ?n ~axis v -> dct ?n ~axis ?norm v) ?s ?axes x

let idctn ?(type_ = 2) ?s ?axes ?norm x =
  check_type type_;
  along_axes (fun ?n ~axis v -> idct ?n ~axis ?norm v) ?s ?axes x
