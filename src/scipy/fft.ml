module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Numpy.Lax_numpy
module NF = Numpy.Fft
module UF = Numpy.Ufuncs

let pi = 4.0 *. atan 1.0
let aval v = C.get_aval v
let shape v = (aval v).T.shape
let dtype v = (aval v).T.dtype
let ndim v = Array.length (shape v)
let is_complex = function D.Complex64 | D.Complex128 -> true | _ -> false

let complexify = function
  | D.F64 | D.Complex128 -> D.Complex128
  | _ -> D.Complex64

let to_inexact = function
  | (D.F32 | D.F64 | D.Complex64 | D.Complex128) as d -> d
  | D.I64 -> D.F64
  | _ -> D.F32

let scalar dt x = T.Concrete (Nd.of_floats dt [||] [| x |])

let canonicalize_axis axis nd =
  let axis = if axis < 0 then axis + nd else axis in
  if axis < 0 || axis >= nd then
    invalid_arg
      (Printf.sprintf "axis %d out of bounds for array of ndim %d" axis nd);
  axis

let canonicalize_axes axes nd =
  let seen = Hashtbl.create (Array.length axes) in
  Array.map
    (fun axis ->
      let axis = canonicalize_axis axis nd in
      if Hashtbl.mem seen axis then invalid_arg "axes must be unique";
      Hashtbl.add seen axis ();
      axis)
    axes

let slice_axis x axis start limit stride =
  let sh = shape x in
  let nd = Array.length sh in
  let starts = Array.make nd 0 in
  let limits = Array.copy sh in
  let strides = Array.make nd 1 in
  starts.(axis) <- start;
  limits.(axis) <- limit;
  strides.(axis) <- stride;
  C.bind1
    (T.Slice
       {
         start_indices = starts;
         limit_indices = limits;
         strides = Some strides;
       })
    [ x ]

let fit_axis x axis n =
  if n < 1 then invalid_arg "transform length must be positive";
  let current = (shape x).(axis) in
  if n = current then x
  else if n < current then slice_axis x axis 0 n 1
  else
    let pads = Array.make (ndim x) (0, 0) in
    pads.(axis) <- (0, n - current);
    NL.pad x pads 0.0

let dct_interleave x axis =
  let n = (shape x).(axis) in
  let v0 = slice_axis x axis 0 n 2 in
  let v1 = slice_axis x axis 1 n 2 |> NL.flip ~axis:[| axis |] in
  NL.concatenate ~axis [ v0; v1 ]

let dct_deinterleave x axis =
  let sh = shape x in
  let n = sh.(axis) in
  let n0 = (n + 1) / 2 in
  let v0 = slice_axis x axis 0 n0 1 in
  let v1 = slice_axis x axis n0 n 1 |> NL.flip ~axis:[| axis |] in
  let v1 =
    if n land 1 = 0 then v1
    else
      let pads = Array.make (ndim x) (0, 0) in
      pads.(axis) <- (0, 1);
      NL.pad v1 pads 0.0
  in
  let stacked = NL.stack ~axis:(axis + 1) [ v0; v1 ] in
  let long_shape = Array.copy sh in
  long_shape.(axis) <- 2 * n0;
  let out = NL.reshape stacked long_shape in
  if n land 1 = 0 then out else slice_axis out axis 0 n 1

let w4 like axis sign =
  let sh = shape like in
  let n = sh.(axis) in
  let wshape = Array.make (Array.length sh) 1 in
  wshape.(axis) <- n;
  let cdt = complexify (dtype like) in
  let values =
    Array.init n (fun k ->
        let angle = sign *. 0.5 *. pi *. float_of_int k /. float_of_int n in
        { Complex.re = cos angle; im = sin angle })
  in
  T.Concrete (Nd.of_complex cdt wshape values)

let dct_ortho_norm out axis =
  let n = (shape out).(axis) in
  let fshape = Array.make (ndim out) 1 in
  fshape.(axis) <- n;
  let dt = dtype out in
  let factor =
    T.Concrete
      (Nd.of_floats dt fshape
         (Array.init n (fun i -> if i = 0 then 4.0 else 2.0)))
  in
  UF.divide out (UF.sqrt (UF.multiply factor (scalar dt (float_of_int n))))

let check_type_norm name type_ norm =
  if type_ <> 2 then failwith "Only DCT type 2 is implemented.";
  match norm with
  | None | Some "backward" | Some "ortho" -> ()
  | Some other ->
      invalid_arg
        (Printf.sprintf "jax.scipy.fft.%s: norm=%S is not implemented" name
           other)

let rec dct ?(type_ = 2) ?n ?(axis = -1) ?norm x =
  check_type_norm "dct" type_ norm;
  if is_complex (dtype x) then
    C.bind1 T.Complex
      [
        dct ~type_ ?n ~axis ?norm (UF.real x);
        dct ~type_ ?n ~axis ?norm (UF.imag x);
      ]
  else
    let axis = canonicalize_axis axis (ndim x) in
    let x = match n with None -> x | Some n -> fit_axis x axis n in
    let v = dct_interleave x axis in
    let transformed = NF.fft ~axis v in
    let out = UF.multiply transformed (w4 transformed axis (-1.0)) |> UF.real in
    let out = UF.multiply out (scalar (dtype out) 2.0) in
    match norm with Some "ortho" -> dct_ortho_norm out axis | _ -> out

let dct2 x axes norm =
  let axis1 = axes.(0) and axis2 = axes.(1) in
  let v = dct_interleave (dct_interleave x axis1) axis2 in
  let transformed = NF.fftn ~axes v in
  let w1 = w4 transformed axis1 (-1.0) in
  let w2 = w4 transformed axis2 (-1.0) in
  let w2_inv = w4 transformed axis2 1.0 in
  let reflected =
    NL.roll ~axis:[| axis2 |] (NL.flip ~axis:[| axis2 |] transformed) [| 1 |]
  in
  let inner =
    UF.add (UF.multiply w2 transformed) (UF.multiply w2_inv reflected)
  in
  let out = UF.multiply w1 inner |> UF.real in
  let out = UF.multiply out (scalar (dtype out) 2.0) in
  match norm with
  | Some "ortho" -> dct_ortho_norm (dct_ortho_norm out axis1) axis2
  | _ -> out

let resolve_axes x s axes =
  let nd = ndim x in
  match axes with
  | Some axes -> canonicalize_axes axes nd
  | None -> (
      match s with
      | None -> Array.init nd Fun.id
      | Some s -> Array.init (Array.length s) (fun i -> nd - Array.length s + i)
      )

let fit_axes x axes s =
  match s with
  | None -> x
  | Some s ->
      if Array.length s > ndim x then
        invalid_arg
          (Printf.sprintf "s must have at most x.ndim (%d) elements, got %d"
             (ndim x) (Array.length s));
      if Array.length s <> Array.length axes then
        invalid_arg "s and axes must have the same length";
      let value = ref x in
      Array.iteri (fun i axis -> value := fit_axis !value axis s.(i)) axes;
      !value

let rec dctn ?(type_ = 2) ?s ?axes ?norm x =
  check_type_norm "dctn" type_ norm;
  if is_complex (dtype x) then
    C.bind1 T.Complex
      [
        dctn ~type_ ?s ?axes ?norm (UF.real x);
        dctn ~type_ ?s ?axes ?norm (UF.imag x);
      ]
  else
    let axes = resolve_axes x s axes in
    let x = fit_axes x axes s in
    match Array.length axes with
    | 0 -> x
    | 1 -> dct ~type_ ~axis:axes.(0) ?norm x
    | 2 -> dct2 x axes norm
    | n ->
        let rec loop i value =
          if i >= n then value
          else
            let count = min 2 (n - i) in
            let block = Array.sub axes i count in
            loop (i + count) (dctn ~type_ ~axes:block ?norm value)
        in
        loop 0 x

let rec idct ?(type_ = 2) ?n ?(axis = -1) ?norm x =
  check_type_norm "idct" type_ norm;
  if is_complex (dtype x) then
    C.bind1 T.Complex
      [
        idct ~type_ ?n ~axis ?norm (UF.real x);
        idct ~type_ ?n ~axis ?norm (UF.imag x);
      ]
  else
    let axis = canonicalize_axis axis (ndim x) in
    let x = match n with None -> x | Some n -> fit_axis x axis n in
    let n = (shape x).(axis) in
    let x = NL.astype x (to_inexact (dtype x)) in
    let x = match norm with Some "ortho" -> x | _ -> dct_ortho_norm x axis in
    let x = dct_ortho_norm x axis in
    let cdt = complexify (dtype x) in
    let xc = NL.astype x cdt in
    let xc = UF.divide xc (w4 xc axis (-1.0)) in
    let xc = UF.multiply xc (scalar (dtype x) (2.0 *. float_of_int n)) in
    let transformed = NF.ifft ~axis xc |> UF.real in
    dct_deinterleave transformed axis

let rec idctn ?(type_ = 2) ?s ?axes ?norm x =
  check_type_norm "idctn" type_ norm;
  if is_complex (dtype x) then
    C.bind1 T.Complex
      [
        idctn ~type_ ?s ?axes ?norm (UF.real x);
        idctn ~type_ ?s ?axes ?norm (UF.imag x);
      ]
  else
    let axes = resolve_axes x s axes in
    let x = fit_axes x axes s in
    Array.fold_left (fun value axis -> idct ~type_ ~axis ?norm value) x axes
