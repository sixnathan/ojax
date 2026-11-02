module NL = Lax_numpy
module UF = Ufuncs
module RED = Reductions
module IDX = Indexing

let all ?axis ?keepdims x = RED.all ?axis ?keepdims x
let any ?axis ?keepdims x = RED.any ?axis ?keepdims x
let sum ?axis ?keepdims x = RED.sum ?axis ?keepdims x
let prod ?axis ?keepdims x = RED.prod ?axis ?keepdims x
let max ?axis ?keepdims x = RED.max ?axis ?keepdims x
let min ?axis ?keepdims x = RED.min ?axis ?keepdims x
let mean ?axis ?keepdims x = RED.mean ?axis ?keepdims x
let var ?axis ?keepdims ?ddof x = RED.var ?axis ?keepdims ?ddof x
let std ?axis ?keepdims ?ddof x = RED.std ?axis ?keepdims ?ddof x
let ptp ?axis ?keepdims x = RED.ptp ?axis ?keepdims x
let cumsum ?axis x = RED.cumsum_by_keyword ?axis x
let cumprod ?axis x = RED.cumprod ?axis x
let argmax ?axis ?keepdims x = NL.argmax ?axis ?keepdims x
let argmin ?axis ?keepdims x = NL.argmin ?axis ?keepdims x
let astype x dtype = NL.astype x dtype
let clip ?min ?max x = NL.clip ?min ?max x
let round ?decimals x = NL.round ?decimals x
let copy x = NL.copy x
let conj x = UF.conj x
let conjugate x = UF.conjugate x
let reshape x shape = NL.reshape x shape
let ravel x = NL.ravel x
let flatten x = NL.ravel x
let transpose ?axes x = NL.transpose ?axes x
let squeeze ?axis x = NL.squeeze ?axis x
let swapaxes axis1 axis2 x = NL.swapaxes axis1 axis2 x
let repeat ?axis x repeats = NL.repeat ?axis x repeats
let diagonal ?offset ?axis1 ?axis2 x = NL.diagonal ?offset ?axis1 ?axis2 x

let trace ?offset ?axis1 ?axis2 ?dtype x =
  NL.trace ?offset ?axis1 ?axis2 ?dtype x

let searchsorted ?side x v = NL.searchsorted ?side x v
let take ?axis ?mode x indices = IDX.take ?axis ?mode x indices
let t x = NL.transpose x
let mt x = NL.matrix_transpose x
let real x = UF.real x
let imag x = UF.imag x
let neg x = UF.negative x
let pos x = UF.positive x
let abs x = UF.abs x
let invert x = UF.invert x
let eq a b = UF.equal a b
let ne a b = UF.not_equal a b
let lt a b = UF.less a b
let le a b = UF.less_equal a b
let gt a b = UF.greater a b
let ge a b = UF.greater_equal a b
let add a b = UF.add a b
let sub a b = UF.subtract a b
let mul a b = UF.multiply a b
let truediv a b = UF.true_divide a b
let floordiv a b = UF.floor_divide a b
let mod_ a b = UF.mod_ a b
let pow a b = UF.power a b
let and_ a b = UF.bitwise_and a b
let or_ a b = UF.bitwise_or a b
let xor a b = UF.bitwise_xor a b
let lshift a b = UF.left_shift a b
let rshift a b = UF.right_shift a b
let aval_of x = Core.get_aval x
let is_complex dt = dt = Dtype.Complex64 || dt = Dtype.Complex128
let complex_part = function Dtype.Complex128 -> Dtype.F64 | _ -> Dtype.F32

let imaginary_unit () =
  Types.Concrete
    (Ndarray.as_weak
       (Ndarray.of_complex Dtype.Complex64 [||]
          [| { Complex.re = 0.0; im = 1.0 } |]))

let every_other x start =
  let s = (aval_of x).shape in
  let n = Array.length s in
  let stop = Array.copy s in
  let begin_ = Array.make n 0 in
  begin_.(n - 1) <- start;
  let strides = Array.make n 1 in
  strides.(n - 1) <- 2;
  Lax_api.slice ~strides x begin_ stop

let abstract_on_keys name x =
  if Key_reuse.is_typed x then
    raise (Py_exn.Not_implemented_error ("Cannot call abstract method " ^ name))

let rec view x dtype =
  abstract_on_keys "view" x;
  let dtype = Dtypes.canonicalize_dtype dtype in
  let a = aval_of x in
  let nbits_in = Dtypes.itemsize_bits a.dtype in
  let nbits_out = Dtypes.itemsize_bits dtype in
  let shape = a.shape in
  let n = Array.length shape in
  if n = 0 then begin
    if nbits_in <> nbits_out then
      raise
        (Py_exn.Value_error
           "view() of a 0d array is only supported if the itemsize is \
            unchanged.");
    NL.squeeze (view (Lax_api.expand_dims x [| 0 |]) dtype)
  end
  else if shape.(n - 1) * nbits_in mod nbits_out <> 0 then
    raise
      (Py_exn.Value_error
         "When changing to a larger dtype, its size must be a divisor of the \
          total size in bytes of the last axis of the array.")
  else if a.dtype = dtype then x
  else if a.dtype = Dtype.Bool then view (NL.astype x Dtype.Uint8) dtype
  else if is_complex a.dtype then
    let part = complex_part a.dtype in
    let re = Lax_api.real x and im = Lax_api.imag x in
    let both =
      NL.reshape
        (NL.stack ~axis:n [ re; im ])
        (Array.append (Array.sub shape 0 (n - 1)) [| shape.(n - 1) * 2 |])
    in
    view (NL.astype both part) dtype
  else if dtype = Dtype.Bool then NL.astype (view x Dtype.Uint8) Dtype.Bool
  else if is_complex dtype then
    let out = NL.astype (view x (complex_part dtype)) dtype in
    UF.add (every_other out 0)
      (UF.multiply (imaginary_unit ()) (every_other out 1))
  else if nbits_in < nbits_out then
    let factor = nbits_out / nbits_in in
    let out =
      NL.reshape x
        (Array.append
           (Array.sub shape 0 (n - 1))
           [| shape.(n - 1) / factor; factor |])
    in
    Lax_api.bitcast_convert_type out dtype
  else if nbits_in > nbits_out then
    let out = Lax_api.bitcast_convert_type x dtype in
    let s = (aval_of out).shape in
    let m = Array.length s in
    NL.reshape out
      (Array.append (Array.sub s 0 (m - 2)) [| s.(m - 2) * s.(m - 1) |])
  else Lax_api.bitcast_convert_type x dtype

let rec byteswap x =
  abstract_on_keys "byteswap" x;
  let a = aval_of x in
  if is_complex a.dtype then
    Lax_api.complex (byteswap (Lax_api.real x)) (byteswap (Lax_api.imag x))
  else if (Dtypes.itemsize_bits a.dtype + 7) / 8 = 1 then NL.copy x
  else
    let bytes = Lax_api.bitcast_convert_type x Dtype.Uint8 in
    let swapped = Lax_api.rev bytes [| Array.length a.shape |] in
    Lax_api.bitcast_convert_type swapped a.dtype

type py_scalar =
  | Py_bool of bool
  | Py_int of string
  | Py_float of float
  | Py_complex of Complex.t

type py_list = Py_item of py_scalar | Py_list of py_list list

let host what x =
  match x with
  | Types.Concrete nd -> nd
  | Types.Device b -> Pjrt.Buffer.to_host b
  | Types.Tracer _ ->
      raise
        (Py_exn.Concretization_type_error
           ("Abstract tracer value encountered where concrete value is \
             expected: This occurred in the " ^ what ^ " method of jax.Array"))

let size_of shape = Array.fold_left ( * ) 1 shape

let unravel shape i =
  let n = Array.length shape in
  let idx = Array.make n 0 in
  let r = ref i in
  for k = n - 1 downto 0 do
    idx.(k) <- !r mod shape.(k);
    r := !r / shape.(k)
  done;
  idx

let scalar_at nd idx =
  match Ndarray.dtype nd with
  | Dtype.Bool -> Py_bool (Ndarray.get_f nd idx <> 0.0)
  | Dtype.Complex64 | Dtype.Complex128 -> Py_complex (Ndarray.get_c nd idx)
  | Dtype.Uint64 -> Py_int (Printf.sprintf "%Lu" (Ndarray.get_i64 nd idx))
  | Dtype.I8 | Dtype.I16 | Dtype.I32 | Dtype.I64 | Dtype.Uint8 | Dtype.Uint16
  | Dtype.Uint32 | Dtype.I2 | Dtype.I4 | Dtype.Uint2 | Dtype.Uint4 ->
      Py_int (Int64.to_string (Ndarray.get_i64 nd idx))
  | _ -> Py_float (Ndarray.get_f nd idx)

let item ?(args = []) x =
  abstract_on_keys "item" x;
  let nd = host "item()" x in
  let shape = Ndarray.shape nd in
  let size = size_of shape in
  match args with
  | [] ->
      if size <> 1 then
        raise
          (Py_exn.Value_error
             "can only convert an array of size 1 to a Python scalar");
      scalar_at nd (Array.make (Array.length shape) 0)
  | [ i ] ->
      let j = if i < 0 then i + size else i in
      if j < 0 || j >= size then
        raise
          (Py_exn.Index_error
             (Printf.sprintf "index %d is out of bounds for size %d" i size));
      scalar_at nd (unravel shape j)
  | _ ->
      if List.length args <> Array.length shape then
        raise (Py_exn.Value_error "incorrect number of indices for array");
      let idx =
        Array.of_list
          (List.mapi
             (fun axis i ->
               let d = shape.(axis) in
               let j = if i < 0 then i + d else i in
               if j < 0 || j >= d then
                 raise
                   (Py_exn.Index_error
                      (Printf.sprintf
                         "index %d is out of bounds for axis %d with size %d" i
                         axis d));
               j)
             args)
      in
      scalar_at nd idx

let tolist x =
  let nd = host "tolist()" x in
  let shape = Ndarray.shape nd in
  let n = Array.length shape in
  let rec build prefix =
    let k = List.length prefix in
    if k = n then Py_item (scalar_at nd (Array.of_list (List.rev prefix)))
    else Py_list (List.init shape.(k) (fun i -> build (i :: prefix)))
  in
  build []

let tobytes x =
  let buf = Pjrt.Buffer.host_bytes (host "tobytes()" x) in
  String.init (Bigarray.Array1.dim buf) (fun i ->
      Char.chr (Bigarray.Array1.get buf i))

let itemsize x = (Dtypes.itemsize_bits (aval_of x).dtype + 7) / 8
let nbytes x = size_of (aval_of x).shape * itemsize x

let flat _ =
  raise
    (Py_exn.Not_implemented_error
       "JAX Arrays do not implement the arr.flat property: consider \
        arr.flatten() instead.")

let shortest_digits f =
  let rec go p =
    let s = Printf.sprintf "%.*e" (p - 1) f in
    if p >= 17 || float_of_string s = f then s else go (p + 1)
  in
  let s = go 1 in
  let e = String.index s 'e' in
  let mant = String.sub s 0 e in
  let exp = int_of_string (String.sub s (e + 1) (String.length s - e - 1)) in
  let digits = String.concat "" (String.split_on_char '.' mant) in
  let len = ref (String.length digits) in
  while !len > 1 && digits.[!len - 1] = '0' do
    decr len
  done;
  (String.sub digits 0 !len, exp)

let float_repr ?(dot0 = true) f =
  if Float.is_nan f then "nan"
  else if f = Float.infinity then "inf"
  else if f = Float.neg_infinity then "-inf"
  else
    let sign = if Float.sign_bit f then "-" else "" in
    let a = Float.abs f in
    if a = 0.0 then sign ^ if dot0 then "0.0" else "0"
    else
      let digits, e = shortest_digits a in
      let nd = String.length digits in
      let body =
        if e < -4 || e >= 16 then
          let mant =
            if nd = 1 then digits
            else String.sub digits 0 1 ^ "." ^ String.sub digits 1 (nd - 1)
          in
          Printf.sprintf "%se%s%02d" mant
            (if e < 0 then "-" else "+")
            (Stdlib.abs e)
        else if e < 0 then "0." ^ String.make (-e - 1) '0' ^ digits
        else if nd <= e + 1 then
          digits ^ String.make (e + 1 - nd) '0' ^ if dot0 then ".0" else ""
        else
          String.sub digits 0 (e + 1)
          ^ "."
          ^ String.sub digits (e + 1) (nd - e - 1)
      in
      sign ^ body

let py_scalar_repr = function
  | Py_bool b -> if b then "True" else "False"
  | Py_int s -> s
  | Py_float f -> float_repr f
  | Py_complex z ->
      let im = float_repr ~dot0:false z.im in
      if z.re = 0.0 && not (Float.sign_bit z.re) then im ^ "j"
      else
        let im = if im.[0] = '-' then im else "+" ^ im in
        "(" ^ float_repr ~dot0:false z.re ^ im ^ "j)"

let rec py_list_repr = function
  | Py_item s -> py_scalar_repr s
  | Py_list l -> "[" ^ String.concat ", " (List.map py_list_repr l) ^ "]"
