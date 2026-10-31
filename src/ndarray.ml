open Bigarray

type buffer =
  | Float_buf of (float, float64_elt, c_layout) Array1.t
  | Int32_buf of (int32, int32_elt, c_layout) Array1.t
  | Int64_buf of (int64, int64_elt, c_layout) Array1.t
  | Int8_buf of (int, int8_signed_elt, c_layout) Array1.t
  | Complex64_buf of (Complex.t, complex32_elt, c_layout) Array1.t
  | Complex128_buf of (Complex.t, complex64_elt, c_layout) Array1.t

type t = { dtype : Dtype.t; shape : int array; data : buffer }

let uint32_mask = 0xFFFF_FFFF
let f32_round v = Int32.float_of_bits (Int32.bits_of_float v)

let round_significant ~bits ~min_exp ~max_exp v =
  if v = 0.0 || not (Float.is_finite v) then v
  else
    let _, e = Float.frexp v in
    let exp = e - 1 in
    if exp > max_exp then Float.copy_sign Float.infinity v
    else
      let lowest = Stdlib.max (exp - bits + 1) (min_exp - bits + 1) in
      let unit = Float.ldexp 1.0 lowest in
      let q = v /. unit in
      let r = Float.round q in
      let tied = Float.abs (q -. Float.trunc q) = 0.5 in
      let even =
        if tied then
          let t = Float.trunc q in
          if Float.rem t 2.0 = 0.0 then t else r
        else r
      in
      let out = even *. unit in
      let _, e' = Float.frexp out in
      if out <> 0.0 && e' - 1 > max_exp then Float.copy_sign Float.infinity v
      else out

let f16_round v = round_significant ~bits:11 ~min_exp:(-14) ~max_exp:15 v
let bf16_round v = round_significant ~bits:8 ~min_exp:(-126) ~max_exp:127 v

let half_of_bits bits =
  let sign = if bits land 0x8000 <> 0 then -1.0 else 1.0 in
  let exponent = (bits lsr 10) land 0x1F and mantissa = bits land 0x3FF in
  let magnitude =
    if exponent = 0x1F then if mantissa = 0 then Float.infinity else Float.nan
    else if exponent = 0 then Float.ldexp (float_of_int mantissa) (-24)
    else Float.ldexp (float_of_int (mantissa + 1024)) (exponent - 25)
  in
  Float.copy_sign magnitude sign

let bits_of_half v =
  let sign = if Float.sign_bit v then 0x8000 else 0 in
  if Float.is_nan v then sign lor 0x7E00
  else if Float.abs v = Float.infinity then sign lor 0x7C00
  else if v = 0.0 then sign
  else
    let fraction, e = Float.frexp (Float.abs v) in
    let exponent = e - 1 in
    if exponent >= -14 then
      let mantissa = int_of_float (Float.ldexp fraction 11) - 1024 in
      sign lor ((exponent + 15) lsl 10) lor mantissa
    else sign lor int_of_float (Float.ldexp (Float.abs v) 24)

let bfloat_of_bits bits =
  Int32.float_of_bits (Int32.shift_left (Int32.of_int bits) 16)

let bits_of_bfloat v =
  Int32.to_int (Int32.shift_right_logical (Int32.bits_of_float v) 16)
  land 0xFFFF

let wrap_int ~bits ~signed v =
  let modulus = 1 lsl bits in
  let raw = int_of_float v land (modulus - 1) in
  if signed && raw >= modulus / 2 then raw - modulus else raw

let two_64 = 18446744073709551616.0

let uint64_to_float x =
  let v = Int64.to_float x in
  if v < 0.0 then v +. two_64 else v

let uint64_of_float v =
  if v >= 9223372036854775808.0 then Int64.of_float (v -. two_64)
  else Int64.of_float v

let create_buffer dtype n =
  match dtype with
  | Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn
  | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
  | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
  | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
      Float_buf (Array1.create Float64 C_layout n)
  | Dtype.I32 | Dtype.Uint32 | Dtype.I8 | Dtype.I16 | Dtype.Uint8 | Dtype.Uint16
  | Dtype.I2 | Dtype.I4 | Dtype.Uint2 | Dtype.Uint4 ->
      Int32_buf (Array1.create Int32 C_layout n)
  | Dtype.I64 | Dtype.Uint64 -> Int64_buf (Array1.create Int64 C_layout n)
  | Dtype.Bool | Dtype.Float0 -> Int8_buf (Array1.create Int8_signed C_layout n)
  | Dtype.Complex64 -> Complex64_buf (Array1.create Complex32 C_layout n)
  | Dtype.Complex128 -> Complex128_buf (Array1.create Complex64 C_layout n)

let read_float dtype data i =
  match data with
  | Float_buf a -> Array1.unsafe_get a i
  | Int32_buf a ->
      let v = Int32.to_int (Array1.unsafe_get a i) in
      if dtype = Dtype.Uint32 then float_of_int (v land uint32_mask)
      else float_of_int v
  | Int64_buf a ->
      let v = Array1.unsafe_get a i in
      if dtype = Dtype.Uint64 then uint64_to_float v else Int64.to_float v
  | Int8_buf a -> float_of_int (Array1.unsafe_get a i)
  | Complex64_buf a -> (Array1.unsafe_get a i).Complex.re
  | Complex128_buf a -> (Array1.unsafe_get a i).Complex.re

let write_float dtype data i v =
  match data with
  | Float_buf a ->
      let stored =
        match dtype with
        | Dtype.F32 -> Int32.float_of_bits (Int32.bits_of_float v)
        | Dtype.F16 -> f16_round v
        | Dtype.Bf16 -> bf16_round v
        | d -> (
            match Float_format.of_dtype d with
            | Some f -> Float_format.round f v
            | None -> v)
      in
      Array1.unsafe_set a i stored
  | Int32_buf a ->
      let stored =
        match dtype with
        | Dtype.Uint32 -> Int32.of_int (int_of_float v land uint32_mask)
        | Dtype.I8 -> Int32.of_int (wrap_int ~bits:8 ~signed:true v)
        | Dtype.I16 -> Int32.of_int (wrap_int ~bits:16 ~signed:true v)
        | Dtype.Uint8 -> Int32.of_int (wrap_int ~bits:8 ~signed:false v)
        | Dtype.Uint16 -> Int32.of_int (wrap_int ~bits:16 ~signed:false v)
        | Dtype.I2 -> Int32.of_int (wrap_int ~bits:2 ~signed:true v)
        | Dtype.I4 -> Int32.of_int (wrap_int ~bits:4 ~signed:true v)
        | Dtype.Uint2 -> Int32.of_int (wrap_int ~bits:2 ~signed:false v)
        | Dtype.Uint4 -> Int32.of_int (wrap_int ~bits:4 ~signed:false v)
        | _ -> Int32.of_int (wrap_int ~bits:32 ~signed:true v)
      in
      Array1.unsafe_set a i stored
  | Int64_buf a ->
      Array1.unsafe_set a i
        (if dtype = Dtype.Uint64 then uint64_of_float v else Int64.of_float v)
  | Int8_buf a ->
      Array1.unsafe_set a i (if v = 0.0 || dtype = Dtype.Float0 then 0 else 1)
  | Complex64_buf a -> Array1.unsafe_set a i { Complex.re = v; im = 0.0 }
  | Complex128_buf a -> Array1.unsafe_set a i { Complex.re = v; im = 0.0 }

let read_int64 dtype data i =
  match data with
  | Float_buf a -> Int64.of_float (Array1.unsafe_get a i)
  | Int32_buf a ->
      let v = Int32.to_int (Array1.unsafe_get a i) in
      if dtype = Dtype.Uint32 then Int64.of_int (v land uint32_mask)
      else Int64.of_int v
  | Int64_buf a -> Array1.unsafe_get a i
  | Int8_buf a -> Int64.of_int (Array1.unsafe_get a i)
  | Complex64_buf a -> Int64.of_float (Array1.unsafe_get a i).Complex.re
  | Complex128_buf a -> Int64.of_float (Array1.unsafe_get a i).Complex.re

let read_complex data i =
  match data with
  | Complex64_buf a -> Array1.unsafe_get a i
  | Complex128_buf a -> Array1.unsafe_get a i
  | Float_buf a -> { Complex.re = Array1.unsafe_get a i; im = 0.0 }
  | Int32_buf _ | Int64_buf _ | Int8_buf _ ->
      { Complex.re = Int64.to_float (read_int64 Dtype.I64 data i); im = 0.0 }

let write_complex data i (z : Complex.t) =
  match data with
  | Complex64_buf a -> Array1.unsafe_set a i z
  | Complex128_buf a -> Array1.unsafe_set a i z
  | Float_buf a -> Array1.unsafe_set a i z.Complex.re
  | Int32_buf _ | Int64_buf _ | Int8_buf _ ->
      write_float Dtype.I64 data i z.Complex.re

let size shape = Array.fold_left ( * ) 1 shape

let flat_index shape idx =
  let off = ref 0 in
  for i = 0 to Array.length shape - 1 do
    off := (!off * shape.(i)) + idx.(i)
  done;
  !off

let of_floats dtype shape floats =
  let n = size shape in
  if Array.length floats <> n then
    invalid_arg "Ndarray.of_floats: length mismatch";
  let data = create_buffer dtype n in
  Array.iteri (fun i v -> write_float dtype data i v) floats;
  { dtype; shape = Array.copy shape; data }

let of_int64s dtype shape values =
  let n = size shape in
  if Array.length values <> n then
    invalid_arg "Ndarray.of_int64s: length mismatch";
  match create_buffer dtype n with
  | Int64_buf a as data ->
      Array.iteri (fun i v -> Array1.unsafe_set a i v) values;
      { dtype; shape = Array.copy shape; data }
  | _ -> invalid_arg "Ndarray.of_int64s: not a 64-bit integer dtype"

let of_complex dtype shape values =
  let n = size shape in
  if Array.length values <> n then
    invalid_arg "Ndarray.of_complex: length mismatch";
  let data = create_buffer dtype n in
  Array.iteri (fun i z -> write_complex data i z) values;
  { dtype; shape = Array.copy shape; data }

let dtype t = t.dtype
let shape t = Array.copy t.shape
let get_f t idx = read_float t.dtype t.data (flat_index t.shape idx)
let set_f t idx v = write_float t.dtype t.data (flat_index t.shape idx) v
let get_i64 t idx = read_int64 t.dtype t.data (flat_index t.shape idx)
let get_c t idx = read_complex t.data (flat_index t.shape idx)
let set_c t idx z = write_complex t.data (flat_index t.shape idx) z

let map dtype f t =
  let n = size t.shape in
  let data = create_buffer dtype n in
  for i = 0 to n - 1 do
    write_float dtype data i (f (read_float t.dtype t.data i))
  done;
  { dtype; shape = Array.copy t.shape; data }

let map2 dtype f a b =
  if a.shape <> b.shape then invalid_arg "Ndarray.map2: shape mismatch";
  let n = size a.shape in
  let data = create_buffer dtype n in
  for i = 0 to n - 1 do
    write_float dtype data i
      (f (read_float a.dtype a.data i) (read_float b.dtype b.data i))
  done;
  { dtype; shape = Array.copy a.shape; data }

let fold f acc t =
  let n = size t.shape in
  let acc = ref acc in
  for i = 0 to n - 1 do
    acc := f !acc (read_float t.dtype t.data i)
  done;
  !acc

let round_to dtype v =
  match dtype with
  | Dtype.F64 -> v
  | Dtype.F32 -> f32_round v
  | Dtype.I32 -> Int32.to_float (Int32.of_float v)
  | Dtype.I64 -> Int64.to_float (Int64.of_float v)
  | Dtype.Uint32 -> float_of_int (int_of_float v land uint32_mask)
  | Dtype.Bool -> if v = 0.0 then 0.0 else 1.0
  | Dtype.Complex64 | Dtype.Complex128 -> v
  | Dtype.F16 -> f16_round v
  | Dtype.Bf16 -> bf16_round v
  | Dtype.I8 -> float_of_int (wrap_int ~bits:8 ~signed:true v)
  | Dtype.I16 -> float_of_int (wrap_int ~bits:16 ~signed:true v)
  | Dtype.Uint8 -> float_of_int (wrap_int ~bits:8 ~signed:false v)
  | Dtype.Uint16 -> float_of_int (wrap_int ~bits:16 ~signed:false v)
  | Dtype.Uint64 -> uint64_to_float (uint64_of_float v)
  | Dtype.I2 -> float_of_int (wrap_int ~bits:2 ~signed:true v)
  | Dtype.I4 -> float_of_int (wrap_int ~bits:4 ~signed:true v)
  | Dtype.Uint2 -> float_of_int (wrap_int ~bits:2 ~signed:false v)
  | Dtype.Uint4 -> float_of_int (wrap_int ~bits:4 ~signed:false v)
  | Dtype.Float0 -> 0.0
  | ( Dtype.F4_e2m1fn | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4
    | Dtype.F8_e4m3 | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz
    | Dtype.F8_e5m2 | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ) as d ->
      Float_format.round (Option.get (Float_format.of_dtype d)) v

let round_complex dtype (z : Complex.t) =
  match dtype with
  | Dtype.Complex64 -> { Complex.re = f32_round z.re; im = f32_round z.im }
  | _ -> z

let map_complex dtype f t =
  let n = size t.shape in
  let data = create_buffer dtype n in
  for i = 0 to n - 1 do
    write_complex data i (f (read_complex t.data i))
  done;
  { dtype; shape = Array.copy t.shape; data }

let to_int64s t =
  let n = Array.fold_left ( * ) 1 t.shape in
  Array.init n (fun i -> read_int64 t.dtype t.data i)

let canonicalize dtype t =
  match dtype with
  | (Dtype.I64 | Dtype.Uint64) when t.dtype = dtype -> t
  | Dtype.Complex64 | Dtype.Complex128 ->
      map_complex dtype (round_complex dtype) t
  | _ -> map dtype (round_to dtype) t
