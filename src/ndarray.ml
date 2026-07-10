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

let create_buffer dtype n =
  match dtype with
  | Dtype.F32 | Dtype.F64 -> Float_buf (Array1.create Float64 C_layout n)
  | Dtype.I32 | Dtype.Uint32 -> Int32_buf (Array1.create Int32 C_layout n)
  | Dtype.I64 -> Int64_buf (Array1.create Int64 C_layout n)
  | Dtype.Bool -> Int8_buf (Array1.create Int8_signed C_layout n)
  | Dtype.Complex64 -> Complex64_buf (Array1.create Complex32 C_layout n)
  | Dtype.Complex128 -> Complex128_buf (Array1.create Complex64 C_layout n)

let read_float dtype data i =
  match data with
  | Float_buf a -> Array1.unsafe_get a i
  | Int32_buf a ->
      let v = Int32.to_int (Array1.unsafe_get a i) in
      if dtype = Dtype.Uint32 then float_of_int (v land uint32_mask)
      else float_of_int v
  | Int64_buf a -> Int64.to_float (Array1.unsafe_get a i)
  | Int8_buf a -> float_of_int (Array1.unsafe_get a i)
  | Complex64_buf a -> (Array1.unsafe_get a i).Complex.re
  | Complex128_buf a -> (Array1.unsafe_get a i).Complex.re

let write_float dtype data i v =
  match data with
  | Float_buf a -> Array1.unsafe_set a i v
  | Int32_buf a ->
      let stored =
        if dtype = Dtype.Uint32 then
          Int32.of_int (int_of_float v land uint32_mask)
        else Int32.of_float v
      in
      Array1.unsafe_set a i stored
  | Int64_buf a -> Array1.unsafe_set a i (Int64.of_float v)
  | Int8_buf a -> Array1.unsafe_set a i (if v = 0.0 then 0 else 1)
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

let canonicalize dtype t =
  match dtype with
  | Dtype.Complex64 | Dtype.Complex128 ->
      map_complex dtype (round_complex dtype) t
  | _ -> map dtype (round_to dtype) t
