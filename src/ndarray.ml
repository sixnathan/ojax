open Bigarray

type buffer =
  | Float_buf of (float, float64_elt, c_layout) Array1.t
  | Int32_buf of (int32, int32_elt, c_layout) Array1.t

type t = { dtype : Dtype.t; shape : int array; data : buffer }

let uint32_mask = 0xFFFF_FFFF
let backs_int32 = function Dtype.Uint32 -> true | _ -> false

let create_buffer dtype n =
  if backs_int32 dtype then Int32_buf (Array1.create Int32 C_layout n)
  else Float_buf (Array1.create Float64 C_layout n)

let read_float data i =
  match data with
  | Float_buf a -> Array1.unsafe_get a i
  | Int32_buf a ->
      float_of_int (Int32.to_int (Array1.unsafe_get a i) land uint32_mask)

let write_float data i v =
  match data with
  | Float_buf a -> Array1.unsafe_set a i v
  | Int32_buf a ->
      Array1.unsafe_set a i (Int32.of_int (int_of_float v land uint32_mask))

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
  Array.iteri (fun i v -> write_float data i v) floats;
  { dtype; shape = Array.copy shape; data }

let dtype t = t.dtype
let shape t = Array.copy t.shape
let get_f t idx = read_float t.data (flat_index t.shape idx)
let set_f t idx v = write_float t.data (flat_index t.shape idx) v
let get_i64 t idx = Int64.of_float (get_f t idx)

let map dtype f t =
  let n = size t.shape in
  let data = create_buffer dtype n in
  for i = 0 to n - 1 do
    write_float data i (f (read_float t.data i))
  done;
  { dtype; shape = Array.copy t.shape; data }

let map2 dtype f a b =
  if a.shape <> b.shape then invalid_arg "Ndarray.map2: shape mismatch";
  let n = size a.shape in
  let data = create_buffer dtype n in
  for i = 0 to n - 1 do
    write_float data i (f (read_float a.data i) (read_float b.data i))
  done;
  { dtype; shape = Array.copy a.shape; data }

let fold f acc t =
  let n = size t.shape in
  let acc = ref acc in
  for i = 0 to n - 1 do
    acc := f !acc (read_float t.data i)
  done;
  !acc

let round_to dtype v =
  match dtype with
  | Dtype.F64 -> v
  | Dtype.F32 -> Int32.float_of_bits (Int32.bits_of_float v)
  | Dtype.I32 -> Int32.to_float (Int32.of_float v)
  | Dtype.I64 -> Int64.to_float (Int64.of_float v)
  | Dtype.Uint32 -> float_of_int (int_of_float v land uint32_mask)
  | Dtype.Bool -> if v = 0.0 then 0.0 else 1.0

let canonicalize dtype t = map dtype (round_to dtype) t
