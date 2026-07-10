type t

type bytebuf =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external of_host_ : Client.t -> bytebuf -> int -> int array -> t
  = "ojax_pjrt_buffer_from_host"

external to_host_ : t -> bytebuf -> unit = "ojax_pjrt_buffer_to_host"
external element_type_ : t -> int = "ojax_pjrt_buffer_element_type"
external dimensions_ : t -> int array = "ojax_pjrt_buffer_dimensions"
external destroy_ : t -> unit = "ojax_pjrt_buffer_destroy"

let u32_mask = 0xFFFF_FFFF

let elt_bytes = function
  | Dtype.F64 | Dtype.I64 -> 8
  | Dtype.F32 | Dtype.I32 | Dtype.Uint32 -> 4
  | Dtype.Bool -> 1
  | Dtype.Complex64 | Dtype.Complex128 ->
      invalid_arg "Pjrt.Buffer: complex unsupported"

let set_byte buf off b = Bigarray.Array1.set buf off (b land 0xff)

let put_i32 buf off (x : int32) =
  set_byte buf off (Int32.to_int x);
  set_byte buf (off + 1) (Int32.to_int (Int32.shift_right_logical x 8));
  set_byte buf (off + 2) (Int32.to_int (Int32.shift_right_logical x 16));
  set_byte buf (off + 3) (Int32.to_int (Int32.shift_right_logical x 24))

let put_i64 buf off (x : int64) =
  for k = 0 to 7 do
    set_byte buf (off + k) (Int64.to_int (Int64.shift_right_logical x (8 * k)))
  done

let get_i32 buf off =
  let b k = Int32.of_int (Bigarray.Array1.get buf (off + k)) in
  Int32.logor (b 0)
    (Int32.logor
       (Int32.shift_left (b 1) 8)
       (Int32.logor (Int32.shift_left (b 2) 16) (Int32.shift_left (b 3) 24)))

let get_i64 buf off =
  let r = ref 0L in
  for k = 7 downto 0 do
    r :=
      Int64.logor (Int64.shift_left !r 8)
        (Int64.of_int (Bigarray.Array1.get buf (off + k)))
  done;
  !r

let encode buf off dtype v =
  match dtype with
  | Dtype.F32 -> put_i32 buf off (Int32.bits_of_float v)
  | Dtype.F64 -> put_i64 buf off (Int64.bits_of_float v)
  | Dtype.I32 -> put_i32 buf off (Int32.of_float v)
  | Dtype.I64 -> put_i64 buf off (Int64.of_float v)
  | Dtype.Uint32 ->
      put_i32 buf off (Int32.of_int (int_of_float v land u32_mask))
  | Dtype.Bool -> set_byte buf off (if v = 0.0 then 0 else 1)
  | Dtype.Complex64 | Dtype.Complex128 ->
      invalid_arg "Pjrt.Buffer: complex unsupported"

let decode buf off dtype =
  match dtype with
  | Dtype.F32 -> Int32.float_of_bits (get_i32 buf off)
  | Dtype.F64 -> Int64.float_of_bits (get_i64 buf off)
  | Dtype.I32 -> Int32.to_float (get_i32 buf off)
  | Dtype.I64 -> Int64.to_float (get_i64 buf off)
  | Dtype.Uint32 -> float_of_int (Int32.to_int (get_i32 buf off) land u32_mask)
  | Dtype.Bool -> if Bigarray.Array1.get buf off = 0 then 0.0 else 1.0
  | Dtype.Complex64 | Dtype.Complex128 ->
      invalid_arg "Pjrt.Buffer: complex unsupported"

let unravel shape flat =
  let r = Array.length shape in
  let idx = Array.make r 0 in
  let rem = ref flat in
  for i = r - 1 downto 0 do
    idx.(i) <- !rem mod shape.(i);
    rem := !rem / shape.(i)
  done;
  idx

let make_bytebuf n =
  Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n

let of_host client nd =
  let dtype = Ndarray.dtype nd in
  let shape = Ndarray.shape nd in
  let n = Array.fold_left ( * ) 1 shape in
  let esz = elt_bytes dtype in
  let buf = make_bytebuf (n * esz) in
  for i = 0 to n - 1 do
    encode buf (i * esz) dtype (Ndarray.get_f nd (unravel shape i))
  done;
  of_host_ client buf (Abi.buffer_type dtype) shape

let dtype_of_buffer buffer =
  let bt = element_type_ buffer in
  match Abi.dtype_of_buffer_type bt with
  | Some d -> d
  | None ->
      failwith
        (Printf.sprintf "Ojax.Pjrt.Buffer: unsupported PJRT_Buffer_Type %d" bt)

let to_host buffer =
  let dtype = dtype_of_buffer buffer in
  let shape = dimensions_ buffer in
  let n = Array.fold_left ( * ) 1 shape in
  let esz = elt_bytes dtype in
  let buf = make_bytebuf (n * esz) in
  to_host_ buffer buf;
  let floats = Array.init n (fun i -> decode buf (i * esz) dtype) in
  Ndarray.of_floats dtype shape floats

let dimensions buffer = dimensions_ buffer
let element_type buffer = dtype_of_buffer buffer
let destroy buffer = destroy_ buffer
