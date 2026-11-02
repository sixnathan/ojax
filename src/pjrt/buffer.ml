type raw

type named = {
  mesh_axes : (string * int) list;
  mesh_devices : int array;
  spec : string list array;
}

type layout = On_device of int | Named of named | Tiled

type shard = {
  device : int;
  index : (int * int) array;
  replica_id : int;
  data : raw;
}

type placed = {
  global_shape : int array;
  dtype : Dtype.t;
  shards : shard array;
  layout : layout;
}

type t = Single of raw | Placed of placed

type bytebuf =
  (int, Bigarray.int8_unsigned_elt, Bigarray.c_layout) Bigarray.Array1.t

external of_host_ : Client.t -> bytebuf -> int -> int array -> int -> raw
  = "ojax_pjrt_buffer_from_host"

external to_host_ : raw -> bytebuf -> unit = "ojax_pjrt_buffer_to_host"
external element_type_ : raw -> int = "ojax_pjrt_buffer_element_type"
external dimensions_ : raw -> int array = "ojax_pjrt_buffer_dimensions"
external destroy_ : raw -> unit = "ojax_pjrt_buffer_destroy"
external device_ : raw -> int = "ojax_pjrt_buffer_device"
external to_dlpack_ : raw -> nativeint = "ojax_pjrt_buffer_to_dlpack"

external of_dlpack_ : Client.t -> nativeint -> int -> raw
  = "ojax_pjrt_buffer_of_dlpack"

external dlpack_delete : nativeint -> unit = "ojax_pjrt_dlpack_delete"
external unsafe_pointer_ : raw -> nativeint = "ojax_pjrt_buffer_unsafe_pointer"
external belongs_to_ : raw -> Client.t -> bool = "ojax_pjrt_buffer_belongs_to"

let u32_mask = 0xFFFF_FFFF

let elt_bytes = function
  | Dtype.F64 | Dtype.I64 | Dtype.Uint64 -> 8
  | Dtype.F32 | Dtype.I32 | Dtype.Uint32 -> 4
  | Dtype.Bool | Dtype.I8 | Dtype.Uint8 | Dtype.I2 | Dtype.I4 | Dtype.Uint2
  | Dtype.Uint4 | Dtype.F4_e2m1fn | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn
  | Dtype.F8_e3m4 | Dtype.F8_e4m3 | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn
  | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2 | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
      1
  | Dtype.Float0 -> 0
  | Dtype.I16 | Dtype.Uint16 | Dtype.F16 | Dtype.Bf16 -> 2
  | Dtype.Complex64 -> 8
  | Dtype.Complex128 -> 16

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

let put_u16 buf off v =
  set_byte buf off (v land 0xff);
  set_byte buf (off + 1) ((v lsr 8) land 0xff)

let get_u16 buf off =
  Bigarray.Array1.get buf off lor (Bigarray.Array1.get buf (off + 1) lsl 8)

let half_bits v =
  let sign = if Float.sign_bit v then 0x8000 else 0 in
  let m = Float.abs v in
  if Float.is_nan v then
    sign lor 0x7c00
    lor Int64.to_int
          (Int64.logand
             (Int64.shift_right_logical (Int64.bits_of_float v) 42)
             0x3ffL)
  else if m = Float.infinity || m >= 65536.0 then sign lor 0x7c00
  else if m = 0.0 then sign
  else
    let _, e = Float.frexp m in
    let exp = e - 1 in
    if exp < -14 then sign lor int_of_float (Float.ldexp m 24)
    else
      let fraction = int_of_float (Float.ldexp m (10 - exp)) - 1024 in
      sign lor ((exp + 15) lsl 10) lor fraction

let half_value bits =
  let sign = if bits land 0x8000 <> 0 then -1.0 else 1.0 in
  let exponent = (bits lsr 10) land 0x1f and fraction = bits land 0x3ff in
  if exponent = 31 && fraction <> 0 then
    Int64.float_of_bits
      (Int64.logor
         (Int64.shift_left (Int64.of_int (((bits lsr 15) lsl 11) lor 0x7FF)) 52)
         (Int64.shift_left (Int64.of_int fraction) 42))
  else
    sign
    *.
    if exponent = 0 then Float.ldexp (float_of_int fraction) (-24)
    else if exponent = 31 then Float.infinity
    else Float.ldexp (float_of_int (fraction + 1024)) (exponent - 25)

let bfloat_bits v =
  Int32.to_int (Int32.shift_right_logical (Int32.bits_of_float v) 16)
  land 0xffff

let bfloat_value bits =
  Int32.float_of_bits (Int32.shift_left (Int32.of_int bits) 16)

let encode buf off dtype nd idx =
  let v () = Ndarray.get_f nd idx in
  match dtype with
  | Dtype.F32 -> put_i32 buf off (Int32.bits_of_float (v ()))
  | Dtype.F64 -> put_i64 buf off (Int64.bits_of_float (v ()))
  | Dtype.I32 -> put_i32 buf off (Int32.of_float (v ()))
  | Dtype.I64 | Dtype.Uint64 -> put_i64 buf off (Ndarray.get_i64 nd idx)
  | Dtype.Uint32 ->
      put_i32 buf off (Int32.of_int (int_of_float (v ()) land u32_mask))
  | Dtype.Bool -> set_byte buf off (if v () = 0.0 then 0 else 1)
  | Dtype.I8 | Dtype.Uint8 | Dtype.I2 | Dtype.I4 | Dtype.Uint2 | Dtype.Uint4 ->
      set_byte buf off (int_of_float (v ()) land 0xff)
  | Dtype.I16 | Dtype.Uint16 -> put_u16 buf off (int_of_float (v ()) land 0xffff)
  | Dtype.F16 -> put_u16 buf off (half_bits (v ()))
  | Dtype.Bf16 -> put_u16 buf off (bfloat_bits (v ()))
  | Dtype.F4_e2m1fn | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4
  | Dtype.F8_e4m3 | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz
  | Dtype.F8_e5m2 | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
      set_byte buf off
        (Float_format.encode (Option.get (Float_format.of_dtype dtype)) (v ()))
  | Dtype.Float0 -> ()
  | Dtype.Complex64 ->
      let z = Ndarray.get_c nd idx in
      put_i32 buf off (Int32.bits_of_float z.Complex.re);
      put_i32 buf (off + 4) (Int32.bits_of_float z.Complex.im)
  | Dtype.Complex128 ->
      let z = Ndarray.get_c nd idx in
      put_i64 buf off (Int64.bits_of_float z.Complex.re);
      put_i64 buf (off + 8) (Int64.bits_of_float z.Complex.im)

let narrow_int buf off bits signed =
  let raw = Bigarray.Array1.get buf off land ((1 lsl bits) - 1) in
  float_of_int
    (if signed && raw >= 1 lsl (bits - 1) then raw - (1 lsl bits) else raw)

let decode buf off dtype =
  match dtype with
  | Dtype.F32 -> Int32.float_of_bits (get_i32 buf off)
  | Dtype.F64 -> Int64.float_of_bits (get_i64 buf off)
  | Dtype.I32 -> Int32.to_float (get_i32 buf off)
  | Dtype.I64 -> Int64.to_float (get_i64 buf off)
  | Dtype.Uint32 -> float_of_int (Int32.to_int (get_i32 buf off) land u32_mask)
  | Dtype.Bool -> if Bigarray.Array1.get buf off = 0 then 0.0 else 1.0
  | Dtype.I8 ->
      let b = Bigarray.Array1.get buf off in
      float_of_int (if b >= 128 then b - 256 else b)
  | Dtype.Uint8 -> float_of_int (Bigarray.Array1.get buf off)
  | Dtype.I16 ->
      let b = get_u16 buf off in
      float_of_int (if b >= 32768 then b - 65536 else b)
  | Dtype.Uint16 -> float_of_int (get_u16 buf off)
  | Dtype.F16 -> half_value (get_u16 buf off)
  | Dtype.Bf16 -> bfloat_value (get_u16 buf off)
  | Dtype.I2 -> narrow_int buf off 2 true
  | Dtype.I4 -> narrow_int buf off 4 true
  | Dtype.Uint2 -> narrow_int buf off 2 false
  | Dtype.Uint4 -> narrow_int buf off 4 false
  | Dtype.F4_e2m1fn | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4
  | Dtype.F8_e4m3 | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz
  | Dtype.F8_e5m2 | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
      Float_format.decode
        (Option.get (Float_format.of_dtype dtype))
        (Bigarray.Array1.get buf off)
  | Dtype.Float0 -> 0.0
  | Dtype.Uint64 | Dtype.Complex64 | Dtype.Complex128 ->
      invalid_arg "Pjrt.Buffer: decode through the wide paths"

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

let host_bytes nd =
  let dtype = Ndarray.dtype nd in
  let shape = Ndarray.shape nd in
  let n = Array.fold_left ( * ) 1 shape in
  let esz = elt_bytes dtype in
  let buf = make_bytebuf (n * esz) in
  for i = 0 to n - 1 do
    encode buf (i * esz) dtype nd (unravel shape i)
  done;
  buf

let of_host client nd =
  Single
    (of_host_ client (host_bytes nd)
       (Abi.buffer_type (Ndarray.dtype nd))
       (Ndarray.shape nd) (-1))

let dtype_of_raw raw =
  let bt = element_type_ raw in
  match Abi.dtype_of_buffer_type bt with
  | Some d -> d
  | None ->
      failwith
        (Printf.sprintf "Ojax.Pjrt.Buffer: unsupported PJRT_Buffer_Type %d" bt)

let host_of_bytes dtype shape buf =
  let n = Array.fold_left ( * ) 1 shape in
  let esz = elt_bytes dtype in
  match dtype with
  | Dtype.I64 | Dtype.Uint64 ->
      Ndarray.of_int64s dtype shape
        (Array.init n (fun i -> get_i64 buf (i * esz)))
  | Dtype.Complex64 ->
      Ndarray.of_complex dtype shape
        (Array.init n (fun i ->
             let off = i * esz in
             {
               Complex.re = Int32.float_of_bits (get_i32 buf off);
               im = Int32.float_of_bits (get_i32 buf (off + 4));
             }))
  | Dtype.Complex128 ->
      Ndarray.of_complex dtype shape
        (Array.init n (fun i ->
             let off = i * esz in
             {
               Complex.re = Int64.float_of_bits (get_i64 buf off);
               im = Int64.float_of_bits (get_i64 buf (off + 8));
             }))
  | _ ->
      let floats = Array.init n (fun i -> decode buf (i * esz) dtype) in
      Ndarray.of_floats dtype shape floats

let raw_bytes raw =
  let dtype = dtype_of_raw raw in
  let shape = dimensions_ raw in
  let n = Array.fold_left ( * ) 1 shape in
  let buf = make_bytebuf (n * elt_bytes dtype) in
  to_host_ raw buf;
  buf

let strides shape =
  let r = Array.length shape in
  let out = Array.make r 1 in
  for i = r - 2 downto 0 do
    out.(i) <- out.(i + 1) * shape.(i + 1)
  done;
  out

let iter_region shape index f =
  let r = Array.length shape in
  let st = strides shape in
  let extent = Array.map (fun (lo, hi) -> hi - lo) index in
  let count = Array.fold_left ( * ) 1 extent in
  for k = 0 to count - 1 do
    let local = unravel extent k in
    let global = ref 0 in
    for d = 0 to r - 1 do
      global := !global + ((fst index.(d) + local.(d)) * st.(d))
    done;
    f k !global
  done

let blit src src_off dst dst_off n =
  Bigarray.Array1.blit
    (Bigarray.Array1.sub src src_off n)
    (Bigarray.Array1.sub dst dst_off n)

let placed_to_host p =
  let esz = elt_bytes p.dtype in
  let n = Array.fold_left ( * ) 1 p.global_shape in
  let out = make_bytebuf (n * esz) in
  Array.iter
    (fun s ->
      if s.replica_id = 0 then
        let local = raw_bytes s.data in
        iter_region p.global_shape s.index (fun k global ->
            blit local (k * esz) out (global * esz) esz))
    p.shards;
  host_of_bytes p.dtype p.global_shape out

let to_host = function
  | Single raw ->
      host_of_bytes (dtype_of_raw raw) (dimensions_ raw) (raw_bytes raw)
  | Placed p -> placed_to_host p

let dimensions = function
  | Single raw -> dimensions_ raw
  | Placed p -> p.global_shape

let element_type = function
  | Single raw -> dtype_of_raw raw
  | Placed p -> p.dtype

let destroy = function
  | Single raw -> destroy_ raw
  | Placed p -> Array.iter (fun s -> destroy_ s.data) p.shards

let raw_dimensions raw = dimensions_ raw
let raw_device raw = device_ raw

let raw_to_host raw =
  host_of_bytes (dtype_of_raw raw) (dimensions_ raw) (raw_bytes raw)

let shards_of_host client nd placements =
  let dtype = Ndarray.dtype nd in
  let shape = Ndarray.shape nd in
  let esz = elt_bytes dtype in
  let whole = host_bytes nd in
  Array.map
    (fun (index, device) ->
      let extent = Array.map (fun (lo, hi) -> hi - lo) index in
      let count = Array.fold_left ( * ) 1 extent in
      let local = make_bytebuf (count * esz) in
      iter_region shape index (fun k global ->
          blit whole (global * esz) local (k * esz) esz);
      of_host_ client local (Abi.buffer_type dtype) extent device)
    placements

let single_raw = function
  | Single raw -> raw
  | Placed { shards = [| s |]; _ } -> s.data
  | Placed _ ->
      invalid_arg "Pjrt.Buffer: a sharded array has many device buffers"

let to_dlpack buffer = to_dlpack_ (single_raw buffer)
let of_dlpack client pointer mode = Single (of_dlpack_ client pointer mode)
let unsafe_pointer buffer = unsafe_pointer_ (single_raw buffer)

let belongs_to buffer client =
  match buffer with
  | Single raw -> belongs_to_ raw client
  | Placed p -> belongs_to_ p.shards.(0).data client
