open Types

type capsule = { pointer : nativeint; mutable used : bool }
type placement = On_device of string | With_sharding of Sharding.t

type external_array = {
  dlpack : unit -> capsule;
  dlpack_device : unit -> int * int;
}

let dlpack_version = (0, 8)
let min_dlpack_version = (0, 5)

let is_supported_dtype = function
  | Dtype.I8 | Dtype.I16 | Dtype.I32 | Dtype.I64 | Dtype.Uint8 | Dtype.Uint16
  | Dtype.Uint32 | Dtype.Uint64 | Dtype.F16 | Dtype.Bf16 | Dtype.F32 | Dtype.F64
  | Dtype.Complex64 | Dtype.Complex128 | Dtype.Bool ->
      true
  | _ -> false

let supported_names =
  [
    "int8"; "i1"; "b"; "byte";
    "int16"; "i2"; "h"; "short";
    "int32"; "i4"; "i"; "intc";
    "int64"; "i8"; "l"; "q"; "int"; "long"; "longlong"; "int_"; "intp"; "p";
    "uint8"; "u1"; "B"; "ubyte";
    "uint16"; "u2"; "H"; "ushort";
    "uint32"; "u4"; "I"; "uintc";
    "uint64"; "u8"; "L"; "Q"; "ulong"; "ulonglong"; "uint"; "uintp"; "P";
    "float16"; "f2"; "e"; "half";
    "bfloat16";
    "float32"; "f4"; "f"; "single";
    "float64"; "f8"; "d"; "double"; "float";
    "complex64"; "c8"; "F"; "csingle";
    "complex128"; "c16"; "D"; "cdouble"; "complex";
    "bool"; "bool_"; "?";
  ]
[@@ocamlformat "disable"]

let unsupported_names =
  [
    "int2"; "int4"; "uint2"; "uint4"; "object"; "O"; "str"; "U"; "S"; "V";
    "bytes"; "datetime64"; "M"; "timedelta64"; "m"; "void"; "longdouble"; "g";
    "clongdouble"; "G"; "float8_e4m3fn"; "float8_e4m3b11fnuz"; "float8_e5m2";
    "float8_e4m3fnuz"; "float8_e5m2fnuz"; "float8_e4m3"; "float8_e3m4";
  ]
[@@ocamlformat "disable"]

let is_flexible_name name =
  let digits_after prefix =
    String.starts_with ~prefix name
    && String.for_all
         (fun c -> c >= '0' && c <= '9')
         (String.sub name (String.length prefix)
            (String.length name - String.length prefix))
  in
  List.exists digits_after [ "U"; "S"; "V"; "<U"; ">U" ]
  || String.starts_with ~prefix:"datetime64[" name
  || String.starts_with ~prefix:"timedelta64[" name

let is_supported_dtype_name name =
  if List.mem name supported_names then true
  else if List.mem name unsupported_names || is_flexible_name name then false
  else invalid_arg (Printf.sprintf "data type '%s' not understood" name)

let delete_unused c =
  if not c.used then (
    c.used <- true;
    Pjrt.Buffer.dlpack_delete c.pointer)

let capsule_of_pointer pointer =
  let c = { pointer; used = false } in
  Gc.finalise delete_unused c;
  c

let capsule_pointer c = c.pointer
let capsule_used c = c.used
let unavailable_device_types = [ (2, "cuda"); (3, "cuda"); (10, "rocm") ]

let check_device_type t =
  match List.assoc_opt t unavailable_device_types with
  | Some backend ->
      failwith
        (Printf.sprintf "Unknown backend %s. Available backends are ['cpu']"
           backend)
  | None -> ()

let check_hardware_id id =
  if id <> 0 then
    failwith
      (Printf.sprintf
         "INVALID_ARGUMENT: No matching device found for local_device_id %d" id)

let buffer_of_value v =
  match v with
  | Device (Pjrt.Buffer.Placed { shards; _ }) when Array.length shards > 1 ->
      invalid_arg "to_dlpack requires an array on a single device"
  | Device b when Pjrt.Buffer.belongs_to b (Backend.xla_client ()) -> b
  | Device b ->
      Pjrt.Buffer.of_host (Backend.xla_client ()) (Pjrt.Buffer.to_host b)
  | Concrete nd -> Pjrt.Buffer.of_host (Backend.xla_client ()) nd
  | Tracer _ ->
      invalid_arg "Argument to to_dlpack must be a jax.Array, got a tracer"

let to_dlpack ?dl_device ?max_version ?copy v =
  (match dl_device with
  | Some (t, id) when t <> 0 ->
      if t <> 1 && not (List.mem_assoc t unavailable_device_types) then
        failwith
          (Printf.sprintf
             "The device specification passed to to_dlpack contains an \
              unsupported device type (DLDeviceType: %d)"
             t);
      check_device_type t;
      check_hardware_id id
  | _ -> ());
  (match max_version with
  | Some ((maj, mnr) as requested) when requested < min_dlpack_version ->
      failwith
        (Printf.sprintf
           "JAX does not support any version below (%d, %d) but version ((%d, \
            %d)) was requested."
           (fst min_dlpack_version) (snd min_dlpack_version) maj mnr)
  | _ -> ());
  let source =
    match (copy, v) with
    | Some true, Device b ->
        Pjrt.Buffer.of_host (Backend.xla_client ()) (Pjrt.Buffer.to_host b)
    | _ -> buffer_of_value v
  in
  capsule_of_pointer (Pjrt.Buffer.to_dlpack source)

let external_of_value v =
  { dlpack = (fun () -> to_dlpack v); dlpack_device = (fun () -> (1, 0)) }

let value_error_prefix = "ojax.dlpack.value_error:"

let import_capsule capsule copy =
  if capsule.used then
    failwith
      "INVALID_ARGUMENT: DLPack tensor must be a capsule with name \
       \"dltensor\", got \"used_dltensor\". Note that a DLPack tensor may be \
       consumed at most once.";
  let mode = match copy with Some false -> 0 | Some true -> 1 | None -> 2 in
  match Pjrt.Buffer.of_dlpack (Backend.xla_client ()) capsule.pointer mode with
  | buffer ->
      capsule.used <- true;
      buffer
  | exception Pjrt.Abi.Error m
    when String.starts_with ~prefix:value_error_prefix m ->
      let n = String.length value_error_prefix in
      invalid_arg (String.sub m n (String.length m - n))
  | exception Pjrt.Abi.Error m -> failwith m

let cpu_device = "cpu:0"

let resolve_device device =
  match device with
  | Some (With_sharding s) ->
      if Sharding.num_devices s > 1 then
        invalid_arg
          (Printf.sprintf
             "from_dlpack can only unpack a dlpack tensor onto a singular \
              device, but a Sharding with %d devices was provided."
             (Sharding.num_devices s));
      None
  | Some (On_device d) -> Some d
  | None -> None

let from_dlpack ?device ?copy ext =
  let device = resolve_device device in
  let device_type, device_id = ext.dlpack_device () in
  if not (device_type = 1 || List.mem_assoc device_type unavailable_device_types)
  then
    invalid_arg
      (Printf.sprintf
         "Array passed to from_dlpack is on unsupported device type \
          (DLDeviceType: %d, array: <external>"
         device_type);
  check_device_type device_type;
  check_hardware_id device_id;
  (match device with
  | Some d when d <> cpu_device && copy = Some false ->
      invalid_arg
        (Printf.sprintf
           "Specified device=%s which requires a copy since the source device \
            is %s, however copy=False. Set copy=True or copy=None to perform \
            the requested operation."
           d cpu_device)
  | Some d when d <> cpu_device ->
      failwith (Printf.sprintf "from_dlpack: unknown device %s" d)
  | _ -> ());
  let buffer = import_capsule (ext.dlpack ()) copy in
  let dt = Pjrt.Buffer.element_type buffer in
  let canonical = Dtypes.canonicalize_dtype dt in
  if canonical = dt then Device buffer
  else
    let host = Pjrt.Buffer.to_host buffer in
    Pjrt.Buffer.destroy buffer;
    Concrete (Ndarray.canonicalize canonical host)

let unsafe_buffer_pointer = function
  | Device b -> Pjrt.Buffer.unsafe_pointer b
  | Concrete _ | Tracer _ ->
      invalid_arg "unsafe_buffer_pointer: value is not on the device"
