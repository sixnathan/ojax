exception Error of string

let () = Callback.register_exception "ojax.pjrt.abi.error" (Error "")

type plugin

external open_plugin : string -> plugin = "ojax_pjrt_open"
external api_version : plugin -> int * int = "ojax_pjrt_api_version"
external struct_size : plugin -> int = "ojax_pjrt_struct_size"
external close : plugin -> unit = "ojax_pjrt_close"
external maxrss_bytes : unit -> int = "ojax_pjrt_maxrss"

let pjrt_api_major = 0
let pjrt_api_minor = 81
let pjrt_api_struct_size = 984

let buffer_type (d : Dtype.t) : int =
  match d with
  | Dtype.F32 -> 11
  | Dtype.F64 -> 12
  | Dtype.I32 -> 4
  | Dtype.I64 -> 5
  | Dtype.Bool -> 1
  | Dtype.Uint32 -> 8
  | Dtype.I8 -> 2
  | Dtype.I16 -> 3
  | Dtype.Uint8 -> 6
  | Dtype.Uint16 -> 7
  | Dtype.Uint64 -> 9
  | Dtype.F16 -> 10
  | Dtype.Bf16 -> 13
  | Dtype.Complex64 -> 14
  | Dtype.Complex128 -> 15
  | Dtype.F8_e5m2 -> 16
  | Dtype.F8_e4m3fn -> 17
  | Dtype.F8_e4m3b11fnuz -> 18
  | Dtype.F8_e5m2fnuz -> 19
  | Dtype.F8_e4m3fnuz -> 20
  | Dtype.I4 -> 21
  | Dtype.Uint4 -> 22
  | Dtype.I2 -> 24
  | Dtype.Uint2 -> 25
  | Dtype.F8_e4m3 -> 26
  | Dtype.F8_e3m4 -> 27
  | Dtype.F8_e8m0fnu -> 28
  | Dtype.F4_e2m1fn -> 29
  | (Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.Float0) as d ->
      invalid_arg
        (Printf.sprintf "PJRT has no buffer type for %s" (Dtype.numpy_name d))

let dtype_of_buffer_type (bt : int) : Dtype.t option =
  match bt with
  | 1 -> Some Dtype.Bool
  | 4 -> Some Dtype.I32
  | 5 -> Some Dtype.I64
  | 8 -> Some Dtype.Uint32
  | 2 -> Some Dtype.I8
  | 3 -> Some Dtype.I16
  | 6 -> Some Dtype.Uint8
  | 7 -> Some Dtype.Uint16
  | 9 -> Some Dtype.Uint64
  | 10 -> Some Dtype.F16
  | 13 -> Some Dtype.Bf16
  | 14 -> Some Dtype.Complex64
  | 15 -> Some Dtype.Complex128
  | 11 -> Some Dtype.F32
  | 12 -> Some Dtype.F64
  | 16 -> Some Dtype.F8_e5m2
  | 17 -> Some Dtype.F8_e4m3fn
  | 18 -> Some Dtype.F8_e4m3b11fnuz
  | 19 -> Some Dtype.F8_e5m2fnuz
  | 20 -> Some Dtype.F8_e4m3fnuz
  | 21 -> Some Dtype.I4
  | 22 -> Some Dtype.Uint4
  | 24 -> Some Dtype.I2
  | 25 -> Some Dtype.Uint2
  | 26 -> Some Dtype.F8_e4m3
  | 27 -> Some Dtype.F8_e3m4
  | 28 -> Some Dtype.F8_e8m0fnu
  | 29 -> Some Dtype.F4_e2m1fn
  | _ -> None
