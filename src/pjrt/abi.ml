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

let dtype_of_buffer_type (bt : int) : Dtype.t option =
  match bt with
  | 1 -> Some Dtype.Bool
  | 4 -> Some Dtype.I32
  | 5 -> Some Dtype.I64
  | 8 -> Some Dtype.Uint32
  | 11 -> Some Dtype.F32
  | 12 -> Some Dtype.F64
  | _ -> None
