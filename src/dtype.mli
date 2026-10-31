type t =
  | F32
  | F64
  | I32
  | I64
  | Bool
  | Uint32
  | Complex64
  | Complex128
  | I8
  | I16
  | Uint8
  | Uint16
  | Uint64
  | F16
  | Bf16

val short_name : t -> string
val numpy_name : t -> string
