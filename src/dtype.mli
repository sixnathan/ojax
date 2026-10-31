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
  | I2
  | I4
  | Uint2
  | Uint4
  | F4_e2m1fn
  | F6_e2m3fn
  | F6_e3m2fn
  | F8_e3m4
  | F8_e4m3
  | F8_e4m3b11fnuz
  | F8_e4m3fn
  | F8_e4m3fnuz
  | F8_e5m2
  | F8_e5m2fnuz
  | F8_e8m0fnu
  | Float0

val short_name : t -> string
val numpy_name : t -> string
val all : t list
val of_numpy_name : string -> t option
