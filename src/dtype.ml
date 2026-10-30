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

let short_name = function
  | F32 -> "f32"
  | F64 -> "f64"
  | I32 -> "i32"
  | I64 -> "i64"
  | Bool -> "bool"
  | Uint32 -> "u32"
  | Complex64 -> "c64"
  | Complex128 -> "c128"
  | I8 -> "i8"
  | I16 -> "i16"
  | Uint8 -> "u8"
  | Uint16 -> "u16"
  | Uint64 -> "u64"
  | F16 -> "f16"
  | Bf16 -> "bf16"
