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

let numpy_name = function
  | F32 -> "float32"
  | F64 -> "float64"
  | I32 -> "int32"
  | I64 -> "int64"
  | Bool -> "bool"
  | Uint32 -> "uint32"
  | Complex64 -> "complex64"
  | Complex128 -> "complex128"
  | I8 -> "int8"
  | I16 -> "int16"
  | Uint8 -> "uint8"
  | Uint16 -> "uint16"
  | Uint64 -> "uint64"
  | F16 -> "float16"
  | Bf16 -> "bfloat16"
