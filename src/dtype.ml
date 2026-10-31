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
  | I2 -> "i2"
  | I4 -> "i4"
  | Uint2 -> "u2"
  | Uint4 -> "u4"
  | F4_e2m1fn -> "f4_e2m1fn"
  | F6_e2m3fn -> "f6_e2m3fn"
  | F6_e3m2fn -> "f6_e3m2fn"
  | F8_e3m4 -> "f8_e3m4"
  | F8_e4m3 -> "f8_e4m3"
  | F8_e4m3b11fnuz -> "f8_e4m3b11fnuz"
  | F8_e4m3fn -> "f8_e4m3fn"
  | F8_e4m3fnuz -> "f8_e4m3fnuz"
  | F8_e5m2 -> "f8_e5m2"
  | F8_e5m2fnuz -> "f8_e5m2fnuz"
  | F8_e8m0fnu -> "f8_e8m0fnu"
  | Float0 -> "float0"

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
  | I2 -> "int2"
  | I4 -> "int4"
  | Uint2 -> "uint2"
  | Uint4 -> "uint4"
  | F4_e2m1fn -> "float4_e2m1fn"
  | F6_e2m3fn -> "float6_e2m3fn"
  | F6_e3m2fn -> "float6_e3m2fn"
  | F8_e3m4 -> "float8_e3m4"
  | F8_e4m3 -> "float8_e4m3"
  | F8_e4m3b11fnuz -> "float8_e4m3b11fnuz"
  | F8_e4m3fn -> "float8_e4m3fn"
  | F8_e4m3fnuz -> "float8_e4m3fnuz"
  | F8_e5m2 -> "float8_e5m2"
  | F8_e5m2fnuz -> "float8_e5m2fnuz"
  | F8_e8m0fnu -> "float8_e8m0fnu"
  | Float0 -> "float0"

let all =
  [
    F32;
    F64;
    I32;
    I64;
    Bool;
    Uint32;
    Complex64;
    Complex128;
    I8;
    I16;
    Uint8;
    Uint16;
    Uint64;
    F16;
    Bf16;
    I2;
    I4;
    Uint2;
    Uint4;
    F4_e2m1fn;
    F6_e2m3fn;
    F6_e3m2fn;
    F8_e3m4;
    F8_e4m3;
    F8_e4m3b11fnuz;
    F8_e4m3fn;
    F8_e4m3fnuz;
    F8_e5m2;
    F8_e5m2fnuz;
    F8_e8m0fnu;
    Float0;
  ]

let of_numpy_name name = List.find_opt (fun d -> numpy_name d = name) all
