module D = Ojax.Dtype
module Dt = Ojax.Dtypes
module FF = Ojax.Float_format
module Nd = Ojax.Ndarray
module Ir = Ojax.Stablehlo.Ir
module C = Ojax.Core
module T = Ojax.Types
module Api = Ojax.Api
module Tree = Ojax.Tree_util
module NL = Ojax.Numpy.Lax_numpy
module ST = Ojax.Numpy.Scalar_types
module U = Ojax.Numpy.Ufuncs
module R = Ojax.Numpy.Reductions
module AC = Ojax.Numpy.Array_creation
module Cfg = Ojax.Config

let () = Ojax.Lax.install ()

let name_order =
  [
    D.Bool;
    D.I2;
    D.I4;
    D.I8;
    D.I16;
    D.I32;
    D.I64;
    D.Uint2;
    D.Uint4;
    D.Uint8;
    D.Uint16;
    D.Uint32;
    D.Uint64;
    D.F4_e2m1fn;
    D.F6_e2m3fn;
    D.F6_e3m2fn;
    D.F8_e3m4;
    D.F8_e4m3;
    D.F8_e4m3b11fnuz;
    D.F8_e4m3fn;
    D.F8_e4m3fnuz;
    D.F8_e5m2;
    D.F8_e5m2fnuz;
    D.F8_e8m0fnu;
    D.F16;
    D.Bf16;
    D.F32;
    D.F64;
    D.Complex64;
    D.Complex128;
  ]

let promote_x32 =
  [
    ( "bool",
      "bool int2 int4 int8 int16 int32 int64 uint2 uint4 uint8 uint16 uint32 \
       uint64 float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 \
       float8_e4m3 float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz \
       float8_e5m2 float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 \
       float64 complex64 complex128 int64 float64 complex128" );
    ( "int2",
      "int2 int2 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR int2 ERR ERR" );
    ( "int4",
      "int4 ERR int4 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR int4 ERR ERR" );
    ( "int8",
      "int8 ERR ERR int8 int16 int32 int64 ERR ERR int16 int32 int32 float64 \
       float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 float8_e4m3 \
       float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz float8_e5m2 \
       float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 float64 \
       complex64 complex128 int8 float64 complex128" );
    ( "int16",
      "int16 ERR ERR int16 int16 int32 int64 ERR ERR int16 int32 int32 float64 \
       float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 float8_e4m3 \
       float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz float8_e5m2 \
       float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 float64 \
       complex64 complex128 int16 float64 complex128" );
    ( "int32",
      "int32 ERR ERR int32 int32 int32 int64 ERR ERR int32 int32 int32 float64 \
       float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 float8_e4m3 \
       float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz float8_e5m2 \
       float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 float64 \
       complex64 complex128 int32 float64 complex128" );
    ( "int64",
      "int64 ERR ERR int64 int64 int64 int64 ERR ERR int64 int64 int64 float64 \
       float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 float8_e4m3 \
       float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz float8_e5m2 \
       float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 float64 \
       complex64 complex128 int64 float64 complex128" );
    ( "uint2",
      "uint2 ERR ERR ERR ERR ERR ERR uint2 ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR uint2 ERR ERR" );
    ( "uint4",
      "uint4 ERR ERR ERR ERR ERR ERR ERR uint4 ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR uint4 ERR ERR" );
    ( "uint8",
      "uint8 ERR ERR int16 int16 int32 int64 ERR ERR uint8 uint16 uint32 \
       uint64 float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 \
       float8_e4m3 float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz \
       float8_e5m2 float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 \
       float64 complex64 complex128 uint8 float64 complex128" );
    ( "uint16",
      "uint16 ERR ERR int32 int32 int32 int64 ERR ERR uint16 uint16 uint32 \
       uint64 float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 \
       float8_e4m3 float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz \
       float8_e5m2 float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 \
       float64 complex64 complex128 uint16 float64 complex128" );
    ( "uint32",
      "uint32 ERR ERR int32 int32 int32 int64 ERR ERR uint32 uint32 uint32 \
       uint64 float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 \
       float8_e4m3 float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz \
       float8_e5m2 float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 \
       float64 complex64 complex128 uint32 float64 complex128" );
    ( "uint64",
      "uint64 ERR ERR float64 float64 float64 float64 ERR ERR uint64 uint64 \
       uint64 uint64 float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 \
       float8_e4m3 float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz \
       float8_e5m2 float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 \
       float64 complex64 complex128 uint64 float64 complex128" );
    ( "float4_e2m1fn",
      "float4_e2m1fn ERR ERR float4_e2m1fn float4_e2m1fn float4_e2m1fn \
       float4_e2m1fn ERR ERR float4_e2m1fn float4_e2m1fn float4_e2m1fn \
       float4_e2m1fn float4_e2m1fn ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR float4_e2m1fn float4_e2m1fn ERR" );
    ( "float6_e2m3fn",
      "float6_e2m3fn ERR ERR float6_e2m3fn float6_e2m3fn float6_e2m3fn \
       float6_e2m3fn ERR ERR float6_e2m3fn float6_e2m3fn float6_e2m3fn \
       float6_e2m3fn ERR float6_e2m3fn ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR float6_e2m3fn float6_e2m3fn ERR" );
    ( "float6_e3m2fn",
      "float6_e3m2fn ERR ERR float6_e3m2fn float6_e3m2fn float6_e3m2fn \
       float6_e3m2fn ERR ERR float6_e3m2fn float6_e3m2fn float6_e3m2fn \
       float6_e3m2fn ERR ERR float6_e3m2fn ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR float6_e3m2fn float6_e3m2fn ERR" );
    ( "float8_e3m4",
      "float8_e3m4 ERR ERR float8_e3m4 float8_e3m4 float8_e3m4 float8_e3m4 ERR \
       ERR float8_e3m4 float8_e3m4 float8_e3m4 float8_e3m4 ERR ERR ERR \
       float8_e3m4 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       float8_e3m4 float8_e3m4 ERR" );
    ( "float8_e4m3",
      "float8_e4m3 ERR ERR float8_e4m3 float8_e4m3 float8_e4m3 float8_e4m3 ERR \
       ERR float8_e4m3 float8_e4m3 float8_e4m3 float8_e4m3 ERR ERR ERR ERR \
       float8_e4m3 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR float8_e4m3 \
       float8_e4m3 ERR" );
    ( "float8_e4m3b11fnuz",
      "float8_e4m3b11fnuz ERR ERR float8_e4m3b11fnuz float8_e4m3b11fnuz \
       float8_e4m3b11fnuz float8_e4m3b11fnuz ERR ERR float8_e4m3b11fnuz \
       float8_e4m3b11fnuz float8_e4m3b11fnuz float8_e4m3b11fnuz ERR ERR ERR \
       ERR ERR float8_e4m3b11fnuz ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       float8_e4m3b11fnuz float8_e4m3b11fnuz ERR" );
    ( "float8_e4m3fn",
      "float8_e4m3fn ERR ERR float8_e4m3fn float8_e4m3fn float8_e4m3fn \
       float8_e4m3fn ERR ERR float8_e4m3fn float8_e4m3fn float8_e4m3fn \
       float8_e4m3fn ERR ERR ERR ERR ERR ERR float8_e4m3fn ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR float8_e4m3fn float8_e4m3fn ERR" );
    ( "float8_e4m3fnuz",
      "float8_e4m3fnuz ERR ERR float8_e4m3fnuz float8_e4m3fnuz float8_e4m3fnuz \
       float8_e4m3fnuz ERR ERR float8_e4m3fnuz float8_e4m3fnuz float8_e4m3fnuz \
       float8_e4m3fnuz ERR ERR ERR ERR ERR ERR ERR float8_e4m3fnuz ERR ERR ERR \
       ERR ERR ERR ERR ERR ERR float8_e4m3fnuz float8_e4m3fnuz ERR" );
    ( "float8_e5m2",
      "float8_e5m2 ERR ERR float8_e5m2 float8_e5m2 float8_e5m2 float8_e5m2 ERR \
       ERR float8_e5m2 float8_e5m2 float8_e5m2 float8_e5m2 ERR ERR ERR ERR ERR \
       ERR ERR ERR float8_e5m2 ERR ERR ERR ERR ERR ERR ERR ERR float8_e5m2 \
       float8_e5m2 ERR" );
    ( "float8_e5m2fnuz",
      "float8_e5m2fnuz ERR ERR float8_e5m2fnuz float8_e5m2fnuz float8_e5m2fnuz \
       float8_e5m2fnuz ERR ERR float8_e5m2fnuz float8_e5m2fnuz float8_e5m2fnuz \
       float8_e5m2fnuz ERR ERR ERR ERR ERR ERR ERR ERR ERR float8_e5m2fnuz ERR \
       ERR ERR ERR ERR ERR ERR float8_e5m2fnuz float8_e5m2fnuz ERR" );
    ( "float8_e8m0fnu",
      "float8_e8m0fnu ERR ERR float8_e8m0fnu float8_e8m0fnu float8_e8m0fnu \
       float8_e8m0fnu ERR ERR float8_e8m0fnu float8_e8m0fnu float8_e8m0fnu \
       float8_e8m0fnu ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR float8_e8m0fnu \
       ERR ERR ERR ERR ERR ERR float8_e8m0fnu float8_e8m0fnu ERR" );
    ( "float16",
      "float16 ERR ERR float16 float16 float16 float16 ERR ERR float16 float16 \
       float16 float16 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR float16 \
       float32 float32 float64 complex64 complex128 float16 float16 complex64"
    );
    ( "bfloat16",
      "bfloat16 ERR ERR bfloat16 bfloat16 bfloat16 bfloat16 ERR ERR bfloat16 \
       bfloat16 bfloat16 bfloat16 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       float32 bfloat16 float32 float64 complex64 complex128 bfloat16 bfloat16 \
       complex64" );
    ( "float32",
      "float32 ERR ERR float32 float32 float32 float32 ERR ERR float32 float32 \
       float32 float32 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR float32 \
       float32 float32 float64 complex64 complex128 float32 float32 complex64"
    );
    ( "float64",
      "float64 ERR ERR float64 float64 float64 float64 ERR ERR float64 float64 \
       float64 float64 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR float64 \
       float64 float64 float64 complex128 complex128 float64 float64 \
       complex128" );
    ( "complex64",
      "complex64 ERR ERR complex64 complex64 complex64 complex64 ERR ERR \
       complex64 complex64 complex64 complex64 ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR complex64 complex64 complex64 complex128 complex64 \
       complex128 complex64 complex64 complex64" );
    ( "complex128",
      "complex128 ERR ERR complex128 complex128 complex128 complex128 ERR ERR \
       complex128 complex128 complex128 complex128 ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR complex128 complex128 complex128 complex128 complex128 \
       complex128 complex128 complex128 complex128" );
  ]

let promote_x64 =
  [
    ( "bool",
      "bool int2 int4 int8 int16 int32 int64 uint2 uint4 uint8 uint16 uint32 \
       uint64 float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 \
       float8_e4m3 float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz \
       float8_e5m2 float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 \
       float64 complex64 complex128 int64 float64 complex128" );
    ( "int2",
      "int2 int2 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR int2 ERR ERR" );
    ( "int4",
      "int4 ERR int4 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR int4 ERR ERR" );
    ( "int8",
      "int8 ERR ERR int8 int16 int32 int64 ERR ERR int16 int32 int64 float64 \
       float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 float8_e4m3 \
       float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz float8_e5m2 \
       float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 float64 \
       complex64 complex128 int8 float64 complex128" );
    ( "int16",
      "int16 ERR ERR int16 int16 int32 int64 ERR ERR int16 int32 int64 float64 \
       float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 float8_e4m3 \
       float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz float8_e5m2 \
       float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 float64 \
       complex64 complex128 int16 float64 complex128" );
    ( "int32",
      "int32 ERR ERR int32 int32 int32 int64 ERR ERR int32 int32 int64 float64 \
       float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 float8_e4m3 \
       float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz float8_e5m2 \
       float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 float64 \
       complex64 complex128 int32 float64 complex128" );
    ( "int64",
      "int64 ERR ERR int64 int64 int64 int64 ERR ERR int64 int64 int64 float64 \
       float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 float8_e4m3 \
       float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz float8_e5m2 \
       float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 float64 \
       complex64 complex128 int64 float64 complex128" );
    ( "uint2",
      "uint2 ERR ERR ERR ERR ERR ERR uint2 ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR uint2 ERR ERR" );
    ( "uint4",
      "uint4 ERR ERR ERR ERR ERR ERR ERR uint4 ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR uint4 ERR ERR" );
    ( "uint8",
      "uint8 ERR ERR int16 int16 int32 int64 ERR ERR uint8 uint16 uint32 \
       uint64 float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 \
       float8_e4m3 float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz \
       float8_e5m2 float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 \
       float64 complex64 complex128 uint8 float64 complex128" );
    ( "uint16",
      "uint16 ERR ERR int32 int32 int32 int64 ERR ERR uint16 uint16 uint32 \
       uint64 float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 \
       float8_e4m3 float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz \
       float8_e5m2 float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 \
       float64 complex64 complex128 uint16 float64 complex128" );
    ( "uint32",
      "uint32 ERR ERR int64 int64 int64 int64 ERR ERR uint32 uint32 uint32 \
       uint64 float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 \
       float8_e4m3 float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz \
       float8_e5m2 float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 \
       float64 complex64 complex128 uint32 float64 complex128" );
    ( "uint64",
      "uint64 ERR ERR float64 float64 float64 float64 ERR ERR uint64 uint64 \
       uint64 uint64 float4_e2m1fn float6_e2m3fn float6_e3m2fn float8_e3m4 \
       float8_e4m3 float8_e4m3b11fnuz float8_e4m3fn float8_e4m3fnuz \
       float8_e5m2 float8_e5m2fnuz float8_e8m0fnu float16 bfloat16 float32 \
       float64 complex64 complex128 uint64 float64 complex128" );
    ( "float4_e2m1fn",
      "float4_e2m1fn ERR ERR float4_e2m1fn float4_e2m1fn float4_e2m1fn \
       float4_e2m1fn ERR ERR float4_e2m1fn float4_e2m1fn float4_e2m1fn \
       float4_e2m1fn float4_e2m1fn ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR float4_e2m1fn float4_e2m1fn ERR" );
    ( "float6_e2m3fn",
      "float6_e2m3fn ERR ERR float6_e2m3fn float6_e2m3fn float6_e2m3fn \
       float6_e2m3fn ERR ERR float6_e2m3fn float6_e2m3fn float6_e2m3fn \
       float6_e2m3fn ERR float6_e2m3fn ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR float6_e2m3fn float6_e2m3fn ERR" );
    ( "float6_e3m2fn",
      "float6_e3m2fn ERR ERR float6_e3m2fn float6_e3m2fn float6_e3m2fn \
       float6_e3m2fn ERR ERR float6_e3m2fn float6_e3m2fn float6_e3m2fn \
       float6_e3m2fn ERR ERR float6_e3m2fn ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR float6_e3m2fn float6_e3m2fn ERR" );
    ( "float8_e3m4",
      "float8_e3m4 ERR ERR float8_e3m4 float8_e3m4 float8_e3m4 float8_e3m4 ERR \
       ERR float8_e3m4 float8_e3m4 float8_e3m4 float8_e3m4 ERR ERR ERR \
       float8_e3m4 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       float8_e3m4 float8_e3m4 ERR" );
    ( "float8_e4m3",
      "float8_e4m3 ERR ERR float8_e4m3 float8_e4m3 float8_e4m3 float8_e4m3 ERR \
       ERR float8_e4m3 float8_e4m3 float8_e4m3 float8_e4m3 ERR ERR ERR ERR \
       float8_e4m3 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR float8_e4m3 \
       float8_e4m3 ERR" );
    ( "float8_e4m3b11fnuz",
      "float8_e4m3b11fnuz ERR ERR float8_e4m3b11fnuz float8_e4m3b11fnuz \
       float8_e4m3b11fnuz float8_e4m3b11fnuz ERR ERR float8_e4m3b11fnuz \
       float8_e4m3b11fnuz float8_e4m3b11fnuz float8_e4m3b11fnuz ERR ERR ERR \
       ERR ERR float8_e4m3b11fnuz ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       float8_e4m3b11fnuz float8_e4m3b11fnuz ERR" );
    ( "float8_e4m3fn",
      "float8_e4m3fn ERR ERR float8_e4m3fn float8_e4m3fn float8_e4m3fn \
       float8_e4m3fn ERR ERR float8_e4m3fn float8_e4m3fn float8_e4m3fn \
       float8_e4m3fn ERR ERR ERR ERR ERR ERR float8_e4m3fn ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR ERR float8_e4m3fn float8_e4m3fn ERR" );
    ( "float8_e4m3fnuz",
      "float8_e4m3fnuz ERR ERR float8_e4m3fnuz float8_e4m3fnuz float8_e4m3fnuz \
       float8_e4m3fnuz ERR ERR float8_e4m3fnuz float8_e4m3fnuz float8_e4m3fnuz \
       float8_e4m3fnuz ERR ERR ERR ERR ERR ERR ERR float8_e4m3fnuz ERR ERR ERR \
       ERR ERR ERR ERR ERR ERR float8_e4m3fnuz float8_e4m3fnuz ERR" );
    ( "float8_e5m2",
      "float8_e5m2 ERR ERR float8_e5m2 float8_e5m2 float8_e5m2 float8_e5m2 ERR \
       ERR float8_e5m2 float8_e5m2 float8_e5m2 float8_e5m2 ERR ERR ERR ERR ERR \
       ERR ERR ERR float8_e5m2 ERR ERR ERR ERR ERR ERR ERR ERR float8_e5m2 \
       float8_e5m2 ERR" );
    ( "float8_e5m2fnuz",
      "float8_e5m2fnuz ERR ERR float8_e5m2fnuz float8_e5m2fnuz float8_e5m2fnuz \
       float8_e5m2fnuz ERR ERR float8_e5m2fnuz float8_e5m2fnuz float8_e5m2fnuz \
       float8_e5m2fnuz ERR ERR ERR ERR ERR ERR ERR ERR ERR float8_e5m2fnuz ERR \
       ERR ERR ERR ERR ERR ERR float8_e5m2fnuz float8_e5m2fnuz ERR" );
    ( "float8_e8m0fnu",
      "float8_e8m0fnu ERR ERR float8_e8m0fnu float8_e8m0fnu float8_e8m0fnu \
       float8_e8m0fnu ERR ERR float8_e8m0fnu float8_e8m0fnu float8_e8m0fnu \
       float8_e8m0fnu ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR float8_e8m0fnu \
       ERR ERR ERR ERR ERR ERR float8_e8m0fnu float8_e8m0fnu ERR" );
    ( "float16",
      "float16 ERR ERR float16 float16 float16 float16 ERR ERR float16 float16 \
       float16 float16 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR float16 \
       float32 float32 float64 complex64 complex128 float16 float16 complex64"
    );
    ( "bfloat16",
      "bfloat16 ERR ERR bfloat16 bfloat16 bfloat16 bfloat16 ERR ERR bfloat16 \
       bfloat16 bfloat16 bfloat16 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR \
       float32 bfloat16 float32 float64 complex64 complex128 bfloat16 bfloat16 \
       complex64" );
    ( "float32",
      "float32 ERR ERR float32 float32 float32 float32 ERR ERR float32 float32 \
       float32 float32 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR float32 \
       float32 float32 float64 complex64 complex128 float32 float32 complex64"
    );
    ( "float64",
      "float64 ERR ERR float64 float64 float64 float64 ERR ERR float64 float64 \
       float64 float64 ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR ERR float64 \
       float64 float64 float64 complex128 complex128 float64 float64 \
       complex128" );
    ( "complex64",
      "complex64 ERR ERR complex64 complex64 complex64 complex64 ERR ERR \
       complex64 complex64 complex64 complex64 ERR ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR complex64 complex64 complex64 complex128 complex64 \
       complex128 complex64 complex64 complex64" );
    ( "complex128",
      "complex128 ERR ERR complex128 complex128 complex128 complex128 ERR ERR \
       complex128 complex128 complex128 complex128 ERR ERR ERR ERR ERR ERR ERR \
       ERR ERR ERR ERR complex128 complex128 complex128 complex128 complex128 \
       complex128 complex128 complex128 complex128" );
  ]

let finfo_table =
  [
    ( D.F4_e2m1fn,
      4,
      1,
      2,
      0,
      0x1.0000000000000p+0,
      0x1.0000000000000p-1,
      0x1.0000000000000p-1,
      0x1.8000000000000p+2,
      -0x1.8000000000000p+2,
      0x1.0000000000000p+0,
      0x1.0000000000000p-1 );
    ( D.F6_e2m3fn,
      6,
      3,
      2,
      0,
      0x1.0000000000000p+0,
      0x1.0000000000000p-3,
      0x1.0000000000000p-3,
      0x1.e000000000000p+2,
      -0x1.e000000000000p+2,
      0x1.0000000000000p+0,
      0x1.0000000000000p-3 );
    ( D.F6_e3m2fn,
      6,
      2,
      3,
      0,
      0x1.0000000000000p+0,
      0x1.0000000000000p-2,
      0x1.0000000000000p-3,
      0x1.c000000000000p+4,
      -0x1.c000000000000p+4,
      0x1.0000000000000p-2,
      0x1.0000000000000p-4 );
    ( D.F8_e3m4,
      8,
      4,
      3,
      1,
      0x1.8000000000000p-4,
      0x1.0000000000000p-4,
      0x1.0000000000000p-5,
      0x1.f000000000000p+3,
      -0x1.f000000000000p+3,
      0x1.0000000000000p-2,
      0x1.0000000000000p-6 );
    ( D.F8_e4m3,
      8,
      3,
      4,
      1,
      0x1.a000000000000p-4,
      0x1.0000000000000p-3,
      0x1.0000000000000p-4,
      0x1.e000000000000p+7,
      -0x1.e000000000000p+7,
      0x1.0000000000000p-6,
      0x1.0000000000000p-9 );
    ( D.F8_e4m3b11fnuz,
      8,
      3,
      4,
      1,
      0x1.a000000000000p-4,
      0x1.0000000000000p-3,
      0x1.0000000000000p-4,
      0x1.e000000000000p+4,
      -0x1.e000000000000p+4,
      0x1.0000000000000p-10,
      0x1.0000000000000p-13 );
    ( D.F8_e4m3fn,
      8,
      3,
      4,
      1,
      0x1.a000000000000p-4,
      0x1.0000000000000p-3,
      0x1.0000000000000p-4,
      0x1.c000000000000p+8,
      -0x1.c000000000000p+8,
      0x1.0000000000000p-6,
      0x1.0000000000000p-9 );
    ( D.F8_e4m3fnuz,
      8,
      3,
      4,
      1,
      0x1.a000000000000p-4,
      0x1.0000000000000p-3,
      0x1.0000000000000p-4,
      0x1.e000000000000p+7,
      -0x1.e000000000000p+7,
      0x1.0000000000000p-7,
      0x1.0000000000000p-10 );
    ( D.F8_e5m2,
      8,
      2,
      5,
      1,
      0x1.8000000000000p-4,
      0x1.0000000000000p-2,
      0x1.0000000000000p-3,
      0x1.c000000000000p+15,
      -0x1.c000000000000p+15,
      0x1.0000000000000p-14,
      0x1.0000000000000p-16 );
    ( D.F8_e5m2fnuz,
      8,
      2,
      5,
      1,
      0x1.8000000000000p-4,
      0x1.0000000000000p-2,
      0x1.0000000000000p-3,
      0x1.c000000000000p+15,
      -0x1.c000000000000p+15,
      0x1.0000000000000p-15,
      0x1.0000000000000p-17 );
    ( D.F8_e8m0fnu,
      8,
      0,
      8,
      1,
      0x1.0000000000000p-3,
      0x1.0000000000000p+0,
      0x1.0000000000000p-1,
      0x1.0000000000000p+127,
      0x1.0000000000000p-127,
      0x1.0000000000000p-127,
      0x1.0000000000000p-127 );
  ]

let iinfo_table =
  [
    (D.I2, -2.0, 1.0, 2);
    (D.I4, -8.0, 7.0, 4);
    (D.Uint2, 0.0, 3.0, 2);
    (D.Uint4, 0.0, 15.0, 4);
  ]

let bits_and_scalar =
  [
    (D.Bool, 8, Dt.Scalar_bool, D.F32);
    (D.I2, 2, Dt.Scalar_int, D.I2);
    (D.I4, 4, Dt.Scalar_int, D.F32);
    (D.I8, 8, Dt.Scalar_int, D.F32);
    (D.I16, 16, Dt.Scalar_int, D.F32);
    (D.I32, 32, Dt.Scalar_int, D.F32);
    (D.I64, 64, Dt.Scalar_int, D.F64);
    (D.Uint2, 2, Dt.Scalar_int, D.Uint2);
    (D.Uint4, 4, Dt.Scalar_int, D.F32);
    (D.Uint8, 8, Dt.Scalar_int, D.F32);
    (D.Uint16, 16, Dt.Scalar_int, D.F32);
    (D.Uint32, 32, Dt.Scalar_int, D.F32);
    (D.Uint64, 64, Dt.Scalar_int, D.F64);
    (D.F4_e2m1fn, 4, Dt.Scalar_float, D.F4_e2m1fn);
    (D.F6_e2m3fn, 6, Dt.Scalar_float, D.F6_e2m3fn);
    (D.F6_e3m2fn, 6, Dt.Scalar_float, D.F6_e3m2fn);
    (D.F8_e3m4, 8, Dt.Scalar_float, D.F8_e3m4);
    (D.F8_e4m3, 8, Dt.Scalar_float, D.F8_e4m3);
    (D.F8_e4m3b11fnuz, 8, Dt.Scalar_float, D.F8_e4m3b11fnuz);
    (D.F8_e4m3fn, 8, Dt.Scalar_float, D.F8_e4m3fn);
    (D.F8_e4m3fnuz, 8, Dt.Scalar_float, D.F8_e4m3fnuz);
    (D.F8_e5m2, 8, Dt.Scalar_float, D.F8_e5m2);
    (D.F8_e5m2fnuz, 8, Dt.Scalar_float, D.F8_e5m2fnuz);
    (D.F8_e8m0fnu, 8, Dt.Scalar_float, D.F8_e8m0fnu);
    (D.F16, 16, Dt.Scalar_float, D.F16);
    (D.Bf16, 16, Dt.Scalar_float, D.Bf16);
    (D.F32, 32, Dt.Scalar_float, D.F32);
    (D.F64, 64, Dt.Scalar_float, D.F64);
    (D.Complex64, 64, Dt.Scalar_complex, D.Complex64);
    (D.Complex128, 128, Dt.Scalar_complex, D.Complex128);
  ]

let short_names =
  [
    (D.Bool, "bool");
    (D.I2, "i2");
    (D.I4, "i4");
    (D.I8, "i8");
    (D.I16, "i16");
    (D.I32, "i32");
    (D.I64, "i64");
    (D.Uint2, "u2");
    (D.Uint4, "u4");
    (D.Uint8, "u8");
    (D.Uint16, "u16");
    (D.Uint32, "u32");
    (D.Uint64, "u64");
    (D.F4_e2m1fn, "f4_e2m1fn");
    (D.F6_e2m3fn, "f6_e2m3fn");
    (D.F6_e3m2fn, "f6_e3m2fn");
    (D.F8_e3m4, "f8_e3m4");
    (D.F8_e4m3, "f8_e4m3");
    (D.F8_e4m3b11fnuz, "f8_e4m3b11fnuz");
    (D.F8_e4m3fn, "f8_e4m3fn");
    (D.F8_e4m3fnuz, "f8_e4m3fnuz");
    (D.F8_e5m2, "f8_e5m2");
    (D.F8_e5m2fnuz, "f8_e5m2fnuz");
    (D.F8_e8m0fnu, "f8_e8m0fnu");
    (D.F16, "f16");
    (D.Bf16, "bf16");
    (D.F32, "f32");
    (D.F64, "f64");
    (D.Complex64, "c64");
    (D.Complex128, "c128");
  ]

let isdtype_table =
  [
    (D.Bool, [ true; false; false; false; false; false; false ]);
    (D.I2, [ false; true; false; true; false; false; true ]);
    (D.I4, [ false; true; false; true; false; false; true ]);
    (D.I8, [ false; true; false; true; false; false; true ]);
    (D.I16, [ false; true; false; true; false; false; true ]);
    (D.I32, [ false; true; false; true; false; false; true ]);
    (D.I64, [ false; true; false; true; false; false; true ]);
    (D.Uint2, [ false; false; true; true; false; false; true ]);
    (D.Uint4, [ false; false; true; true; false; false; true ]);
    (D.Uint8, [ false; false; true; true; false; false; true ]);
    (D.Uint16, [ false; false; true; true; false; false; true ]);
    (D.Uint32, [ false; false; true; true; false; false; true ]);
    (D.Uint64, [ false; false; true; true; false; false; true ]);
    (D.F4_e2m1fn, [ false; false; false; false; true; false; true ]);
    (D.F6_e2m3fn, [ false; false; false; false; true; false; true ]);
    (D.F6_e3m2fn, [ false; false; false; false; true; false; true ]);
    (D.F8_e3m4, [ false; false; false; false; true; false; true ]);
    (D.F8_e4m3, [ false; false; false; false; true; false; true ]);
    (D.F8_e4m3b11fnuz, [ false; false; false; false; true; false; true ]);
    (D.F8_e4m3fn, [ false; false; false; false; true; false; true ]);
    (D.F8_e4m3fnuz, [ false; false; false; false; true; false; true ]);
    (D.F8_e5m2, [ false; false; false; false; true; false; true ]);
    (D.F8_e5m2fnuz, [ false; false; false; false; true; false; true ]);
    (D.F8_e8m0fnu, [ false; false; false; false; true; false; true ]);
    (D.F16, [ false; false; false; false; true; false; true ]);
    (D.Bf16, [ false; false; false; false; true; false; true ]);
    (D.F32, [ false; false; false; false; true; false; true ]);
    (D.F64, [ false; false; false; false; true; false; true ]);
    (D.Complex64, [ false; false; false; false; false; true; true ]);
    (D.Complex128, [ false; false; false; false; false; true; true ]);
  ]

let kinds =
  [
    "bool";
    "signed integer";
    "unsigned integer";
    "integral";
    "real floating";
    "complex floating";
    "numeric";
  ]

let issub_table =
  [
    ( D.Bool,
      [
        true;
        true;
        false;
        false;
        false;
        false;
        false;
        false;
        false;
        false;
        false;
      ] );
    ( D.I2,
      [
        true; false; true; true; true; false; false; false; false; false; false;
      ] );
    ( D.I4,
      [
        true; false; true; true; true; false; false; false; false; false; false;
      ] );
    ( D.I8,
      [
        true; false; true; true; true; false; false; false; false; false; false;
      ] );
    ( D.I16,
      [
        true; false; true; true; true; false; false; false; false; false; false;
      ] );
    ( D.I32,
      [
        true; false; true; true; true; false; false; false; false; false; false;
      ] );
    ( D.I64,
      [
        true; false; true; true; true; false; false; false; false; false; false;
      ] );
    ( D.Uint2,
      [
        true; false; true; true; false; true; false; false; false; false; false;
      ] );
    ( D.Uint4,
      [
        true; false; true; true; false; true; false; false; false; false; false;
      ] );
    ( D.Uint8,
      [
        true; false; true; true; false; true; false; false; false; false; false;
      ] );
    ( D.Uint16,
      [
        true; false; true; true; false; true; false; false; false; false; false;
      ] );
    ( D.Uint32,
      [
        true; false; true; true; false; true; false; false; false; false; false;
      ] );
    ( D.Uint64,
      [
        true; false; true; true; false; true; false; false; false; false; false;
      ] );
    ( D.F4_e2m1fn,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F6_e2m3fn,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F6_e3m2fn,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F8_e3m4,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F8_e4m3,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F8_e4m3b11fnuz,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F8_e4m3fn,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F8_e4m3fnuz,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F8_e5m2,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F8_e5m2fnuz,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F8_e8m0fnu,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F16,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.Bf16,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F32,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.F64,
      [
        true; false; true; false; false; false; true; true; false; false; false;
      ] );
    ( D.Complex64,
      [
        true; false; true; false; false; false; true; false; true; false; false;
      ] );
    ( D.Complex128,
      [
        true; false; true; false; false; false; true; false; true; false; false;
      ] );
  ]

let issub_kinds =
  [
    `Generic;
    `Bool;
    `Number;
    `Integer;
    `Signed;
    `Unsigned;
    `Inexact;
    `Floating;
    `Complex;
    `Extended;
    `Prng_key;
  ]

let class_kinds =
  [
    NL.Generic;
    NL.Number;
    NL.Integer;
    NL.Signedinteger;
    NL.unsignedinteger;
    NL.Inexact;
    NL.Floating;
    NL.complexfloating;
    NL.extended;
  ]

let class_table =
  [
    (D.Bool, [ true; false; false; false; false; false; false; false; false ]);
    (D.I2, [ true; true; true; true; false; false; false; false; false ]);
    (D.I4, [ true; true; true; true; false; false; false; false; false ]);
    (D.I8, [ true; true; true; true; false; false; false; false; false ]);
    (D.I16, [ true; true; true; true; false; false; false; false; false ]);
    (D.I32, [ true; true; true; true; false; false; false; false; false ]);
    (D.I64, [ true; true; true; true; false; false; false; false; false ]);
    (D.Uint2, [ true; true; true; false; true; false; false; false; false ]);
    (D.Uint4, [ true; true; true; false; true; false; false; false; false ]);
    (D.Uint8, [ true; true; true; false; true; false; false; false; false ]);
    (D.Uint16, [ true; true; true; false; true; false; false; false; false ]);
    (D.Uint32, [ true; true; true; false; true; false; false; false; false ]);
    (D.Uint64, [ true; true; true; false; true; false; false; false; false ]);
    (D.F4_e2m1fn, [ true; true; false; false; false; true; true; false; false ]);
    (D.F6_e2m3fn, [ true; true; false; false; false; true; true; false; false ]);
    (D.F6_e3m2fn, [ true; true; false; false; false; true; true; false; false ]);
    (D.F8_e3m4, [ true; true; false; false; false; true; true; false; false ]);
    (D.F8_e4m3, [ true; true; false; false; false; true; true; false; false ]);
    ( D.F8_e4m3b11fnuz,
      [ true; true; false; false; false; true; true; false; false ] );
    (D.F8_e4m3fn, [ true; true; false; false; false; true; true; false; false ]);
    ( D.F8_e4m3fnuz,
      [ true; true; false; false; false; true; true; false; false ] );
    (D.F8_e5m2, [ true; true; false; false; false; true; true; false; false ]);
    ( D.F8_e5m2fnuz,
      [ true; true; false; false; false; true; true; false; false ] );
    (D.F8_e8m0fnu, [ true; true; false; false; false; true; true; false; false ]);
    (D.F16, [ true; true; false; false; false; true; true; false; false ]);
    (D.Bf16, [ true; true; false; false; false; true; true; false; false ]);
    (D.F32, [ true; true; false; false; false; true; true; false; false ]);
    (D.F64, [ true; true; false; false; false; true; true; false; false ]);
    (D.Complex64, [ true; true; false; false; false; true; false; true; false ]);
    (D.Complex128, [ true; true; false; false; false; true; false; true; false ]);
  ]

let class_pairs =
  [
    (NL.complexfloating, NL.Inexact, true);
    (NL.complexfloating, NL.Number, true);
    (NL.complexfloating, NL.Floating, false);
    (NL.unsignedinteger, NL.Integer, true);
    (NL.unsignedinteger, NL.Signedinteger, false);
    (NL.unsignedinteger, NL.Number, true);
    (NL.extended, NL.Generic, true);
    (NL.extended, NL.Number, false);
    (NL.Prng_key_class, NL.extended, true);
    (NL.Generic, NL.extended, false);
  ]

let float0_class =
  [ true; false; false; false; false; false; false; false; false ]

let round_inputs =
  [|
    0x0.0p+0;
    -0x0.0p+0;
    0x1.4484bfeebc2a0p-100;
    -0x1.4484bfeebc2a0p-100;
    0x1.3333333333333p-2;
    -0x1.3333333333333p-2;
    0x1.0000000000000p+0;
    0x1.8000000000000p+0;
    -0x1.4cccccccccccdp+0;
    0x1.4000000000000p+1;
    0x1.8000000000000p+1;
    0x1.c000000000000p+1;
    0x1.4000000000000p+2;
    0x1.8000000000000p+2;
    0x1.c000000000000p+2;
    0x1.0000000000000p+3;
    0x1.4000000000000p+3;
    0x1.8000000000000p+3;
    0x1.f800000000000p+3;
    0x1.f000000000000p+4;
    0x1.9000000000000p+6;
    0x1.2c00000000000p+8;
    0x1.d000000000000p+8;
    0x1.e000000000000p+8;
    0x1.3880000000000p+13;
    0x1.d4c0000000000p+15;
    0x1.e848000000000p+19;
    0x1.c363cbf21f28ap+127;
    Float.infinity;
    Float.neg_infinity;
    Float.nan;
    0x1.100000006df38p+0;
    0x1.400000055e63cp+0;
    0x1.0000000000000p-127;
    0x1.4000000000000p-127;
    0x1.0000000000000p-149;
    0x1.0000000000000p-150;
    0x1.56e1fc2f8f359p-997;
  |]

let round_table =
  [
    ( D.F4_e2m1fn,
      [|
        0x0.0p+0;
        -0x0.0p+0;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.0000000000000p-1;
        -0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.8000000000000p+0;
        0x1.0000000000000p+1;
        0x1.8000000000000p+1;
        0x1.0000000000000p+2;
        0x1.0000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        -0x1.8000000000000p+2;
        -0x0.0p+0;
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
      |],
      [|
        0;
        8;
        0;
        8;
        1;
        9;
        2;
        3;
        11;
        4;
        5;
        6;
        6;
        7;
        7;
        7;
        7;
        7;
        7;
        7;
        7;
        7;
        7;
        7;
        7;
        7;
        7;
        7;
        7;
        15;
        8;
        2;
        2;
        0;
        0;
        0;
        0;
        0;
      |] );
    ( D.F6_e2m3fn,
      [|
        0x0.0p+0;
        -0x0.0p+0;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.0000000000000p-2;
        -0x1.0000000000000p-2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        0x1.e000000000000p+2;
        -0x1.e000000000000p+2;
        -0x0.0p+0;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
      |],
      [|
        0;
        32;
        0;
        32;
        2;
        34;
        8;
        12;
        42;
        18;
        20;
        22;
        26;
        28;
        30;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        63;
        32;
        8;
        10;
        0;
        0;
        0;
        0;
        0;
      |] );
    ( D.F6_e3m2fn,
      [|
        0x0.0p+0;
        -0x0.0p+0;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.4000000000000p-2;
        -0x1.4000000000000p-2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.8000000000000p+3;
        0x1.0000000000000p+4;
        0x1.c000000000000p+4;
        0x1.c000000000000p+4;
        0x1.c000000000000p+4;
        0x1.c000000000000p+4;
        0x1.c000000000000p+4;
        0x1.c000000000000p+4;
        0x1.c000000000000p+4;
        0x1.c000000000000p+4;
        0x1.c000000000000p+4;
        0x1.c000000000000p+4;
        -0x1.c000000000000p+4;
        -0x0.0p+0;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
      |],
      [|
        0;
        32;
        0;
        32;
        5;
        37;
        12;
        14;
        45;
        17;
        18;
        19;
        21;
        22;
        23;
        24;
        25;
        26;
        28;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        31;
        63;
        32;
        12;
        13;
        0;
        0;
        0;
        0;
        0;
      |] );
    ( D.F8_e3m4,
      [|
        0x0.0p+0;
        -0x0.0p+0;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3000000000000p-2;
        -0x1.3000000000000p-2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.5000000000000p+0;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.8000000000000p+3;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.neg_infinity;
        Float.nan;
        0x1.1000000000000p+0;
        0x1.4000000000000p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
      |],
      [|
        0;
        128;
        0;
        128;
        19;
        147;
        48;
        56;
        181;
        68;
        72;
        76;
        84;
        88;
        92;
        96;
        100;
        104;
        112;
        112;
        112;
        112;
        112;
        112;
        112;
        112;
        112;
        112;
        112;
        240;
        120;
        49;
        52;
        0;
        0;
        0;
        0;
        0;
      |] );
    ( D.F8_e4m3,
      [|
        0x0.0p+0;
        -0x0.0p+0;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.4000000000000p-2;
        -0x1.4000000000000p-2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.8000000000000p+3;
        0x1.0000000000000p+4;
        0x1.0000000000000p+5;
        0x1.8000000000000p+6;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.neg_infinity;
        Float.nan;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
      |],
      [|
        0;
        128;
        0;
        128;
        42;
        170;
        56;
        60;
        186;
        66;
        68;
        70;
        74;
        76;
        78;
        80;
        82;
        84;
        88;
        96;
        108;
        120;
        120;
        120;
        120;
        120;
        120;
        120;
        120;
        248;
        124;
        56;
        58;
        0;
        0;
        0;
        0;
        0;
      |] );
    ( D.F8_e4m3b11fnuz,
      [|
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x1.4000000000000p-2;
        -0x1.4000000000000p-2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.8000000000000p+3;
        0x1.0000000000000p+4;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
      |],
      [|
        0;
        0;
        0;
        0;
        74;
        202;
        88;
        92;
        218;
        98;
        100;
        102;
        106;
        108;
        110;
        112;
        114;
        116;
        120;
        128;
        128;
        128;
        128;
        128;
        128;
        128;
        128;
        128;
        128;
        128;
        128;
        88;
        90;
        0;
        0;
        0;
        0;
        0;
      |] );
    ( D.F8_e4m3fn,
      [|
        0x0.0p+0;
        -0x0.0p+0;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.4000000000000p-2;
        -0x1.4000000000000p-2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.8000000000000p+3;
        0x1.0000000000000p+4;
        0x1.0000000000000p+5;
        0x1.8000000000000p+6;
        0x1.2000000000000p+8;
        0x1.c000000000000p+8;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
      |],
      [|
        0;
        128;
        0;
        128;
        42;
        170;
        56;
        60;
        186;
        66;
        68;
        70;
        74;
        76;
        78;
        80;
        82;
        84;
        88;
        96;
        108;
        121;
        126;
        127;
        127;
        127;
        127;
        127;
        127;
        255;
        127;
        56;
        58;
        0;
        0;
        0;
        0;
        0;
      |] );
    ( D.F8_e4m3fnuz,
      [|
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x1.4000000000000p-2;
        -0x1.4000000000000p-2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.8000000000000p+3;
        0x1.0000000000000p+4;
        0x1.0000000000000p+5;
        0x1.8000000000000p+6;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
      |],
      [|
        0;
        0;
        0;
        0;
        50;
        178;
        64;
        68;
        194;
        74;
        76;
        78;
        82;
        84;
        86;
        88;
        90;
        92;
        96;
        104;
        116;
        128;
        128;
        128;
        128;
        128;
        128;
        128;
        128;
        128;
        128;
        64;
        66;
        0;
        0;
        0;
        0;
        0;
      |] );
    ( D.F8_e5m2,
      [|
        0x0.0p+0;
        -0x0.0p+0;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.4000000000000p-2;
        -0x1.4000000000000p-2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.8000000000000p+3;
        0x1.0000000000000p+4;
        0x1.0000000000000p+5;
        0x1.8000000000000p+6;
        0x1.4000000000000p+8;
        0x1.c000000000000p+8;
        0x1.0000000000000p+9;
        0x1.4000000000000p+13;
        0x1.c000000000000p+15;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.neg_infinity;
        Float.nan;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
      |],
      [|
        0;
        128;
        0;
        128;
        53;
        181;
        60;
        62;
        189;
        65;
        66;
        67;
        69;
        70;
        71;
        72;
        73;
        74;
        76;
        80;
        86;
        93;
        95;
        96;
        113;
        123;
        124;
        124;
        124;
        252;
        126;
        60;
        61;
        0;
        0;
        0;
        0;
        0;
      |] );
    ( D.F8_e5m2fnuz,
      [|
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x1.4000000000000p-2;
        -0x1.4000000000000p-2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.8000000000000p+3;
        0x1.0000000000000p+4;
        0x1.0000000000000p+5;
        0x1.8000000000000p+6;
        0x1.4000000000000p+8;
        0x1.c000000000000p+8;
        0x1.0000000000000p+9;
        0x1.4000000000000p+13;
        0x1.c000000000000p+15;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
      |],
      [|
        0;
        0;
        0;
        0;
        57;
        185;
        64;
        66;
        193;
        69;
        70;
        71;
        73;
        74;
        75;
        76;
        77;
        78;
        80;
        84;
        90;
        97;
        99;
        100;
        117;
        127;
        128;
        128;
        128;
        128;
        128;
        64;
        65;
        0;
        0;
        0;
        0;
        0;
      |] );
    ( D.F8_e8m0fnu,
      [|
        Float.nan;
        Float.nan;
        0x1.0000000000000p-100;
        Float.nan;
        0x1.0000000000000p-2;
        Float.nan;
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
        Float.nan;
        0x1.0000000000000p+1;
        0x1.0000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+3;
        0x1.0000000000000p+3;
        0x1.0000000000000p+3;
        0x1.0000000000000p+3;
        0x1.0000000000000p+4;
        0x1.0000000000000p+4;
        0x1.0000000000000p+5;
        0x1.0000000000000p+7;
        0x1.0000000000000p+8;
        0x1.0000000000000p+9;
        0x1.0000000000000p+9;
        0x1.0000000000000p+13;
        0x1.0000000000000p+16;
        0x1.0000000000000p+20;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x1.0000000000000p-127;
        0x1.0000000000000p-126;
        0x1.0000000000000p-127;
        Float.nan;
        Float.nan;
      |],
      [|
        255;
        255;
        27;
        255;
        125;
        255;
        127;
        128;
        255;
        128;
        129;
        129;
        129;
        130;
        130;
        130;
        130;
        131;
        131;
        132;
        134;
        135;
        136;
        136;
        140;
        143;
        147;
        255;
        255;
        255;
        255;
        127;
        127;
        0;
        1;
        0;
        255;
        255;
      |] );
  ]

let decode_table =
  [
    ( D.F4_e2m1fn,
      [|
        0x0.0p+0;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        0x1.0000000000000p+1;
        0x1.8000000000000p+1;
        0x1.0000000000000p+2;
        0x1.8000000000000p+2;
        -0x0.0p+0;
        -0x1.0000000000000p-1;
        -0x1.0000000000000p+0;
        -0x1.8000000000000p+0;
        -0x1.0000000000000p+1;
        -0x1.8000000000000p+1;
        -0x1.0000000000000p+2;
        -0x1.8000000000000p+2;
      |] );
    ( D.F6_e2m3fn,
      [|
        0x0.0p+0;
        0x1.0000000000000p-3;
        0x1.0000000000000p-2;
        0x1.8000000000000p-2;
        0x1.0000000000000p-1;
        0x1.4000000000000p-1;
        0x1.8000000000000p-1;
        0x1.c000000000000p-1;
        0x1.0000000000000p+0;
        0x1.2000000000000p+0;
        0x1.4000000000000p+0;
        0x1.6000000000000p+0;
        0x1.8000000000000p+0;
        0x1.a000000000000p+0;
        0x1.c000000000000p+0;
        0x1.e000000000000p+0;
        0x1.0000000000000p+1;
        0x1.2000000000000p+1;
        0x1.4000000000000p+1;
        0x1.6000000000000p+1;
        0x1.8000000000000p+1;
        0x1.a000000000000p+1;
        0x1.c000000000000p+1;
        0x1.e000000000000p+1;
        0x1.0000000000000p+2;
        0x1.2000000000000p+2;
        0x1.4000000000000p+2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+2;
        0x1.a000000000000p+2;
        0x1.c000000000000p+2;
        0x1.e000000000000p+2;
        -0x0.0p+0;
        -0x1.0000000000000p-3;
        -0x1.0000000000000p-2;
        -0x1.8000000000000p-2;
        -0x1.0000000000000p-1;
        -0x1.4000000000000p-1;
        -0x1.8000000000000p-1;
        -0x1.c000000000000p-1;
        -0x1.0000000000000p+0;
        -0x1.2000000000000p+0;
        -0x1.4000000000000p+0;
        -0x1.6000000000000p+0;
        -0x1.8000000000000p+0;
        -0x1.a000000000000p+0;
        -0x1.c000000000000p+0;
        -0x1.e000000000000p+0;
        -0x1.0000000000000p+1;
        -0x1.2000000000000p+1;
        -0x1.4000000000000p+1;
        -0x1.6000000000000p+1;
        -0x1.8000000000000p+1;
        -0x1.a000000000000p+1;
        -0x1.c000000000000p+1;
        -0x1.e000000000000p+1;
        -0x1.0000000000000p+2;
        -0x1.2000000000000p+2;
        -0x1.4000000000000p+2;
        -0x1.6000000000000p+2;
        -0x1.8000000000000p+2;
        -0x1.a000000000000p+2;
        -0x1.c000000000000p+2;
        -0x1.e000000000000p+2;
      |] );
    ( D.F6_e3m2fn,
      [|
        0x0.0p+0;
        0x1.0000000000000p-4;
        0x1.0000000000000p-3;
        0x1.8000000000000p-3;
        0x1.0000000000000p-2;
        0x1.4000000000000p-2;
        0x1.8000000000000p-2;
        0x1.c000000000000p-2;
        0x1.0000000000000p-1;
        0x1.4000000000000p-1;
        0x1.8000000000000p-1;
        0x1.c000000000000p-1;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x1.8000000000000p+0;
        0x1.c000000000000p+0;
        0x1.0000000000000p+1;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.0000000000000p+2;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.8000000000000p+3;
        0x1.c000000000000p+3;
        0x1.0000000000000p+4;
        0x1.4000000000000p+4;
        0x1.8000000000000p+4;
        0x1.c000000000000p+4;
        -0x0.0p+0;
        -0x1.0000000000000p-4;
        -0x1.0000000000000p-3;
        -0x1.8000000000000p-3;
        -0x1.0000000000000p-2;
        -0x1.4000000000000p-2;
        -0x1.8000000000000p-2;
        -0x1.c000000000000p-2;
        -0x1.0000000000000p-1;
        -0x1.4000000000000p-1;
        -0x1.8000000000000p-1;
        -0x1.c000000000000p-1;
        -0x1.0000000000000p+0;
        -0x1.4000000000000p+0;
        -0x1.8000000000000p+0;
        -0x1.c000000000000p+0;
        -0x1.0000000000000p+1;
        -0x1.4000000000000p+1;
        -0x1.8000000000000p+1;
        -0x1.c000000000000p+1;
        -0x1.0000000000000p+2;
        -0x1.4000000000000p+2;
        -0x1.8000000000000p+2;
        -0x1.c000000000000p+2;
        -0x1.0000000000000p+3;
        -0x1.4000000000000p+3;
        -0x1.8000000000000p+3;
        -0x1.c000000000000p+3;
        -0x1.0000000000000p+4;
        -0x1.4000000000000p+4;
        -0x1.8000000000000p+4;
        -0x1.c000000000000p+4;
      |] );
    ( D.F8_e3m4,
      [|
        0x0.0p+0;
        0x1.0000000000000p-6;
        0x1.0000000000000p-5;
        0x1.8000000000000p-5;
        0x1.0000000000000p-4;
        0x1.4000000000000p-4;
        0x1.8000000000000p-4;
        0x1.c000000000000p-4;
        0x1.0000000000000p-3;
        0x1.2000000000000p-3;
        0x1.4000000000000p-3;
        0x1.6000000000000p-3;
        0x1.8000000000000p-3;
        0x1.a000000000000p-3;
        0x1.c000000000000p-3;
        0x1.e000000000000p-3;
        0x1.0000000000000p-2;
        0x1.1000000000000p-2;
        0x1.2000000000000p-2;
        0x1.3000000000000p-2;
        0x1.4000000000000p-2;
        0x1.5000000000000p-2;
        0x1.6000000000000p-2;
        0x1.7000000000000p-2;
        0x1.8000000000000p-2;
        0x1.9000000000000p-2;
        0x1.a000000000000p-2;
        0x1.b000000000000p-2;
        0x1.c000000000000p-2;
        0x1.d000000000000p-2;
        0x1.e000000000000p-2;
        0x1.f000000000000p-2;
        0x1.0000000000000p-1;
        0x1.1000000000000p-1;
        0x1.2000000000000p-1;
        0x1.3000000000000p-1;
        0x1.4000000000000p-1;
        0x1.5000000000000p-1;
        0x1.6000000000000p-1;
        0x1.7000000000000p-1;
        0x1.8000000000000p-1;
        0x1.9000000000000p-1;
        0x1.a000000000000p-1;
        0x1.b000000000000p-1;
        0x1.c000000000000p-1;
        0x1.d000000000000p-1;
        0x1.e000000000000p-1;
        0x1.f000000000000p-1;
        0x1.0000000000000p+0;
        0x1.1000000000000p+0;
        0x1.2000000000000p+0;
        0x1.3000000000000p+0;
        0x1.4000000000000p+0;
        0x1.5000000000000p+0;
        0x1.6000000000000p+0;
        0x1.7000000000000p+0;
        0x1.8000000000000p+0;
        0x1.9000000000000p+0;
        0x1.a000000000000p+0;
        0x1.b000000000000p+0;
        0x1.c000000000000p+0;
        0x1.d000000000000p+0;
        0x1.e000000000000p+0;
        0x1.f000000000000p+0;
        0x1.0000000000000p+1;
        0x1.1000000000000p+1;
        0x1.2000000000000p+1;
        0x1.3000000000000p+1;
        0x1.4000000000000p+1;
        0x1.5000000000000p+1;
        0x1.6000000000000p+1;
        0x1.7000000000000p+1;
        0x1.8000000000000p+1;
        0x1.9000000000000p+1;
        0x1.a000000000000p+1;
        0x1.b000000000000p+1;
        0x1.c000000000000p+1;
        0x1.d000000000000p+1;
        0x1.e000000000000p+1;
        0x1.f000000000000p+1;
        0x1.0000000000000p+2;
        0x1.1000000000000p+2;
        0x1.2000000000000p+2;
        0x1.3000000000000p+2;
        0x1.4000000000000p+2;
        0x1.5000000000000p+2;
        0x1.6000000000000p+2;
        0x1.7000000000000p+2;
        0x1.8000000000000p+2;
        0x1.9000000000000p+2;
        0x1.a000000000000p+2;
        0x1.b000000000000p+2;
        0x1.c000000000000p+2;
        0x1.d000000000000p+2;
        0x1.e000000000000p+2;
        0x1.f000000000000p+2;
        0x1.0000000000000p+3;
        0x1.1000000000000p+3;
        0x1.2000000000000p+3;
        0x1.3000000000000p+3;
        0x1.4000000000000p+3;
        0x1.5000000000000p+3;
        0x1.6000000000000p+3;
        0x1.7000000000000p+3;
        0x1.8000000000000p+3;
        0x1.9000000000000p+3;
        0x1.a000000000000p+3;
        0x1.b000000000000p+3;
        0x1.c000000000000p+3;
        0x1.d000000000000p+3;
        0x1.e000000000000p+3;
        0x1.f000000000000p+3;
        Float.infinity;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        -0x0.0p+0;
        -0x1.0000000000000p-6;
        -0x1.0000000000000p-5;
        -0x1.8000000000000p-5;
        -0x1.0000000000000p-4;
        -0x1.4000000000000p-4;
        -0x1.8000000000000p-4;
        -0x1.c000000000000p-4;
        -0x1.0000000000000p-3;
        -0x1.2000000000000p-3;
        -0x1.4000000000000p-3;
        -0x1.6000000000000p-3;
        -0x1.8000000000000p-3;
        -0x1.a000000000000p-3;
        -0x1.c000000000000p-3;
        -0x1.e000000000000p-3;
        -0x1.0000000000000p-2;
        -0x1.1000000000000p-2;
        -0x1.2000000000000p-2;
        -0x1.3000000000000p-2;
        -0x1.4000000000000p-2;
        -0x1.5000000000000p-2;
        -0x1.6000000000000p-2;
        -0x1.7000000000000p-2;
        -0x1.8000000000000p-2;
        -0x1.9000000000000p-2;
        -0x1.a000000000000p-2;
        -0x1.b000000000000p-2;
        -0x1.c000000000000p-2;
        -0x1.d000000000000p-2;
        -0x1.e000000000000p-2;
        -0x1.f000000000000p-2;
        -0x1.0000000000000p-1;
        -0x1.1000000000000p-1;
        -0x1.2000000000000p-1;
        -0x1.3000000000000p-1;
        -0x1.4000000000000p-1;
        -0x1.5000000000000p-1;
        -0x1.6000000000000p-1;
        -0x1.7000000000000p-1;
        -0x1.8000000000000p-1;
        -0x1.9000000000000p-1;
        -0x1.a000000000000p-1;
        -0x1.b000000000000p-1;
        -0x1.c000000000000p-1;
        -0x1.d000000000000p-1;
        -0x1.e000000000000p-1;
        -0x1.f000000000000p-1;
        -0x1.0000000000000p+0;
        -0x1.1000000000000p+0;
        -0x1.2000000000000p+0;
        -0x1.3000000000000p+0;
        -0x1.4000000000000p+0;
        -0x1.5000000000000p+0;
        -0x1.6000000000000p+0;
        -0x1.7000000000000p+0;
        -0x1.8000000000000p+0;
        -0x1.9000000000000p+0;
        -0x1.a000000000000p+0;
        -0x1.b000000000000p+0;
        -0x1.c000000000000p+0;
        -0x1.d000000000000p+0;
        -0x1.e000000000000p+0;
        -0x1.f000000000000p+0;
        -0x1.0000000000000p+1;
        -0x1.1000000000000p+1;
        -0x1.2000000000000p+1;
        -0x1.3000000000000p+1;
        -0x1.4000000000000p+1;
        -0x1.5000000000000p+1;
        -0x1.6000000000000p+1;
        -0x1.7000000000000p+1;
        -0x1.8000000000000p+1;
        -0x1.9000000000000p+1;
        -0x1.a000000000000p+1;
        -0x1.b000000000000p+1;
        -0x1.c000000000000p+1;
        -0x1.d000000000000p+1;
        -0x1.e000000000000p+1;
        -0x1.f000000000000p+1;
        -0x1.0000000000000p+2;
        -0x1.1000000000000p+2;
        -0x1.2000000000000p+2;
        -0x1.3000000000000p+2;
        -0x1.4000000000000p+2;
        -0x1.5000000000000p+2;
        -0x1.6000000000000p+2;
        -0x1.7000000000000p+2;
        -0x1.8000000000000p+2;
        -0x1.9000000000000p+2;
        -0x1.a000000000000p+2;
        -0x1.b000000000000p+2;
        -0x1.c000000000000p+2;
        -0x1.d000000000000p+2;
        -0x1.e000000000000p+2;
        -0x1.f000000000000p+2;
        -0x1.0000000000000p+3;
        -0x1.1000000000000p+3;
        -0x1.2000000000000p+3;
        -0x1.3000000000000p+3;
        -0x1.4000000000000p+3;
        -0x1.5000000000000p+3;
        -0x1.6000000000000p+3;
        -0x1.7000000000000p+3;
        -0x1.8000000000000p+3;
        -0x1.9000000000000p+3;
        -0x1.a000000000000p+3;
        -0x1.b000000000000p+3;
        -0x1.c000000000000p+3;
        -0x1.d000000000000p+3;
        -0x1.e000000000000p+3;
        -0x1.f000000000000p+3;
        Float.neg_infinity;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
      |] );
    ( D.F8_e4m3,
      [|
        0x0.0p+0;
        0x1.0000000000000p-9;
        0x1.0000000000000p-8;
        0x1.8000000000000p-8;
        0x1.0000000000000p-7;
        0x1.4000000000000p-7;
        0x1.8000000000000p-7;
        0x1.c000000000000p-7;
        0x1.0000000000000p-6;
        0x1.2000000000000p-6;
        0x1.4000000000000p-6;
        0x1.6000000000000p-6;
        0x1.8000000000000p-6;
        0x1.a000000000000p-6;
        0x1.c000000000000p-6;
        0x1.e000000000000p-6;
        0x1.0000000000000p-5;
        0x1.2000000000000p-5;
        0x1.4000000000000p-5;
        0x1.6000000000000p-5;
        0x1.8000000000000p-5;
        0x1.a000000000000p-5;
        0x1.c000000000000p-5;
        0x1.e000000000000p-5;
        0x1.0000000000000p-4;
        0x1.2000000000000p-4;
        0x1.4000000000000p-4;
        0x1.6000000000000p-4;
        0x1.8000000000000p-4;
        0x1.a000000000000p-4;
        0x1.c000000000000p-4;
        0x1.e000000000000p-4;
        0x1.0000000000000p-3;
        0x1.2000000000000p-3;
        0x1.4000000000000p-3;
        0x1.6000000000000p-3;
        0x1.8000000000000p-3;
        0x1.a000000000000p-3;
        0x1.c000000000000p-3;
        0x1.e000000000000p-3;
        0x1.0000000000000p-2;
        0x1.2000000000000p-2;
        0x1.4000000000000p-2;
        0x1.6000000000000p-2;
        0x1.8000000000000p-2;
        0x1.a000000000000p-2;
        0x1.c000000000000p-2;
        0x1.e000000000000p-2;
        0x1.0000000000000p-1;
        0x1.2000000000000p-1;
        0x1.4000000000000p-1;
        0x1.6000000000000p-1;
        0x1.8000000000000p-1;
        0x1.a000000000000p-1;
        0x1.c000000000000p-1;
        0x1.e000000000000p-1;
        0x1.0000000000000p+0;
        0x1.2000000000000p+0;
        0x1.4000000000000p+0;
        0x1.6000000000000p+0;
        0x1.8000000000000p+0;
        0x1.a000000000000p+0;
        0x1.c000000000000p+0;
        0x1.e000000000000p+0;
        0x1.0000000000000p+1;
        0x1.2000000000000p+1;
        0x1.4000000000000p+1;
        0x1.6000000000000p+1;
        0x1.8000000000000p+1;
        0x1.a000000000000p+1;
        0x1.c000000000000p+1;
        0x1.e000000000000p+1;
        0x1.0000000000000p+2;
        0x1.2000000000000p+2;
        0x1.4000000000000p+2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+2;
        0x1.a000000000000p+2;
        0x1.c000000000000p+2;
        0x1.e000000000000p+2;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.4000000000000p+3;
        0x1.6000000000000p+3;
        0x1.8000000000000p+3;
        0x1.a000000000000p+3;
        0x1.c000000000000p+3;
        0x1.e000000000000p+3;
        0x1.0000000000000p+4;
        0x1.2000000000000p+4;
        0x1.4000000000000p+4;
        0x1.6000000000000p+4;
        0x1.8000000000000p+4;
        0x1.a000000000000p+4;
        0x1.c000000000000p+4;
        0x1.e000000000000p+4;
        0x1.0000000000000p+5;
        0x1.2000000000000p+5;
        0x1.4000000000000p+5;
        0x1.6000000000000p+5;
        0x1.8000000000000p+5;
        0x1.a000000000000p+5;
        0x1.c000000000000p+5;
        0x1.e000000000000p+5;
        0x1.0000000000000p+6;
        0x1.2000000000000p+6;
        0x1.4000000000000p+6;
        0x1.6000000000000p+6;
        0x1.8000000000000p+6;
        0x1.a000000000000p+6;
        0x1.c000000000000p+6;
        0x1.e000000000000p+6;
        0x1.0000000000000p+7;
        0x1.2000000000000p+7;
        0x1.4000000000000p+7;
        0x1.6000000000000p+7;
        0x1.8000000000000p+7;
        0x1.a000000000000p+7;
        0x1.c000000000000p+7;
        0x1.e000000000000p+7;
        Float.infinity;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        -0x0.0p+0;
        -0x1.0000000000000p-9;
        -0x1.0000000000000p-8;
        -0x1.8000000000000p-8;
        -0x1.0000000000000p-7;
        -0x1.4000000000000p-7;
        -0x1.8000000000000p-7;
        -0x1.c000000000000p-7;
        -0x1.0000000000000p-6;
        -0x1.2000000000000p-6;
        -0x1.4000000000000p-6;
        -0x1.6000000000000p-6;
        -0x1.8000000000000p-6;
        -0x1.a000000000000p-6;
        -0x1.c000000000000p-6;
        -0x1.e000000000000p-6;
        -0x1.0000000000000p-5;
        -0x1.2000000000000p-5;
        -0x1.4000000000000p-5;
        -0x1.6000000000000p-5;
        -0x1.8000000000000p-5;
        -0x1.a000000000000p-5;
        -0x1.c000000000000p-5;
        -0x1.e000000000000p-5;
        -0x1.0000000000000p-4;
        -0x1.2000000000000p-4;
        -0x1.4000000000000p-4;
        -0x1.6000000000000p-4;
        -0x1.8000000000000p-4;
        -0x1.a000000000000p-4;
        -0x1.c000000000000p-4;
        -0x1.e000000000000p-4;
        -0x1.0000000000000p-3;
        -0x1.2000000000000p-3;
        -0x1.4000000000000p-3;
        -0x1.6000000000000p-3;
        -0x1.8000000000000p-3;
        -0x1.a000000000000p-3;
        -0x1.c000000000000p-3;
        -0x1.e000000000000p-3;
        -0x1.0000000000000p-2;
        -0x1.2000000000000p-2;
        -0x1.4000000000000p-2;
        -0x1.6000000000000p-2;
        -0x1.8000000000000p-2;
        -0x1.a000000000000p-2;
        -0x1.c000000000000p-2;
        -0x1.e000000000000p-2;
        -0x1.0000000000000p-1;
        -0x1.2000000000000p-1;
        -0x1.4000000000000p-1;
        -0x1.6000000000000p-1;
        -0x1.8000000000000p-1;
        -0x1.a000000000000p-1;
        -0x1.c000000000000p-1;
        -0x1.e000000000000p-1;
        -0x1.0000000000000p+0;
        -0x1.2000000000000p+0;
        -0x1.4000000000000p+0;
        -0x1.6000000000000p+0;
        -0x1.8000000000000p+0;
        -0x1.a000000000000p+0;
        -0x1.c000000000000p+0;
        -0x1.e000000000000p+0;
        -0x1.0000000000000p+1;
        -0x1.2000000000000p+1;
        -0x1.4000000000000p+1;
        -0x1.6000000000000p+1;
        -0x1.8000000000000p+1;
        -0x1.a000000000000p+1;
        -0x1.c000000000000p+1;
        -0x1.e000000000000p+1;
        -0x1.0000000000000p+2;
        -0x1.2000000000000p+2;
        -0x1.4000000000000p+2;
        -0x1.6000000000000p+2;
        -0x1.8000000000000p+2;
        -0x1.a000000000000p+2;
        -0x1.c000000000000p+2;
        -0x1.e000000000000p+2;
        -0x1.0000000000000p+3;
        -0x1.2000000000000p+3;
        -0x1.4000000000000p+3;
        -0x1.6000000000000p+3;
        -0x1.8000000000000p+3;
        -0x1.a000000000000p+3;
        -0x1.c000000000000p+3;
        -0x1.e000000000000p+3;
        -0x1.0000000000000p+4;
        -0x1.2000000000000p+4;
        -0x1.4000000000000p+4;
        -0x1.6000000000000p+4;
        -0x1.8000000000000p+4;
        -0x1.a000000000000p+4;
        -0x1.c000000000000p+4;
        -0x1.e000000000000p+4;
        -0x1.0000000000000p+5;
        -0x1.2000000000000p+5;
        -0x1.4000000000000p+5;
        -0x1.6000000000000p+5;
        -0x1.8000000000000p+5;
        -0x1.a000000000000p+5;
        -0x1.c000000000000p+5;
        -0x1.e000000000000p+5;
        -0x1.0000000000000p+6;
        -0x1.2000000000000p+6;
        -0x1.4000000000000p+6;
        -0x1.6000000000000p+6;
        -0x1.8000000000000p+6;
        -0x1.a000000000000p+6;
        -0x1.c000000000000p+6;
        -0x1.e000000000000p+6;
        -0x1.0000000000000p+7;
        -0x1.2000000000000p+7;
        -0x1.4000000000000p+7;
        -0x1.6000000000000p+7;
        -0x1.8000000000000p+7;
        -0x1.a000000000000p+7;
        -0x1.c000000000000p+7;
        -0x1.e000000000000p+7;
        Float.neg_infinity;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
      |] );
    ( D.F8_e4m3b11fnuz,
      [|
        0x0.0p+0;
        0x1.0000000000000p-13;
        0x1.0000000000000p-12;
        0x1.8000000000000p-12;
        0x1.0000000000000p-11;
        0x1.4000000000000p-11;
        0x1.8000000000000p-11;
        0x1.c000000000000p-11;
        0x1.0000000000000p-10;
        0x1.2000000000000p-10;
        0x1.4000000000000p-10;
        0x1.6000000000000p-10;
        0x1.8000000000000p-10;
        0x1.a000000000000p-10;
        0x1.c000000000000p-10;
        0x1.e000000000000p-10;
        0x1.0000000000000p-9;
        0x1.2000000000000p-9;
        0x1.4000000000000p-9;
        0x1.6000000000000p-9;
        0x1.8000000000000p-9;
        0x1.a000000000000p-9;
        0x1.c000000000000p-9;
        0x1.e000000000000p-9;
        0x1.0000000000000p-8;
        0x1.2000000000000p-8;
        0x1.4000000000000p-8;
        0x1.6000000000000p-8;
        0x1.8000000000000p-8;
        0x1.a000000000000p-8;
        0x1.c000000000000p-8;
        0x1.e000000000000p-8;
        0x1.0000000000000p-7;
        0x1.2000000000000p-7;
        0x1.4000000000000p-7;
        0x1.6000000000000p-7;
        0x1.8000000000000p-7;
        0x1.a000000000000p-7;
        0x1.c000000000000p-7;
        0x1.e000000000000p-7;
        0x1.0000000000000p-6;
        0x1.2000000000000p-6;
        0x1.4000000000000p-6;
        0x1.6000000000000p-6;
        0x1.8000000000000p-6;
        0x1.a000000000000p-6;
        0x1.c000000000000p-6;
        0x1.e000000000000p-6;
        0x1.0000000000000p-5;
        0x1.2000000000000p-5;
        0x1.4000000000000p-5;
        0x1.6000000000000p-5;
        0x1.8000000000000p-5;
        0x1.a000000000000p-5;
        0x1.c000000000000p-5;
        0x1.e000000000000p-5;
        0x1.0000000000000p-4;
        0x1.2000000000000p-4;
        0x1.4000000000000p-4;
        0x1.6000000000000p-4;
        0x1.8000000000000p-4;
        0x1.a000000000000p-4;
        0x1.c000000000000p-4;
        0x1.e000000000000p-4;
        0x1.0000000000000p-3;
        0x1.2000000000000p-3;
        0x1.4000000000000p-3;
        0x1.6000000000000p-3;
        0x1.8000000000000p-3;
        0x1.a000000000000p-3;
        0x1.c000000000000p-3;
        0x1.e000000000000p-3;
        0x1.0000000000000p-2;
        0x1.2000000000000p-2;
        0x1.4000000000000p-2;
        0x1.6000000000000p-2;
        0x1.8000000000000p-2;
        0x1.a000000000000p-2;
        0x1.c000000000000p-2;
        0x1.e000000000000p-2;
        0x1.0000000000000p-1;
        0x1.2000000000000p-1;
        0x1.4000000000000p-1;
        0x1.6000000000000p-1;
        0x1.8000000000000p-1;
        0x1.a000000000000p-1;
        0x1.c000000000000p-1;
        0x1.e000000000000p-1;
        0x1.0000000000000p+0;
        0x1.2000000000000p+0;
        0x1.4000000000000p+0;
        0x1.6000000000000p+0;
        0x1.8000000000000p+0;
        0x1.a000000000000p+0;
        0x1.c000000000000p+0;
        0x1.e000000000000p+0;
        0x1.0000000000000p+1;
        0x1.2000000000000p+1;
        0x1.4000000000000p+1;
        0x1.6000000000000p+1;
        0x1.8000000000000p+1;
        0x1.a000000000000p+1;
        0x1.c000000000000p+1;
        0x1.e000000000000p+1;
        0x1.0000000000000p+2;
        0x1.2000000000000p+2;
        0x1.4000000000000p+2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+2;
        0x1.a000000000000p+2;
        0x1.c000000000000p+2;
        0x1.e000000000000p+2;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.4000000000000p+3;
        0x1.6000000000000p+3;
        0x1.8000000000000p+3;
        0x1.a000000000000p+3;
        0x1.c000000000000p+3;
        0x1.e000000000000p+3;
        0x1.0000000000000p+4;
        0x1.2000000000000p+4;
        0x1.4000000000000p+4;
        0x1.6000000000000p+4;
        0x1.8000000000000p+4;
        0x1.a000000000000p+4;
        0x1.c000000000000p+4;
        0x1.e000000000000p+4;
        Float.nan;
        -0x1.0000000000000p-13;
        -0x1.0000000000000p-12;
        -0x1.8000000000000p-12;
        -0x1.0000000000000p-11;
        -0x1.4000000000000p-11;
        -0x1.8000000000000p-11;
        -0x1.c000000000000p-11;
        -0x1.0000000000000p-10;
        -0x1.2000000000000p-10;
        -0x1.4000000000000p-10;
        -0x1.6000000000000p-10;
        -0x1.8000000000000p-10;
        -0x1.a000000000000p-10;
        -0x1.c000000000000p-10;
        -0x1.e000000000000p-10;
        -0x1.0000000000000p-9;
        -0x1.2000000000000p-9;
        -0x1.4000000000000p-9;
        -0x1.6000000000000p-9;
        -0x1.8000000000000p-9;
        -0x1.a000000000000p-9;
        -0x1.c000000000000p-9;
        -0x1.e000000000000p-9;
        -0x1.0000000000000p-8;
        -0x1.2000000000000p-8;
        -0x1.4000000000000p-8;
        -0x1.6000000000000p-8;
        -0x1.8000000000000p-8;
        -0x1.a000000000000p-8;
        -0x1.c000000000000p-8;
        -0x1.e000000000000p-8;
        -0x1.0000000000000p-7;
        -0x1.2000000000000p-7;
        -0x1.4000000000000p-7;
        -0x1.6000000000000p-7;
        -0x1.8000000000000p-7;
        -0x1.a000000000000p-7;
        -0x1.c000000000000p-7;
        -0x1.e000000000000p-7;
        -0x1.0000000000000p-6;
        -0x1.2000000000000p-6;
        -0x1.4000000000000p-6;
        -0x1.6000000000000p-6;
        -0x1.8000000000000p-6;
        -0x1.a000000000000p-6;
        -0x1.c000000000000p-6;
        -0x1.e000000000000p-6;
        -0x1.0000000000000p-5;
        -0x1.2000000000000p-5;
        -0x1.4000000000000p-5;
        -0x1.6000000000000p-5;
        -0x1.8000000000000p-5;
        -0x1.a000000000000p-5;
        -0x1.c000000000000p-5;
        -0x1.e000000000000p-5;
        -0x1.0000000000000p-4;
        -0x1.2000000000000p-4;
        -0x1.4000000000000p-4;
        -0x1.6000000000000p-4;
        -0x1.8000000000000p-4;
        -0x1.a000000000000p-4;
        -0x1.c000000000000p-4;
        -0x1.e000000000000p-4;
        -0x1.0000000000000p-3;
        -0x1.2000000000000p-3;
        -0x1.4000000000000p-3;
        -0x1.6000000000000p-3;
        -0x1.8000000000000p-3;
        -0x1.a000000000000p-3;
        -0x1.c000000000000p-3;
        -0x1.e000000000000p-3;
        -0x1.0000000000000p-2;
        -0x1.2000000000000p-2;
        -0x1.4000000000000p-2;
        -0x1.6000000000000p-2;
        -0x1.8000000000000p-2;
        -0x1.a000000000000p-2;
        -0x1.c000000000000p-2;
        -0x1.e000000000000p-2;
        -0x1.0000000000000p-1;
        -0x1.2000000000000p-1;
        -0x1.4000000000000p-1;
        -0x1.6000000000000p-1;
        -0x1.8000000000000p-1;
        -0x1.a000000000000p-1;
        -0x1.c000000000000p-1;
        -0x1.e000000000000p-1;
        -0x1.0000000000000p+0;
        -0x1.2000000000000p+0;
        -0x1.4000000000000p+0;
        -0x1.6000000000000p+0;
        -0x1.8000000000000p+0;
        -0x1.a000000000000p+0;
        -0x1.c000000000000p+0;
        -0x1.e000000000000p+0;
        -0x1.0000000000000p+1;
        -0x1.2000000000000p+1;
        -0x1.4000000000000p+1;
        -0x1.6000000000000p+1;
        -0x1.8000000000000p+1;
        -0x1.a000000000000p+1;
        -0x1.c000000000000p+1;
        -0x1.e000000000000p+1;
        -0x1.0000000000000p+2;
        -0x1.2000000000000p+2;
        -0x1.4000000000000p+2;
        -0x1.6000000000000p+2;
        -0x1.8000000000000p+2;
        -0x1.a000000000000p+2;
        -0x1.c000000000000p+2;
        -0x1.e000000000000p+2;
        -0x1.0000000000000p+3;
        -0x1.2000000000000p+3;
        -0x1.4000000000000p+3;
        -0x1.6000000000000p+3;
        -0x1.8000000000000p+3;
        -0x1.a000000000000p+3;
        -0x1.c000000000000p+3;
        -0x1.e000000000000p+3;
        -0x1.0000000000000p+4;
        -0x1.2000000000000p+4;
        -0x1.4000000000000p+4;
        -0x1.6000000000000p+4;
        -0x1.8000000000000p+4;
        -0x1.a000000000000p+4;
        -0x1.c000000000000p+4;
        -0x1.e000000000000p+4;
      |] );
    ( D.F8_e4m3fn,
      [|
        0x0.0p+0;
        0x1.0000000000000p-9;
        0x1.0000000000000p-8;
        0x1.8000000000000p-8;
        0x1.0000000000000p-7;
        0x1.4000000000000p-7;
        0x1.8000000000000p-7;
        0x1.c000000000000p-7;
        0x1.0000000000000p-6;
        0x1.2000000000000p-6;
        0x1.4000000000000p-6;
        0x1.6000000000000p-6;
        0x1.8000000000000p-6;
        0x1.a000000000000p-6;
        0x1.c000000000000p-6;
        0x1.e000000000000p-6;
        0x1.0000000000000p-5;
        0x1.2000000000000p-5;
        0x1.4000000000000p-5;
        0x1.6000000000000p-5;
        0x1.8000000000000p-5;
        0x1.a000000000000p-5;
        0x1.c000000000000p-5;
        0x1.e000000000000p-5;
        0x1.0000000000000p-4;
        0x1.2000000000000p-4;
        0x1.4000000000000p-4;
        0x1.6000000000000p-4;
        0x1.8000000000000p-4;
        0x1.a000000000000p-4;
        0x1.c000000000000p-4;
        0x1.e000000000000p-4;
        0x1.0000000000000p-3;
        0x1.2000000000000p-3;
        0x1.4000000000000p-3;
        0x1.6000000000000p-3;
        0x1.8000000000000p-3;
        0x1.a000000000000p-3;
        0x1.c000000000000p-3;
        0x1.e000000000000p-3;
        0x1.0000000000000p-2;
        0x1.2000000000000p-2;
        0x1.4000000000000p-2;
        0x1.6000000000000p-2;
        0x1.8000000000000p-2;
        0x1.a000000000000p-2;
        0x1.c000000000000p-2;
        0x1.e000000000000p-2;
        0x1.0000000000000p-1;
        0x1.2000000000000p-1;
        0x1.4000000000000p-1;
        0x1.6000000000000p-1;
        0x1.8000000000000p-1;
        0x1.a000000000000p-1;
        0x1.c000000000000p-1;
        0x1.e000000000000p-1;
        0x1.0000000000000p+0;
        0x1.2000000000000p+0;
        0x1.4000000000000p+0;
        0x1.6000000000000p+0;
        0x1.8000000000000p+0;
        0x1.a000000000000p+0;
        0x1.c000000000000p+0;
        0x1.e000000000000p+0;
        0x1.0000000000000p+1;
        0x1.2000000000000p+1;
        0x1.4000000000000p+1;
        0x1.6000000000000p+1;
        0x1.8000000000000p+1;
        0x1.a000000000000p+1;
        0x1.c000000000000p+1;
        0x1.e000000000000p+1;
        0x1.0000000000000p+2;
        0x1.2000000000000p+2;
        0x1.4000000000000p+2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+2;
        0x1.a000000000000p+2;
        0x1.c000000000000p+2;
        0x1.e000000000000p+2;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.4000000000000p+3;
        0x1.6000000000000p+3;
        0x1.8000000000000p+3;
        0x1.a000000000000p+3;
        0x1.c000000000000p+3;
        0x1.e000000000000p+3;
        0x1.0000000000000p+4;
        0x1.2000000000000p+4;
        0x1.4000000000000p+4;
        0x1.6000000000000p+4;
        0x1.8000000000000p+4;
        0x1.a000000000000p+4;
        0x1.c000000000000p+4;
        0x1.e000000000000p+4;
        0x1.0000000000000p+5;
        0x1.2000000000000p+5;
        0x1.4000000000000p+5;
        0x1.6000000000000p+5;
        0x1.8000000000000p+5;
        0x1.a000000000000p+5;
        0x1.c000000000000p+5;
        0x1.e000000000000p+5;
        0x1.0000000000000p+6;
        0x1.2000000000000p+6;
        0x1.4000000000000p+6;
        0x1.6000000000000p+6;
        0x1.8000000000000p+6;
        0x1.a000000000000p+6;
        0x1.c000000000000p+6;
        0x1.e000000000000p+6;
        0x1.0000000000000p+7;
        0x1.2000000000000p+7;
        0x1.4000000000000p+7;
        0x1.6000000000000p+7;
        0x1.8000000000000p+7;
        0x1.a000000000000p+7;
        0x1.c000000000000p+7;
        0x1.e000000000000p+7;
        0x1.0000000000000p+8;
        0x1.2000000000000p+8;
        0x1.4000000000000p+8;
        0x1.6000000000000p+8;
        0x1.8000000000000p+8;
        0x1.a000000000000p+8;
        0x1.c000000000000p+8;
        Float.nan;
        -0x0.0p+0;
        -0x1.0000000000000p-9;
        -0x1.0000000000000p-8;
        -0x1.8000000000000p-8;
        -0x1.0000000000000p-7;
        -0x1.4000000000000p-7;
        -0x1.8000000000000p-7;
        -0x1.c000000000000p-7;
        -0x1.0000000000000p-6;
        -0x1.2000000000000p-6;
        -0x1.4000000000000p-6;
        -0x1.6000000000000p-6;
        -0x1.8000000000000p-6;
        -0x1.a000000000000p-6;
        -0x1.c000000000000p-6;
        -0x1.e000000000000p-6;
        -0x1.0000000000000p-5;
        -0x1.2000000000000p-5;
        -0x1.4000000000000p-5;
        -0x1.6000000000000p-5;
        -0x1.8000000000000p-5;
        -0x1.a000000000000p-5;
        -0x1.c000000000000p-5;
        -0x1.e000000000000p-5;
        -0x1.0000000000000p-4;
        -0x1.2000000000000p-4;
        -0x1.4000000000000p-4;
        -0x1.6000000000000p-4;
        -0x1.8000000000000p-4;
        -0x1.a000000000000p-4;
        -0x1.c000000000000p-4;
        -0x1.e000000000000p-4;
        -0x1.0000000000000p-3;
        -0x1.2000000000000p-3;
        -0x1.4000000000000p-3;
        -0x1.6000000000000p-3;
        -0x1.8000000000000p-3;
        -0x1.a000000000000p-3;
        -0x1.c000000000000p-3;
        -0x1.e000000000000p-3;
        -0x1.0000000000000p-2;
        -0x1.2000000000000p-2;
        -0x1.4000000000000p-2;
        -0x1.6000000000000p-2;
        -0x1.8000000000000p-2;
        -0x1.a000000000000p-2;
        -0x1.c000000000000p-2;
        -0x1.e000000000000p-2;
        -0x1.0000000000000p-1;
        -0x1.2000000000000p-1;
        -0x1.4000000000000p-1;
        -0x1.6000000000000p-1;
        -0x1.8000000000000p-1;
        -0x1.a000000000000p-1;
        -0x1.c000000000000p-1;
        -0x1.e000000000000p-1;
        -0x1.0000000000000p+0;
        -0x1.2000000000000p+0;
        -0x1.4000000000000p+0;
        -0x1.6000000000000p+0;
        -0x1.8000000000000p+0;
        -0x1.a000000000000p+0;
        -0x1.c000000000000p+0;
        -0x1.e000000000000p+0;
        -0x1.0000000000000p+1;
        -0x1.2000000000000p+1;
        -0x1.4000000000000p+1;
        -0x1.6000000000000p+1;
        -0x1.8000000000000p+1;
        -0x1.a000000000000p+1;
        -0x1.c000000000000p+1;
        -0x1.e000000000000p+1;
        -0x1.0000000000000p+2;
        -0x1.2000000000000p+2;
        -0x1.4000000000000p+2;
        -0x1.6000000000000p+2;
        -0x1.8000000000000p+2;
        -0x1.a000000000000p+2;
        -0x1.c000000000000p+2;
        -0x1.e000000000000p+2;
        -0x1.0000000000000p+3;
        -0x1.2000000000000p+3;
        -0x1.4000000000000p+3;
        -0x1.6000000000000p+3;
        -0x1.8000000000000p+3;
        -0x1.a000000000000p+3;
        -0x1.c000000000000p+3;
        -0x1.e000000000000p+3;
        -0x1.0000000000000p+4;
        -0x1.2000000000000p+4;
        -0x1.4000000000000p+4;
        -0x1.6000000000000p+4;
        -0x1.8000000000000p+4;
        -0x1.a000000000000p+4;
        -0x1.c000000000000p+4;
        -0x1.e000000000000p+4;
        -0x1.0000000000000p+5;
        -0x1.2000000000000p+5;
        -0x1.4000000000000p+5;
        -0x1.6000000000000p+5;
        -0x1.8000000000000p+5;
        -0x1.a000000000000p+5;
        -0x1.c000000000000p+5;
        -0x1.e000000000000p+5;
        -0x1.0000000000000p+6;
        -0x1.2000000000000p+6;
        -0x1.4000000000000p+6;
        -0x1.6000000000000p+6;
        -0x1.8000000000000p+6;
        -0x1.a000000000000p+6;
        -0x1.c000000000000p+6;
        -0x1.e000000000000p+6;
        -0x1.0000000000000p+7;
        -0x1.2000000000000p+7;
        -0x1.4000000000000p+7;
        -0x1.6000000000000p+7;
        -0x1.8000000000000p+7;
        -0x1.a000000000000p+7;
        -0x1.c000000000000p+7;
        -0x1.e000000000000p+7;
        -0x1.0000000000000p+8;
        -0x1.2000000000000p+8;
        -0x1.4000000000000p+8;
        -0x1.6000000000000p+8;
        -0x1.8000000000000p+8;
        -0x1.a000000000000p+8;
        -0x1.c000000000000p+8;
        Float.nan;
      |] );
    ( D.F8_e4m3fnuz,
      [|
        0x0.0p+0;
        0x1.0000000000000p-10;
        0x1.0000000000000p-9;
        0x1.8000000000000p-9;
        0x1.0000000000000p-8;
        0x1.4000000000000p-8;
        0x1.8000000000000p-8;
        0x1.c000000000000p-8;
        0x1.0000000000000p-7;
        0x1.2000000000000p-7;
        0x1.4000000000000p-7;
        0x1.6000000000000p-7;
        0x1.8000000000000p-7;
        0x1.a000000000000p-7;
        0x1.c000000000000p-7;
        0x1.e000000000000p-7;
        0x1.0000000000000p-6;
        0x1.2000000000000p-6;
        0x1.4000000000000p-6;
        0x1.6000000000000p-6;
        0x1.8000000000000p-6;
        0x1.a000000000000p-6;
        0x1.c000000000000p-6;
        0x1.e000000000000p-6;
        0x1.0000000000000p-5;
        0x1.2000000000000p-5;
        0x1.4000000000000p-5;
        0x1.6000000000000p-5;
        0x1.8000000000000p-5;
        0x1.a000000000000p-5;
        0x1.c000000000000p-5;
        0x1.e000000000000p-5;
        0x1.0000000000000p-4;
        0x1.2000000000000p-4;
        0x1.4000000000000p-4;
        0x1.6000000000000p-4;
        0x1.8000000000000p-4;
        0x1.a000000000000p-4;
        0x1.c000000000000p-4;
        0x1.e000000000000p-4;
        0x1.0000000000000p-3;
        0x1.2000000000000p-3;
        0x1.4000000000000p-3;
        0x1.6000000000000p-3;
        0x1.8000000000000p-3;
        0x1.a000000000000p-3;
        0x1.c000000000000p-3;
        0x1.e000000000000p-3;
        0x1.0000000000000p-2;
        0x1.2000000000000p-2;
        0x1.4000000000000p-2;
        0x1.6000000000000p-2;
        0x1.8000000000000p-2;
        0x1.a000000000000p-2;
        0x1.c000000000000p-2;
        0x1.e000000000000p-2;
        0x1.0000000000000p-1;
        0x1.2000000000000p-1;
        0x1.4000000000000p-1;
        0x1.6000000000000p-1;
        0x1.8000000000000p-1;
        0x1.a000000000000p-1;
        0x1.c000000000000p-1;
        0x1.e000000000000p-1;
        0x1.0000000000000p+0;
        0x1.2000000000000p+0;
        0x1.4000000000000p+0;
        0x1.6000000000000p+0;
        0x1.8000000000000p+0;
        0x1.a000000000000p+0;
        0x1.c000000000000p+0;
        0x1.e000000000000p+0;
        0x1.0000000000000p+1;
        0x1.2000000000000p+1;
        0x1.4000000000000p+1;
        0x1.6000000000000p+1;
        0x1.8000000000000p+1;
        0x1.a000000000000p+1;
        0x1.c000000000000p+1;
        0x1.e000000000000p+1;
        0x1.0000000000000p+2;
        0x1.2000000000000p+2;
        0x1.4000000000000p+2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+2;
        0x1.a000000000000p+2;
        0x1.c000000000000p+2;
        0x1.e000000000000p+2;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.4000000000000p+3;
        0x1.6000000000000p+3;
        0x1.8000000000000p+3;
        0x1.a000000000000p+3;
        0x1.c000000000000p+3;
        0x1.e000000000000p+3;
        0x1.0000000000000p+4;
        0x1.2000000000000p+4;
        0x1.4000000000000p+4;
        0x1.6000000000000p+4;
        0x1.8000000000000p+4;
        0x1.a000000000000p+4;
        0x1.c000000000000p+4;
        0x1.e000000000000p+4;
        0x1.0000000000000p+5;
        0x1.2000000000000p+5;
        0x1.4000000000000p+5;
        0x1.6000000000000p+5;
        0x1.8000000000000p+5;
        0x1.a000000000000p+5;
        0x1.c000000000000p+5;
        0x1.e000000000000p+5;
        0x1.0000000000000p+6;
        0x1.2000000000000p+6;
        0x1.4000000000000p+6;
        0x1.6000000000000p+6;
        0x1.8000000000000p+6;
        0x1.a000000000000p+6;
        0x1.c000000000000p+6;
        0x1.e000000000000p+6;
        0x1.0000000000000p+7;
        0x1.2000000000000p+7;
        0x1.4000000000000p+7;
        0x1.6000000000000p+7;
        0x1.8000000000000p+7;
        0x1.a000000000000p+7;
        0x1.c000000000000p+7;
        0x1.e000000000000p+7;
        Float.nan;
        -0x1.0000000000000p-10;
        -0x1.0000000000000p-9;
        -0x1.8000000000000p-9;
        -0x1.0000000000000p-8;
        -0x1.4000000000000p-8;
        -0x1.8000000000000p-8;
        -0x1.c000000000000p-8;
        -0x1.0000000000000p-7;
        -0x1.2000000000000p-7;
        -0x1.4000000000000p-7;
        -0x1.6000000000000p-7;
        -0x1.8000000000000p-7;
        -0x1.a000000000000p-7;
        -0x1.c000000000000p-7;
        -0x1.e000000000000p-7;
        -0x1.0000000000000p-6;
        -0x1.2000000000000p-6;
        -0x1.4000000000000p-6;
        -0x1.6000000000000p-6;
        -0x1.8000000000000p-6;
        -0x1.a000000000000p-6;
        -0x1.c000000000000p-6;
        -0x1.e000000000000p-6;
        -0x1.0000000000000p-5;
        -0x1.2000000000000p-5;
        -0x1.4000000000000p-5;
        -0x1.6000000000000p-5;
        -0x1.8000000000000p-5;
        -0x1.a000000000000p-5;
        -0x1.c000000000000p-5;
        -0x1.e000000000000p-5;
        -0x1.0000000000000p-4;
        -0x1.2000000000000p-4;
        -0x1.4000000000000p-4;
        -0x1.6000000000000p-4;
        -0x1.8000000000000p-4;
        -0x1.a000000000000p-4;
        -0x1.c000000000000p-4;
        -0x1.e000000000000p-4;
        -0x1.0000000000000p-3;
        -0x1.2000000000000p-3;
        -0x1.4000000000000p-3;
        -0x1.6000000000000p-3;
        -0x1.8000000000000p-3;
        -0x1.a000000000000p-3;
        -0x1.c000000000000p-3;
        -0x1.e000000000000p-3;
        -0x1.0000000000000p-2;
        -0x1.2000000000000p-2;
        -0x1.4000000000000p-2;
        -0x1.6000000000000p-2;
        -0x1.8000000000000p-2;
        -0x1.a000000000000p-2;
        -0x1.c000000000000p-2;
        -0x1.e000000000000p-2;
        -0x1.0000000000000p-1;
        -0x1.2000000000000p-1;
        -0x1.4000000000000p-1;
        -0x1.6000000000000p-1;
        -0x1.8000000000000p-1;
        -0x1.a000000000000p-1;
        -0x1.c000000000000p-1;
        -0x1.e000000000000p-1;
        -0x1.0000000000000p+0;
        -0x1.2000000000000p+0;
        -0x1.4000000000000p+0;
        -0x1.6000000000000p+0;
        -0x1.8000000000000p+0;
        -0x1.a000000000000p+0;
        -0x1.c000000000000p+0;
        -0x1.e000000000000p+0;
        -0x1.0000000000000p+1;
        -0x1.2000000000000p+1;
        -0x1.4000000000000p+1;
        -0x1.6000000000000p+1;
        -0x1.8000000000000p+1;
        -0x1.a000000000000p+1;
        -0x1.c000000000000p+1;
        -0x1.e000000000000p+1;
        -0x1.0000000000000p+2;
        -0x1.2000000000000p+2;
        -0x1.4000000000000p+2;
        -0x1.6000000000000p+2;
        -0x1.8000000000000p+2;
        -0x1.a000000000000p+2;
        -0x1.c000000000000p+2;
        -0x1.e000000000000p+2;
        -0x1.0000000000000p+3;
        -0x1.2000000000000p+3;
        -0x1.4000000000000p+3;
        -0x1.6000000000000p+3;
        -0x1.8000000000000p+3;
        -0x1.a000000000000p+3;
        -0x1.c000000000000p+3;
        -0x1.e000000000000p+3;
        -0x1.0000000000000p+4;
        -0x1.2000000000000p+4;
        -0x1.4000000000000p+4;
        -0x1.6000000000000p+4;
        -0x1.8000000000000p+4;
        -0x1.a000000000000p+4;
        -0x1.c000000000000p+4;
        -0x1.e000000000000p+4;
        -0x1.0000000000000p+5;
        -0x1.2000000000000p+5;
        -0x1.4000000000000p+5;
        -0x1.6000000000000p+5;
        -0x1.8000000000000p+5;
        -0x1.a000000000000p+5;
        -0x1.c000000000000p+5;
        -0x1.e000000000000p+5;
        -0x1.0000000000000p+6;
        -0x1.2000000000000p+6;
        -0x1.4000000000000p+6;
        -0x1.6000000000000p+6;
        -0x1.8000000000000p+6;
        -0x1.a000000000000p+6;
        -0x1.c000000000000p+6;
        -0x1.e000000000000p+6;
        -0x1.0000000000000p+7;
        -0x1.2000000000000p+7;
        -0x1.4000000000000p+7;
        -0x1.6000000000000p+7;
        -0x1.8000000000000p+7;
        -0x1.a000000000000p+7;
        -0x1.c000000000000p+7;
        -0x1.e000000000000p+7;
      |] );
    ( D.F8_e5m2,
      [|
        0x0.0p+0;
        0x1.0000000000000p-16;
        0x1.0000000000000p-15;
        0x1.8000000000000p-15;
        0x1.0000000000000p-14;
        0x1.4000000000000p-14;
        0x1.8000000000000p-14;
        0x1.c000000000000p-14;
        0x1.0000000000000p-13;
        0x1.4000000000000p-13;
        0x1.8000000000000p-13;
        0x1.c000000000000p-13;
        0x1.0000000000000p-12;
        0x1.4000000000000p-12;
        0x1.8000000000000p-12;
        0x1.c000000000000p-12;
        0x1.0000000000000p-11;
        0x1.4000000000000p-11;
        0x1.8000000000000p-11;
        0x1.c000000000000p-11;
        0x1.0000000000000p-10;
        0x1.4000000000000p-10;
        0x1.8000000000000p-10;
        0x1.c000000000000p-10;
        0x1.0000000000000p-9;
        0x1.4000000000000p-9;
        0x1.8000000000000p-9;
        0x1.c000000000000p-9;
        0x1.0000000000000p-8;
        0x1.4000000000000p-8;
        0x1.8000000000000p-8;
        0x1.c000000000000p-8;
        0x1.0000000000000p-7;
        0x1.4000000000000p-7;
        0x1.8000000000000p-7;
        0x1.c000000000000p-7;
        0x1.0000000000000p-6;
        0x1.4000000000000p-6;
        0x1.8000000000000p-6;
        0x1.c000000000000p-6;
        0x1.0000000000000p-5;
        0x1.4000000000000p-5;
        0x1.8000000000000p-5;
        0x1.c000000000000p-5;
        0x1.0000000000000p-4;
        0x1.4000000000000p-4;
        0x1.8000000000000p-4;
        0x1.c000000000000p-4;
        0x1.0000000000000p-3;
        0x1.4000000000000p-3;
        0x1.8000000000000p-3;
        0x1.c000000000000p-3;
        0x1.0000000000000p-2;
        0x1.4000000000000p-2;
        0x1.8000000000000p-2;
        0x1.c000000000000p-2;
        0x1.0000000000000p-1;
        0x1.4000000000000p-1;
        0x1.8000000000000p-1;
        0x1.c000000000000p-1;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x1.8000000000000p+0;
        0x1.c000000000000p+0;
        0x1.0000000000000p+1;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.0000000000000p+2;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.8000000000000p+3;
        0x1.c000000000000p+3;
        0x1.0000000000000p+4;
        0x1.4000000000000p+4;
        0x1.8000000000000p+4;
        0x1.c000000000000p+4;
        0x1.0000000000000p+5;
        0x1.4000000000000p+5;
        0x1.8000000000000p+5;
        0x1.c000000000000p+5;
        0x1.0000000000000p+6;
        0x1.4000000000000p+6;
        0x1.8000000000000p+6;
        0x1.c000000000000p+6;
        0x1.0000000000000p+7;
        0x1.4000000000000p+7;
        0x1.8000000000000p+7;
        0x1.c000000000000p+7;
        0x1.0000000000000p+8;
        0x1.4000000000000p+8;
        0x1.8000000000000p+8;
        0x1.c000000000000p+8;
        0x1.0000000000000p+9;
        0x1.4000000000000p+9;
        0x1.8000000000000p+9;
        0x1.c000000000000p+9;
        0x1.0000000000000p+10;
        0x1.4000000000000p+10;
        0x1.8000000000000p+10;
        0x1.c000000000000p+10;
        0x1.0000000000000p+11;
        0x1.4000000000000p+11;
        0x1.8000000000000p+11;
        0x1.c000000000000p+11;
        0x1.0000000000000p+12;
        0x1.4000000000000p+12;
        0x1.8000000000000p+12;
        0x1.c000000000000p+12;
        0x1.0000000000000p+13;
        0x1.4000000000000p+13;
        0x1.8000000000000p+13;
        0x1.c000000000000p+13;
        0x1.0000000000000p+14;
        0x1.4000000000000p+14;
        0x1.8000000000000p+14;
        0x1.c000000000000p+14;
        0x1.0000000000000p+15;
        0x1.4000000000000p+15;
        0x1.8000000000000p+15;
        0x1.c000000000000p+15;
        Float.infinity;
        Float.nan;
        Float.nan;
        Float.nan;
        -0x0.0p+0;
        -0x1.0000000000000p-16;
        -0x1.0000000000000p-15;
        -0x1.8000000000000p-15;
        -0x1.0000000000000p-14;
        -0x1.4000000000000p-14;
        -0x1.8000000000000p-14;
        -0x1.c000000000000p-14;
        -0x1.0000000000000p-13;
        -0x1.4000000000000p-13;
        -0x1.8000000000000p-13;
        -0x1.c000000000000p-13;
        -0x1.0000000000000p-12;
        -0x1.4000000000000p-12;
        -0x1.8000000000000p-12;
        -0x1.c000000000000p-12;
        -0x1.0000000000000p-11;
        -0x1.4000000000000p-11;
        -0x1.8000000000000p-11;
        -0x1.c000000000000p-11;
        -0x1.0000000000000p-10;
        -0x1.4000000000000p-10;
        -0x1.8000000000000p-10;
        -0x1.c000000000000p-10;
        -0x1.0000000000000p-9;
        -0x1.4000000000000p-9;
        -0x1.8000000000000p-9;
        -0x1.c000000000000p-9;
        -0x1.0000000000000p-8;
        -0x1.4000000000000p-8;
        -0x1.8000000000000p-8;
        -0x1.c000000000000p-8;
        -0x1.0000000000000p-7;
        -0x1.4000000000000p-7;
        -0x1.8000000000000p-7;
        -0x1.c000000000000p-7;
        -0x1.0000000000000p-6;
        -0x1.4000000000000p-6;
        -0x1.8000000000000p-6;
        -0x1.c000000000000p-6;
        -0x1.0000000000000p-5;
        -0x1.4000000000000p-5;
        -0x1.8000000000000p-5;
        -0x1.c000000000000p-5;
        -0x1.0000000000000p-4;
        -0x1.4000000000000p-4;
        -0x1.8000000000000p-4;
        -0x1.c000000000000p-4;
        -0x1.0000000000000p-3;
        -0x1.4000000000000p-3;
        -0x1.8000000000000p-3;
        -0x1.c000000000000p-3;
        -0x1.0000000000000p-2;
        -0x1.4000000000000p-2;
        -0x1.8000000000000p-2;
        -0x1.c000000000000p-2;
        -0x1.0000000000000p-1;
        -0x1.4000000000000p-1;
        -0x1.8000000000000p-1;
        -0x1.c000000000000p-1;
        -0x1.0000000000000p+0;
        -0x1.4000000000000p+0;
        -0x1.8000000000000p+0;
        -0x1.c000000000000p+0;
        -0x1.0000000000000p+1;
        -0x1.4000000000000p+1;
        -0x1.8000000000000p+1;
        -0x1.c000000000000p+1;
        -0x1.0000000000000p+2;
        -0x1.4000000000000p+2;
        -0x1.8000000000000p+2;
        -0x1.c000000000000p+2;
        -0x1.0000000000000p+3;
        -0x1.4000000000000p+3;
        -0x1.8000000000000p+3;
        -0x1.c000000000000p+3;
        -0x1.0000000000000p+4;
        -0x1.4000000000000p+4;
        -0x1.8000000000000p+4;
        -0x1.c000000000000p+4;
        -0x1.0000000000000p+5;
        -0x1.4000000000000p+5;
        -0x1.8000000000000p+5;
        -0x1.c000000000000p+5;
        -0x1.0000000000000p+6;
        -0x1.4000000000000p+6;
        -0x1.8000000000000p+6;
        -0x1.c000000000000p+6;
        -0x1.0000000000000p+7;
        -0x1.4000000000000p+7;
        -0x1.8000000000000p+7;
        -0x1.c000000000000p+7;
        -0x1.0000000000000p+8;
        -0x1.4000000000000p+8;
        -0x1.8000000000000p+8;
        -0x1.c000000000000p+8;
        -0x1.0000000000000p+9;
        -0x1.4000000000000p+9;
        -0x1.8000000000000p+9;
        -0x1.c000000000000p+9;
        -0x1.0000000000000p+10;
        -0x1.4000000000000p+10;
        -0x1.8000000000000p+10;
        -0x1.c000000000000p+10;
        -0x1.0000000000000p+11;
        -0x1.4000000000000p+11;
        -0x1.8000000000000p+11;
        -0x1.c000000000000p+11;
        -0x1.0000000000000p+12;
        -0x1.4000000000000p+12;
        -0x1.8000000000000p+12;
        -0x1.c000000000000p+12;
        -0x1.0000000000000p+13;
        -0x1.4000000000000p+13;
        -0x1.8000000000000p+13;
        -0x1.c000000000000p+13;
        -0x1.0000000000000p+14;
        -0x1.4000000000000p+14;
        -0x1.8000000000000p+14;
        -0x1.c000000000000p+14;
        -0x1.0000000000000p+15;
        -0x1.4000000000000p+15;
        -0x1.8000000000000p+15;
        -0x1.c000000000000p+15;
        Float.neg_infinity;
        Float.nan;
        Float.nan;
        Float.nan;
      |] );
    ( D.F8_e5m2fnuz,
      [|
        0x0.0p+0;
        0x1.0000000000000p-17;
        0x1.0000000000000p-16;
        0x1.8000000000000p-16;
        0x1.0000000000000p-15;
        0x1.4000000000000p-15;
        0x1.8000000000000p-15;
        0x1.c000000000000p-15;
        0x1.0000000000000p-14;
        0x1.4000000000000p-14;
        0x1.8000000000000p-14;
        0x1.c000000000000p-14;
        0x1.0000000000000p-13;
        0x1.4000000000000p-13;
        0x1.8000000000000p-13;
        0x1.c000000000000p-13;
        0x1.0000000000000p-12;
        0x1.4000000000000p-12;
        0x1.8000000000000p-12;
        0x1.c000000000000p-12;
        0x1.0000000000000p-11;
        0x1.4000000000000p-11;
        0x1.8000000000000p-11;
        0x1.c000000000000p-11;
        0x1.0000000000000p-10;
        0x1.4000000000000p-10;
        0x1.8000000000000p-10;
        0x1.c000000000000p-10;
        0x1.0000000000000p-9;
        0x1.4000000000000p-9;
        0x1.8000000000000p-9;
        0x1.c000000000000p-9;
        0x1.0000000000000p-8;
        0x1.4000000000000p-8;
        0x1.8000000000000p-8;
        0x1.c000000000000p-8;
        0x1.0000000000000p-7;
        0x1.4000000000000p-7;
        0x1.8000000000000p-7;
        0x1.c000000000000p-7;
        0x1.0000000000000p-6;
        0x1.4000000000000p-6;
        0x1.8000000000000p-6;
        0x1.c000000000000p-6;
        0x1.0000000000000p-5;
        0x1.4000000000000p-5;
        0x1.8000000000000p-5;
        0x1.c000000000000p-5;
        0x1.0000000000000p-4;
        0x1.4000000000000p-4;
        0x1.8000000000000p-4;
        0x1.c000000000000p-4;
        0x1.0000000000000p-3;
        0x1.4000000000000p-3;
        0x1.8000000000000p-3;
        0x1.c000000000000p-3;
        0x1.0000000000000p-2;
        0x1.4000000000000p-2;
        0x1.8000000000000p-2;
        0x1.c000000000000p-2;
        0x1.0000000000000p-1;
        0x1.4000000000000p-1;
        0x1.8000000000000p-1;
        0x1.c000000000000p-1;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x1.8000000000000p+0;
        0x1.c000000000000p+0;
        0x1.0000000000000p+1;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.c000000000000p+1;
        0x1.0000000000000p+2;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.8000000000000p+3;
        0x1.c000000000000p+3;
        0x1.0000000000000p+4;
        0x1.4000000000000p+4;
        0x1.8000000000000p+4;
        0x1.c000000000000p+4;
        0x1.0000000000000p+5;
        0x1.4000000000000p+5;
        0x1.8000000000000p+5;
        0x1.c000000000000p+5;
        0x1.0000000000000p+6;
        0x1.4000000000000p+6;
        0x1.8000000000000p+6;
        0x1.c000000000000p+6;
        0x1.0000000000000p+7;
        0x1.4000000000000p+7;
        0x1.8000000000000p+7;
        0x1.c000000000000p+7;
        0x1.0000000000000p+8;
        0x1.4000000000000p+8;
        0x1.8000000000000p+8;
        0x1.c000000000000p+8;
        0x1.0000000000000p+9;
        0x1.4000000000000p+9;
        0x1.8000000000000p+9;
        0x1.c000000000000p+9;
        0x1.0000000000000p+10;
        0x1.4000000000000p+10;
        0x1.8000000000000p+10;
        0x1.c000000000000p+10;
        0x1.0000000000000p+11;
        0x1.4000000000000p+11;
        0x1.8000000000000p+11;
        0x1.c000000000000p+11;
        0x1.0000000000000p+12;
        0x1.4000000000000p+12;
        0x1.8000000000000p+12;
        0x1.c000000000000p+12;
        0x1.0000000000000p+13;
        0x1.4000000000000p+13;
        0x1.8000000000000p+13;
        0x1.c000000000000p+13;
        0x1.0000000000000p+14;
        0x1.4000000000000p+14;
        0x1.8000000000000p+14;
        0x1.c000000000000p+14;
        0x1.0000000000000p+15;
        0x1.4000000000000p+15;
        0x1.8000000000000p+15;
        0x1.c000000000000p+15;
        Float.nan;
        -0x1.0000000000000p-17;
        -0x1.0000000000000p-16;
        -0x1.8000000000000p-16;
        -0x1.0000000000000p-15;
        -0x1.4000000000000p-15;
        -0x1.8000000000000p-15;
        -0x1.c000000000000p-15;
        -0x1.0000000000000p-14;
        -0x1.4000000000000p-14;
        -0x1.8000000000000p-14;
        -0x1.c000000000000p-14;
        -0x1.0000000000000p-13;
        -0x1.4000000000000p-13;
        -0x1.8000000000000p-13;
        -0x1.c000000000000p-13;
        -0x1.0000000000000p-12;
        -0x1.4000000000000p-12;
        -0x1.8000000000000p-12;
        -0x1.c000000000000p-12;
        -0x1.0000000000000p-11;
        -0x1.4000000000000p-11;
        -0x1.8000000000000p-11;
        -0x1.c000000000000p-11;
        -0x1.0000000000000p-10;
        -0x1.4000000000000p-10;
        -0x1.8000000000000p-10;
        -0x1.c000000000000p-10;
        -0x1.0000000000000p-9;
        -0x1.4000000000000p-9;
        -0x1.8000000000000p-9;
        -0x1.c000000000000p-9;
        -0x1.0000000000000p-8;
        -0x1.4000000000000p-8;
        -0x1.8000000000000p-8;
        -0x1.c000000000000p-8;
        -0x1.0000000000000p-7;
        -0x1.4000000000000p-7;
        -0x1.8000000000000p-7;
        -0x1.c000000000000p-7;
        -0x1.0000000000000p-6;
        -0x1.4000000000000p-6;
        -0x1.8000000000000p-6;
        -0x1.c000000000000p-6;
        -0x1.0000000000000p-5;
        -0x1.4000000000000p-5;
        -0x1.8000000000000p-5;
        -0x1.c000000000000p-5;
        -0x1.0000000000000p-4;
        -0x1.4000000000000p-4;
        -0x1.8000000000000p-4;
        -0x1.c000000000000p-4;
        -0x1.0000000000000p-3;
        -0x1.4000000000000p-3;
        -0x1.8000000000000p-3;
        -0x1.c000000000000p-3;
        -0x1.0000000000000p-2;
        -0x1.4000000000000p-2;
        -0x1.8000000000000p-2;
        -0x1.c000000000000p-2;
        -0x1.0000000000000p-1;
        -0x1.4000000000000p-1;
        -0x1.8000000000000p-1;
        -0x1.c000000000000p-1;
        -0x1.0000000000000p+0;
        -0x1.4000000000000p+0;
        -0x1.8000000000000p+0;
        -0x1.c000000000000p+0;
        -0x1.0000000000000p+1;
        -0x1.4000000000000p+1;
        -0x1.8000000000000p+1;
        -0x1.c000000000000p+1;
        -0x1.0000000000000p+2;
        -0x1.4000000000000p+2;
        -0x1.8000000000000p+2;
        -0x1.c000000000000p+2;
        -0x1.0000000000000p+3;
        -0x1.4000000000000p+3;
        -0x1.8000000000000p+3;
        -0x1.c000000000000p+3;
        -0x1.0000000000000p+4;
        -0x1.4000000000000p+4;
        -0x1.8000000000000p+4;
        -0x1.c000000000000p+4;
        -0x1.0000000000000p+5;
        -0x1.4000000000000p+5;
        -0x1.8000000000000p+5;
        -0x1.c000000000000p+5;
        -0x1.0000000000000p+6;
        -0x1.4000000000000p+6;
        -0x1.8000000000000p+6;
        -0x1.c000000000000p+6;
        -0x1.0000000000000p+7;
        -0x1.4000000000000p+7;
        -0x1.8000000000000p+7;
        -0x1.c000000000000p+7;
        -0x1.0000000000000p+8;
        -0x1.4000000000000p+8;
        -0x1.8000000000000p+8;
        -0x1.c000000000000p+8;
        -0x1.0000000000000p+9;
        -0x1.4000000000000p+9;
        -0x1.8000000000000p+9;
        -0x1.c000000000000p+9;
        -0x1.0000000000000p+10;
        -0x1.4000000000000p+10;
        -0x1.8000000000000p+10;
        -0x1.c000000000000p+10;
        -0x1.0000000000000p+11;
        -0x1.4000000000000p+11;
        -0x1.8000000000000p+11;
        -0x1.c000000000000p+11;
        -0x1.0000000000000p+12;
        -0x1.4000000000000p+12;
        -0x1.8000000000000p+12;
        -0x1.c000000000000p+12;
        -0x1.0000000000000p+13;
        -0x1.4000000000000p+13;
        -0x1.8000000000000p+13;
        -0x1.c000000000000p+13;
        -0x1.0000000000000p+14;
        -0x1.4000000000000p+14;
        -0x1.8000000000000p+14;
        -0x1.c000000000000p+14;
        -0x1.0000000000000p+15;
        -0x1.4000000000000p+15;
        -0x1.8000000000000p+15;
        -0x1.c000000000000p+15;
      |] );
    ( D.F8_e8m0fnu,
      [|
        0x1.0000000000000p-127;
        0x1.0000000000000p-126;
        0x1.0000000000000p-125;
        0x1.0000000000000p-124;
        0x1.0000000000000p-123;
        0x1.0000000000000p-122;
        0x1.0000000000000p-121;
        0x1.0000000000000p-120;
        0x1.0000000000000p-119;
        0x1.0000000000000p-118;
        0x1.0000000000000p-117;
        0x1.0000000000000p-116;
        0x1.0000000000000p-115;
        0x1.0000000000000p-114;
        0x1.0000000000000p-113;
        0x1.0000000000000p-112;
        0x1.0000000000000p-111;
        0x1.0000000000000p-110;
        0x1.0000000000000p-109;
        0x1.0000000000000p-108;
        0x1.0000000000000p-107;
        0x1.0000000000000p-106;
        0x1.0000000000000p-105;
        0x1.0000000000000p-104;
        0x1.0000000000000p-103;
        0x1.0000000000000p-102;
        0x1.0000000000000p-101;
        0x1.0000000000000p-100;
        0x1.0000000000000p-99;
        0x1.0000000000000p-98;
        0x1.0000000000000p-97;
        0x1.0000000000000p-96;
        0x1.0000000000000p-95;
        0x1.0000000000000p-94;
        0x1.0000000000000p-93;
        0x1.0000000000000p-92;
        0x1.0000000000000p-91;
        0x1.0000000000000p-90;
        0x1.0000000000000p-89;
        0x1.0000000000000p-88;
        0x1.0000000000000p-87;
        0x1.0000000000000p-86;
        0x1.0000000000000p-85;
        0x1.0000000000000p-84;
        0x1.0000000000000p-83;
        0x1.0000000000000p-82;
        0x1.0000000000000p-81;
        0x1.0000000000000p-80;
        0x1.0000000000000p-79;
        0x1.0000000000000p-78;
        0x1.0000000000000p-77;
        0x1.0000000000000p-76;
        0x1.0000000000000p-75;
        0x1.0000000000000p-74;
        0x1.0000000000000p-73;
        0x1.0000000000000p-72;
        0x1.0000000000000p-71;
        0x1.0000000000000p-70;
        0x1.0000000000000p-69;
        0x1.0000000000000p-68;
        0x1.0000000000000p-67;
        0x1.0000000000000p-66;
        0x1.0000000000000p-65;
        0x1.0000000000000p-64;
        0x1.0000000000000p-63;
        0x1.0000000000000p-62;
        0x1.0000000000000p-61;
        0x1.0000000000000p-60;
        0x1.0000000000000p-59;
        0x1.0000000000000p-58;
        0x1.0000000000000p-57;
        0x1.0000000000000p-56;
        0x1.0000000000000p-55;
        0x1.0000000000000p-54;
        0x1.0000000000000p-53;
        0x1.0000000000000p-52;
        0x1.0000000000000p-51;
        0x1.0000000000000p-50;
        0x1.0000000000000p-49;
        0x1.0000000000000p-48;
        0x1.0000000000000p-47;
        0x1.0000000000000p-46;
        0x1.0000000000000p-45;
        0x1.0000000000000p-44;
        0x1.0000000000000p-43;
        0x1.0000000000000p-42;
        0x1.0000000000000p-41;
        0x1.0000000000000p-40;
        0x1.0000000000000p-39;
        0x1.0000000000000p-38;
        0x1.0000000000000p-37;
        0x1.0000000000000p-36;
        0x1.0000000000000p-35;
        0x1.0000000000000p-34;
        0x1.0000000000000p-33;
        0x1.0000000000000p-32;
        0x1.0000000000000p-31;
        0x1.0000000000000p-30;
        0x1.0000000000000p-29;
        0x1.0000000000000p-28;
        0x1.0000000000000p-27;
        0x1.0000000000000p-26;
        0x1.0000000000000p-25;
        0x1.0000000000000p-24;
        0x1.0000000000000p-23;
        0x1.0000000000000p-22;
        0x1.0000000000000p-21;
        0x1.0000000000000p-20;
        0x1.0000000000000p-19;
        0x1.0000000000000p-18;
        0x1.0000000000000p-17;
        0x1.0000000000000p-16;
        0x1.0000000000000p-15;
        0x1.0000000000000p-14;
        0x1.0000000000000p-13;
        0x1.0000000000000p-12;
        0x1.0000000000000p-11;
        0x1.0000000000000p-10;
        0x1.0000000000000p-9;
        0x1.0000000000000p-8;
        0x1.0000000000000p-7;
        0x1.0000000000000p-6;
        0x1.0000000000000p-5;
        0x1.0000000000000p-4;
        0x1.0000000000000p-3;
        0x1.0000000000000p-2;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
        0x1.0000000000000p+2;
        0x1.0000000000000p+3;
        0x1.0000000000000p+4;
        0x1.0000000000000p+5;
        0x1.0000000000000p+6;
        0x1.0000000000000p+7;
        0x1.0000000000000p+8;
        0x1.0000000000000p+9;
        0x1.0000000000000p+10;
        0x1.0000000000000p+11;
        0x1.0000000000000p+12;
        0x1.0000000000000p+13;
        0x1.0000000000000p+14;
        0x1.0000000000000p+15;
        0x1.0000000000000p+16;
        0x1.0000000000000p+17;
        0x1.0000000000000p+18;
        0x1.0000000000000p+19;
        0x1.0000000000000p+20;
        0x1.0000000000000p+21;
        0x1.0000000000000p+22;
        0x1.0000000000000p+23;
        0x1.0000000000000p+24;
        0x1.0000000000000p+25;
        0x1.0000000000000p+26;
        0x1.0000000000000p+27;
        0x1.0000000000000p+28;
        0x1.0000000000000p+29;
        0x1.0000000000000p+30;
        0x1.0000000000000p+31;
        0x1.0000000000000p+32;
        0x1.0000000000000p+33;
        0x1.0000000000000p+34;
        0x1.0000000000000p+35;
        0x1.0000000000000p+36;
        0x1.0000000000000p+37;
        0x1.0000000000000p+38;
        0x1.0000000000000p+39;
        0x1.0000000000000p+40;
        0x1.0000000000000p+41;
        0x1.0000000000000p+42;
        0x1.0000000000000p+43;
        0x1.0000000000000p+44;
        0x1.0000000000000p+45;
        0x1.0000000000000p+46;
        0x1.0000000000000p+47;
        0x1.0000000000000p+48;
        0x1.0000000000000p+49;
        0x1.0000000000000p+50;
        0x1.0000000000000p+51;
        0x1.0000000000000p+52;
        0x1.0000000000000p+53;
        0x1.0000000000000p+54;
        0x1.0000000000000p+55;
        0x1.0000000000000p+56;
        0x1.0000000000000p+57;
        0x1.0000000000000p+58;
        0x1.0000000000000p+59;
        0x1.0000000000000p+60;
        0x1.0000000000000p+61;
        0x1.0000000000000p+62;
        0x1.0000000000000p+63;
        0x1.0000000000000p+64;
        0x1.0000000000000p+65;
        0x1.0000000000000p+66;
        0x1.0000000000000p+67;
        0x1.0000000000000p+68;
        0x1.0000000000000p+69;
        0x1.0000000000000p+70;
        0x1.0000000000000p+71;
        0x1.0000000000000p+72;
        0x1.0000000000000p+73;
        0x1.0000000000000p+74;
        0x1.0000000000000p+75;
        0x1.0000000000000p+76;
        0x1.0000000000000p+77;
        0x1.0000000000000p+78;
        0x1.0000000000000p+79;
        0x1.0000000000000p+80;
        0x1.0000000000000p+81;
        0x1.0000000000000p+82;
        0x1.0000000000000p+83;
        0x1.0000000000000p+84;
        0x1.0000000000000p+85;
        0x1.0000000000000p+86;
        0x1.0000000000000p+87;
        0x1.0000000000000p+88;
        0x1.0000000000000p+89;
        0x1.0000000000000p+90;
        0x1.0000000000000p+91;
        0x1.0000000000000p+92;
        0x1.0000000000000p+93;
        0x1.0000000000000p+94;
        0x1.0000000000000p+95;
        0x1.0000000000000p+96;
        0x1.0000000000000p+97;
        0x1.0000000000000p+98;
        0x1.0000000000000p+99;
        0x1.0000000000000p+100;
        0x1.0000000000000p+101;
        0x1.0000000000000p+102;
        0x1.0000000000000p+103;
        0x1.0000000000000p+104;
        0x1.0000000000000p+105;
        0x1.0000000000000p+106;
        0x1.0000000000000p+107;
        0x1.0000000000000p+108;
        0x1.0000000000000p+109;
        0x1.0000000000000p+110;
        0x1.0000000000000p+111;
        0x1.0000000000000p+112;
        0x1.0000000000000p+113;
        0x1.0000000000000p+114;
        0x1.0000000000000p+115;
        0x1.0000000000000p+116;
        0x1.0000000000000p+117;
        0x1.0000000000000p+118;
        0x1.0000000000000p+119;
        0x1.0000000000000p+120;
        0x1.0000000000000p+121;
        0x1.0000000000000p+122;
        0x1.0000000000000p+123;
        0x1.0000000000000p+124;
        0x1.0000000000000p+125;
        0x1.0000000000000p+126;
        0x1.0000000000000p+127;
        Float.nan;
      |] );
  ]

let literal_table =
  [
    ( D.F4_e2m1fn,
      "f4E2M1FN",
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.0000000000000p-1;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3333333333333p-2;
        0x1.0000000000000p+0;
        0x1.0000000000000p-1;
        0x1.8000000000000p+2;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
      |],
      [|
        "1.000000e+00";
        "1.500000e+00";
        "-5.000000e-01";
        "0.000000e+00";
        "-0.000000e+00";
        "5.000000e-01";
        "1.000000e+00";
        "5.000000e-01";
        "6.000000e+00";
        "-0.000000e+00";
        "6.000000e+00";
        "-6.000000e+00";
      |] );
    ( D.F6_e2m3fn,
      "f6E2M3FN",
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.0000000000000p-1;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3333333333333p-2;
        0x1.0000000000000p+0;
        0x1.0000000000000p-3;
        0x1.e000000000000p+2;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
      |],
      [|
        "1.000000e+00";
        "1.500000e+00";
        "-5.000000e-01";
        "0.000000e+00";
        "-0.000000e+00";
        "2.500000e-01";
        "1.000000e+00";
        "1.250000e-01";
        "7.500000e+00";
        "-0.000000e+00";
        "7.500000e+00";
        "-7.500000e+00";
      |] );
    ( D.F6_e3m2fn,
      "f6E3M2FN",
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.0000000000000p-1;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3333333333333p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-4;
        0x1.c000000000000p+4;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
      |],
      [|
        "1.000000e+00";
        "1.500000e+00";
        "-5.000000e-01";
        "0.000000e+00";
        "-0.000000e+00";
        "3.125000e-01";
        "2.500000e-01";
        "6.250000e-02";
        "2.800000e+01";
        "-0.000000e+00";
        "2.800000e+01";
        "-2.800000e+01";
      |] );
    ( D.F8_e3m4,
      "f8E3M4",
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.0000000000000p-1;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3333333333333p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-6;
        0x1.f000000000000p+3;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
      |],
      [|
        "1.000000e+00";
        "1.500000e+00";
        "-5.000000e-01";
        "0.000000e+00";
        "-0.000000e+00";
        "2.968750e-01";
        "2.500000e-01";
        "1.562500e-02";
        "1.550000e+01";
        "0x78";
        "0x70";
        "0xF0";
      |] );
    ( D.F8_e4m3,
      "f8E4M3",
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.0000000000000p-1;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3333333333333p-2;
        0x1.0000000000000p-6;
        0x1.0000000000000p-9;
        0x1.e000000000000p+7;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
      |],
      [|
        "1.000000e+00";
        "1.500000e+00";
        "-5.000000e-01";
        "0.000000e+00";
        "-0.000000e+00";
        "3.125000e-01";
        "1.562500e-02";
        "1.953130e-03";
        "2.400000e+02";
        "0x7C";
        "0x78";
        "0xF8";
      |] );
    ( D.F8_e4m3b11fnuz,
      "f8E4M3B11FNUZ",
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.0000000000000p-1;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3333333333333p-2;
        0x1.0000000000000p-10;
        0x1.0000000000000p-13;
        0x1.e000000000000p+4;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
      |],
      [|
        "1.000000e+00";
        "1.500000e+00";
        "-5.000000e-01";
        "0.000000e+00";
        "0.000000e+00";
        "3.125000e-01";
        "9.765620e-04";
        "1.220700e-04";
        "3.000000e+01";
        "0x80";
        "0x80";
        "0x80";
      |] );
    ( D.F8_e4m3fn,
      "f8E4M3FN",
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.0000000000000p-1;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3333333333333p-2;
        0x1.0000000000000p-6;
        0x1.0000000000000p-9;
        0x1.c000000000000p+8;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
      |],
      [|
        "1.000000e+00";
        "1.500000e+00";
        "-5.000000e-01";
        "0.000000e+00";
        "-0.000000e+00";
        "3.125000e-01";
        "1.562500e-02";
        "1.953130e-03";
        "4.480000e+02";
        "0x7F";
        "0x7F";
        "0xFF";
      |] );
    ( D.F8_e4m3fnuz,
      "f8E4M3FNUZ",
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.0000000000000p-1;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3333333333333p-2;
        0x1.0000000000000p-7;
        0x1.0000000000000p-10;
        0x1.e000000000000p+7;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
      |],
      [|
        "1.000000e+00";
        "1.500000e+00";
        "-5.000000e-01";
        "0.000000e+00";
        "0.000000e+00";
        "3.125000e-01";
        "7.812500e-03";
        "9.765620e-04";
        "2.400000e+02";
        "0x80";
        "0x80";
        "0x80";
      |] );
    ( D.F8_e5m2,
      "f8E5M2",
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.0000000000000p-1;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3333333333333p-2;
        0x1.0000000000000p-14;
        0x1.0000000000000p-16;
        0x1.c000000000000p+15;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
      |],
      [|
        "1.000000e+00";
        "1.500000e+00";
        "-5.000000e-01";
        "0.000000e+00";
        "-0.000000e+00";
        "3.125000e-01";
        "6.103520e-05";
        "1.525880e-05";
        "5.734400e+04";
        "0x7E";
        "0x7C";
        "0xFC";
      |] );
    ( D.F8_e5m2fnuz,
      "f8E5M2FNUZ",
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.0000000000000p-1;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3333333333333p-2;
        0x1.0000000000000p-15;
        0x1.0000000000000p-17;
        0x1.c000000000000p+15;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
      |],
      [|
        "1.000000e+00";
        "1.500000e+00";
        "-5.000000e-01";
        "0.000000e+00";
        "0.000000e+00";
        "3.125000e-01";
        "3.051760e-05";
        "7.629390e-06";
        "5.734400e+04";
        "0x80";
        "0x80";
        "0x80";
      |] );
    ( D.F8_e8m0fnu,
      "f8E8M0FNU",
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        -0x1.0000000000000p-1;
        0x0.0p+0;
        -0x0.0p+0;
        0x1.3333333333333p-2;
        0x1.0000000000000p-127;
        0x1.0000000000000p-127;
        0x1.0000000000000p+127;
        Float.nan;
        Float.infinity;
        Float.neg_infinity;
      |],
      [|
        "1.000000e+00";
        "2.000000e+00";
        "0xFF";
        "0xFF";
        "0xFF";
        "2.500000e-01";
        "5.877470e-39";
        "5.877470e-39";
        "1.701410e+38";
        "0xFF";
        "0xFF";
        "0xFF";
      |] );
  ]

let narrow_int_types =
  [ (D.I2, "i2"); (D.I4, "i4"); (D.Uint2, "ui2"); (D.Uint4, "ui4") ]

let program_table =
  [
    (D.F4_e2m1fn, "stablehlo.multiply X, X : tensor<3xf4E2M1FN>");
    (D.F6_e2m3fn, "stablehlo.multiply X, X : tensor<3xf6E2M3FN>");
    (D.F6_e3m2fn, "stablehlo.multiply X, X : tensor<3xf6E3M2FN>");
    (D.F8_e3m4, "stablehlo.multiply X, X : tensor<3xf8E3M4>");
    (D.F8_e4m3, "stablehlo.multiply X, X : tensor<3xf8E4M3>");
    (D.F8_e4m3b11fnuz, "stablehlo.multiply X, X : tensor<3xf8E4M3B11FNUZ>");
    (D.F8_e4m3fn, "stablehlo.multiply X, X : tensor<3xf8E4M3FN>");
    (D.F8_e4m3fnuz, "stablehlo.multiply X, X : tensor<3xf8E4M3FNUZ>");
    (D.F8_e5m2, "stablehlo.multiply X, X : tensor<3xf8E5M2>");
    (D.F8_e5m2fnuz, "stablehlo.multiply X, X : tensor<3xf8E5M2FNUZ>");
    (D.F8_e8m0fnu, "stablehlo.multiply X, X : tensor<3xf8E8M0FNU>");
    (D.I2, "stablehlo.multiply X, X : tensor<3xi2>");
    (D.I4, "stablehlo.multiply X, X : tensor<3xi4>");
    (D.Uint2, "stablehlo.multiply X, X : tensor<3xui2>");
    (D.Uint4, "stablehlo.multiply X, X : tensor<3xui4>");
  ]

let int_wrap =
  [
    ( D.I2,
      [|
        -0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        -0x1.0000000000000p+0;
        -0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x0.0p+0;
        -0x1.0000000000000p+0;
        0x0.0p+0;
      |] );
    ( D.I4,
      [|
        -0x1.4000000000000p+2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+1;
        0x1.c000000000000p+2;
        -0x1.c000000000000p+2;
        0x1.0000000000000p+2;
        0x1.c000000000000p+2;
        0x0.0p+0;
      |] );
    ( D.Uint2,
      [|
        0x1.8000000000000p+1;
        0x1.0000000000000p+0;
        0x1.8000000000000p+1;
        0x1.8000000000000p+1;
        0x1.0000000000000p+0;
        0x0.0p+0;
        0x1.8000000000000p+1;
        0x0.0p+0;
      |] );
    ( D.Uint4,
      [|
        0x1.6000000000000p+3;
        0x1.0000000000000p+0;
        0x1.8000000000000p+1;
        0x1.c000000000000p+2;
        0x1.2000000000000p+3;
        0x1.0000000000000p+2;
        0x1.c000000000000p+2;
        0x0.0p+0;
      |] );
  ]

let int_wrap_src =
  [|
    -0x1.4000000000000p+2;
    0x1.0000000000000p+0;
    0x1.8000000000000p+1;
    0x1.c000000000000p+2;
    0x1.2000000000000p+3;
    0x1.4000000000000p+4;
    -0x1.2000000000000p+3;
    0x1.0000000000000p+4;
  |]

let float_to_int_src =
  [|
    -0x1.8000000000000p+0;
    0x1.e848000000000p+19;
    -0x1.e848000000000p+19;
    0x1.599999999999ap+1;
    Float.nan;
    0x1.f333333333333p+1;
  |]

let float_to_int =
  [
    ( D.I2,
      [|
        -0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        -0x1.0000000000000p+1;
        0x1.0000000000000p+0;
        0x0.0p+0;
        0x1.0000000000000p+0;
      |] );
    ( D.I4,
      [|
        -0x1.0000000000000p+0;
        0x1.c000000000000p+2;
        -0x1.0000000000000p+3;
        0x1.0000000000000p+1;
        0x0.0p+0;
        0x1.8000000000000p+1;
      |] );
    ( D.Uint2,
      [|
        0x0.0p+0;
        0x1.8000000000000p+1;
        0x0.0p+0;
        0x1.0000000000000p+1;
        0x0.0p+0;
        0x1.8000000000000p+1;
      |] );
    ( D.Uint4,
      [|
        0x0.0p+0;
        0x1.e000000000000p+3;
        0x0.0p+0;
        0x1.0000000000000p+1;
        0x0.0p+0;
        0x1.8000000000000p+1;
      |] );
  ]

let op_input =
  [|
    0x1.0000000000000p-1;
    0x1.0000000000000p-2;
    0x1.2000000000000p+0;
    0x1.c000000000000p+1;
    0x1.0000000000000p-3;
    0x1.c000000000000p+2;
    0x1.0000000000000p-4;
    0x1.0000000000000p+0;
    0x1.c000000000000p+0;
    0x1.4000000000000p+1;
    0x1.8000000000000p-2;
    0x1.4000000000000p+2;
    0x1.8000000000000p+0;
    0x1.8000000000000p+2;
    0x1.8000000000000p-1;
    0x1.8000000000000p+3;
  |]

let op_results =
  [
    ( D.F4_e2m1fn,
      [| 0x1.8000000000000p+2 |],
      [| 0x0.0p+0 |],
      [| 0x1.8000000000000p+2 |],
      [| 0x0.0p+0 |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.8000000000000p+0;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.0000000000000p+2;
        0x1.8000000000000p+2;
        0x1.0000000000000p+1;
        0x1.8000000000000p+1;
        0x1.8000000000000p+1;
        0x1.0000000000000p+1;
        0x1.8000000000000p+2;
        0x1.0000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.8000000000000p+1;
        0x0.0p+0;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x0.0p+0;
        0x1.8000000000000p+1;
        0x0.0p+0;
        0x1.0000000000000p+1;
        0x1.0000000000000p+1;
        0x0.0p+0;
        0x1.8000000000000p+1;
        0x0.0p+0;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x0.0p+0;
        0x1.8000000000000p+1;
      |],
      [|
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x1.8000000000000p+1;
        0x0.0p+0;
        0x1.8000000000000p+2;
        0x0.0p+0;
        0x1.0000000000000p-1;
        0x1.0000000000000p+1;
        0x1.8000000000000p+2;
        0x0.0p+0;
        0x1.8000000000000p+2;
        0x1.0000000000000p-1;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.0000000000000p-1;
        0x0.0p+0;
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
        0x0.0p+0;
        0x1.0000000000000p+1;
        0x0.0p+0;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        0x1.8000000000000p+0;
        0x1.0000000000000p-1;
        0x1.0000000000000p+1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
      |],
      [| 0x1.0000000000000p+0 |],
      [| 0x1.8000000000000p+2 |] );
    ( D.F8_e3m4,
      [| Float.infinity |],
      [| 0x1.0000000000000p-5 |],
      [| 0x1.8000000000000p+3 |],
      [| 0x1.0000000000000p-4 |],
      [|
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.e000000000000p+0;
        0x1.6000000000000p+2;
        0x1.6000000000000p+2;
        0x1.9000000000000p+3;
        0x1.9000000000000p+3;
        0x1.b000000000000p+3;
        0x1.e000000000000p+3;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
        Float.infinity;
      |],
      [|
        0x1.9000000000000p+3;
        0x1.0000000000000p+0;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.e000000000000p+2;
        0x1.4000000000000p+1;
        0x1.6000000000000p+1;
        0x1.6000000000000p+1;
        0x1.4000000000000p+1;
        0x1.e000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+0;
        0x1.9000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.8000000000000p-3;
        0x1.b000000000000p+2;
        0x1.5000000000000p+2;
        0x1.4000000000000p-1;
        0x1.5000000000000p+1;
        0x1.4000000000000p-3;
        0x1.c000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p-3;
        0x1.5000000000000p+1;
        0x1.4000000000000p-1;
        0x1.5000000000000p+2;
        0x1.b000000000000p+2;
        0x1.8000000000000p-3;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.8000000000000p-5;
        0x1.5000000000000p-2;
        0x1.8000000000000p-3;
        0x1.3000000000000p+1;
        0x1.0000000000000p-5;
        Float.infinity;
        0x1.0000000000000p-5;
        0x1.2000000000000p-1;
        0x1.c000000000000p+0;
        Float.infinity;
        0x1.8000000000000p-5;
        Float.infinity;
        0x1.b000000000000p-2;
        0x1.5000000000000p+2;
        0x1.8000000000000p+1;
        Float.infinity;
      |],
      [|
        0x1.7000000000000p-1;
        0x1.0000000000000p-1;
        0x1.1000000000000p+0;
        0x1.e000000000000p+0;
        0x1.7000000000000p-2;
        0x1.5000000000000p+1;
        0x1.0000000000000p-2;
        0x1.0000000000000p+0;
        0x1.5000000000000p+0;
        0x1.9000000000000p+0;
        0x1.4000000000000p-1;
        0x1.2000000000000p+1;
        0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p-1;
        0x1.c000000000000p+1;
      |],
      [| Float.nan |],
      [| Float.infinity |] );
    ( D.F8_e4m3,
      [| 0x1.6000000000000p+5 |],
      [| 0x1.c000000000000p-6 |],
      [| 0x1.8000000000000p+3 |],
      [| 0x1.0000000000000p-4 |],
      [|
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.e000000000000p+0;
        0x1.6000000000000p+2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+3;
        0x1.8000000000000p+3;
        0x1.a000000000000p+3;
        0x1.e000000000000p+3;
        0x1.2000000000000p+4;
        0x1.2000000000000p+4;
        0x1.8000000000000p+4;
        0x1.a000000000000p+4;
        0x1.0000000000000p+5;
        0x1.0000000000000p+5;
        0x1.6000000000000p+5;
      |],
      [|
        0x1.8000000000000p+3;
        0x1.0000000000000p+0;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.e000000000000p+2;
        0x1.4000000000000p+1;
        0x1.6000000000000p+1;
        0x1.6000000000000p+1;
        0x1.4000000000000p+1;
        0x1.e000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.8000000000000p-3;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p-1;
        0x1.4000000000000p+1;
        0x1.4000000000000p-3;
        0x1.c000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p-3;
        0x1.4000000000000p+1;
        0x1.4000000000000p-1;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p-3;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.6000000000000p-5;
        0x1.6000000000000p-2;
        0x1.8000000000000p-3;
        0x1.2000000000000p+1;
        0x1.a000000000000p-6;
        0x1.2000000000000p+4;
        0x1.a000000000000p-6;
        0x1.2000000000000p-1;
        0x1.c000000000000p+0;
        0x1.4000000000000p+5;
        0x1.c000000000000p-5;
        0x1.4000000000000p+5;
        0x1.c000000000000p-2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+1;
        0x1.8000000000000p+4;
      |],
      [|
        0x1.6000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.e000000000000p+0;
        0x1.6000000000000p-2;
        0x1.6000000000000p+1;
        0x1.0000000000000p-2;
        0x1.0000000000000p+0;
        0x1.6000000000000p+0;
        0x1.a000000000000p+0;
        0x1.4000000000000p-1;
        0x1.2000000000000p+1;
        0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p-1;
        0x1.c000000000000p+1;
      |],
      [| 0x1.6000000000000p+1 |],
      [| Float.infinity |] );
    ( D.F8_e4m3b11fnuz,
      [| Float.nan |],
      [| 0x1.c000000000000p-6 |],
      [| 0x1.8000000000000p+3 |],
      [| 0x1.0000000000000p-4 |],
      [|
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.e000000000000p+0;
        0x1.6000000000000p+2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+3;
        0x1.8000000000000p+3;
        0x1.a000000000000p+3;
        0x1.e000000000000p+3;
        0x1.2000000000000p+4;
        0x1.2000000000000p+4;
        0x1.8000000000000p+4;
        0x1.a000000000000p+4;
        Float.nan;
        Float.nan;
        Float.nan;
      |],
      [|
        0x1.8000000000000p+3;
        0x1.0000000000000p+0;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.e000000000000p+2;
        0x1.4000000000000p+1;
        0x1.6000000000000p+1;
        0x1.6000000000000p+1;
        0x1.4000000000000p+1;
        0x1.e000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.8000000000000p-3;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p-1;
        0x1.4000000000000p+1;
        0x1.4000000000000p-3;
        0x1.c000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p-3;
        0x1.4000000000000p+1;
        0x1.4000000000000p-1;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p-3;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.6000000000000p-5;
        0x1.6000000000000p-2;
        0x1.8000000000000p-3;
        0x1.2000000000000p+1;
        0x1.a000000000000p-6;
        0x1.2000000000000p+4;
        0x1.a000000000000p-6;
        0x1.2000000000000p-1;
        0x1.c000000000000p+0;
        Float.nan;
        0x1.c000000000000p-5;
        Float.nan;
        0x1.c000000000000p-2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+1;
        0x1.8000000000000p+4;
      |],
      [|
        0x1.6000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.e000000000000p+0;
        0x1.6000000000000p-2;
        0x1.6000000000000p+1;
        0x1.0000000000000p-2;
        0x1.0000000000000p+0;
        0x1.6000000000000p+0;
        0x1.a000000000000p+0;
        0x1.4000000000000p-1;
        0x1.2000000000000p+1;
        0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p-1;
        0x1.c000000000000p+1;
      |],
      [| Float.nan |],
      [| Float.nan |] );
    ( D.F8_e4m3fn,
      [| 0x1.6000000000000p+5 |],
      [| 0x1.c000000000000p-6 |],
      [| 0x1.8000000000000p+3 |],
      [| 0x1.0000000000000p-4 |],
      [|
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.e000000000000p+0;
        0x1.6000000000000p+2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+3;
        0x1.8000000000000p+3;
        0x1.a000000000000p+3;
        0x1.e000000000000p+3;
        0x1.2000000000000p+4;
        0x1.2000000000000p+4;
        0x1.8000000000000p+4;
        0x1.a000000000000p+4;
        0x1.0000000000000p+5;
        0x1.0000000000000p+5;
        0x1.6000000000000p+5;
      |],
      [|
        0x1.8000000000000p+3;
        0x1.0000000000000p+0;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.e000000000000p+2;
        0x1.4000000000000p+1;
        0x1.6000000000000p+1;
        0x1.6000000000000p+1;
        0x1.4000000000000p+1;
        0x1.e000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.8000000000000p-3;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p-1;
        0x1.4000000000000p+1;
        0x1.4000000000000p-3;
        0x1.c000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p-3;
        0x1.4000000000000p+1;
        0x1.4000000000000p-1;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p-3;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.6000000000000p-5;
        0x1.6000000000000p-2;
        0x1.8000000000000p-3;
        0x1.2000000000000p+1;
        0x1.a000000000000p-6;
        0x1.2000000000000p+4;
        0x1.a000000000000p-6;
        0x1.2000000000000p-1;
        0x1.c000000000000p+0;
        0x1.4000000000000p+5;
        0x1.c000000000000p-5;
        0x1.4000000000000p+5;
        0x1.c000000000000p-2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+1;
        0x1.8000000000000p+4;
      |],
      [|
        0x1.6000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.e000000000000p+0;
        0x1.6000000000000p-2;
        0x1.6000000000000p+1;
        0x1.0000000000000p-2;
        0x1.0000000000000p+0;
        0x1.6000000000000p+0;
        0x1.a000000000000p+0;
        0x1.4000000000000p-1;
        0x1.2000000000000p+1;
        0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p-1;
        0x1.c000000000000p+1;
      |],
      [| 0x1.6000000000000p+1 |],
      [| 0x1.2000000000000p+8 |] );
    ( D.F8_e4m3fnuz,
      [| 0x1.6000000000000p+5 |],
      [| 0x1.c000000000000p-6 |],
      [| 0x1.8000000000000p+3 |],
      [| 0x1.0000000000000p-4 |],
      [|
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.e000000000000p+0;
        0x1.6000000000000p+2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+3;
        0x1.8000000000000p+3;
        0x1.a000000000000p+3;
        0x1.e000000000000p+3;
        0x1.2000000000000p+4;
        0x1.2000000000000p+4;
        0x1.8000000000000p+4;
        0x1.a000000000000p+4;
        0x1.0000000000000p+5;
        0x1.0000000000000p+5;
        0x1.6000000000000p+5;
      |],
      [|
        0x1.8000000000000p+3;
        0x1.0000000000000p+0;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.e000000000000p+2;
        0x1.4000000000000p+1;
        0x1.6000000000000p+1;
        0x1.6000000000000p+1;
        0x1.4000000000000p+1;
        0x1.e000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.8000000000000p-3;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p-1;
        0x1.4000000000000p+1;
        0x1.4000000000000p-3;
        0x1.c000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p-3;
        0x1.4000000000000p+1;
        0x1.4000000000000p-1;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p-3;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.6000000000000p-5;
        0x1.6000000000000p-2;
        0x1.8000000000000p-3;
        0x1.2000000000000p+1;
        0x1.a000000000000p-6;
        0x1.2000000000000p+4;
        0x1.a000000000000p-6;
        0x1.2000000000000p-1;
        0x1.c000000000000p+0;
        0x1.4000000000000p+5;
        0x1.c000000000000p-5;
        0x1.4000000000000p+5;
        0x1.c000000000000p-2;
        0x1.6000000000000p+2;
        0x1.8000000000000p+1;
        0x1.8000000000000p+4;
      |],
      [|
        0x1.6000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.e000000000000p+0;
        0x1.6000000000000p-2;
        0x1.6000000000000p+1;
        0x1.0000000000000p-2;
        0x1.0000000000000p+0;
        0x1.6000000000000p+0;
        0x1.a000000000000p+0;
        0x1.4000000000000p-1;
        0x1.2000000000000p+1;
        0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p-1;
        0x1.c000000000000p+1;
      |],
      [| 0x1.6000000000000p+1 |],
      [| Float.nan |] );
    ( D.F8_e5m2,
      [| 0x1.0000000000000p+5 |],
      [| 0x1.8000000000000p-6 |],
      [| 0x1.8000000000000p+3 |],
      [| 0x1.0000000000000p-4 |],
      [|
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.c000000000000p+0;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.8000000000000p+3;
        0x1.8000000000000p+3;
        0x1.8000000000000p+3;
        0x1.c000000000000p+3;
        0x1.0000000000000p+4;
        0x1.0000000000000p+4;
        0x1.4000000000000p+4;
        0x1.4000000000000p+4;
        0x1.8000000000000p+4;
        0x1.8000000000000p+4;
        0x1.0000000000000p+5;
      |],
      [|
        0x1.8000000000000p+3;
        0x1.0000000000000p+0;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.8000000000000p+1;
        0x1.4000000000000p+1;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.8000000000000p-3;
        0x1.8000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p-1;
        0x1.4000000000000p+1;
        0x1.4000000000000p-3;
        0x1.c000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p-3;
        0x1.4000000000000p+1;
        0x1.4000000000000p-1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p-3;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.4000000000000p-5;
        0x1.4000000000000p-2;
        0x1.4000000000000p-3;
        0x1.4000000000000p+1;
        0x1.8000000000000p-6;
        0x1.4000000000000p+4;
        0x1.8000000000000p-6;
        0x1.4000000000000p-1;
        0x1.c000000000000p+0;
        0x1.4000000000000p+5;
        0x1.c000000000000p-5;
        0x1.4000000000000p+5;
        0x1.c000000000000p-2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+1;
        0x1.8000000000000p+4;
      |],
      [|
        0x1.8000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.c000000000000p+0;
        0x1.8000000000000p-2;
        0x1.4000000000000p+1;
        0x1.0000000000000p-2;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x1.8000000000000p+0;
        0x1.4000000000000p-1;
        0x1.0000000000000p+1;
        0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p-1;
        0x1.c000000000000p+1;
      |],
      [| 0x1.0000000000000p+1 |],
      [| 0x1.0000000000000p+8 |] );
    ( D.F8_e5m2fnuz,
      [| 0x1.0000000000000p+5 |],
      [| 0x1.8000000000000p-6 |],
      [| 0x1.8000000000000p+3 |],
      [| 0x1.0000000000000p-4 |],
      [|
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.c000000000000p+0;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.8000000000000p+3;
        0x1.8000000000000p+3;
        0x1.8000000000000p+3;
        0x1.c000000000000p+3;
        0x1.0000000000000p+4;
        0x1.0000000000000p+4;
        0x1.4000000000000p+4;
        0x1.4000000000000p+4;
        0x1.8000000000000p+4;
        0x1.8000000000000p+4;
        0x1.0000000000000p+5;
      |],
      [|
        0x1.8000000000000p+3;
        0x1.0000000000000p+0;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.4000000000000p+1;
        0x1.8000000000000p+1;
        0x1.8000000000000p+1;
        0x1.4000000000000p+1;
        0x1.c000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p+2;
        0x1.c000000000000p+2;
        0x1.0000000000000p+0;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.8000000000000p-3;
        0x1.8000000000000p+2;
        0x1.4000000000000p+2;
        0x1.4000000000000p-1;
        0x1.4000000000000p+1;
        0x1.4000000000000p-3;
        0x1.c000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p-3;
        0x1.4000000000000p+1;
        0x1.4000000000000p-1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p-3;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.4000000000000p-5;
        0x1.4000000000000p-2;
        0x1.4000000000000p-3;
        0x1.4000000000000p+1;
        0x1.8000000000000p-6;
        0x1.4000000000000p+4;
        0x1.8000000000000p-6;
        0x1.4000000000000p-1;
        0x1.c000000000000p+0;
        0x1.4000000000000p+5;
        0x1.c000000000000p-5;
        0x1.4000000000000p+5;
        0x1.c000000000000p-2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+1;
        0x1.8000000000000p+4;
      |],
      [|
        0x1.8000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.c000000000000p+0;
        0x1.8000000000000p-2;
        0x1.4000000000000p+1;
        0x1.0000000000000p-2;
        0x1.0000000000000p+0;
        0x1.4000000000000p+0;
        0x1.8000000000000p+0;
        0x1.4000000000000p-1;
        0x1.0000000000000p+1;
        0x1.4000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p-1;
        0x1.c000000000000p+1;
      |],
      [| 0x1.0000000000000p+1 |],
      [| 0x1.0000000000000p+8 |] );
    ( D.F8_e8m0fnu,
      [| Float.nan |],
      [| 0x1.0000000000000p-5 |],
      [| Float.nan |],
      [| Float.nan |],
      [|
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
      |],
      [|
        0x1.0000000000000p+4;
        0x1.0000000000000p+0;
        0x1.0000000000000p+3;
        0x1.0000000000000p+3;
        0x1.0000000000000p+2;
        0x1.0000000000000p+3;
        0x1.0000000000000p+1;
        0x1.0000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+1;
        0x1.0000000000000p+3;
        0x1.0000000000000p+2;
        0x1.0000000000000p+3;
        0x1.0000000000000p+3;
        0x1.0000000000000p+0;
        0x1.0000000000000p+4;
      |],
      [|
        0x1.0000000000000p+3;
        0x1.0000000000000p-2;
        0x1.0000000000000p+3;
        0x1.0000000000000p+3;
        0x1.0000000000000p-1;
        0x1.0000000000000p+2;
        0x1.0000000000000p-3;
        0x1.0000000000000p+1;
        0x1.0000000000000p+1;
        0x1.0000000000000p-3;
        0x1.0000000000000p+2;
        0x1.0000000000000p-1;
        0x1.0000000000000p+3;
        0x1.0000000000000p+3;
        0x1.0000000000000p-2;
        0x1.0000000000000p+3;
      |],
      [|
        0x1.0000000000000p-5;
        0x1.0000000000000p-2;
        0x1.0000000000000p-3;
        0x1.0000000000000p+1;
        0x1.0000000000000p-5;
        0x1.0000000000000p+4;
        0x1.0000000000000p-5;
        0x1.0000000000000p-1;
        0x1.0000000000000p+1;
        0x1.0000000000000p+5;
        0x1.0000000000000p-4;
        0x1.0000000000000p+5;
        0x1.0000000000000p-1;
        0x1.0000000000000p+3;
        0x1.0000000000000p+2;
        0x1.0000000000000p+5;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
        0x1.0000000000000p-2;
        0x1.0000000000000p+1;
        0x1.0000000000000p-2;
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x1.0000000000000p-1;
        0x1.0000000000000p+1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+2;
      |],
      [| Float.nan |],
      [| 0x1.0000000000000p+9 |] );
  ]

let special_table =
  [
    ( D.F4_e2m1fn,
      [| true; true; true |],
      [| false; true; false |],
      [| false; false; false |],
      [| 0x1.8000000000000p+0; 0x1.8000000000000p+0; 0x1.0000000000000p-1 |],
      [| 0x1.0000000000000p+1; 0x1.0000000000000p+2; 0x1.0000000000000p+0 |],
      [| 0x1.0000000000000p+0; 0x1.0000000000000p+1; 0x0.0p+0 |] );
    ( D.F8_e3m4,
      [| true; false; false |],
      [| false; true; false |],
      [| false; false; true |],
      [| 0x1.1000000000000p+0; 0x1.f000000000000p+0; 0x1.0000000000000p-1 |],
      [| 0x1.8000000000000p+5; 0x1.0000000000000p+6; 0x1.0000000000000p+5 |],
      [| 0x1.0000000000000p+0; 0x1.0000000000000p+1; 0x0.0p+0 |] );
    ( D.F8_e4m3,
      [| true; false; false |],
      [| false; true; false |],
      [| false; false; true |],
      [| 0x1.2000000000000p+0; 0x1.e000000000000p+0; 0x1.0000000000000p-1 |],
      [| 0x1.c000000000000p+5; 0x1.0000000000000p+6; 0x1.8000000000000p+5 |],
      [| 0x1.0000000000000p+0; 0x1.0000000000000p+1; 0x0.0p+0 |] );
    ( D.F8_e4m3b11fnuz,
      [| true; false; false |],
      [| false; false; false |],
      [| false; true; true |],
      [| 0x1.2000000000000p+0; 0x1.e000000000000p+0; 0x1.0000000000000p-1 |],
      [| 0x1.6000000000000p+6; 0x1.8000000000000p+6; 0x1.4000000000000p+6 |],
      [| 0x1.0000000000000p+0; 0x1.0000000000000p+1; 0x0.0p+0 |] );
    ( D.F8_e4m3fn,
      [| true; false; false |],
      [| false; false; false |],
      [| false; true; true |],
      [| 0x1.2000000000000p+0; 0x1.e000000000000p+0; 0x1.0000000000000p-1 |],
      [| 0x1.c000000000000p+5; 0x1.0000000000000p+6; 0x1.8000000000000p+5 |],
      [| 0x1.0000000000000p+0; 0x1.0000000000000p+1; 0x0.0p+0 |] );
    ( D.F8_e4m3fnuz,
      [| true; false; false |],
      [| false; false; false |],
      [| false; true; true |],
      [| 0x1.2000000000000p+0; 0x1.e000000000000p+0; 0x1.0000000000000p-1 |],
      [| 0x1.0000000000000p+6; 0x1.2000000000000p+6; 0x1.c000000000000p+5 |],
      [| 0x1.0000000000000p+0; 0x1.0000000000000p+1; 0x0.0p+0 |] );
    ( D.F8_e5m2,
      [| true; false; false |],
      [| false; true; false |],
      [| false; false; true |],
      [| 0x1.4000000000000p+0; 0x1.c000000000000p+0; 0x1.0000000000000p-1 |],
      [| 0x1.e000000000000p+5; 0x1.0000000000000p+6; 0x1.c000000000000p+5 |],
      [| 0x1.0000000000000p+0; 0x1.0000000000000p+1; 0x0.0p+0 |] );
    ( D.F8_e5m2fnuz,
      [| true; false; false |],
      [| false; false; false |],
      [| false; true; true |],
      [| 0x1.4000000000000p+0; 0x1.c000000000000p+0; 0x1.0000000000000p-1 |],
      [| 0x1.0000000000000p+6; 0x1.1000000000000p+6; 0x1.e000000000000p+5 |],
      [| 0x1.0000000000000p+0; 0x1.0000000000000p+1; 0x0.0p+0 |] );
    ( D.F8_e8m0fnu,
      [| true; false; false |],
      [| false; false; false |],
      [| false; true; true |],
      [| 0x1.0000000000000p-1; 0x1.0000000000000p-126; 0x1.0000000000000p-1 |],
      [| 0x1.fc00000000000p+6; 0x1.0000000000000p+7; 0x1.f800000000000p+6 |],
      [| 0x1.0000000000000p+0; 0x1.0000000000000p+1; Float.nan |] );
  ]

let int_ops =
  [
    ( D.I2,
      [|
        0x1.0000000000000p+0;
        -0x1.0000000000000p+1;
        0x0.0p+0;
        -0x1.0000000000000p+0;
      |],
      [| 0x0.0p+0; -0x1.0000000000000p+1; -0x1.0000000000000p+1; 0x0.0p+0 |],
      [|
        -0x1.0000000000000p+1;
        -0x1.0000000000000p+1;
        -0x1.0000000000000p+1;
        -0x1.0000000000000p+1;
      |],
      [| -0x1.0000000000000p+0; 0x0.0p+0; 0x0.0p+0; -0x1.0000000000000p+0 |],
      [|
        -0x1.0000000000000p+1;
        0x1.0000000000000p+0;
        -0x1.0000000000000p+0;
        0x0.0p+0;
      |],
      [| -0x1.0000000000000p+1; 0x0.0p+0; 0x0.0p+0; -0x1.0000000000000p+1 |],
      [| 0x0.0p+0; 0x1.0000000000000p+0; 0x0.0p+0; 0x1.0000000000000p+0 |],
      [| 0x0.0p+0; -0x1.0000000000000p+0; 0x0.0p+0; -0x1.0000000000000p+0 |],
      [|
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x0.0p+0;
        -0x1.0000000000000p+1;
      |],
      [| 0x1.0000000000000p+0; 0x0.0p+0; -0x1.0000000000000p+1; 0x0.0p+0 |],
      D.Uint2,
      [|
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
        0x0.0p+0;
        0x1.8000000000000p+1;
      |] );
    ( D.Uint2,
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+1;
        0x0.0p+0;
        0x1.0000000000000p+1;
      |],
      [|
        0x1.8000000000000p+1;
        0x1.8000000000000p+1;
        0x1.8000000000000p+1;
        0x1.8000000000000p+1;
      |],
      [|
        0x1.8000000000000p+1;
        0x1.8000000000000p+1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
      |],
      [| 0x1.0000000000000p+1; 0x0.0p+0; 0x0.0p+0; 0x1.0000000000000p+1 |],
      [|
        0x1.0000000000000p+1;
        0x0.0p+0;
        0x1.8000000000000p+1;
        0x1.0000000000000p+0;
      |],
      [| 0x1.0000000000000p+1; 0x1.0000000000000p+1; 0x0.0p+0; 0x0.0p+0 |],
      [| 0x0.0p+0; 0x1.0000000000000p+0; 0x0.0p+0; 0x1.0000000000000p+0 |],
      [| 0x0.0p+0; 0x1.8000000000000p+1; 0x0.0p+0; 0x1.8000000000000p+1 |],
      [|
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
        0x0.0p+0;
        0x1.0000000000000p+0;
      |],
      [| 0x1.0000000000000p+0; 0x0.0p+0; 0x1.0000000000000p+1; 0x0.0p+0 |],
      D.I2,
      [|
        0x1.0000000000000p+0;
        -0x1.0000000000000p+0;
        0x0.0p+0;
        -0x1.0000000000000p+1;
      |] );
    ( D.I4,
      [|
        0x1.0000000000000p+0;
        -0x1.0000000000000p+3;
        0x1.c000000000000p+2;
        -0x1.8000000000000p+1;
        0x1.4000000000000p+2;
        0x1.0000000000000p+1;
      |],
      [|
        0x1.8000000000000p+1;
        -0x1.8000000000000p+1;
        0x1.0000000000000p+2;
        0x1.0000000000000p+2;
        -0x1.8000000000000p+1;
        0x1.8000000000000p+1;
      |],
      [|
        -0x1.0000000000000p+0;
        0x1.8000000000000p+1;
        -0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        -0x1.8000000000000p+1;
        0x1.0000000000000p+0;
      |],
      [|
        0x1.0000000000000p+1;
        -0x1.0000000000000p+3;
        -0x1.4000000000000p+2;
        -0x1.4000000000000p+2;
        -0x1.0000000000000p+3;
        0x1.0000000000000p+1;
      |],
      [|
        -0x1.0000000000000p+1;
        0x1.c000000000000p+2;
        -0x1.0000000000000p+3;
        0x1.0000000000000p+1;
        -0x1.8000000000000p+2;
        -0x1.8000000000000p+1;
      |],
      [|
        0x1.0000000000000p+1;
        0x0.0p+0;
        -0x1.0000000000000p+1;
        -0x1.8000000000000p+2;
        -0x1.8000000000000p+2;
        0x1.0000000000000p+2;
      |],
      [|
        0x0.0p+0;
        0x1.0000000000000p+2;
        0x1.8000000000000p+1;
        0x1.8000000000000p+2;
        0x1.0000000000000p+1;
        0x1.0000000000000p+0;
      |],
      [|
        0x0.0p+0;
        -0x1.0000000000000p+2;
        0x1.8000000000000p+1;
        -0x1.0000000000000p+1;
        0x1.0000000000000p+1;
        0x1.0000000000000p+0;
      |],
      [|
        0x1.0000000000000p+0;
        0x1.4000000000000p+2;
        0x1.8000000000000p+1;
        0x1.c000000000000p+2;
        0x1.0000000000000p+1;
        0x1.0000000000000p+0;
      |],
      [|
        0x1.8000000000000p+1;
        0x0.0p+0;
        0x1.0000000000000p+0;
        0x0.0p+0;
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
      |],
      D.Uint4,
      [|
        0x1.0000000000000p+0;
        0x1.0000000000000p+3;
        0x1.c000000000000p+2;
        0x1.a000000000000p+3;
        0x1.4000000000000p+2;
        0x1.0000000000000p+1;
      |] );
    ( D.Uint4,
      [|
        0x1.0000000000000p+0;
        0x1.e000000000000p+3;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
        0x1.4000000000000p+2;
        0x1.0000000000000p+1;
      |],
      [|
        0x1.8000000000000p+1;
        0x1.0000000000000p+2;
        0x1.8000000000000p+1;
        0x1.8000000000000p+1;
        0x1.0000000000000p+2;
        0x1.8000000000000p+1;
      |],
      [|
        0x1.e000000000000p+3;
        0x1.4000000000000p+3;
        0x1.6000000000000p+3;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.0000000000000p+0;
      |],
      [|
        0x1.0000000000000p+1;
        0x1.6000000000000p+3;
        0x1.0000000000000p+2;
        0x1.0000000000000p+2;
        0x1.6000000000000p+3;
        0x1.0000000000000p+1;
      |],
      [|
        0x1.c000000000000p+3;
        0x0.0p+0;
        0x1.0000000000000p+3;
        0x1.8000000000000p+1;
        0x1.4000000000000p+3;
        0x1.a000000000000p+3;
      |],
      [|
        0x1.0000000000000p+1;
        0x1.c000000000000p+3;
        0x1.c000000000000p+3;
        0x1.0000000000000p+3;
        0x1.4000000000000p+3;
        0x1.0000000000000p+2;
      |],
      [|
        0x0.0p+0;
        0x1.c000000000000p+2;
        0x1.8000000000000p+1;
        0x1.8000000000000p+2;
        0x1.0000000000000p+1;
        0x1.0000000000000p+0;
      |],
      [|
        0x0.0p+0;
        0x1.e000000000000p+3;
        0x1.8000000000000p+1;
        0x1.c000000000000p+3;
        0x1.0000000000000p+1;
        0x1.0000000000000p+0;
      |],
      [|
        0x1.0000000000000p+0;
        0x1.0000000000000p+2;
        0x1.8000000000000p+1;
        0x1.0000000000000p+1;
        0x1.0000000000000p+1;
        0x1.0000000000000p+0;
      |],
      [|
        0x1.8000000000000p+1;
        0x0.0p+0;
        0x1.0000000000000p+0;
        0x0.0p+0;
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
      |],
      D.I4,
      [|
        0x1.0000000000000p+0;
        -0x1.0000000000000p+0;
        0x1.c000000000000p+2;
        -0x1.0000000000000p+2;
        0x1.4000000000000p+2;
        0x1.0000000000000p+1;
      |] );
  ]

let alias_table =
  [
    ( "bfloat16",
      ST.bfloat16,
      0x1.199999999999ap+0,
      D.Bf16,
      D.Bf16,
      0x1.1a00000000000p+0 );
    ( "float16",
      ST.float16,
      0x1.199999999999ap+0,
      D.F16,
      D.F16,
      0x1.1980000000000p+0 );
    ("int8", ST.int8, 0x1.9000000000000p+6, D.I8, D.I8, 0x1.9000000000000p+6);
    ( "int16",
      ST.int16,
      0x1.b580000000000p+12,
      D.I16,
      D.I16,
      0x1.b580000000000p+12 );
    ( "uint",
      ST.uint,
      0x1.8000000000000p+1,
      D.Uint32,
      D.Uint64,
      0x1.8000000000000p+1 );
    ( "uint8",
      ST.uint8,
      0x1.9000000000000p+7,
      D.Uint8,
      D.Uint8,
      0x1.9000000000000p+7 );
    ( "uint16",
      ST.uint16,
      0x1.d4c0000000000p+15,
      D.Uint16,
      D.Uint16,
      0x1.d4c0000000000p+15 );
    ( "uint32",
      ST.uint32,
      0x1.c000000000000p+2,
      D.Uint32,
      D.Uint32,
      0x1.c000000000000p+2 );
    ( "uint64",
      ST.uint64,
      0x1.8000000000000p+1,
      D.Uint32,
      D.Uint64,
      0x1.8000000000000p+1 );
    ( "complex64",
      ST.complex64,
      0x1.8000000000000p+0,
      D.Complex64,
      D.Complex64,
      0x1.8000000000000p+0 );
    ( "complex128",
      ST.complex128,
      0x1.8000000000000p+0,
      D.Complex64,
      D.Complex128,
      0x1.8000000000000p+0 );
    ( "csingle",
      ST.csingle,
      0x1.8000000000000p+0,
      D.Complex64,
      D.Complex64,
      0x1.8000000000000p+0 );
    ( "cdouble",
      ST.cdouble,
      0x1.8000000000000p+0,
      D.Complex64,
      D.Complex128,
      0x1.8000000000000p+0 );
    ("int2", ST.int2, 0x1.0000000000000p+0, D.I2, D.I2, 0x1.0000000000000p+0);
    ("int4", ST.int4, 0x1.c000000000000p+2, D.I4, D.I4, 0x1.c000000000000p+2);
    ( "uint2",
      ST.uint2,
      0x1.8000000000000p+1,
      D.Uint2,
      D.Uint2,
      0x1.8000000000000p+1 );
    ( "uint4",
      ST.uint4,
      0x1.e000000000000p+3,
      D.Uint4,
      D.Uint4,
      0x1.e000000000000p+3 );
    ( "float4_e2m1fn",
      ST.float4_e2m1fn,
      0x1.999999999999ap-1,
      D.F4_e2m1fn,
      D.F4_e2m1fn,
      0x1.0000000000000p+0 );
    ( "float8_e3m4",
      ST.float8_e3m4,
      0x1.199999999999ap+0,
      D.F8_e3m4,
      D.F8_e3m4,
      0x1.2000000000000p+0 );
    ( "float8_e4m3",
      ST.float8_e4m3,
      0x1.199999999999ap+0,
      D.F8_e4m3,
      D.F8_e4m3,
      0x1.2000000000000p+0 );
    ( "float8_e4m3b11fnuz",
      ST.float8_e4m3b11fnuz,
      0x1.199999999999ap+0,
      D.F8_e4m3b11fnuz,
      D.F8_e4m3b11fnuz,
      0x1.2000000000000p+0 );
    ( "float8_e4m3fn",
      ST.float8_e4m3fn,
      0x1.9000000000000p+6,
      D.F8_e4m3fn,
      D.F8_e4m3fn,
      0x1.8000000000000p+6 );
    ( "float8_e4m3fnuz",
      ST.float8_e4m3fnuz,
      0x1.199999999999ap+0,
      D.F8_e4m3fnuz,
      D.F8_e4m3fnuz,
      0x1.2000000000000p+0 );
    ( "float8_e5m2",
      ST.float8_e5m2,
      0x1.199999999999ap+0,
      D.F8_e5m2,
      D.F8_e5m2,
      0x1.0000000000000p+0 );
    ( "float8_e5m2fnuz",
      ST.float8_e5m2fnuz,
      0x1.199999999999ap+0,
      D.F8_e5m2fnuz,
      D.F8_e5m2fnuz,
      0x1.0000000000000p+0 );
    ( "float8_e8m0fnu",
      ST.float8_e8m0fnu,
      0x1.199999999999ap+0,
      D.F8_e8m0fnu,
      D.F8_e8m0fnu,
      0x1.0000000000000p+0 );
  ]

let promotion_messages =
  [
    ( D.F8_e4m3,
      D.F8_e5m2,
      "Input dtypes ('float8_e4m3', 'float8_e5m2') have no available implicit \
       dtype promotion path. To avoid unintended promotion, 8-bit floats do \
       not support implicit promotion. If you'd like your inputs to be \
       promoted to another type, you can do so explicitly using e.g. \
       x.astype('float32')" );
    ( D.F4_e2m1fn,
      D.F16,
      "Input dtypes ('float4_e2m1fn', 'float16') have no available implicit \
       dtype promotion path. To avoid unintended promotion, 4-bit floats do \
       not support implicit promotion. If you'd like your inputs to be \
       promoted to another type, you can do so explicitly using e.g. \
       x.astype('float32')" );
    ( D.F6_e3m2fn,
      D.F32,
      "Input dtypes ('float6_e3m2fn', 'float32') have no available implicit \
       dtype promotion path. To avoid unintended promotion, 6-bit floats do \
       not support implicit promotion. If you'd like your inputs to be \
       promoted to another type, you can do so explicitly using e.g. \
       x.astype('float32')" );
    ( D.I4,
      D.I2,
      "Input dtypes ('int4', 'int2') have no available implicit dtype \
       promotion path. To avoid unintended promotion, 1-bit, 2-bit and 4-bit \
       integers do not support implicit promotion. If you'd like your inputs \
       to be promoted to another type, you can do so explicitly using e.g. \
       x.astype('int32')" );
    ( D.Uint4,
      D.I8,
      "Input dtypes ('uint4', 'int8') have no available implicit dtype \
       promotion path. To avoid unintended promotion, 1-bit, 2-bit and 4-bit \
       integers do not support implicit promotion. If you'd like your inputs \
       to be promoted to another type, you can do so explicitly using e.g. \
       x.astype('int32')" );
  ]

let sort_table =
  [
    ( D.F4_e2m1fn,
      [|
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        0x1.0000000000000p+1;
        0x1.0000000000000p+1;
        0x1.0000000000000p+2;
        0x1.0000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.0000000000000p+0;
        0x1.0000000000000p+2;
        0x1.8000000000000p+2;
        0x0.0p+0;
        0x1.4000000000000p+3;
        0x1.0000000000000p+1;
        0x1.c000000000000p+2;
        0x1.c000000000000p+3;
        0x1.8000000000000p+3;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.8000000000000p+1;
        0x1.6000000000000p+3;
        0x1.4000000000000p+2;
        0x1.a000000000000p+3;
        0x1.e000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+2;
        0x1.0000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
        0x1.8000000000000p+2;
      |],
      [|
        0x1.0000000000000p-1;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
        0x0.0p+0;
      |],
      [| 0x1.4000000000000p+2 |],
      [| 0x1.0000000000000p+0 |],
      [| 0x0.0p+0; 0x1.0000000000000p-1; 0x1.0000000000000p+0 |] );
    ( D.F8_e3m4,
      [|
        0x1.0000000000000p-4;
        0x1.0000000000000p-3;
        0x1.0000000000000p-2;
        0x1.8000000000000p-2;
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.0000000000000p+0;
        0x1.2000000000000p+0;
        0x1.8000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+0;
        0x1.4000000000000p+3;
        0x0.0p+0;
        0x1.c000000000000p+3;
        0x1.c000000000000p+2;
        0x1.0000000000000p+1;
        0x1.8000000000000p+3;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.8000000000000p+1;
        0x1.6000000000000p+3;
        0x1.a000000000000p+3;
        0x1.4000000000000p+2;
        0x1.e000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.2000000000000p+0;
        0x1.c000000000000p+1;
        0x1.c000000000000p+1;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-3;
        0x1.0000000000000p-3;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
      |],
      [| 0x1.e000000000000p+3 |],
      [| 0x1.8000000000000p+2 |],
      [| 0x0.0p+0; 0x1.0000000000000p-1; 0x1.0000000000000p+0 |] );
    ( D.F8_e4m3,
      [|
        0x1.0000000000000p-4;
        0x1.0000000000000p-3;
        0x1.0000000000000p-2;
        0x1.8000000000000p-2;
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.0000000000000p+0;
        0x1.2000000000000p+0;
        0x1.8000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+0;
        0x1.4000000000000p+3;
        0x0.0p+0;
        0x1.c000000000000p+3;
        0x1.c000000000000p+2;
        0x1.0000000000000p+1;
        0x1.8000000000000p+3;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.8000000000000p+1;
        0x1.6000000000000p+3;
        0x1.a000000000000p+3;
        0x1.4000000000000p+2;
        0x1.e000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.2000000000000p+0;
        0x1.c000000000000p+1;
        0x1.c000000000000p+1;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-3;
        0x1.0000000000000p-3;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
      |],
      [| 0x1.e000000000000p+3 |],
      [| 0x1.8000000000000p+2 |],
      [| 0x0.0p+0; 0x1.0000000000000p-1; 0x1.0000000000000p+0 |] );
    ( D.F8_e4m3fn,
      [|
        0x1.0000000000000p-4;
        0x1.0000000000000p-3;
        0x1.0000000000000p-2;
        0x1.8000000000000p-2;
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.0000000000000p+0;
        0x1.2000000000000p+0;
        0x1.8000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+0;
        0x1.4000000000000p+3;
        0x0.0p+0;
        0x1.c000000000000p+3;
        0x1.c000000000000p+2;
        0x1.0000000000000p+1;
        0x1.8000000000000p+3;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.8000000000000p+1;
        0x1.6000000000000p+3;
        0x1.a000000000000p+3;
        0x1.4000000000000p+2;
        0x1.e000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.2000000000000p+0;
        0x1.c000000000000p+1;
        0x1.c000000000000p+1;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-3;
        0x1.0000000000000p-3;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
      |],
      [| 0x1.e000000000000p+3 |],
      [| 0x1.8000000000000p+2 |],
      [| 0x0.0p+0; 0x1.0000000000000p-1; 0x1.0000000000000p+0 |] );
    ( D.F8_e4m3fnuz,
      [|
        0x1.0000000000000p-4;
        0x1.0000000000000p-3;
        0x1.0000000000000p-2;
        0x1.8000000000000p-2;
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.0000000000000p+0;
        0x1.2000000000000p+0;
        0x1.8000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+0;
        0x1.4000000000000p+3;
        0x0.0p+0;
        0x1.c000000000000p+3;
        0x1.c000000000000p+2;
        0x1.0000000000000p+1;
        0x1.8000000000000p+3;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.8000000000000p+1;
        0x1.6000000000000p+3;
        0x1.a000000000000p+3;
        0x1.4000000000000p+2;
        0x1.e000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.2000000000000p+0;
        0x1.c000000000000p+1;
        0x1.c000000000000p+1;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-3;
        0x1.0000000000000p-3;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
      |],
      [| 0x1.e000000000000p+3 |],
      [| 0x1.8000000000000p+2 |],
      [| 0x0.0p+0; 0x1.0000000000000p-1; 0x1.0000000000000p+0 |] );
    ( D.F8_e5m2,
      [|
        0x1.0000000000000p-4;
        0x1.0000000000000p-3;
        0x1.0000000000000p-2;
        0x1.8000000000000p-2;
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+0;
        0x1.4000000000000p+3;
        0x0.0p+0;
        0x1.c000000000000p+3;
        0x1.0000000000000p+1;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.8000000000000p+1;
        0x1.6000000000000p+3;
        0x1.a000000000000p+3;
        0x1.4000000000000p+2;
        0x1.e000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.c000000000000p+1;
        0x1.c000000000000p+1;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-3;
        0x1.0000000000000p-3;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
      |],
      [| 0x1.e000000000000p+3 |],
      [| 0x1.8000000000000p+2 |],
      [| 0x0.0p+0; 0x1.0000000000000p-1; 0x1.0000000000000p+0 |] );
    ( D.F8_e8m0fnu,
      [|
        0x1.0000000000000p-4;
        0x1.0000000000000p-3;
        0x1.0000000000000p-2;
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x1.0000000000000p+1;
        0x1.0000000000000p+1;
        0x1.0000000000000p+1;
        0x1.0000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+3;
        0x1.0000000000000p+3;
        0x1.0000000000000p+4;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+0;
        0x0.0p+0;
        0x1.4000000000000p+3;
        0x1.0000000000000p+1;
        0x1.c000000000000p+2;
        0x1.c000000000000p+3;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.8000000000000p+3;
        0x1.8000000000000p+1;
        0x1.6000000000000p+3;
        0x1.4000000000000p+2;
        0x1.a000000000000p+3;
        0x1.e000000000000p+3;
      |],
      [|
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
      |],
      [|
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
        Float.nan;
      |],
      [| 0x0.0p+0 |],
      [| 0x0.0p+0 |],
      [| Float.nan; Float.nan; 0x1.0000000000000p+0 |] );
    ( D.F8_e4m3b11fnuz,
      [|
        0x1.0000000000000p-4;
        0x1.0000000000000p-3;
        0x1.0000000000000p-2;
        0x1.8000000000000p-2;
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.0000000000000p+0;
        0x1.2000000000000p+0;
        0x1.8000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+0;
        0x1.4000000000000p+3;
        0x0.0p+0;
        0x1.c000000000000p+3;
        0x1.c000000000000p+2;
        0x1.0000000000000p+1;
        0x1.8000000000000p+3;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.8000000000000p+1;
        0x1.6000000000000p+3;
        0x1.a000000000000p+3;
        0x1.4000000000000p+2;
        0x1.e000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.2000000000000p+0;
        0x1.c000000000000p+1;
        0x1.c000000000000p+1;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-3;
        0x1.0000000000000p-3;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
      |],
      [| 0x1.e000000000000p+3 |],
      [| 0x1.8000000000000p+2 |],
      [| 0x0.0p+0; 0x1.0000000000000p-1; 0x1.0000000000000p+0 |] );
    ( D.F8_e5m2fnuz,
      [|
        0x1.0000000000000p-4;
        0x1.0000000000000p-3;
        0x1.0000000000000p-2;
        0x1.8000000000000p-2;
        0x1.0000000000000p-1;
        0x1.8000000000000p-1;
        0x1.0000000000000p+0;
        0x1.0000000000000p+0;
        0x1.8000000000000p+0;
        0x1.c000000000000p+0;
        0x1.4000000000000p+1;
        0x1.c000000000000p+1;
        0x1.4000000000000p+2;
        0x1.8000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.8000000000000p+2;
        0x1.0000000000000p+2;
        0x1.0000000000000p+0;
        0x1.4000000000000p+3;
        0x0.0p+0;
        0x1.c000000000000p+3;
        0x1.0000000000000p+1;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
        0x1.0000000000000p+3;
        0x1.2000000000000p+3;
        0x1.8000000000000p+1;
        0x1.6000000000000p+3;
        0x1.a000000000000p+3;
        0x1.4000000000000p+2;
        0x1.e000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-1;
        0x1.0000000000000p+0;
        0x1.c000000000000p+1;
        0x1.c000000000000p+1;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.c000000000000p+2;
        0x1.8000000000000p+3;
      |],
      [|
        0x1.0000000000000p-1;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-2;
        0x1.0000000000000p-3;
        0x1.0000000000000p-3;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
        0x1.0000000000000p-4;
      |],
      [| 0x1.e000000000000p+3 |],
      [| 0x1.8000000000000p+2 |],
      [| 0x0.0p+0; 0x1.0000000000000p-1; 0x1.0000000000000p+0 |] );
  ]

let uint_floor_div =
  [
    ( D.Uint2,
      [|
        0x1.0000000000000p+0;
        0x1.8000000000000p+1;
        0x0.0p+0;
        0x1.0000000000000p+1;
      |],
      [| 0x0.0p+0; 0x1.8000000000000p+1; 0x0.0p+0; 0x1.0000000000000p+1 |] );
    ( D.Uint4,
      [|
        0x1.0000000000000p+0;
        0x1.e000000000000p+3;
        0x0.0p+0;
        0x1.0000000000000p+1;
      |],
      [| 0x0.0p+0; 0x1.e000000000000p+3; 0x0.0p+0; 0x1.0000000000000p+1 |] );
    ( D.Uint8,
      [|
        0x1.0000000000000p+0;
        0x1.9000000000000p+7;
        0x0.0p+0;
        0x1.0000000000000p+1;
      |],
      [| 0x0.0p+0; 0x1.fe00000000000p+7; 0x0.0p+0; 0x1.0000000000000p+1 |] );
  ]

let narrow_finite =
  [
    ( D.F16,
      [| 0x1.0000000000000p+0; 0x0.0p+0; 0x0.0p+0 |],
      [| 0x0.0p+0; 0x1.0000000000000p+0; 0x0.0p+0 |] );
    ( D.Bf16,
      [| 0x1.0000000000000p+0; 0x0.0p+0; 0x0.0p+0 |],
      [| 0x0.0p+0; 0x1.0000000000000p+0; 0x0.0p+0 |] );
  ]

let convert_tangents =
  [
    (D.F16, 0x1.8000000000000p+0, 0x1.0000000000000p+1);
    (D.Bf16, 0x1.8000000000000p+0, 0x1.0000000000000p+1);
    (D.F8_e4m3, 0x1.8000000000000p+0, 0x1.0000000000000p+1);
    (D.F4_e2m1fn, 0x1.8000000000000p+0, 0x1.0000000000000p+1);
  ]

let signbit_message =
  "jax.numpy.signbit only supports 16, 32, and 64-bit types."

let grad_e4m3 = [| 0x1.0000000000000p+0; 0x1.8000000000000p+1 |]

let dt_t =
  Alcotest.testable
    (fun ppf d -> Format.pp_print_string ppf (D.numpy_name d))
    ( = )

let same a b =
  (Float.is_nan a && Float.is_nan b)
  || (a = b && Float.sign_bit a = Float.sign_bit b)

let check_floats name expected actual =
  Alcotest.(check int)
    (name ^ " length") (Array.length expected) (Array.length actual);
  Array.iteri
    (fun i e ->
      if not (same e actual.(i)) then
        Alcotest.failf "%s[%d]: expected %h, got %h" name i e actual.(i))
    expected

let to_nd = function
  | T.Concrete nd -> nd
  | T.Device b -> Ojax.Pjrt.Buffer.to_host b
  | T.Tracer _ -> Alcotest.fail "expected a concrete value"

let flat nd = List.rev (Nd.fold (fun acc x -> x :: acc) [] nd) |> Array.of_list
let flat_v v = flat (to_nd v)

let dtype_of_string s =
  match D.of_numpy_name s with
  | Some d -> d
  | None -> Alcotest.failf "unknown dtype %s" s

let check_promotion_row x64 (a_name, row) =
  let a = dtype_of_string a_name in
  let results = Array.of_list (String.split_on_char ' ' row) in
  let others = Array.of_list name_order in
  let n = Array.length others in
  Array.iteri
    (fun i expected ->
      let label = Printf.sprintf "x64=%b %s column %d" x64 a_name i in
      if i < n then
        match Dt.promote_types a others.(i) with
        | actual -> Alcotest.check dt_t label (dtype_of_string expected) actual
        | exception Dt.Type_promotion_error _ ->
            Alcotest.(check string) label expected "ERR"
      else
        let weak =
          match i - n with 0 -> D.I32 | 1 -> D.F32 | _ -> D.Complex64
        in
        match Dt.result_type [ (a, false); (weak, true) ] with
        | actual, _ ->
            Alcotest.check dt_t label
              (Dt.canonicalize_dtype (dtype_of_string expected))
              (Dt.canonicalize_dtype actual)
        | exception Dt.Type_promotion_error _ ->
            Alcotest.(check string) label expected "ERR")
    results

let promotion_x32 () =
  Cfg.with_value Cfg.enable_x64 false (fun () ->
      List.iter (check_promotion_row false) promote_x32)

let promotion_x64 () =
  Cfg.with_value Cfg.enable_x64 true (fun () ->
      List.iter (check_promotion_row true) promote_x64)

let promotion_errors () =
  List.iter
    (fun (a, b, message) ->
      Alcotest.check_raises message (Dt.Type_promotion_error message) (fun () ->
          ignore (Dt.promote_types a b)))
    promotion_messages

let result_type_weak () =
  Alcotest.(check (pair dt_t bool))
    "weak f8 goes to default float" (D.F32, true)
    (Dt.result_type [ (D.F8_e4m3, true) ]);
  Alcotest.(check (pair dt_t bool))
    "f8 with weak float" (D.F8_e4m3, false)
    (Dt.result_type [ (D.F8_e4m3, false); (D.F32, true) ]);
  Alcotest.(check (pair dt_t bool))
    "i4 with weak int" (D.I4, false)
    (Dt.result_type [ (D.I4, false); (D.I32, true) ]);
  Alcotest.check_raises "i4 with weak float"
    (Dt.Type_promotion_error
       "Input dtypes ('int4', \"<class 'float'>\") have no available implicit \
        dtype promotion path. To avoid unintended promotion, 1-bit, 2-bit and \
        4-bit integers do not support implicit promotion. If you'd like your \
        inputs to be promoted to another type, you can do so explicitly using \
        e.g. x.astype('int32')") (fun () ->
      ignore (Dt.result_type [ (D.I4, false); (D.F32, true) ]))

let finfo_values () =
  List.iter
    (fun ( d,
           bits,
           nmant,
           nexp,
           precision,
           resolution,
           eps,
           epsneg,
           fmax,
           fmin,
           tiny,
           sub ) ->
      let f = Dt.finfo d in
      let name = D.numpy_name d in
      Alcotest.(check int) (name ^ " bits") bits f.Dt.bits;
      Alcotest.(check int) (name ^ " nmant") nmant f.Dt.nmant;
      Alcotest.(check int) (name ^ " nexp") nexp f.Dt.nexp;
      Alcotest.(check int) (name ^ " precision") precision f.Dt.precision;
      check_floats (name ^ " values")
        [| resolution; eps; epsneg; fmax; fmin; tiny; sub |]
        [|
          f.Dt.resolution;
          f.Dt.eps;
          f.Dt.epsneg;
          f.Dt.fmax;
          f.Dt.fmin;
          f.Dt.tiny;
          f.Dt.smallest_subnormal;
        |])
    finfo_table

let iinfo_values () =
  List.iter
    (fun (d, lo, hi, bits) ->
      let i = Dt.iinfo d in
      check_floats (D.numpy_name d) [| lo; hi |] [| i.Dt.imin; i.Dt.imax |];
      Alcotest.(check int) (D.numpy_name d ^ " bits") bits i.Dt.ibits)
    iinfo_table

let sizes_and_scalars () =
  List.iter
    (fun (d, bits, scalar, inexact) ->
      let name = D.numpy_name d in
      Alcotest.(check int) (name ^ " itemsize_bits") bits (Dt.itemsize_bits d);
      Alcotest.(check bool)
        (name ^ " scalar_type_of") true
        (Dt.scalar_type_of d = scalar);
      Alcotest.check dt_t (name ^ " to_inexact") inexact (Dt.to_inexact_dtype d);
      let v = T.Concrete (Nd.of_floats d [| 1 |] [| 0.0 |]) in
      Alcotest.(check bool)
        (name ^ " Api.scalar_type_of")
        true
        (Api.scalar_type_of v = scalar))
    bits_and_scalar

let names () =
  List.iter
    (fun (d, short) ->
      Alcotest.(check string) (D.numpy_name d) short (D.short_name d);
      Alcotest.(check (option dt_t))
        (D.numpy_name d ^ " lookup")
        (Some d)
        (D.of_numpy_name (D.numpy_name d)))
    short_names

let isdtype_kinds () =
  List.iter
    (fun (d, row) ->
      List.iter2
        (fun kind expected ->
          Alcotest.(check bool)
            (D.numpy_name d ^ " " ^ kind)
            expected (Dt.isdtype d kind))
        kinds row)
    isdtype_table

let issubdtype_kinds () =
  List.iter
    (fun (d, row) ->
      List.iteri
        (fun i expected ->
          Alcotest.(check bool)
            (Printf.sprintf "%s kind %d" (D.numpy_name d) i)
            expected
            (Dt.issubdtype d (List.nth issub_kinds i)))
        row)
    issub_table

let numpy_classes () =
  List.iter
    (fun (d, row) ->
      List.iteri
        (fun i expected ->
          Alcotest.(check bool)
            (Printf.sprintf "%s class %d" (D.numpy_name d) i)
            expected
            (NL.issubdtype (NL.Cdtype d) (List.nth class_kinds i)))
        row)
    class_table;
  List.iter
    (fun (a, b, expected) ->
      Alcotest.(check bool) "class pair" expected (NL.issubdtype a b))
    class_pairs

let rounding () =
  List.iter
    (fun (d, expected, codes) ->
      let name = D.numpy_name d in
      let nd = Nd.of_floats d [| Array.length round_inputs |] round_inputs in
      check_floats (name ^ " storage") expected (flat nd);
      let f = Option.get (FF.of_dtype d) in
      Array.iteri
        (fun i v ->
          Alcotest.(check int)
            (Printf.sprintf "%s encode %d" name i)
            codes.(i) (FF.encode f v))
        round_inputs)
    round_table

let decoding () =
  List.iter
    (fun (d, expected) ->
      let f = Option.get (FF.of_dtype d) in
      check_floats
        (D.numpy_name d ^ " decode")
        expected
        (Array.init (Array.length expected) (FF.decode f)))
    decode_table

let literals () =
  List.iter
    (fun (d, ty, values, expected) ->
      Alcotest.(check string) "type" ty (Ir.element_type d);
      Array.iteri
        (fun i v ->
          Alcotest.(check string)
            (Printf.sprintf "%s literal %d" (D.numpy_name d) i)
            expected.(i) (Ir.float_literal d v))
        values)
    literal_table;
  List.iter
    (fun (d, ty) ->
      Alcotest.(check string) (D.numpy_name d) ty (Ir.element_type d))
    narrow_int_types

let contains text sub =
  let n = String.length sub in
  let rec scan i =
    i + n <= String.length text && (String.sub text i n = sub || scan (i + 1))
  in
  scan 0

let program_text () =
  List.iter
    (fun (d, expected) ->
      let aval = { T.shape = [| 3 |]; dtype = d; weak_type = false } in
      let lowered =
        Ojax.Stages.lower
          (function [ x ] -> [ C.bind1 T.Mul [ x; x ] ] | _ -> assert false)
          [ aval ]
      in
      let text = Ojax.Stages.as_text lowered in
      let wanted = String.concat "%0" (String.split_on_char 'X' expected) in
      if not (contains text wanted) then
        Alcotest.failf "%s: %s not in %s" (D.numpy_name d) wanted text;
      if Option.is_some (FF.of_dtype d) && d <> D.F8_e8m0fnu then begin
        let half = T.Concrete (Nd.of_floats d [||] [| 0.5 |]) in
        let lowered =
          Ojax.Stages.lower
            (function
              | [ x ] ->
                  [
                    C.bind1 T.Add
                      [
                        x;
                        C.bind1
                          (T.Broadcast_in_dim { shape = [| 3 |]; dims = [||] })
                          [ half ];
                      ];
                  ]
              | _ -> assert false)
            [ aval ]
        in
        let wanted =
          Printf.sprintf "dense<5.000000e-01> : tensor<%s>" (Ir.element_type d)
        in
        let text = Ojax.Stages.as_text lowered in
        if not (contains text wanted) then
          Alcotest.failf "%s: %s not in %s" (D.numpy_name d) wanted text
      end)
    program_table

let conversions () =
  List.iter
    (fun (d, expected) ->
      let src =
        T.Concrete
          (Nd.of_floats D.I32 [| Array.length int_wrap_src |] int_wrap_src)
      in
      check_floats
        (D.numpy_name d ^ " wrap")
        expected
        (flat_v (C.bind1 (T.Convert_element_type d) [ src ])))
    int_wrap;
  List.iter
    (fun (d, expected) ->
      let src =
        T.Concrete
          (Nd.of_floats D.F32
             [| Array.length float_to_int_src |]
             float_to_int_src)
      in
      check_floats
        (D.numpy_name d ^ " saturate")
        expected
        (flat_v (C.bind1 (T.Convert_element_type d) [ src ])))
    float_to_int

let rev x = NL.flip ~axis:[| 0 |] x

let eager_and_jit name expected f x =
  check_floats (name ^ " eager") expected (flat_v (f x));
  let out =
    List.hd
      (Api.jit_flat (function [ a ] -> [ f a ] | _ -> assert false) [ x ])
  in
  check_floats (name ^ " jit") expected (flat_v out)

let float_ops () =
  List.iter
    (fun (d, sum, prod, max_, min_, cumsum, add, mul, div, sqrt, mean, dot) ->
      let name = D.numpy_name d in
      let x =
        T.Concrete (Nd.of_floats d [| Array.length op_input |] op_input)
      in
      let head k v =
        C.bind1
          (T.Slice
             {
               start_indices = [| 0 |];
               limit_indices = [| k |];
               strides = None;
             })
          [ v ]
      in
      eager_and_jit (name ^ " sum") sum (fun v -> R.sum v) x;
      eager_and_jit (name ^ " prod") prod (fun v -> R.prod (head 8 v)) x;
      eager_and_jit (name ^ " max") max_ (fun v -> R.max v) x;
      eager_and_jit (name ^ " min") min_ (fun v -> R.min v) x;
      eager_and_jit (name ^ " cumsum") cumsum
        (fun v -> C.bind1 (T.Cumsum { axis = 0; reverse = false }) [ v ])
        x;
      eager_and_jit (name ^ " add") add (fun v -> U.add v (rev v)) x;
      eager_and_jit (name ^ " mul") mul (fun v -> U.multiply v (rev v)) x;
      eager_and_jit (name ^ " div") div (fun v -> U.divide v (rev v)) x;
      eager_and_jit (name ^ " sqrt") sqrt U.sqrt x;
      eager_and_jit (name ^ " mean") mean (fun v -> R.mean v) x;
      eager_and_jit (name ^ " dot") dot
        (fun v -> Ojax.Numpy.Tensor_contractions.dot v v)
        x)
    op_results

let bools a = Array.map (fun b -> if b then 1.0 else 0.0) a

let special_ops () =
  List.iter
    (fun (d, isfinite, isinf, isnan, next, bits, reduced) ->
      let name = D.numpy_name d in
      let raw =
        T.Concrete
          (Nd.of_floats D.F32 [| 3 |] [| 1.0; Float.infinity; Float.nan |])
      in
      let a = C.bind1 (T.Convert_element_type d) [ raw ] in
      check_floats (name ^ " isfinite") (bools isfinite) (flat_v (U.isfinite a));
      check_floats (name ^ " isinf") (bools isinf) (flat_v (U.isinf a));
      check_floats (name ^ " isnan") (bools isnan) (flat_v (U.isnan a));
      let x = T.Concrete (Nd.of_floats d [| 3 |] [| 1.0; 2.0; 0.5 |]) in
      let y = T.Concrete (Nd.of_floats d [| 3 |] [| 2.0; 1.0; 0.5 |]) in
      check_floats (name ^ " nextafter") next
        (flat_v (C.bind1 T.Nextafter [ x; y ]));
      let target = if (Dt.finfo d).Dt.bits = 4 then D.Uint4 else D.Uint8 in
      check_floats (name ^ " bitcast") bits
        (flat_v (C.bind1 (T.Bitcast_convert_type target) [ x ]));
      check_floats
        (name ^ " reduce_precision")
        reduced
        (flat_v
           (C.bind1
              (T.Reduce_precision { exponent_bits = 2; mantissa_bits = 1 })
              [ x ])))
    special_table

let integer_ops () =
  List.iter
    (fun (d, values, add, sub, mul, not_, shl, shr, sra, pop, clz, other, bits)
       ->
      let name = D.numpy_name d in
      let shape = [| Array.length values |] in
      let x = T.Concrete (Nd.of_floats d shape values) in
      let one = T.Concrete (Nd.of_floats d shape (Array.make shape.(0) 1.0)) in
      eager_and_jit (name ^ " add") add (fun v -> U.add v (rev v)) x;
      eager_and_jit (name ^ " sub") sub (fun v -> U.subtract v (rev v)) x;
      eager_and_jit (name ^ " mul") mul (fun v -> U.multiply v (rev v)) x;
      eager_and_jit (name ^ " not") not_ (fun v -> C.bind1 T.Not [ v ]) x;
      eager_and_jit (name ^ " shl") shl
        (fun v -> C.bind1 T.Shift_left [ v; one ])
        x;
      eager_and_jit (name ^ " shr") shr
        (fun v -> C.bind1 T.Shift_right_logical [ v; one ])
        x;
      eager_and_jit (name ^ " sra") sra
        (fun v -> C.bind1 T.Shift_right_arithmetic [ v; one ])
        x;
      eager_and_jit (name ^ " pop") pop
        (fun v -> C.bind1 T.Population_count [ v ])
        x;
      eager_and_jit (name ^ " clz") clz (fun v -> C.bind1 T.Clz [ v ]) x;
      check_floats (name ^ " bitcast") bits
        (flat_v (C.bind1 (T.Bitcast_convert_type other) [ x ])))
    int_ops

let aliases () =
  List.iter
    (fun (name, f, input, x32, x64, value) ->
      let v = T.Concrete (Nd.of_floats D.F64 [||] [| input |]) in
      let run () = to_nd (f v) in
      let got = Cfg.with_value Cfg.enable_x64 false run in
      Alcotest.check dt_t (name ^ " x32") x32 (Nd.dtype got);
      check_floats (name ^ " value") [| value |] [| Nd.get_f got [||] |];
      Alcotest.check dt_t (name ^ " x64") x64
        (Nd.dtype (Cfg.with_value Cfg.enable_x64 true run)))
    alias_table;
  let one = T.Concrete (Nd.of_floats D.F32 [||] [| 1.0 |]) in
  Cfg.with_value Cfg.enable_x64 false (fun () ->
      Alcotest.check dt_t "float64 canonicalizes" D.F32
        (Nd.dtype (to_nd (ST.float64 one)));
      Alcotest.check dt_t "int64 canonicalizes" D.I32
        (Nd.dtype (to_nd (ST.int64 one))))

let float0_dtype () =
  Alcotest.check dt_t "top level" Dt.float0 Api.float0;
  Alcotest.(check string) "name" "float0" (D.numpy_name Dt.float0);
  Alcotest.check dt_t "bfloat16" D.Bf16 Dt.bfloat16;
  let zeros = AC.zeros ~dtype:Dt.float0 [| 2 |] in
  Alcotest.check dt_t "zeros dtype" Dt.float0 (Nd.dtype (to_nd zeros));
  Alcotest.(check (array int)) "zeros shape" [| 2 |] (Nd.shape (to_nd zeros));
  let ones = Nd.of_floats Dt.float0 [| 2 |] [| 1.0; 1.0 |] in
  check_floats "float0 holds no value" [| 0.0; 0.0 |] (flat ones);
  List.iter2
    (fun cls expected ->
      Alcotest.(check bool)
        "float0 class" expected
        (NL.issubdtype (NL.Cdtype Dt.float0) cls))
    class_kinds float0_class;
  Alcotest.(check bool) "not numeric" false (Dt.isdtype Dt.float0 "numeric");
  let raises name f =
    match f () with
    | _ -> Alcotest.failf "%s should raise" name
    | exception Invalid_argument _ -> ()
  in
  raises "promote" (fun () -> ignore (Dt.promote_types Dt.float0 D.I32));
  raises "itemsize" (fun () -> ignore (Dt.itemsize_bits Dt.float0));
  raises "scalar_type_of" (fun () -> ignore (Dt.scalar_type_of Dt.float0));
  raises "finfo" (fun () -> ignore (Dt.finfo Dt.float0));
  raises "iinfo" (fun () -> ignore (Dt.iinfo Dt.float0));
  raises "stablehlo" (fun () -> ignore (Ir.element_type Dt.float0))

let leaf_nd = function
  | Tree.Leaf v -> to_nd v
  | _ -> Alcotest.fail "expected a leaf"

let leaf x = Tree.Leaf x

let float0_jvp () =
  let primal = T.Concrete (Nd.of_floats D.I32 [||] [| 1.0 |]) in
  let tangent = T.Concrete (Nd.of_floats Dt.float0 [||] [| 0.0 |]) in
  let f = function
    | [ Tree.Leaf x ] -> Tree.Leaf (C.bind1 T.Mul [ x; x ])
    | _ -> assert false
  in
  let po, to_ = Api.jvp f [ leaf primal ] [ leaf tangent ] in
  check_floats "jvp primal" [| 1.0 |] (flat (leaf_nd po));
  Alcotest.check dt_t "jvp tangent dtype" Dt.float0 (Nd.dtype (leaf_nd to_));
  let real = T.Concrete (Nd.of_floats D.F32 [||] [| 1.0 |]) in
  match Api.jvp f [ leaf real ] [ leaf tangent ] with
  | _ -> Alcotest.fail "float0 tangent for a float primal should raise"
  | exception Invalid_argument _ -> ()

let float0_vjp () =
  let ints = T.Concrete (Nd.of_floats D.I32 [| 2 |] [| 1.0; 2.0 |]) in
  let reals = T.Concrete (Nd.of_floats D.F32 [| 2 |] [| 3.0; 4.0 |]) in
  let f = function
    | [ Tree.Leaf x; Tree.Leaf y ] ->
        Tree.Leaf
          (C.bind1 T.Mul [ C.bind1 (T.Convert_element_type D.F32) [ x ]; y ])
    | _ -> assert false
  in
  let out, pull = Api.vjp f [ leaf ints; leaf reals ] in
  check_floats "vjp out" [| 3.0; 8.0 |] (flat (leaf_nd out));
  (match
     pull (leaf (T.Concrete (Nd.of_floats D.F32 [| 2 |] [| 1.0; 1.0 |])))
   with
  | [ gx; gy ] ->
      Alcotest.check dt_t "int input cotangent" Dt.float0
        (Nd.dtype (leaf_nd gx));
      Alcotest.(check (array int)) "shape" [| 2 |] (Nd.shape (leaf_nd gx));
      check_floats "real input cotangent" [| 1.0; 2.0 |] (flat (leaf_nd gy))
  | _ -> Alcotest.fail "expected two cotangents");
  let g = function
    | [ Tree.Leaf x ] -> Tree.Leaf (C.bind1 T.Mul [ x; x ])
    | _ -> assert false
  in
  let _, pull_int = Api.vjp g [ leaf ints ] in
  let zero_ct = T.Concrete (Nd.of_floats Dt.float0 [| 2 |] [| 0.0; 0.0 |]) in
  match pull_int (leaf zero_ct) with
  | [ gx ] ->
      Alcotest.check dt_t "float0 cotangent in" Dt.float0
        (Nd.dtype (leaf_nd gx))
  | _ -> Alcotest.fail "expected one cotangent"

let float0_grad () =
  let f = function
    | [ Tree.Leaf x ] ->
        Tree.Leaf
          (C.bind1 T.Mul
             [
               C.bind1 (T.Convert_element_type D.F32) [ x ];
               T.Concrete (Nd.of_floats D.F32 [||] [| 1.0 |]);
             ])
    | _ -> assert false
  in
  let one_int = T.Concrete (Nd.of_floats D.I32 [||] [| 1.0 |]) in
  let g = Api.grad ~allow_int:true f [ leaf one_int ] in
  Alcotest.check dt_t "grad allow_int" Dt.float0 (Nd.dtype (leaf_nd g));
  Alcotest.check_raises "grad without allow_int"
    (Invalid_argument
       "grad requires real- or complex-valued inputs (input dtype that is a \
        sub-dtype of np.inexact), but got int32. If you want to use Boolean- \
        or integer-valued inputs, use vjp or set allow_int to True.") (fun () ->
      ignore (Api.grad f [ leaf one_int ]))

let f6_host_only () =
  List.iter
    (fun d ->
      Alcotest.(check bool)
        (D.numpy_name d ^ " no pjrt type")
        true
        (match Ojax.Pjrt.Abi.buffer_type d with
        | _ -> false
        | exception Invalid_argument _ -> true);
      let x = T.Concrete (Nd.of_floats d [| 2 |] [| 1.5; 2.0 |]) in
      let out =
        Api.jit_flat
          (function [ a ] -> [ C.bind1 T.Add [ a; a ] ] | _ -> assert false)
          [ x ]
      in
      check_floats
        (D.numpy_name d ^ " jit")
        [| 3.0; 4.0 |]
        (flat_v (List.hd out)))
    [ D.F6_e2m3fn; D.F6_e3m2fn ]

let abi_codes () =
  List.iter
    (fun (d, code) ->
      Alcotest.(check int) (D.numpy_name d) code (Ojax.Pjrt.Abi.buffer_type d);
      Alcotest.(check (option dt_t))
        (D.numpy_name d ^ " back")
        (Some d)
        (Ojax.Pjrt.Abi.dtype_of_buffer_type code))
    [
      (D.F8_e5m2, 16);
      (D.F8_e4m3fn, 17);
      (D.F8_e4m3b11fnuz, 18);
      (D.F8_e5m2fnuz, 19);
      (D.F8_e4m3fnuz, 20);
      (D.I4, 21);
      (D.Uint4, 22);
      (D.I2, 24);
      (D.Uint2, 25);
      (D.F8_e4m3, 26);
      (D.F8_e3m4, 27);
      (D.F8_e8m0fnu, 28);
      (D.F4_e2m1fn, 29);
    ]

let pjrt_roundtrip () =
  if Ojax.Pjrt.Discover.find () = None then ()
  else
    List.iter
      (fun (d, values) ->
        let nd = Nd.of_floats d [| Array.length values |] values in
        let device = Ojax.Backend.of_host_value (T.Concrete nd) in
        (match device with
        | T.Device _ -> ()
        | _ -> Alcotest.fail "expected a device buffer");
        let back = to_nd device in
        Alcotest.check dt_t "dtype back" d (Nd.dtype back);
        check_floats (D.numpy_name d ^ " roundtrip") (flat nd) (flat back))
      [
        (D.I2, [| 1.0; -2.0; 0.0; -1.0 |]);
        (D.Uint2, [| 1.0; 3.0; 0.0; 2.0 |]);
        (D.I4, [| 1.0; -8.0; 7.0; -3.0 |]);
        (D.Uint4, [| 1.0; 15.0; 7.0; 12.0 |]);
        (D.F4_e2m1fn, [| 1.5; -6.0; -0.0; 0.5 |]);
        (D.F8_e3m4, [| 1.5; -15.5; Float.infinity; Float.nan |]);
        (D.F8_e4m3, [| 1.5; -240.0; Float.neg_infinity; Float.nan |]);
        (D.F8_e4m3b11fnuz, [| 1.5; -30.0; 0.0; Float.nan |]);
        (D.F8_e4m3fn, [| 1.5; -448.0; -0.0; Float.nan |]);
        (D.F8_e4m3fnuz, [| 1.5; -240.0; 0.0; Float.nan |]);
        (D.F8_e5m2, [| 1.5; -57344.0; Float.infinity; Float.nan |]);
        (D.F8_e5m2fnuz, [| 1.5; -57344.0; 0.0; Float.nan |]);
        (D.F8_e8m0fnu, [| 2.0; 0.25; 1024.0; Float.nan |]);
      ]

let sort_and_cumulative () =
  List.iter
    (fun (d, sorted, order, cmax, cmin, amax, amin, spaced) ->
      let name = D.numpy_name d in
      let x =
        T.Concrete (Nd.of_floats d [| Array.length op_input |] op_input)
      in
      eager_and_jit (name ^ " sort") sorted
        (fun v -> Ojax.Numpy.Sorting.sort v)
        x;
      eager_and_jit (name ^ " argsort") order
        (fun v -> Ojax.Numpy.Sorting.argsort v)
        x;
      eager_and_jit (name ^ " cummax") cmax
        (fun v -> C.bind1 (T.Cummax { axis = 0; reverse = false }) [ v ])
        x;
      eager_and_jit (name ^ " cummin") cmin
        (fun v -> C.bind1 (T.Cummin { axis = 0; reverse = false }) [ v ])
        x;
      eager_and_jit (name ^ " argmax") amax (fun v -> NL.argmax v) x;
      eager_and_jit (name ^ " argmin") amin (fun v -> NL.argmin v) x;
      check_floats (name ^ " linspace") spaced
        (flat_v (AC.linspace ~num:3 ~dtype:d 0.0 1.0)))
    sort_table

let unsigned_floor_divide () =
  List.iter
    (fun (d, values, expected) ->
      let x = T.Concrete (Nd.of_floats d [| 4 |] values) in
      eager_and_jit
        (D.numpy_name d ^ " floor_divide")
        expected
        (fun v -> U.floor_divide v (rev v))
        x)
    uint_floor_div

let finite_checks_for_narrow_floats () =
  List.iter
    (fun (d, isfinite, isinf) ->
      let raw =
        T.Concrete
          (Nd.of_floats D.F32 [| 3 |] [| 1.0; Float.infinity; Float.nan |])
      in
      let a = C.bind1 (T.Convert_element_type d) [ raw ] in
      check_floats
        (D.numpy_name d ^ " isfinite")
        isfinite
        (flat_v (U.isfinite a));
      check_floats (D.numpy_name d ^ " isinf") isinf (flat_v (U.isinf a)))
    narrow_finite

let tangent_conversion () =
  List.iter
    (fun (d, primal, tangent) ->
      let f = function
        | [ Tree.Leaf x ] ->
            Tree.Leaf (C.bind1 (T.Convert_element_type d) [ x ])
        | _ -> assert false
      in
      let p = T.Concrete (Nd.of_floats D.F32 [||] [| 1.5 |]) in
      let t = T.Concrete (Nd.of_floats D.F32 [||] [| 2.0 |]) in
      let po, to_ = Api.jvp f [ leaf p ] [ leaf t ] in
      check_floats (D.numpy_name d ^ " primal") [| primal |] (flat (leaf_nd po));
      check_floats
        (D.numpy_name d ^ " tangent")
        [| tangent |]
        (flat (leaf_nd to_)))
    convert_tangents

let signbit_rejects_small_floats () =
  List.iter
    (fun d ->
      let x = T.Concrete (Nd.of_floats d [| 1 |] [| 1.0 |]) in
      Alcotest.check_raises (D.numpy_name d) (Failure signbit_message)
        (fun () -> ignore (U.signbit x));
      Alcotest.check_raises
        (D.numpy_name d ^ " spacing")
        (Failure signbit_message)
        (fun () -> ignore (U.spacing x)))
    [ D.F8_e4m3; D.F4_e2m1fn ]

let grad_small_float () =
  let f = function
    | [ Tree.Leaf z ] -> Tree.Leaf (R.sum (U.multiply z z))
    | _ -> assert false
  in
  let x = T.Concrete (Nd.of_floats D.F8_e4m3 [| 2 |] [| 0.5; 1.5 |]) in
  check_floats "grad" grad_e4m3 (flat (leaf_nd (Api.grad f [ leaf x ])))

let device_buffers_survive_collection () =
  if Ojax.Pjrt.Discover.find () <> None then begin
    for _ = 1 to 3 do
      ignore
        (Ojax.Backend.of_host_value
           (T.Concrete (Nd.of_floats D.F32 [| 2 |] [| 1.0; 2.0 |])))
    done;
    Gc.full_major ();
    Gc.full_major ();
    let out =
      Ojax.Backend.of_host_value
        (T.Concrete (Nd.of_floats D.F8_e4m3 [| 1 |] [| 1.5 |]))
    in
    check_floats "after collection" [| 1.5 |] (flat_v out)
  end

let () =
  let c name f = Alcotest.test_case name `Quick f in
  Alcotest.run "dtypes_narrow"
    [
      ( "promotion",
        [
          c "x32 table" promotion_x32;
          c "x64 table" promotion_x64;
          c "errors" promotion_errors;
          c "result_type" result_type_weak;
        ] );
      ( "info",
        [
          c "finfo" finfo_values;
          c "iinfo" iinfo_values;
          c "sizes and scalars" sizes_and_scalars;
          c "names" names;
          c "isdtype" isdtype_kinds;
          c "issubdtype" issubdtype_kinds;
          c "numpy classes" numpy_classes;
        ] );
      ( "storage",
        [
          c "rounding" rounding;
          c "decoding" decoding;
          c "conversions" conversions;
          c "abi codes" abi_codes;
          c "pjrt roundtrip" pjrt_roundtrip;
          c "device buffers survive collection"
            device_buffers_survive_collection;
          c "f6 stays on the host" f6_host_only;
        ] );
      ("stablehlo", [ c "literals" literals; c "program text" program_text ]);
      ( "ops",
        [
          c "float ops" float_ops;
          c "special" special_ops;
          c "integer ops" integer_ops;
          c "aliases" aliases;
          c "sort and cumulative" sort_and_cumulative;
          c "unsigned floor_divide" unsigned_floor_divide;
          c "isfinite on narrow floats" finite_checks_for_narrow_floats;
          c "convert keeps tangents" tangent_conversion;
          c "signbit rejects small floats" signbit_rejects_small_floats;
          c "grad on a small float" grad_small_float;
        ] );
      ( "float0",
        [
          c "dtype" float0_dtype;
          c "jvp" float0_jvp;
          c "vjp" float0_vjp;
          c "grad" float0_grad;
        ] );
    ]
