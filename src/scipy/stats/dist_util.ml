module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Numpy.Lax_numpy

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let shape v = (get_aval v).T.shape
let ndim v = Array.length (shape v)
let sc v x = T.Concrete (Nd.of_floats (dtype v) [||] [| x |])
let ninf v = sc v Float.neg_infinity
let nan v = sc v Float.nan
let f32s x = T.Concrete (Nd.of_floats D.F32 [||] [| x |])
let default_float () = if Config.x64_enabled () then D.F64 else D.F32

let to_inexact = function
  | D.I32 | D.Bool | D.Uint32 | D.I8 | D.I16 | D.I4 | D.Uint8 | D.Uint16
  | D.Uint4 ->
      D.F32
  | (D.I2 | D.Uint2) as narrow -> narrow
  | D.Float0 -> invalid_arg "float0 has no inexact dtype"
  | D.I64 | D.Uint64 -> D.F64
  | ( D.F32 | D.F64 | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn
    | D.F8_e3m4 | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz
    | D.F8_e5m2 | D.F8_e5m2fnuz | D.F8_e8m0fnu | D.Complex64 | D.Complex128 ) as
    d ->
      d

let promote_dtypes vs =
  let dt = to_inexact (NL.result_type vs) in
  List.map (fun v -> if dtype v = dt then v else NL.astype v dt) vs

let promote vs = NL.broadcast_arrays (promote_dtypes vs)
let lgamma v = C.bind1 T.Lgamma [ v ]
