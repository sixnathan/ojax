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
  | D.I32 | D.Bool | D.Uint32 -> D.F32
  | D.I64 -> D.F64
  | (D.F32 | D.F64 | D.Complex64 | D.Complex128) as d -> d

let promote_dtypes vs =
  let dt = to_inexact (NL.result_type vs) in
  List.map (fun v -> if dtype v = dt then v else NL.astype v dt) vs

let promote vs = NL.broadcast_arrays (promote_dtypes vs)
let lgamma v = C.bind1 T.Lgamma [ v ]
