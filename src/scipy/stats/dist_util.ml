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
let weak_defaults : (float * T.value) list ref = ref []

let f32s x =
  match List.assoc_opt x !weak_defaults with
  | Some v -> v
  | None ->
      let v = T.Concrete (Nd.of_floats D.F32 [||] [| x |]) in
      weak_defaults := (x, v) :: !weak_defaults;
      v

let is_weak_default v = List.exists (fun (_, d) -> d == v) !weak_defaults
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

let promote_dtypes_any vs =
  let strong = List.filter (fun v -> not (is_weak_default v)) vs in
  let dt =
    match NL.result_type (if strong = [] then vs else strong) with
    | D.Bool when List.length strong < List.length vs -> default_float ()
    | d -> to_inexact d
  in
  List.map (fun v -> if dtype v = dt then v else NL.astype v dt) vs

let promote_complex vs = NL.broadcast_arrays (promote_dtypes_any vs)

let real_only vs =
  List.iter
    (fun v ->
      match dtype v with
      | D.Complex64 | D.Complex128 ->
          raise
            (Errors.jax_type_error
               "scipy.stats does not accept complex input here.")
      | _ -> ())
    vs;
  vs

let promote_dtypes vs = real_only (promote_dtypes_any vs)
let promote vs = Numpy.Ufuncs.promote_shapes (promote_dtypes vs)
let lgamma v = C.bind1 T.Lgamma [ v ]
