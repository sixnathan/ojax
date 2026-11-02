module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Numpy.Lax_numpy
module JU = Numpy.Jnp_util

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let shape v = (get_aval v).T.shape
let ndim v = Array.length (shape v)
let sc v x = T.Concrete (Nd.of_floats (dtype v) [||] [| x |])
let ninf v = sc v Float.neg_infinity
let nan v = sc v Float_format.nan
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

let default_value v =
  List.find_map (fun (x, d) -> if d == v then Some x else None) !weak_defaults

let promote_dtypes_any vs =
  let strong = List.filter (fun v -> not (is_weak_default v)) vs in
  let dt =
    match NL.result_type (if strong = [] then vs else strong) with
    | D.Bool when List.length strong < List.length vs -> default_float ()
    | d -> to_inexact d
  in
  let all_weak =
    List.for_all (fun v -> is_weak_default v || (get_aval v).T.weak_type) vs
  in
  List.map
    (fun v ->
      match default_value v with
      | Some x ->
          let lit = Nd.of_floats dt [||] [| x |] in
          T.Concrete (if all_weak then Nd.as_weak lit else lit)
      | None -> C.convert_like ~weak:all_weak v dt)
    vs

let promote_complex vs = Numpy.Ufuncs.promote_shapes (promote_dtypes_any vs)
let promote_dtypes vs = promote_dtypes_any vs
let promote vs = Numpy.Ufuncs.promote_shapes (promote_dtypes vs)
let lgamma v = C.bind1 T.Lgamma [ v ]
let py_float = JU.py_float
let where_px c v y = JU.where_weak ~x_weak:true c (JU.py_float v) y
let where_py c x v = JU.where_weak ~y_weak:true c x (JU.py_float v)
let where_ninf c y = where_px c Float.neg_infinity y
let where_nan c y = where_px c Float_format.nan y
let where_or_ninf c x = where_py c x Float.neg_infinity
let where_or_nan c x = where_py c x Float_format.nan

let defaults loc scale =
  (Option.value loc ~default:(f32s 0.0), Option.value scale ~default:(f32s 1.0))

let qbad q =
  let module UF = Numpy.Ufuncs in
  UF.bitwise_or
    (UF.bitwise_or (UF.isnan q) (UF.less q (sc q 0.0)))
    (UF.greater q (sc q 1.0))

let promote_dtypes_any_like n v =
  match default_value v with Some x -> Some (sc n x) | None -> None
