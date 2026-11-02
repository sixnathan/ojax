module C = Core
module T = Types
module D = Dtype
module Nd = Ndarray
module NL = Numpy.Lax_numpy
module UF = Numpy.Ufuncs
module RED = Numpy.Reductions
module L = Lax_api
module JU = Numpy.Jnp_util

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let const_full dt sh x = Core.splat dt sh x
let scalar_of v x = T.Concrete (Nd.of_floats (dtype v) [||] [| x |])

let promote_inexact v =
  let dt = Dtypes.to_inexact_dtype (dtype v) in
  if dt = dtype v then v else NL.astype v dt

let canon_axis ndim a = if a < 0 then a + ndim else a

let reduction_dims ndim axis =
  match axis with
  | None -> Array.init ndim (fun i -> i)
  | Some a -> Array.map (canon_axis ndim) a

let logsumexp ?axis ?b ?(keepdims = false) a =
  let a_arr, b_arr =
    match b with
    | None -> (promote_inexact a, None)
    | Some bv -> (
        match UF.promote_shapes [ promote_inexact a; promote_inexact bv ] with
        | [ a_arr; b_arr ] ->
            let dt = Dtypes.promote_types (dtype a_arr) (dtype b_arr) in
            let a_arr = NL.astype a_arr dt and b_arr = NL.astype b_arr dt in
            let mask = UF.not_equal b_arr (scalar_of b_arr 0.0) in
            ( JU.where_weak ~y_weak:true mask a_arr
                (JU.py_float Float.neg_infinity),
              Some b_arr )
        | _ -> assert false)
  in
  let ndim = Array.length (shape a_arr) in
  let dims = reduction_dims ndim axis in
  let real_a = UF.real a_arr in
  let amax =
    RED.max ~axis:dims ~keepdims
      ~initial:(scalar_of real_a Float.neg_infinity)
      real_a
  in
  let finite = UF.isfinite amax in
  let amax = L.stop_gradient (L.select finite amax (L.full_like amax 0.0)) in
  let amax_with_dims = if keepdims then amax else L.expand_dims amax dims in
  let diff = C.bind1 T.Sub [ a_arr; NL.astype amax_with_dims (dtype a_arr) ] in
  let exp_a = L.exp diff in
  let exp_a = match b_arr with None -> exp_a | Some bb -> L.mul exp_a bb in
  let sumexp = RED.sum ~axis:dims ~keepdims exp_a in
  let sign = L.sign sumexp in
  let sumexp =
    match dtype a_arr with
    | D.Complex64 | D.Complex128 -> sumexp
    | _ -> UF.abs sumexp
  in
  let lg = L.log sumexp in
  let out = L.add lg (NL.astype amax (dtype sumexp)) in
  match (b, dtype out) with
  | Some _, (D.F16 | D.Bf16 | D.F32 | D.F64) ->
      let neg = UF.less sign (scalar_of sign 0.0) in
      NL.where_ neg (scalar_of out Float_format.nan) out
  | _ -> out
