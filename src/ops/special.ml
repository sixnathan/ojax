module C = Core
module T = Types
module D = Dtype
module Nd = Ndarray
module NL = Numpy.Lax_numpy
module UF = Numpy.Ufuncs
module RED = Numpy.Reductions

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (Array.fold_left ( * ) 1 sh) x))

let scalar_like v x = const_full (dtype v) (shape v) x

let promote_inexact v =
  match dtype v with
  | D.F32 | D.F64 -> v
  | _ -> NL.astype v (Dtypes.default_float_dtype ())

let canon_axis ndim a = if a < 0 then a + ndim else a

let reduction_dims ndim axis =
  match axis with
  | None -> Array.init ndim (fun i -> i)
  | Some a -> Array.map (canon_axis ndim) a

let logsumexp ?axis ?b ?(keepdims = false) a =
  let a_arr = promote_inexact a in
  let a_arr, b_arr =
    match b with
    | None -> (a_arr, None)
    | Some bv ->
        let b_arr = promote_inexact bv in
        let mask = UF.not_equal b_arr (scalar_like b_arr 0.0) in
        let neg_inf = scalar_like a_arr Float.neg_infinity in
        (NL.where_ mask a_arr neg_inf, Some b_arr)
  in
  let ndim = Array.length (shape a_arr) in
  let dims = reduction_dims ndim axis in
  let amax = RED.max ~axis:dims ~keepdims a_arr in
  let amax = NL.where_ (UF.isfinite amax) amax (scalar_like amax 0.0) in
  let amax_wd = if keepdims then amax else NL.expand_dims amax dims in
  let exp_a = UF.exp (UF.subtract a_arr amax_wd) in
  let exp_a =
    match b_arr with None -> exp_a | Some bb -> UF.multiply exp_a bb
  in
  let sumexp = RED.sum ~axis:dims ~keepdims exp_a in
  let sign = UF.sign sumexp in
  let sumexp = UF.abs sumexp in
  let out = UF.add (UF.log sumexp) amax in
  match b with
  | None -> out
  | Some _ ->
      let neg = UF.less sign (scalar_like sign 0.0) in
      NL.where_ neg (scalar_like out Float.nan) out
