module T = Types
module C = Core
module D = Dtype
module LN = Numpy.Lax_numpy
module U = Numpy.Ufuncs
module RED = Numpy.Reductions
module NLIN = Numpy.Linalg

let get_aval = C.get_aval
let ndim v = Array.length (get_aval v).T.shape

let is_inexact = function
  | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> true
  | _ -> false

let promote_inexact vs =
  let dt = LN.result_type vs in
  let dt = if is_inexact dt then dt else Dtypes.default_float_dtype () in
  List.map (fun v -> LN.astype v dt) vs

let vq ?(check_finite = true) obs code_book =
  ignore check_finite;
  let obs, cb =
    match promote_inexact [ obs; code_book ] with
    | [ a; b ] -> (a, b)
    | _ -> assert false
  in
  if ndim obs <> ndim cb then
    invalid_arg "Observation and code_book should have the same rank";
  let obs, cb =
    if ndim obs = 1 then
      (LN.expand_dims obs [| ndim obs |], LN.expand_dims cb [| ndim cb |])
    else (obs, cb)
  in
  if ndim obs <> 2 then
    invalid_arg "ndim different than 1 or 2 are not supported";
  let diff =
    U.subtract (LN.expand_dims obs [| 1 |]) (LN.expand_dims cb [| 0 |])
  in
  let dist = NLIN.norm ~axis:(NLIN.Aint (-1)) diff in
  let code = LN.argmin ~axis:1 dist in
  let dist_min = RED.amin ~axis:[| 1 |] dist in
  (code, dist_min)
