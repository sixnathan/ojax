module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
open DU

let promote3 x loc scale =
  match promote [ x; loc; scale ] with
  | [ x; loc; scale ] -> (x, loc, scale)
  | _ -> assert false

let defaults loc scale =
  (Option.value loc ~default:(f32s 0.0), Option.value scale ~default:(f32s 1.0))

let logpdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let ok = UF.greater scale (sc scale 0.0) in
  let z = UF.divide (UF.subtract x loc) scale in
  let neg_log_scale = SP.xlogy (sc scale (-1.0)) scale in
  let t2 = UF.subtract z (UF.exp z) in
  NL.where_ ok (UF.add neg_log_scale t2) (nan x)

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x)

let logcdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let ok = UF.greater scale (sc scale 0.0) in
  let z = UF.divide (UF.subtract x loc) scale in
  let neg_exp_z = UF.negative (UF.exp z) in
  let log_cdf = UF.log (UF.negative (UF.expm1 neg_exp_z)) in
  NL.where_ ok log_cdf (nan x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.exp (logcdf ~loc ~scale x)

let ppf ?loc ?scale p =
  let loc, scale = defaults loc scale in
  let p, loc, scale = promote3 p loc scale in
  let ok = UF.logical_and (UF.greater p (sc p 0.0)) (UF.less p (sc p 1.0)) in
  let t1 = SP.xlog1py (sc p (-1.0)) (UF.negative p) in
  let t = UF.multiply scale (UF.log t1) in
  NL.where_ ok (UF.add loc t) (nan p)

let logsf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let ok = UF.greater scale (sc scale 0.0) in
  let z = UF.divide (UF.subtract x loc) scale in
  let log_sf = UF.negative (UF.exp z) in
  NL.where_ ok log_sf (nan x)

let sf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.exp (logsf ~loc ~scale x)
