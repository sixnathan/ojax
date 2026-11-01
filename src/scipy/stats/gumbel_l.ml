module DU = Dist_util
module UF = Numpy.Ufuncs
module SP = Special
module L = Lax_api
open DU

let promote3 x loc scale =
  match promote [ x; loc; scale ] with
  | [ x; loc; scale ] -> (x, loc, scale)
  | _ -> assert false

let logpdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let ok = L.gt scale (sc scale 0.0) in
  let z = L.div (L.sub x loc) scale in
  let neg_log_scale = SP.xlogy (sc scale (-1.0)) scale in
  let t2 = L.sub z (L.exp z) in
  where_or_nan ok (L.add neg_log_scale t2)

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x)

let logcdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let ok = L.gt scale (sc scale 0.0) in
  let z = L.div (L.sub x loc) scale in
  let neg_exp_z = L.neg (L.exp z) in
  where_or_nan ok (L.log (UF.negative (L.expm1 neg_exp_z)))

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  L.exp (logcdf ~loc ~scale x)

let ppf ?loc ?scale p =
  let loc, scale = defaults loc scale in
  let p, loc, scale = promote3 p loc scale in
  let ok = L.bitwise_and (L.gt p (sc p 0.0)) (L.lt p (sc p 1.0)) in
  let t1 = SP.xlog1py (sc p (-1.0)) (L.neg p) in
  let t = L.mul scale (L.log t1) in
  where_or_nan ok (L.add loc t)

let logsf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let ok = L.gt scale (sc scale 0.0) in
  let z = L.div (L.sub x loc) scale in
  where_or_nan ok (L.neg (L.exp z))

let sf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.exp (logsf ~loc ~scale x)
