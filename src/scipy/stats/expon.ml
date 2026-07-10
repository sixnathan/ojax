module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
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
  let log_scale = UF.log scale in
  let linear_term = UF.divide (UF.subtract x loc) scale in
  let log_probs = UF.negative (UF.add linear_term log_scale) in
  NL.where_ (UF.less x loc) (ninf x) log_probs

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let neg_scaled_x = UF.divide (UF.subtract loc x) scale in
  NL.where_ (UF.less x loc) (sc neg_scaled_x 0.0)
    (UF.negative (UF.expm1 neg_scaled_x))

let logsf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let neg_scaled_x = UF.divide (UF.subtract loc x) scale in
  NL.where_ (UF.less x loc) (sc neg_scaled_x 0.0) neg_scaled_x

let sf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.exp (logsf ~loc ~scale x)

let logcdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.log1p (UF.negative (sf ~loc ~scale x))

let ppf ?loc ?scale q =
  let loc, scale = defaults loc scale in
  let q, loc, scale = promote3 q loc scale in
  let bad =
    UF.logical_or
      (UF.logical_or (UF.isnan q) (UF.less q (sc q 0.0)))
      (UF.greater q (sc q 1.0))
  in
  NL.where_ bad (nan q)
    (UF.subtract loc (UF.multiply scale (UF.log1p (UF.negative q))))
