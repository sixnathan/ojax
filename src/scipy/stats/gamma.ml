module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
open DU

let promote4 x a loc scale =
  match promote [ x; a; loc; scale ] with
  | [ x; a; loc; scale ] -> (x, a, loc, scale)
  | _ -> assert false

let defaults loc scale =
  (Option.value loc ~default:(f32s 0.0), Option.value scale ~default:(f32s 1.0))

let logpdf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  let x, a, loc, scale = promote4 x a loc scale in
  let one = sc x 1.0 in
  let ok = UF.greater_equal x loc in
  let y = NL.where_ ok (UF.divide (UF.subtract x loc) scale) one in
  let log_linear_term = UF.subtract (SP.xlogy (UF.subtract a one) y) y in
  let shape_terms = UF.add (SP.gammaln a) (UF.log scale) in
  let log_probs = UF.subtract log_linear_term shape_terms in
  NL.where_ ok log_probs (ninf x)

let pdf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x a)

let cdf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  let x, a, loc, scale = promote4 x a loc scale in
  SP.gammainc a (NL.clip ~min:0.0 (UF.divide (UF.subtract x loc) scale))

let logcdf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  UF.log (cdf ~loc ~scale x a)

let sf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  let x, a, loc, scale = promote4 x a loc scale in
  let y = UF.divide (UF.subtract x loc) scale in
  NL.where_ (UF.less y (sc y 0.0)) (sc y 1.0) (SP.gammaincc a y)

let logsf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  UF.log (sf ~loc ~scale x a)
