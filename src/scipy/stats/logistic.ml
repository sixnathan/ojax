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
  let z = UF.divide (UF.subtract x loc) scale in
  let two = sc z 2.0 in
  let half_x = UF.divide z two in
  UF.subtract
    (UF.multiply (UF.negative two) (UF.logaddexp half_x (UF.negative half_x)))
    (UF.log scale)

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  SP.expit (UF.divide (UF.subtract x loc) scale)

let sf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  SP.expit (UF.negative (UF.divide (UF.subtract x loc) scale))

let ppf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  UF.add (UF.multiply (SP.logit x) scale) loc

let isf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  UF.add (UF.multiply (UF.negative (SP.logit x)) scale) loc
