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
  let two = sc x 2.0 in
  let linear_term = UF.divide (UF.abs (UF.subtract x loc)) scale in
  UF.negative (UF.add linear_term (UF.log (UF.multiply two scale)))

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let half = sc x 0.5 in
  let one = sc x 1.0 in
  let zero = sc x 0.0 in
  let diff = UF.divide (UF.subtract x loc) scale in
  NL.where_ (UF.less_equal diff zero)
    (UF.multiply half (UF.exp diff))
    (UF.subtract one (UF.multiply half (UF.exp (UF.negative diff))))
