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
  let pi = sc x Float.pi in
  let scaled_x = UF.divide (UF.subtract x loc) scale in
  let normalize_term = UF.log (UF.multiply pi scale) in
  UF.negative (UF.add normalize_term (UF.log1p (UF.multiply scaled_x scaled_x)))

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let pi = sc x Float.pi in
  let scaled_x = UF.divide (UF.subtract x loc) scale in
  UF.add (sc x 0.5) (UF.multiply (UF.divide (sc x 1.0) pi) (UF.arctan scaled_x))

let logcdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.log (cdf ~loc ~scale x)

let sf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  cdf ~loc:(UF.negative loc) ~scale (UF.negative x)

let logsf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  logcdf ~loc:(UF.negative loc) ~scale (UF.negative x)

let isf ?loc ?scale q =
  let loc, scale = defaults loc scale in
  let q, loc, scale = promote3 q loc scale in
  let pi = sc q Float.pi in
  let half_pi = sc q (Float.pi /. 2.0) in
  let unscaled = UF.tan (UF.subtract half_pi (UF.multiply pi q)) in
  UF.add (UF.multiply unscaled scale) loc

let ppf ?loc ?scale q =
  let loc, scale = defaults loc scale in
  let q, loc, scale = promote3 q loc scale in
  let pi = sc q Float.pi in
  let half_pi = sc q (Float.pi /. 2.0) in
  let unscaled = UF.tan (UF.subtract (UF.multiply pi q) half_pi) in
  UF.add (UF.multiply unscaled scale) loc
