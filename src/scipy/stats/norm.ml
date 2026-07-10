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
  let scale_sqrd = UF.square scale in
  let log_normalizer =
    UF.log (UF.multiply (sc x (2.0 *. Float.pi)) scale_sqrd)
  in
  let quadratic = UF.divide (UF.square (UF.subtract x loc)) scale_sqrd in
  UF.divide (UF.add log_normalizer quadratic) (sc x (-2.0))

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  SP.ndtr (UF.divide (UF.subtract x loc) scale)

let logcdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  SP.log_ndtr (UF.divide (UF.subtract x loc) scale)

let ppf ?loc ?scale q =
  let loc, scale = defaults loc scale in
  let q, loc, scale = promote3 q loc scale in
  NL.astype (UF.add (UF.multiply (SP.ndtri q) scale) loc) (default_float ())

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
  ppf ~loc ~scale (UF.subtract (sc q 1.0) q)
