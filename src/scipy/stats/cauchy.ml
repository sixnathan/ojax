module DU = Dist_util
module UF = Numpy.Ufuncs
module L = Lax_api
open DU

let promote3 x loc scale =
  match promote_complex [ x; loc; scale ] with
  | [ x; loc; scale ] -> (x, loc, scale)
  | _ -> assert false

let logpdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let pi = sc x Float.pi in
  let scaled_x = L.div (L.sub x loc) scale in
  let normalize_term = L.log (L.mul pi scale) in
  L.neg (L.add normalize_term (L.log1p (L.mul scaled_x scaled_x)))

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let pi = sc x Float.pi in
  let scaled_x = L.div (L.sub x loc) scale in
  L.add (sc x 0.5) (L.mul (L.div (sc x 1.0) pi) (UF.arctan scaled_x))

let logcdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  L.log (cdf ~loc ~scale x)

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
  let unscaled = L.tan (L.sub half_pi (L.mul pi q)) in
  L.add (L.mul unscaled scale) loc

let ppf ?loc ?scale q =
  let loc, scale = defaults loc scale in
  let q, loc, scale = promote3 q loc scale in
  let pi = sc q Float.pi in
  let half_pi = sc q (Float.pi /. 2.0) in
  let unscaled = L.tan (L.sub (L.mul pi q) half_pi) in
  L.add (L.mul unscaled scale) loc
