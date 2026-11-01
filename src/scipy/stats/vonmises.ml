module UF = Numpy.Ufuncs
module L = Lax_api
open Dist_util

let promote2 a b =
  match promote [ a; b ] with [ a; b ] -> (a, b) | _ -> assert false

let logpdf x kappa =
  let x, kappa = promote2 x kappa in
  let zero = sc kappa 0.0 in
  let cos_term = UF.subtract (UF.cos x) (sc x 1.0) in
  let value =
    UF.subtract
      (UF.multiply kappa cos_term)
      (UF.log
         (UF.multiply
            (sc kappa (2.0 *. Float.pi))
            (Core.bind1 Types.Bessel_i0e [ kappa ])))
  in
  where_or_nan (L.gt kappa zero) value

let pdf x kappa = L.exp (logpdf x kappa)
