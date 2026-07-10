module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
open Dist_util

let promote2 a b =
  match promote [ a; b ] with [ a; b ] -> (a, b) | _ -> assert false

let logpdf x kappa =
  let x, kappa = promote2 x kappa in
  let zero = sc kappa 0.0 in
  let value =
    UF.subtract
      (UF.multiply kappa (UF.subtract (UF.cos x) (sc x 1.0)))
      (UF.log (UF.multiply (sc kappa (2.0 *. Float.pi)) (SP.i0e kappa)))
  in
  NL.where_ (UF.greater kappa zero) value (nan kappa)

let pdf x kappa = UF.exp (logpdf x kappa)
