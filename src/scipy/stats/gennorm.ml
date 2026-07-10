module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
open DU

let promote2 x beta =
  match promote [ x; beta ] with [ x; beta ] -> (x, beta) | _ -> assert false

let logpdf x beta =
  let x, beta = promote2 x beta in
  let half = sc beta 0.5 in
  let one = sc beta 1.0 in
  UF.subtract
    (UF.subtract (UF.log (UF.multiply half beta)) (lgamma (UF.divide one beta)))
    (UF.power (UF.abs x) beta)

let cdf x beta =
  let x, beta = promote2 x beta in
  let half = sc x 0.5 in
  let one = sc x 1.0 in
  UF.multiply half
    (UF.add one
       (UF.multiply (UF.sign x)
          (SP.gammainc (UF.divide one beta) (UF.power (UF.abs x) beta))))

let pdf x beta = UF.exp (logpdf x beta)
