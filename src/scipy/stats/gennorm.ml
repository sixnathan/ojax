module DU = Dist_util
module UF = Numpy.Ufuncs
module SP = Special
module L = Lax_api
open DU

let promote2 x beta =
  match promote [ x; beta ] with [ x; beta ] -> (x, beta) | _ -> assert false

let logpdf x beta =
  let x, beta = promote2 x beta in
  let half = sc beta 0.5 in
  let one = sc beta 1.0 in
  UF.subtract
    (UF.subtract
       (L.log (UF.multiply half beta))
       (lgamma (UF.true_divide one beta)))
    (UF.power (L.abs x) beta)

let cdf x beta =
  let x, beta = promote2 x beta in
  let half = sc x 0.5 in
  let one = sc x 1.0 in
  UF.multiply half
    (UF.add one
       (UF.multiply (L.sign x)
          (SP.lax_igamma
             (UF.true_divide (sc beta 1.0) beta)
             (UF.power (L.abs x) beta))))

let pdf x beta = L.exp (logpdf x beta)
