module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
open DU

let promote4 k n p loc =
  match promote [ k; n; p; loc ] with
  | [ k; n; p; loc ] -> (k, n, p, loc)
  | _ -> assert false

let logpmf ?loc k n p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  let k, n, p, loc = promote4 k n p loc in
  let one = sc k 1.0 in
  let y = UF.subtract k loc in
  let comb_term =
    UF.subtract
      (UF.subtract (SP.gammaln (UF.add y n)) (SP.gammaln n))
      (SP.gammaln (UF.add y one))
  in
  let log_linear_term =
    UF.add (SP.xlogy n p) (SP.xlogy y (UF.subtract one p))
  in
  let log_probs = UF.add comb_term log_linear_term in
  NL.where_ (UF.less k loc) (ninf k) log_probs

let pmf ?loc k n p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  UF.exp (logpmf ~loc k n p)
