module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
module R = Numpy.Reductions
open DU

let logpmf x n p =
  let p = match promote_dtypes [ p ] with [ p ] -> p | _ -> assert false in
  let pdt = dtype p in
  let x = NL.astype x pdt in
  let n = NL.astype n pdt in
  let one = sc p 1.0 in
  let term = UF.subtract (SP.xlogy x p) (SP.gammaln (UF.add x one)) in
  let last = [| ndim term - 1 |] in
  let logprobs = UF.add (SP.gammaln (UF.add n one)) (R.sum ~axis:last term) in
  NL.where_ (UF.equal (R.sum x) n) logprobs (ninf p)

let pmf x n p = UF.exp (logpmf x n p)
