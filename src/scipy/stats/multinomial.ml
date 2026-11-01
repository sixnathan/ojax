module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
module R = Numpy.Reductions
module L = Lax_api
open DU

let logpmf x n p =
  let p = match promote_dtypes [ p ] with [ p ] -> p | _ -> assert false in
  let pdt = dtype p in
  let x = NL.astype x pdt in
  let n = NL.astype n pdt in
  let p1 v = UF.add v (sc v 1.0) in
  let term = UF.subtract (SP.xlogy x p) (SP.gammaln (p1 x)) in
  let last = [| ndim term - 1 |] in
  let logprobs = UF.add (SP.gammaln (p1 n)) (R.sum ~axis:last term) in
  where_or_ninf (UF.equal (R.sum x) n) logprobs

let pmf x n p = L.exp (logpmf x n p)
