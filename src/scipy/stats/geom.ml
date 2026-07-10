module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
open DU

let promote3 k p loc =
  match promote [ k; p; loc ] with
  | [ k; p; loc ] -> (k, p, loc)
  | _ -> assert false

let logpmf ?loc k p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  let k, p, loc = promote3 k p loc in
  let zero = sc k 0.0 in
  let one = sc k 1.0 in
  let x = UF.subtract k loc in
  let log_probs =
    UF.add (SP.xlog1py (UF.subtract x one) (UF.negative p)) (UF.log p)
  in
  NL.where_ (UF.less_equal x zero) (ninf k) log_probs

let pmf ?loc k p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  UF.exp (logpmf ~loc k p)
