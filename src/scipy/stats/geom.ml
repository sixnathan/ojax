module DU = Dist_util
module UF = Numpy.Ufuncs
module SP = Special
module L = Lax_api
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
  let x = L.sub k loc in
  let log_probs = UF.add (SP.xlog1py (L.sub x one) (UF.negative p)) (L.log p) in
  where_ninf (L.le x zero) log_probs

let pmf ?loc k p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  UF.exp (logpmf ~loc k p)
