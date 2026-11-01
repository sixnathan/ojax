module DU = Dist_util
module SP = Special
module L = Lax_api
open DU

let promote4 k n p loc =
  match promote [ k; n; p; loc ] with
  | [ k; n; p; loc ] -> (k, n, p, loc)
  | _ -> assert false

let logpmf ?loc k n p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  let k, n, p, loc = promote4 k n p loc in
  let one = sc k 1.0 in
  let y = L.sub k loc in
  let comb_term =
    L.sub
      (L.sub (SP.gammaln (L.add y n)) (SP.gammaln n))
      (SP.gammaln (L.add y one))
  in
  let log_linear_term = L.add (SP.xlogy n p) (SP.xlogy y (L.sub one p)) in
  let log_probs = L.add comb_term log_linear_term in
  where_ninf (L.lt k loc) log_probs

let pmf ?loc k n p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  L.exp (logpmf ~loc k n p)
