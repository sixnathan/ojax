module DU = Dist_util
module UF = Numpy.Ufuncs
module SP = Special
module L = Lax_api
open DU

let logpmf ?loc k n p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  match promote [ k; n; p; loc ] with
  | [ k; n; p; loc ] ->
      let y = L.sub k loc in
      let zero = sc y 0.0 in
      let p1 v = UF.add v (sc v 1.0) in
      let comb_term =
        L.sub
          (SP.gammaln (p1 n))
          (L.add (SP.gammaln (p1 y)) (SP.gammaln (p1 (UF.subtract n y))))
      in
      let log_linear_term =
        L.add (SP.xlogy y p) (SP.xlog1py (L.sub n y) (L.neg p))
      in
      let log_probs = L.add comb_term log_linear_term in
      let y_n_cond =
        UF.logical_or
          (UF.logical_and (L.eq y zero) (L.eq n zero))
          (L.eq log_linear_term zero)
      in
      let log_probs = where_px y_n_cond 0.0 log_probs in
      where_or_ninf
        (UF.bitwise_and (L.ge k loc) (L.lt k (p1 (UF.add loc n))))
        log_probs
  | _ -> assert false

let pmf ?loc k n p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  L.exp (logpmf ~loc k n p)
