module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
open DU

let logpmf ?loc k n p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  match promote [ k; n; p; loc ] with
  | [ k; n; p; loc ] ->
      let y = UF.subtract k loc in
      let zero = sc y 0.0 in
      let one = sc k 1.0 in
      let comb_term =
        UF.subtract
          (SP.gammaln (UF.add n one))
          (UF.add
             (SP.gammaln (UF.add y one))
             (SP.gammaln (UF.add (UF.subtract n y) one)))
      in
      let log_linear_term =
        UF.add (SP.xlogy y p) (SP.xlog1py (UF.subtract n y) (UF.negative p))
      in
      let log_probs = UF.add comb_term log_linear_term in
      let y_n_cond =
        UF.logical_or
          (UF.logical_and (UF.equal y zero) (UF.equal n zero))
          (UF.equal log_linear_term zero)
      in
      let log_probs = NL.where_ y_n_cond (sc y 0.0) log_probs in
      NL.where_
        (UF.logical_and (UF.greater_equal k loc)
           (UF.less k (UF.add (UF.add loc n) one)))
        log_probs (ninf y)
  | _ -> assert false

let pmf ?loc k n p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  UF.exp (logpmf ~loc k n p)
