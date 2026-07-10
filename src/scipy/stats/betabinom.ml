module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
open DU

let logpmf ?loc k n a b =
  let loc = Option.value loc ~default:(f32s 0.0) in
  match promote [ k; n; a; b; loc ] with
  | [ k; n; a; b; loc ] ->
      let y = UF.subtract (UF.floor k) loc in
      let one = sc y 1.0 in
      let zero = sc y 0.0 in
      let combiln =
        UF.negative
          (UF.add (UF.log1p n)
             (SP.betaln (UF.add (UF.subtract n y) one) (UF.add y one)))
      in
      let beta_lns =
        UF.subtract
          (SP.betaln (UF.add y a) (UF.add (UF.subtract n y) b))
          (SP.betaln a b)
      in
      let log_probs = UF.add combiln beta_lns in
      let log_probs =
        NL.where_
          (UF.logical_and (UF.equal y zero) (UF.equal n zero))
          (sc y 0.0) log_probs
      in
      let y_cond =
        UF.logical_or
          (UF.logical_or (UF.less y (UF.negative loc)) (UF.greater y n))
          (UF.less_equal (UF.add y a) zero)
      in
      let log_probs = NL.where_ y_cond (ninf y) log_probs in
      let n_a_b_cond =
        UF.logical_or
          (UF.logical_or (UF.less n zero) (UF.less_equal a zero))
          (UF.less_equal b zero)
      in
      NL.where_ n_a_b_cond (nan y) log_probs
  | _ -> assert false

let pmf ?loc k n a b =
  let loc = Option.value loc ~default:(f32s 0.0) in
  UF.exp (logpmf ~loc k n a b)
