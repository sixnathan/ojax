module DU = Dist_util
module UF = Numpy.Ufuncs
module SP = Special
module L = Lax_api
open DU

let logpmf ?loc k n a b =
  let loc = Option.value loc ~default:(f32s 0.0) in
  match promote [ k; n; a; b; loc ] with
  | [ k; n; a; b; loc ] ->
      let y = L.sub (L.floor k) loc in
      let one = sc y 1.0 in
      let zero = sc y 0.0 in
      let combiln =
        L.neg
          (L.add (L.log1p n) (SP.betaln (L.add (L.sub n y) one) (L.add y one)))
      in
      let beta_lns =
        L.sub (SP.betaln (L.add y a) (L.add (L.sub n y) b)) (SP.betaln a b)
      in
      let log_probs = L.add combiln beta_lns in
      let log_probs =
        where_px (UF.logical_and (L.eq y zero) (L.eq n zero)) 0.0 log_probs
      in
      let y_cond =
        UF.logical_or
          (UF.logical_or (L.lt y (L.neg loc)) (L.gt y n))
          (L.le (L.add y a) zero)
      in
      let log_probs = where_ninf y_cond log_probs in
      let n_a_b_cond =
        UF.logical_or (UF.logical_or (L.lt n zero) (L.le a zero)) (L.le b zero)
      in
      where_nan n_a_b_cond log_probs
  | _ -> assert false

let logpmf ?loc a0 a1 a2 a3 =
  Eager_jit.one ~jit:"logpmf" "scipy.stats.betabinom.logpmf" () [ loc ]
    [ a0; a1; a2; a3 ] (fun opts vals ->
      match (opts, vals) with
      | [ loc ], [ a0; a1; a2; a3 ] -> logpmf ?loc a0 a1 a2 a3
      | _ -> assert false)

let pmf ?loc k n a b = L.exp (logpmf ?loc k n a b)
