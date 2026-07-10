module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module RD = Numpy.Reductions
module SP = Special
open Dist_util

let promote3 a b c =
  match promote [ a; b; c ] with [ a; b; c ] -> (a, b, c) | _ -> assert false

let logpmf ?loc k mu =
  let loc = Option.value loc ~default:(f32s 0.0) in
  let k, mu, loc = promote3 k mu loc in
  let zero = sc k 0.0 in
  let one = sc k 1.0 in
  let x = UF.subtract k loc in
  let log_probs =
    UF.subtract (UF.subtract (SP.xlogy x mu) (SP.gammaln (UF.add x one))) mu
  in
  NL.where_
    (UF.logical_or (UF.less x zero) (UF.not_equal (NL.round k) k))
    (ninf k) log_probs

let pmf ?loc k mu =
  let loc = Option.value loc ~default:(f32s 0.0) in
  UF.exp (logpmf ~loc k mu)

let cdf ?loc k mu =
  let loc = Option.value loc ~default:(f32s 0.0) in
  let k, mu, loc = promote3 k mu loc in
  let zero = sc k 0.0 in
  let one = sc k 1.0 in
  let x = UF.subtract k loc in
  let p = SP.gammaincc (UF.floor (UF.add one x)) mu in
  NL.where_ (UF.less x zero) zero p

let entropy_helper max_k bound_expr mu =
  let dt = dtype mu in
  let n = (shape mu).(0) in
  let k =
    NL.reshape (NL.arange ~dtype:dt (float_of_int max_k)) [| max_k; 1 |]
  in
  let probs = pmf k mu in
  let upper_bounds = UF.ceil (bound_expr mu) in
  let ub_row = NL.reshape upper_bounds [| 1; n |] in
  let mask = UF.less k ub_row in
  let probs_masked = NL.where_ mask probs (sc probs 0.0) in
  RD.sum ~axis:[| 0 |] (SP.entr probs_masked)

let entropy ?loc mu =
  let loc = Option.value loc ~default:(f32s 0.0) in
  let pmu =
    match promote_dtypes [ mu; loc ] with a :: _ -> a | _ -> assert false
  in
  let mu_flat = NL.ravel pmu in
  let small = entropy_helper 35 (fun m -> UF.add m (sc m 20.0)) mu_flat in
  let medium =
    entropy_helper 250
      (fun m ->
        UF.add (UF.add m (UF.multiply (sc m 10.0) (UF.sqrt m))) (sc m 20.0))
      mu_flat
  in
  let e = Float.exp 1.0 in
  let large =
    UF.subtract
      (UF.multiply (sc mu_flat 0.5)
         (UF.log (UF.multiply (sc mu_flat (2.0 *. Float.pi *. e)) mu_flat)))
      (UF.divide (sc mu_flat 1.0) (UF.multiply (sc mu_flat 12.0) mu_flat))
  in
  let zero = sc mu_flat 0.0 in
  let result =
    NL.where_ (UF.equal mu_flat zero) zero
      (NL.where_
         (UF.less mu_flat (sc mu_flat 10.0))
         small
         (NL.where_ (UF.less mu_flat (sc mu_flat 100.0)) medium large))
  in
  NL.reshape result (shape pmu)
