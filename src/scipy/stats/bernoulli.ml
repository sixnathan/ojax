module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
open DU

let logpmf ?loc k p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  match promote [ k; p; loc ] with
  | [ k; p; loc ] ->
      let zero = sc k 0.0 in
      let one = sc k 1.0 in
      let x = UF.subtract k loc in
      let log_probs =
        UF.add (SP.xlogy x p) (SP.xlog1py (UF.subtract one x) (UF.negative p))
      in
      NL.where_
        (UF.logical_or (UF.less x zero) (UF.greater x one))
        (ninf k) log_probs
  | _ -> assert false

let pmf ?loc k p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  UF.exp (logpmf ~loc k p)

let cdf k p =
  match promote [ k; p ] with
  | [ k; p ] ->
      let zero = sc k 0.0 in
      let one = sc k 1.0 in
      let conds =
        [
          UF.logical_or
            (UF.logical_or (UF.isnan k) (UF.isnan p))
            (UF.logical_or (UF.less p zero) (UF.greater p one));
          UF.less k zero;
          UF.logical_and (UF.greater_equal k zero) (UF.less k one);
          UF.greater_equal k one;
        ]
      in
      let vals = [ nan k; zero; UF.subtract one p; one ] in
      NL.select conds vals
  | _ -> assert false

let ppf q p =
  match promote [ q; p ] with
  | [ q; p ] ->
      let zero = sc q 0.0 in
      let one = sc q 1.0 in
      let bad =
        UF.logical_or
          (UF.logical_or (UF.isnan q) (UF.isnan p))
          (UF.logical_or
             (UF.logical_or (UF.less p zero) (UF.greater p one))
             (UF.logical_or (UF.less q zero) (UF.greater q one)))
      in
      NL.where_ bad (nan q)
        (NL.where_ (UF.less_equal q (UF.subtract one p)) zero one)
  | _ -> assert false
