module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
module L = Lax_api
open DU

let logpmf ?loc k p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  match promote [ k; p; loc ] with
  | [ k; p; loc ] ->
      let zero = sc k 0.0 in
      let one = sc k 1.0 in
      let x = L.sub k loc in
      let log_probs =
        UF.add (SP.xlogy x p) (SP.xlog1py (L.sub one x) (UF.negative p))
      in
      where_ninf (UF.logical_or (L.lt x zero) (L.gt x one)) log_probs
  | _ -> assert false

let pmf ?loc k p =
  let loc = Option.value loc ~default:(f32s 0.0) in
  UF.exp (logpmf ~loc k p)

let cdf k p =
  match promote [ k; p ] with
  | [ k; p ] ->
      let zero = sc k 0.0 in
      let one = sc k 1.0 in
      let bor = UF.bitwise_or in
      let conds =
        [
          bor
            (bor (bor (UF.isnan k) (UF.isnan p)) (UF.less p zero))
            (UF.greater p one);
          L.lt k zero;
          UF.logical_and (L.ge k zero) (L.lt k one);
          L.ge k one;
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
      let bor = UF.bitwise_or in
      let bad =
        bor
          (bor
             (bor
                (bor (bor (UF.isnan q) (UF.isnan p)) (UF.less p zero))
                (UF.greater p one))
             (UF.less q zero))
          (UF.greater q one)
      in
      where_nan bad (NL.where_ (L.le q (UF.subtract one p)) zero one)
  | _ -> assert false
