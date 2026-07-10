module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
open Dist_util

let promote3 a b c =
  match promote [ a; b; c ] with [ a; b; c ] -> (a, b, c) | _ -> assert false

let defaults loc scale =
  (Option.value loc ~default:(f32s 0.0), Option.value scale ~default:(f32s 1.0))

let logpdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let log_probs = UF.negative (UF.log scale) in
  NL.where_
    (UF.logical_or (UF.greater x (UF.add loc scale)) (UF.less x loc))
    (ninf x) log_probs

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let zero = sc x 0.0 in
  let one = sc x 1.0 in
  let conds =
    [
      UF.less x loc;
      UF.greater x (UF.add loc scale);
      UF.logical_and (UF.greater_equal x loc)
        (UF.less_equal x (UF.add loc scale));
    ]
  in
  let vals = [ zero; one; UF.divide (UF.subtract x loc) scale ] in
  NL.select conds vals

let ppf ?loc ?scale q =
  let loc, scale = defaults loc scale in
  let q, loc, scale = promote3 q loc scale in
  let bad =
    UF.logical_or
      (UF.logical_or (UF.isnan q) (UF.less q (sc q 0.0)))
      (UF.greater q (sc q 1.0))
  in
  NL.where_ bad (nan q) (UF.add loc (UF.multiply scale q))
