module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
open Dist_util

let promote4 a b c d =
  match promote [ a; b; c; d ] with
  | [ a; b; c; d ] -> (a, b, c, d)
  | _ -> assert false

let defaults loc scale =
  (Option.value loc ~default:(f32s 0.0), Option.value scale ~default:(f32s 1.0))

let logpdf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  let x, b, loc, scale = promote4 x b loc scale in
  let one = sc x 1.0 in
  let scaled_x = UF.divide (UF.subtract x loc) scale in
  let normalize_term = UF.log (UF.divide scale b) in
  let log_probs =
    UF.negative
      (UF.add normalize_term (UF.multiply (UF.add b one) (UF.log scaled_x)))
  in
  NL.where_ (UF.less x (UF.add loc scale)) (ninf x) log_probs

let pdf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x b)

let cdf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  let x, b, loc, scale = promote4 x b loc scale in
  let one = sc x 1.0 in
  let zero = sc x 0.0 in
  let scaled_x = UF.divide (UF.subtract x loc) scale in
  let cdf = UF.subtract one (UF.power scaled_x (UF.negative b)) in
  NL.where_ (UF.less x (UF.add loc scale)) zero cdf

let logcdf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  let x, b, loc, scale = promote4 x b loc scale in
  let scaled_x = UF.divide (UF.subtract x loc) scale in
  let logcdf_val = UF.log1p (UF.negative (UF.power scaled_x (UF.negative b))) in
  NL.where_ (UF.less x (UF.add loc scale)) (ninf x) logcdf_val

let logsf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  let x, b, loc, scale = promote4 x b loc scale in
  let zero = sc x 0.0 in
  let scaled_x = UF.divide (UF.subtract x loc) scale in
  let logsf_val = UF.negative (UF.multiply b (UF.log scaled_x)) in
  NL.where_ (UF.less x (UF.add loc scale)) zero logsf_val

let sf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  UF.exp (logsf ~loc ~scale x b)

let ppf ?loc ?scale q b =
  let loc, scale = defaults loc scale in
  let q, b, loc, scale = promote4 q b loc scale in
  let one = sc q 1.0 in
  let ppf_val =
    UF.add loc
      (UF.multiply scale
         (UF.power (UF.subtract one q) (UF.negative (UF.divide one b))))
  in
  let bad =
    UF.logical_or
      (UF.logical_or (UF.isnan q) (UF.less q (sc q 0.0)))
      (UF.greater q (sc q 1.0))
  in
  NL.where_ bad (nan q) ppf_val
