module L = Lax_api
open Dist_util

let promote4 a b c d =
  match promote [ a; b; c; d ] with
  | [ a; b; c; d ] -> (a, b, c, d)
  | _ -> assert false

let logpdf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  let x, b, loc, scale = promote4 x b loc scale in
  let one = sc x 1.0 in
  let scaled_x = L.div (L.sub x loc) scale in
  let normalize_term = L.log (L.div scale b) in
  let log_probs =
    L.neg (L.add normalize_term (L.mul (L.add b one) (L.log scaled_x)))
  in
  where_ninf (L.lt x (L.add loc scale)) log_probs

let pdf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x b)

let cdf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  let x, b, loc, scale = promote4 x b loc scale in
  let one = sc x 1.0 in
  let zero = sc x 0.0 in
  let scaled_x = L.div (L.sub x loc) scale in
  let cdf = L.sub one (L.pow scaled_x (L.neg b)) in
  Numpy.Lax_numpy.where_ (L.lt x (L.add loc scale)) zero cdf

let logcdf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  let x, b, loc, scale = promote4 x b loc scale in
  let scaled_x = L.div (L.sub x loc) scale in
  let logcdf_val = L.log1p (L.neg (L.pow scaled_x (L.neg b))) in
  where_ninf (L.lt x (L.add loc scale)) logcdf_val

let logsf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  let x, b, loc, scale = promote4 x b loc scale in
  let zero = sc x 0.0 in
  let scaled_x = L.div (L.sub x loc) scale in
  let logsf_val = L.neg (L.mul b (L.log scaled_x)) in
  Numpy.Lax_numpy.where_ (L.lt x (L.add loc scale)) zero logsf_val

let sf ?loc ?scale x b =
  let loc, scale = defaults loc scale in
  L.exp (logsf ~loc ~scale x b)

let ppf ?loc ?scale q b =
  let loc, scale = defaults loc scale in
  let q, b, loc, scale =
    match promote_complex [ q; b; loc; scale ] with
    | [ q; b; loc; scale ] -> (q, b, loc, scale)
    | _ -> assert false
  in
  let one = sc q 1.0 in
  let ppf_val =
    L.add loc (L.mul scale (L.pow (L.sub one q) (L.neg (L.div one b))))
  in
  where_nan (qbad q) ppf_val
