module DU = Dist_util
module UF = Numpy.Ufuncs
module SP = Special
module L = Lax_api
open DU

let promote5 x a b loc scale =
  match promote [ x; a; b; loc; scale ] with
  | [ x; a; b; loc; scale ] -> (x, a, b, loc, scale)
  | _ -> assert false

let logpdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  let x, a, b, loc, scale = promote5 x a b loc scale in
  let one = sc x 1.0 in
  let zero = sc a 0.0 in
  let shape_term = L.neg (SP.betaln a b) in
  let y = L.div (L.sub x loc) scale in
  let log_linear_term =
    L.add (SP.xlogy (L.sub a one) y) (SP.xlog1py (L.sub b one) (L.neg y))
  in
  let log_probs = L.sub (L.add shape_term log_linear_term) (L.log scale) in
  let result =
    where_ninf (UF.logical_or (L.gt x (L.add loc scale)) (L.lt x loc)) log_probs
  in
  where_nan
    (UF.logical_or
       (UF.logical_or (L.le a zero) (L.le b zero))
       (L.le scale zero))
    result

let pdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x a b)

let unit_clamp x loc scale =
  L.clamp (sc x 0.0) (L.div (L.sub x loc) scale) (sc x 1.0)

let cdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  let x, a, b, loc, scale = promote5 x a b loc scale in
  SP.betainc a b (unit_clamp x loc scale)

let logcdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  L.log (cdf ~loc ~scale x a b)

let sf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  let x, a, b, loc, scale = promote5 x a b loc scale in
  SP.betainc b a (UF.subtract (sc x 1.0) (unit_clamp x loc scale))

let logsf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  L.log (sf ~loc ~scale x a b)
