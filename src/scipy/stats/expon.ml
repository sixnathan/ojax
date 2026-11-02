module DU = Dist_util
module UF = Numpy.Ufuncs
module AC = Numpy.Array_creation
module NL = Numpy.Lax_numpy
module L = Lax_api
open DU

let promote3 ?name x loc scale =
  match promote ?name [ x; loc; scale ] with
  | [ x; loc; scale ] -> (x, loc, scale)
  | _ -> assert false

let logpdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 ~name:"expon.logpdf" x loc scale in
  let log_scale = L.log scale in
  let linear_term = L.div (L.sub x loc) scale in
  let log_probs = L.neg (L.add linear_term log_scale) in
  where_ninf (L.lt x loc) log_probs

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 ~name:"expon.cdf" x loc scale in
  let neg_scaled_x = L.div (L.sub loc x) scale in
  NL.where_ (L.lt x loc)
    (AC.zeros_like neg_scaled_x)
    (L.neg (L.expm1 neg_scaled_x))

let logsf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 ~name:"expon.sf" x loc scale in
  let neg_scaled_x = L.div (L.sub loc x) scale in
  NL.where_ (L.lt x loc) (AC.zeros_like neg_scaled_x) neg_scaled_x

let sf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  L.exp (logsf ~loc ~scale x)

let logcdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  L.log1p (L.neg (sf ~loc ~scale x))

let ppf ?loc ?scale q =
  let loc, scale = defaults loc scale in
  let q, loc, scale =
    match promote_complex ~name:"expon.ppf" [ q; loc; scale ] with
    | [ q; loc; scale ] -> (q, loc, scale)
    | _ -> assert false
  in
  where_nan (qbad q) (L.sub loc (L.mul scale (L.log1p (L.neg q))))
