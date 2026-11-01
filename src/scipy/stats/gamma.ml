module DU = Dist_util
module NL = Numpy.Lax_numpy
module SP = Special
module L = Lax_api
open DU

let promote4 x a loc scale =
  match promote [ x; a; loc; scale ] with
  | [ x; a; loc; scale ] -> (x, a, loc, scale)
  | _ -> assert false

let logpdf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  let x, a, loc, scale = promote4 x a loc scale in
  let ok = L.ge x loc in
  let one = sc x 1.0 in
  let y = NL.where_ ok (L.div (L.sub x loc) scale) one in
  let log_linear_term = L.sub (SP.xlogy (L.sub a one) y) y in
  let shape_terms = L.add (SP.gammaln a) (L.log scale) in
  let log_probs = L.sub log_linear_term shape_terms in
  where_or_ninf ok log_probs

let pdf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x a)

let cdf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  let x, a, loc, scale = promote4 x a loc scale in
  SP.gammainc a
    (L.clamp (sc x 0.0) (L.div (L.sub x loc) scale) (sc x Float.infinity))

let logcdf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  L.log (cdf ~loc ~scale x a)

let sf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  let x, a, loc, scale = promote4 x a loc scale in
  let y = L.div (L.sub x loc) scale in
  Numpy.Jnp_util.where_weak ~x_weak:true
    (L.lt y (sc y 0.0))
    (Numpy.Jnp_util.py_int 1) (SP.gammaincc a y)

let logsf ?loc ?scale x a =
  let loc, scale = defaults loc scale in
  L.log (sf ~loc ~scale x a)
