module DU = Dist_util
module L = Lax_api
open DU

let promote3 x loc scale =
  match promote [ x; loc; scale ] with
  | [ x; loc; scale ] -> (x, loc, scale)
  | _ -> assert false

let logpdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let two = sc x 2.0 in
  let linear_term = L.div (L.abs (L.sub x loc)) scale in
  L.neg (L.add linear_term (L.log (L.mul two scale)))

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let half = sc x 0.5 in
  let one = sc x 1.0 in
  let zero = sc x 0.0 in
  let diff = L.div (L.sub x loc) scale in
  L.select (L.le diff zero)
    (L.mul half (L.exp diff))
    (L.sub one (L.mul half (L.exp (L.neg diff))))
