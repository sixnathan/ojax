module DU = Dist_util
module UF = Numpy.Ufuncs
module SP = Special
module L = Lax_api
open DU

let promote3 ?name x loc scale =
  match promote_complex ?name [ x; loc; scale ] with
  | [ x; loc; scale ] -> (x, loc, scale)
  | _ -> assert false

let logpdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 ~name:"logistic.logpdf" x loc scale in
  let z = L.div (L.sub x loc) scale in
  let two = sc z 2.0 in
  let half_x = L.div z two in
  L.sub (L.mul (L.neg two) (UF.logaddexp half_x (L.neg half_x))) (L.log scale)

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 ~name:"logistic.cdf" x loc scale in
  SP.expit (L.div (L.sub x loc) scale)

let sf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 ~name:"logistic.sf" x loc scale in
  SP.expit (L.neg (L.div (L.sub x loc) scale))

let ppf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 ~name:"logistic.ppf" x loc scale in
  L.add (L.mul (SP.logit x) scale) loc

let isf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 ~name:"logistic.isf" x loc scale in
  L.add (L.mul (L.neg (SP.logit x)) scale) loc
