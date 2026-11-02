module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module L = Lax_api
open Dist_util

let promote3 ?name a b c =
  match promote ?name [ a; b; c ] with
  | [ a; b; c ] -> (a, b, c)
  | _ -> assert false

let logpdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 ~name:"uniform.logpdf" x loc scale in
  let log_probs = L.neg (L.log scale) in
  where_ninf (UF.logical_or (L.gt x (L.add loc scale)) (L.lt x loc)) log_probs

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 ~name:"uniform.cdf" x loc scale in
  let zero = sc x 0.0 in
  let one = sc x 1.0 in
  let conds =
    [
      L.lt x loc;
      L.gt x (L.add loc scale);
      UF.bitwise_and (L.ge x loc) (L.le x (L.add loc scale));
    ]
  in
  let vals = [ zero; one; L.div (L.sub x loc) scale ] in
  NL.select conds vals

let ppf ?loc ?scale q =
  let loc, scale = defaults loc scale in
  let q, loc, scale =
    match promote_complex ~name:"uniform.ppf" [ q; loc; scale ] with
    | [ q; loc; scale ] -> (q, loc, scale)
    | _ -> assert false
  in
  where_nan (qbad q) (L.add loc (L.mul scale q))
