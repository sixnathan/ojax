module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
open DU

let promote5 x a b loc scale =
  match promote [ x; a; b; loc; scale ] with
  | [ x; a; b; loc; scale ] -> (x, a, b, loc, scale)
  | _ -> assert false

let defaults loc scale =
  (Option.value loc ~default:(f32s 0.0), Option.value scale ~default:(f32s 1.0))

let logpdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  let x, a, b, loc, scale = promote5 x a b loc scale in
  let one = sc x 1.0 in
  let zero = sc a 0.0 in
  let shape_term = UF.negative (SP.betaln a b) in
  let y = UF.divide (UF.subtract x loc) scale in
  let log_linear_term =
    UF.add
      (SP.xlogy (UF.subtract a one) y)
      (SP.xlog1py (UF.subtract b one) (UF.negative y))
  in
  let log_probs =
    UF.subtract (UF.add shape_term log_linear_term) (UF.log scale)
  in
  let result =
    NL.where_
      (UF.logical_or (UF.greater x (UF.add loc scale)) (UF.less x loc))
      (ninf x) log_probs
  in
  NL.where_
    (UF.logical_or
       (UF.logical_or (UF.less_equal a zero) (UF.less_equal b zero))
       (UF.less_equal scale zero))
    (nan x) result

let pdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x a b)

let cdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  let x, a, b, loc, scale = promote5 x a b loc scale in
  let y = UF.divide (UF.subtract x loc) scale in
  SP.betainc a b (NL.clip ~min:0.0 ~max:1.0 y)

let logcdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  UF.log (cdf ~loc ~scale x a b)

let sf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  let x, a, b, loc, scale = promote5 x a b loc scale in
  let y = UF.divide (UF.subtract x loc) scale in
  SP.betainc b a (UF.subtract (sc x 1.0) (NL.clip ~min:0.0 ~max:1.0 y))

let logsf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  UF.log (sf ~loc ~scale x a b)
