module DU = Dist_util
module SP = Special
module L = Lax_api
open DU

let promote4 x df loc scale =
  match promote [ x; df; loc; scale ] with
  | [ x; df; loc; scale ] -> (x, df, loc, scale)
  | _ -> assert false

let logpdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  let x, df, loc, scale = promote4 x df loc scale in
  let one = sc x 1.0 in
  let two = sc x 2.0 in
  let y = L.div (L.sub x loc) scale in
  let df_on_two = L.div df two in
  let kernel = L.sub (L.mul (L.sub df_on_two one) (L.log y)) (L.div y two) in
  let nrml_cnst =
    L.neg (L.add (lgamma df_on_two) (L.div (L.mul (L.log two) df) two))
  in
  let log_probs = L.add (L.sub nrml_cnst (L.log scale)) kernel in
  where_ninf (L.lt x loc) log_probs

let pdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x df)

let clamped x loc scale two =
  L.clamp (sc x 0.0)
    (L.div (L.sub x loc) (L.mul scale two))
    (sc x Float.infinity)

let cdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  let x, df, loc, scale = promote4 x df loc scale in
  let two = sc scale 2.0 in
  SP.gammainc (L.div df two) (clamped x loc scale two)

let logcdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  L.log (cdf ~loc ~scale x df)

let sf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  let x, df, loc, scale = promote4 x df loc scale in
  let two = sc scale 2.0 in
  SP.gammaincc (L.div df two) (clamped x loc scale two)

let logsf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  L.log (sf ~loc ~scale x df)
