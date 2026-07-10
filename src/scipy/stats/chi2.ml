module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
open DU

let promote4 x df loc scale =
  match promote [ x; df; loc; scale ] with
  | [ x; df; loc; scale ] -> (x, df, loc, scale)
  | _ -> assert false

let defaults loc scale =
  (Option.value loc ~default:(f32s 0.0), Option.value scale ~default:(f32s 1.0))

let logpdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  let x, df, loc, scale = promote4 x df loc scale in
  let one = sc x 1.0 in
  let two = sc x 2.0 in
  let y = UF.divide (UF.subtract x loc) scale in
  let df_on_two = UF.divide df two in
  let kernel =
    UF.subtract
      (UF.multiply (UF.subtract df_on_two one) (UF.log y))
      (UF.divide y two)
  in
  let nrml_cnst =
    UF.negative
      (UF.add (lgamma df_on_two) (UF.divide (UF.multiply (UF.log two) df) two))
  in
  let log_probs = UF.add (UF.subtract nrml_cnst (UF.log scale)) kernel in
  NL.where_ (UF.less x loc) (ninf x) log_probs

let pdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x df)

let cdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  let x, df, loc, scale = promote4 x df loc scale in
  let two = sc scale 2.0 in
  let y = UF.divide (UF.subtract x loc) (UF.multiply scale two) in
  SP.gammainc (UF.divide df two) (NL.clip ~min:0.0 y)

let logcdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  UF.log (cdf ~loc ~scale x df)

let sf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  let x, df, loc, scale = promote4 x df loc scale in
  let two = sc scale 2.0 in
  let y = UF.divide (UF.subtract x loc) (UF.multiply scale two) in
  SP.gammaincc (UF.divide df two) (NL.clip ~min:0.0 y)

let logsf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  UF.log (sf ~loc ~scale x df)
