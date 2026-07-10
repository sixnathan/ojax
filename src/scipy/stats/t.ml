module UF = Numpy.Ufuncs
open Dist_util

let promote4 a b c d =
  match promote [ a; b; c; d ] with
  | [ a; b; c; d ] -> (a, b, c, d)
  | _ -> assert false

let defaults loc scale =
  (Option.value loc ~default:(f32s 0.0), Option.value scale ~default:(f32s 1.0))

let logpdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  let x, df, loc, scale = promote4 x df loc scale in
  let two = sc x 2.0 in
  let scaled_x = UF.divide (UF.subtract x loc) scale in
  let df_over_two = UF.divide df two in
  let df_plus_one_over_two = UF.add df_over_two (sc x 0.5) in
  let normalize_term_const =
    UF.multiply (UF.multiply scale scale) (sc x Float.pi)
  in
  let normalize_term_tmp =
    UF.divide (UF.log (UF.multiply normalize_term_const df)) two
  in
  let normalize_term =
    UF.subtract
      (UF.add (lgamma df_over_two) normalize_term_tmp)
      (lgamma df_plus_one_over_two)
  in
  let quadratic = UF.divide (UF.multiply scaled_x scaled_x) df in
  UF.negative
    (UF.add normalize_term
       (UF.multiply df_plus_one_over_two (UF.log1p quadratic)))

let pdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  UF.exp (logpdf ~loc ~scale x df)
