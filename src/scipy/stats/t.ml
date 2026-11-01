module L = Lax_api
open Dist_util

let promote4 a b c d =
  match promote [ a; b; c; d ] with
  | [ a; b; c; d ] -> (a, b, c, d)
  | _ -> assert false

let logpdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  let x, df, loc, scale = promote4 x df loc scale in
  let two = sc x 2.0 in
  let scaled_x = L.div (L.sub x loc) scale in
  let df_over_two = L.div df two in
  let df_plus_one_over_two = L.add df_over_two (sc x 0.5) in
  let normalize_term_const = L.mul (L.mul scale scale) (sc x Float.pi) in
  let normalize_term_tmp = L.div (L.log (L.mul normalize_term_const df)) two in
  let normalize_term =
    L.sub
      (L.add (lgamma df_over_two) normalize_term_tmp)
      (lgamma df_plus_one_over_two)
  in
  let quadratic = L.div (L.mul scaled_x scaled_x) df in
  L.neg (L.add normalize_term (L.mul df_plus_one_over_two (L.log1p quadratic)))

let pdf ?loc ?scale x df =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x df)
