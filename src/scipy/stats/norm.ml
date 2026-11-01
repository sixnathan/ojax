module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module SP = Special
module L = Lax_api
open DU

let promote3 x loc scale =
  match promote [ x; loc; scale ] with
  | [ x; loc; scale ] -> (x, loc, scale)
  | _ -> assert false

let logpdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale =
    match promote_complex [ x; loc; scale ] with
    | [ x; loc; scale ] -> (x, loc, scale)
    | _ -> assert false
  in
  let scale_sqrd = L.square scale in
  let log_normalizer = L.log (L.mul (sc x (2.0 *. Float.pi)) scale_sqrd) in
  let quadratic = L.div (L.square (L.sub x loc)) scale_sqrd in
  L.div (L.add log_normalizer quadratic) (sc x (-2.0))

let pdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x)

let cdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  SP.ndtr (L.div (L.sub x loc) scale)

let logcdf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  SP.log_ndtr (L.div (L.sub x loc) scale)

let scalar_like n v =
  match promote_dtypes_any_like n v with
  | Some c -> c
  | None -> NL.astype v (dtype n)

let ppf ?loc ?scale q =
  let loc, scale = defaults loc scale in
  let n = SP.ndtri q in
  let scale = scalar_like n scale and loc = scalar_like n loc in
  NL.astype (UF.add (UF.multiply n scale) loc) (default_float ())

let sf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let neg_x = UF.negative x in
  cdf ~loc:(UF.negative loc) ~scale neg_x

let logsf ?loc ?scale x =
  let loc, scale = defaults loc scale in
  let x, loc, scale = promote3 x loc scale in
  let neg_x = UF.negative x in
  logcdf ~loc:(UF.negative loc) ~scale neg_x

let isf ?loc ?scale q =
  let loc, scale = defaults loc scale in
  ppf ~loc ~scale (L.sub (sc q 1.0) q)
