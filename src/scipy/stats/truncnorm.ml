module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module AC = Numpy.Array_creation
module SP = Special
module LSE = Ops.Special
module N = Norm
module L = Lax_api
open Dist_util

let promote5 a b c d e =
  match promote [ a; b; c; d; e ] with
  | [ a; b; c; d; e ] -> (a, b, c, d, e)
  | _ -> assert false

let log_diff x y =
  let stacked = NL.stack ~axis:0 [ x; y ] in
  let weights =
    NL.stack ~axis:0 [ AC.ones_like x; UF.negative (AC.ones_like y) ]
  in
  LSE.logsumexp ~axis:[| 0 |] ~b:weights stacked

let log_gauss_mass a b =
  let a, b =
    match NL.broadcast_arrays [ a; b ] with
    | [ a; b ] -> (a, b)
    | _ -> assert false
  in
  let case_left = UF.less_equal b (sc b 0.0) in
  let case_right = UF.greater a (sc a 0.0) in
  let case_central = UF.bitwise_not (UF.bitwise_or case_left case_right) in
  let a_tail = NL.where_ case_right (UF.negative b) a in
  let b_tail = NL.where_ case_right (UF.negative a) b in
  let mass_tail = log_diff (SP.log_ndtr b_tail) (SP.log_ndtr a_tail) in
  let mass_central =
    UF.log1p (UF.subtract (UF.negative (SP.ndtr a)) (SP.ndtr (UF.negative b)))
  in
  NL.where_ case_central mass_central mass_tail

let logpdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  let x, a, b, loc, scale = promote5 x a b loc scale in
  let value = L.sub (N.logpdf ~loc ~scale x) (log_gauss_mass a b) in
  let x_scaled = L.div (L.sub x loc) scale in
  let value =
    where_ninf
      (UF.bitwise_or (UF.less x_scaled a) (UF.greater x_scaled b))
      value
  in
  where_nan (UF.greater_equal a b) value

let pdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  L.exp (logpdf ~loc ~scale x a b)

let logcdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  let x, a, b, loc, scale = promote5 x a b loc scale in
  let x, a, b =
    match NL.broadcast_arrays [ x; a; b ] with
    | [ x; a; b ] -> (x, a, b)
    | _ -> assert false
  in
  let x = L.div (L.sub x loc) scale in
  let lgm_ab = log_gauss_mass a b in
  let logcdf_v = UF.subtract (log_gauss_mass a x) lgm_ab in
  let logsf_v = UF.subtract (log_gauss_mass x b) lgm_ab in
  let conds =
    [
      UF.greater_equal x b;
      UF.less_equal x a;
      UF.greater logcdf_v (sc x (-0.1));
      UF.greater x a;
    ]
  in
  let vals =
    [ sc x 0.0; ninf x; UF.log1p (UF.negative (UF.exp logsf_v)); logcdf_v ]
  in
  let logcdf = NL.select conds vals in
  where_nan (UF.greater_equal a b) logcdf

let cdf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  L.exp (logcdf ~loc ~scale x a b)

let logsf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  let x, a, b, loc, scale = promote5 x a b loc scale in
  logcdf ~loc:(UF.negative loc) ~scale (UF.negative x) (UF.negative b)
    (UF.negative a)

let sf ?loc ?scale x a b =
  let loc, scale = defaults loc scale in
  L.exp (logsf ~loc ~scale x a b)

let logpdf ?loc ?scale a0 a1 a2 =
  Eager_jit.one ~jit:"logpdf" "scipy.stats.truncnorm.logpdf" () [ loc; scale ]
    [ a0; a1; a2 ] (fun opts vals ->
      match (opts, vals) with
      | [ loc; scale ], [ a0; a1; a2 ] -> logpdf ?loc ?scale a0 a1 a2
      | _ -> assert false)

let logsf ?loc ?scale a0 a1 a2 =
  Eager_jit.one ~jit:"logsf" "scipy.stats.truncnorm.logsf" () [ loc; scale ]
    [ a0; a1; a2 ] (fun opts vals ->
      match (opts, vals) with
      | [ loc; scale ], [ a0; a1; a2 ] -> logsf ?loc ?scale a0 a1 a2
      | _ -> assert false)

let sf ?loc ?scale a0 a1 a2 =
  Eager_jit.one ~jit:"sf" "scipy.stats.truncnorm.sf" () [ loc; scale ]
    [ a0; a1; a2 ] (fun opts vals ->
      match (opts, vals) with
      | [ loc; scale ], [ a0; a1; a2 ] -> sf ?loc ?scale a0 a1 a2
      | _ -> assert false)

let logcdf ?loc ?scale a0 a1 a2 =
  Eager_jit.one ~jit:"logcdf" "scipy.stats.truncnorm.logcdf" () [ loc; scale ]
    [ a0; a1; a2 ] (fun opts vals ->
      match (opts, vals) with
      | [ loc; scale ], [ a0; a1; a2 ] -> logcdf ?loc ?scale a0 a1 a2
      | _ -> assert false)

let cdf ?loc ?scale a0 a1 a2 =
  Eager_jit.one ~jit:"cdf" "scipy.stats.truncnorm.cdf" () [ loc; scale ]
    [ a0; a1; a2 ] (fun opts vals ->
      match (opts, vals) with
      | [ loc; scale ], [ a0; a1; a2 ] -> cdf ?loc ?scale a0 a1 a2
      | _ -> assert false)
