module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module UF = Ufuncs
module NL = Lax_numpy

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let bind1 = C.bind1
let prod sh = Array.fold_left ( * ) 1 sh

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (prod sh) x))

let iota dt m = bind1 (T.Iota { dtype = dt; shape = [| m |]; dimension = 0 }) []
let smul dt s v = bind1 T.Mul [ const_full dt (shape v) s; v ]
let sdiv_r dt v s = bind1 T.Div [ v; const_full dt (shape v) s ]
let ssub_l dt s v = bind1 T.Sub [ const_full dt (shape v) s; v ]
let ssub_r dt v s = bind1 T.Sub [ v; const_full dt (shape v) s ]
let sadd dt s v = bind1 T.Add [ const_full dt (shape v) s; v ]
let add v w = bind1 T.Add [ v; w ]

let blackman m =
  let dt = Dtypes.default_float_dtype () in
  if m <= 1 then const_full dt [| m |] 1.0
  else begin
    let n = iota dt m in
    let d = float_of_int (m - 1) in
    let theta1 = sdiv_r dt (smul dt (2.0 *. Float.pi) n) d in
    let theta2 = sdiv_r dt (smul dt (4.0 *. Float.pi) n) d in
    let term1 = ssub_l dt 0.42 (smul dt 0.5 (UF.cos theta1)) in
    add term1 (smul dt 0.08 (UF.cos theta2))
  end

let bartlett m =
  let dt = Dtypes.default_float_dtype () in
  if m <= 1 then const_full dt [| m |] 1.0
  else begin
    let n = iota dt m in
    let d = float_of_int (m - 1) in
    let t = ssub_r dt (sadd dt 1.0 (smul dt 2.0 n)) (float_of_int m) in
    ssub_l dt 1.0 (sdiv_r dt (UF.abs t) d)
  end

let hamming m =
  let dt = Dtypes.default_float_dtype () in
  if m <= 1 then const_full dt [| m |] 1.0
  else begin
    let n = iota dt m in
    let d = float_of_int (m - 1) in
    let theta = sdiv_r dt (smul dt (2.0 *. Float.pi) n) d in
    ssub_l dt 0.54 (smul dt 0.46 (UF.cos theta))
  end

let hanning m =
  let dt = Dtypes.default_float_dtype () in
  if m <= 1 then const_full dt [| m |] 1.0
  else begin
    let n = iota dt m in
    let d = float_of_int (m - 1) in
    let theta = sdiv_r dt (smul dt (2.0 *. Float.pi) n) d in
    smul dt 0.5 (ssub_l dt 1.0 (UF.cos theta))
  end

let kaiser m beta =
  let dt = Dtypes.default_float_dtype () in
  if m <= 1 then const_full dt [| m |] 1.0
  else begin
    let n = iota dt m in
    let alpha = 0.5 *. float_of_int (m - 1) in
    let z = sdiv_r dt (ssub_r dt n alpha) alpha in
    let z2 = bind1 (T.Integer_pow 2) [ z ] in
    let s = UF.sqrt (ssub_l dt 1.0 z2) in
    let num = NL.i0 (smul dt beta s) in
    let den = NL.i0 (const_full dt [||] beta) in
    bind1 T.Div [ num; NL.broadcast_to den (shape num) ]
  end
