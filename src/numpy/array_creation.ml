module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Lax_numpy
module UF = Ufuncs

let get_aval = C.get_aval
let dtype_of v = (get_aval v).T.dtype
let shape_of v = (get_aval v).T.shape
let bind1 = C.bind1

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (Array.fold_left ( * ) 1 sh) x))

let scalar dt x = const_full dt [||] x

let convert v dt =
  if dtype_of v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let is_inexact = function D.F32 | D.F64 -> true | _ -> false
let to_inexact dt = if is_inexact dt then dt else Dtypes.default_float_dtype ()
let default_float () = Dtypes.default_float_dtype ()

let full ?dtype shape fill =
  let dt = match dtype with Some d -> d | None -> default_float () in
  const_full dt shape fill

let zeros ?dtype shape = full ?dtype shape 0.0
let ones ?dtype shape = full ?dtype shape 1.0
let empty ?dtype shape = zeros ?dtype shape

let full_like ?dtype ?shape a fill =
  let dt = match dtype with Some d -> d | None -> dtype_of a in
  let sh = match shape with Some s -> s | None -> shape_of a in
  const_full dt sh fill

let zeros_like ?dtype ?shape a = full_like ?dtype ?shape a 0.0
let ones_like ?dtype ?shape a = full_like ?dtype ?shape a 1.0
let empty_like ?dtype ?shape a = zeros_like ?dtype ?shape a

let linspace_core ~num ~endpoint ~comp startv stopv =
  let div = if endpoint then num - 1 else num in
  if num > 1 then begin
    let iota =
      bind1 (T.Iota { dtype = comp; shape = [| div |]; dimension = 0 }) []
    in
    let divv = NL.broadcast_to (scalar comp (float_of_int div)) [| div |] in
    let step = bind1 T.Div [ iota; divv ] in
    let one = NL.broadcast_to (scalar comp 1.0) [| div |] in
    let sb = NL.broadcast_to startv [| div |] in
    let tb = NL.broadcast_to stopv [| div |] in
    let body =
      bind1 T.Add
        [
          bind1 T.Mul [ sb; bind1 T.Sub [ one; step ] ];
          bind1 T.Mul [ tb; step ];
        ]
    in
    if endpoint then NL.concatenate ~axis:0 [ body; NL.reshape stopv [| 1 |] ]
    else body
  end
  else if num = 1 then NL.reshape startv [| 1 |]
  else const_full comp [| 0 |] 0.0

let linspace ?(num = 50) ?(endpoint = true) ?dtype start stop =
  if num < 0 then invalid_arg "linspace: num must be non-negative";
  let out_dt = match dtype with Some d -> d | None -> default_float () in
  let comp = to_inexact out_dt in
  let startv = scalar comp start and stopv = scalar comp stop in
  let body = linspace_core ~num ~endpoint ~comp startv stopv in
  let body = if is_inexact out_dt then body else bind1 T.Floor [ body ] in
  convert body out_dt

let logspace ?(num = 50) ?(endpoint = true) ?(base = 10.0) ?dtype start stop =
  let out_dt = match dtype with Some d -> d | None -> default_float () in
  let comp = to_inexact out_dt in
  let startv = scalar comp start and stopv = scalar comp stop in
  let lin = linspace_core ~num ~endpoint ~comp startv stopv in
  let basev = NL.broadcast_to (scalar comp base) (shape_of lin) in
  convert (UF.power basev lin) out_dt

let geomspace ?(num = 50) ?(endpoint = true) ?dtype start stop =
  let out_dt = match dtype with Some d -> d | None -> default_float () in
  let comp = to_inexact out_dt in
  let startv = scalar comp start and stopv = scalar comp stop in
  let sign = UF.sign startv in
  let ls = UF.log10 (bind1 T.Div [ startv; sign ]) in
  let le = UF.log10 (bind1 T.Div [ stopv; sign ]) in
  let lin = linspace_core ~num ~endpoint ~comp ls le in
  let basev = NL.broadcast_to (scalar comp 10.0) (shape_of lin) in
  let logsp = UF.power basev lin in
  let res = UF.multiply (NL.broadcast_to sign (shape_of logsp)) logsp in
  convert res out_dt
