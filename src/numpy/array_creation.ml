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

let is_inexact = function
  | D.F16 | D.Bf16 | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> true
  | _ -> false

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

let linspace_v ?(num = 50) ?(endpoint = true) ?(axis = 0) ?dtype start stop =
  if num < 0 then invalid_arg "linspace: num must be non-negative";
  let joined =
    let dt, _ =
      Dtypes.result_type
        [
          ((get_aval start).T.dtype, (get_aval start).T.weak_type);
          ((get_aval stop).T.dtype, (get_aval stop).T.weak_type);
        ]
    in
    dt
  in
  let out_dt = match dtype with Some d -> d | None -> to_inexact joined in
  let comp = to_inexact out_dt in
  let bounds = NL.broadcast_shapes_n [ shape_of start; shape_of stop ] in
  let rank = Array.length bounds in
  let axis = if axis < 0 then axis + rank + 1 else axis in
  let start = NL.broadcast_to (convert start comp) bounds in
  let stop = NL.broadcast_to (convert stop comp) bounds in
  let lifted v = NL.expand_dims v [| axis |] in
  let div = if endpoint then num - 1 else num in
  let body =
    if num > 1 then begin
      let rdt =
        match comp with D.Complex128 -> D.F64 | D.Complex64 -> D.F32 | d -> d
      in
      let iota =
        bind1 (T.Iota { dtype = rdt; shape = [| div |]; dimension = 0 }) []
      in
      let fraction =
        bind1 T.Div
          [ iota; NL.broadcast_to (scalar rdt (float_of_int div)) [| div |] ]
      in
      let shape_for =
        Array.init (rank + 1) (fun i -> if i = axis then div else 1)
      in
      let step = convert (NL.reshape fraction shape_for) comp in
      let one = NL.broadcast_to (scalar comp 1.0) shape_for in
      let inner =
        UF.add
          (UF.multiply (lifted start) (bind1 T.Sub [ one; step ]))
          (UF.multiply (lifted stop) step)
      in
      if endpoint then NL.concatenate ~axis [ inner; lifted stop ] else inner
    end
    else if num = 1 then lifted start
    else
      const_full comp
        (Array.init (rank + 1) (fun i ->
             if i = axis then 0
             else if i < axis then bounds.(i)
             else bounds.(i - 1)))
        0.0
  in
  let body = if is_inexact out_dt then body else bind1 T.Floor [ body ] in
  convert body out_dt

let logspace_v ?(num = 50) ?(endpoint = true) ?(base = 10.0) ?(axis = 0) ?dtype
    start stop =
  let lin = linspace_v ~num ~endpoint ~axis ?dtype start stop in
  let dt = dtype_of lin in
  let basev = NL.broadcast_to (scalar dt base) (shape_of lin) in
  UF.power basev lin

let geomspace_v ?(num = 50) ?(endpoint = true) ?(axis = 0) ?dtype start stop =
  let joined, _ =
    Dtypes.result_type
      [
        ((get_aval start).T.dtype, (get_aval start).T.weak_type);
        ((get_aval stop).T.dtype, (get_aval stop).T.weak_type);
      ]
  in
  let out_dt = match dtype with Some d -> d | None -> to_inexact joined in
  let comp = to_inexact out_dt in
  let start = convert start comp and stop = convert stop comp in
  let sign = UF.sign start in
  let res =
    UF.multiply sign
      (logspace_v ~num ~endpoint ~base:10.0 ~axis:0 ~dtype:comp
         (UF.log10 (UF.divide start sign))
         (UF.log10 (UF.divide stop sign)))
  in
  let rank = Array.length (shape_of res) in
  let axis = if axis < 0 then axis + rank else axis in
  let res = if axis = 0 then res else NL.moveaxis [| 0 |] [| axis |] res in
  convert res out_dt
