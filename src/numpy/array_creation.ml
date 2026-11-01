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
let const_full dt sh x = Core.splat dt sh x
let scalar dt x = const_full dt [||] x

let convert v dt =
  if dtype_of v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let is_inexact = function
  | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn | D.F8_e3m4
  | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz | D.F8_e5m2
  | D.F8_e5m2fnuz | D.F8_e8m0fnu | D.F32 | D.F64 | D.Complex64 | D.Complex128 ->
      true
  | _ -> false

let to_inexact = Dtypes.to_inexact_dtype
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
let weak_scalar dt x = scalar dt x

let reshape_same v sh =
  if shape_of v = sh then v else bind1 (T.Reshape sh) [ v ]

let result_of a b =
  fst
    (Dtypes.result_type
       [
         ((get_aval a).T.dtype, (get_aval a).T.weak_type);
         ((get_aval b).T.dtype, (get_aval b).T.weak_type);
       ])

let dtype_key = function None -> "none" | Some d -> Dtype.short_name d

let linspace_v ?(num = 50) ?(endpoint = true) ?(axis = 0) ?dtype start stop =
  if num < 0 then invalid_arg "linspace: num must be non-negative";
  let key = Printf.sprintf "%d %b %d %s" num endpoint axis (dtype_key dtype) in
  Jnp_util.jit1 "_linspace" ~key
    (function
      | [ start; stop ] ->
          let out_dt =
            match dtype with
            | Some d -> d
            | None -> to_inexact (result_of start stop)
          in
          let comp = to_inexact out_dt in
          let start = convert start comp and stop = convert stop comp in
          let bounds =
            NL.broadcast_shapes_n [ shape_of start; shape_of stop ]
          in
          let broadcast_start = Jnp_util.broadcast_to start bounds in
          let broadcast_stop = Jnp_util.broadcast_to stop bounds in
          let rank = Array.length bounds in
          let axis = if axis < 0 then rank + axis + 1 else axis in
          let insert size =
            Array.init (rank + 1) (fun i ->
                if i < axis then bounds.(i)
                else if i = axis then size
                else bounds.(i - 1))
          in
          let bounds_shape = insert 1 in
          let div = if endpoint then num - 1 else num in
          let out =
            if num > 1 then begin
              let difference = UF.subtract stop start in
              ignore
                (bind1 T.Div [ difference; scalar comp (float_of_int div) ]);
              let rdt =
                match comp with
                | D.Complex128 -> D.F64
                | D.Complex64 -> D.F32
                | d -> d
              in
              let iota_shape =
                Array.init (rank + 1) (fun i -> if i = axis then div else 1)
              in
              let iota =
                bind1
                  (T.Iota { dtype = rdt; shape = [| div |]; dimension = 0 })
                  []
              in
              let reshaped = reshape_same iota iota_shape in
              let step =
                bind1 T.Div [ reshaped; scalar rdt (float_of_int div) ]
              in
              let step = convert step comp in
              let start_part = reshape_same broadcast_start bounds_shape in
              let remainder = bind1 T.Sub [ scalar comp 1.0; step ] in
              let left = bind1 T.Mul [ start_part; remainder ] in
              let stop_part = reshape_same broadcast_stop bounds_shape in
              let right = bind1 T.Mul [ stop_part; step ] in
              let body = bind1 T.Add [ left; right ] in
              if endpoint then
                let tail =
                  bind1
                    (T.Broadcast_in_dim
                       {
                         shape = bounds_shape;
                         dims =
                           Array.init rank (fun i ->
                               if i < axis then i else i + 1);
                       })
                    [ broadcast_stop ]
                in
                bind1 (T.Concatenate axis) [ body; tail ]
              else body
            end
            else if num = 1 then reshape_same broadcast_start bounds_shape
            else const_full out_dt (insert 0) 0.0
          in
          let out =
            if is_inexact out_dt || not (is_inexact (dtype_of out)) then out
            else bind1 T.Floor [ out ]
          in
          convert out out_dt
      | _ -> invalid_arg "linspace: arity")
    [ start; stop ]

let logspace_core ?(num = 50) ?(endpoint = true) ?(axis = 0) ?dtype start stop
    base =
  let key = Printf.sprintf "%d %b %d %s" num endpoint axis (dtype_key dtype) in
  Jnp_util.jit1 "_logspace" ~key
    (function
      | [ start; stop; base ] ->
          let out_dt =
            match dtype with
            | Some d -> d
            | None -> to_inexact (result_of start stop)
          in
          let comp = to_inexact out_dt in
          let start = convert start comp and stop = convert stop comp in
          let lin = linspace_v ~num ~endpoint ~axis start stop in
          let base = convert base (dtype_of lin) in
          convert (bind1 T.Pow [ base; lin ]) out_dt
      | _ -> invalid_arg "logspace: arity")
    [ start; stop; base ]

let logspace_v ?(num = 50) ?(endpoint = true) ?(base = 10.0) ?(axis = 0) ?dtype
    start stop =
  logspace_core ~num ~endpoint ~axis ?dtype start stop
    (weak_scalar (default_float ()) base)

let geomspace_v ?(num = 50) ?(endpoint = true) ?(axis = 0) ?dtype start stop =
  let key = Printf.sprintf "%d %b %d %s" num endpoint axis (dtype_key dtype) in
  Jnp_util.jit1 "_geomspace" ~key
    (function
      | [ start; stop ] ->
          let out_dt =
            match dtype with
            | Some d -> d
            | None -> to_inexact (result_of start stop)
          in
          let comp = to_inexact out_dt in
          let start = convert start comp and stop = convert stop comp in
          let sign = UF.sign start in
          let lower = UF.log10 (UF.divide start sign) in
          let upper = UF.log10 (UF.divide stop sign) in
          let logs =
            logspace_core ~num ~endpoint ~dtype:comp lower upper
              (weak_scalar (default_float ()) 10.0)
          in
          let res = UF.multiply sign logs in
          let rank = Array.length (shape_of res) in
          let axis = if axis < 0 then axis + rank else axis in
          let res =
            if axis = 0 then res
            else
              bind1
                (T.Transpose
                   (Array.init rank (fun i ->
                        if i < axis then i + 1 else if i = axis then 0 else i)))
                [ res ]
          in
          convert res out_dt
      | _ -> invalid_arg "geomspace: arity")
    [ start; stop ]

let float_bound dtype x =
  match dtype with
  | Some d when is_inexact d -> scalar (to_inexact d) x
  | _ -> scalar (default_float ()) x

let linspace ?num ?endpoint ?dtype start stop =
  linspace_v ?num ?endpoint ?dtype (float_bound dtype start)
    (float_bound dtype stop)

let logspace ?num ?endpoint ?base ?dtype start stop =
  logspace_v ?num ?endpoint ?base ?dtype (float_bound dtype start)
    (float_bound dtype stop)

let geomspace ?num ?endpoint ?dtype start stop =
  geomspace_v ?num ?endpoint ?dtype (float_bound dtype start)
    (float_bound dtype stop)
