module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let shape v = (get_aval v).T.shape
let weak v = (get_aval v).T.weak_type
let bind1 = C.bind1

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (Array.fold_left ( * ) 1 sh) x))

let zeros_like v = const_full (dtype v) (shape v) 0.0

let to_inexact_dtype = function
  | D.F32 -> D.F32
  | D.F64 -> D.F64
  | D.I32 | D.Bool -> D.F32
  | D.I64 -> D.F64

let to_numeric_dtype = function D.Bool -> D.I32 | d -> d

let convert v dt =
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let broadcast_shapes a b =
  let na = Array.length a and nb = Array.length b in
  let n = max na nb in
  Array.init n (fun i ->
      let da = if i < n - na then 1 else a.(i - (n - na)) in
      let db = if i < n - nb then 1 else b.(i - (n - nb)) in
      if da = db then da
      else if da = 1 then db
      else if db = 1 then da
      else invalid_arg "broadcast: incompatible shapes")

let broadcast_to v sh =
  let s = shape v in
  if s = sh then v
  else begin
    let nd_in = Array.length s and nd_out = Array.length sh in
    let dims = Array.init nd_in (fun i -> i + (nd_out - nd_in)) in
    bind1 (T.Broadcast_in_dim { shape = sh; dims }) [ v ]
  end

let common_shape vs =
  List.fold_left (fun acc v -> broadcast_shapes acc (shape v)) [||] vs

let promote_to dt vs =
  let sh = common_shape vs in
  List.map (fun v -> broadcast_to (convert v dt) sh) vs

let result_dtype vs =
  let dt, _ = Dtypes.result_type (List.map (fun v -> (dtype v, weak v)) vs) in
  dt

let promote vs = promote_to (result_dtype vs) vs
let promote_inexact vs = promote_to (to_inexact_dtype (result_dtype vs)) vs
let promote_numeric vs = promote_to (to_numeric_dtype (result_dtype vs)) vs
let p1 = function [ a ] -> a | _ -> assert false
let p2 = function [ a; b ] -> (a, b) | _ -> assert false
let unary_inexact prim x = bind1 prim [ p1 (promote_inexact [ x ]) ]
let unary_promote prim x = bind1 prim [ p1 (promote [ x ]) ]
let fabs x = unary_inexact T.Abs x
let floor x = unary_inexact T.Floor x
let ceil x = unary_inexact T.Ceil x
let exp x = unary_inexact T.Exp x
let expm1 x = unary_inexact T.Expm1 x
let log x = unary_inexact T.Log x
let log1p x = unary_inexact T.Log1p x
let sin x = unary_inexact T.Sin x
let cos x = unary_inexact T.Cos x
let tan x = unary_inexact T.Tan x
let arcsin x = unary_inexact T.Asin x
let arccos x = unary_inexact T.Acos x
let arctan x = unary_inexact T.Atan x
let sinh x = unary_inexact T.Sinh x
let cosh x = unary_inexact T.Cosh x
let arcsinh x = unary_inexact T.Asinh x
let arccosh x = unary_inexact T.Acosh x
let tanh x = unary_inexact T.Tanh x
let arctanh x = unary_inexact T.Atanh x
let sqrt x = unary_inexact T.Sqrt x
let cbrt x = unary_inexact T.Cbrt x
let negative x = unary_promote T.Neg x
let positive x = p1 (promote [ x ])
let sign x = unary_promote T.Sign x
let bitwise_not x = unary_promote T.Not x
let bitwise_invert = bitwise_not
let invert = bitwise_not
let to_bool v = if dtype v = D.Bool then v else bind1 T.Ne [ v; zeros_like v ]
let logical_not x = bind1 T.Not [ to_bool (p1 (promote [ x ])) ]

let add x y =
  let a, b = p2 (promote [ x; y ]) in
  if dtype a = D.Bool then bind1 T.Or [ a; b ] else bind1 T.Add [ a; b ]

let subtract x y =
  let a, b = p2 (promote [ x; y ]) in
  bind1 T.Sub [ a; b ]

let multiply x y =
  let a, b = p2 (promote [ x; y ]) in
  if dtype a = D.Bool then bind1 T.And [ a; b ] else bind1 T.Mul [ a; b ]

let maximum x y =
  let a, b = p2 (promote [ x; y ]) in
  bind1 T.Max [ a; b ]

let minimum x y =
  let a, b = p2 (promote [ x; y ]) in
  bind1 T.Min [ a; b ]

let binary_inexact prim x y =
  let a, b = p2 (promote_inexact [ x; y ]) in
  bind1 prim [ a; b ]

let arctan2 x y = binary_inexact T.Atan2 x y
let float_power x y = binary_inexact T.Pow x y
let nextafter x y = binary_inexact T.Nextafter x y

let comparison prim x y =
  let a, b = p2 (promote [ x; y ]) in
  bind1 prim [ a; b ]

let equal x y = comparison T.Eq x y
let not_equal x y = comparison T.Ne x y
let greater x y = comparison T.Gt x y
let greater_equal x y = comparison T.Ge x y

let bitwise prim x y =
  let a, b = p2 (promote [ x; y ]) in
  bind1 prim [ a; b ]

let bitwise_and x y = bitwise T.And x y
let bitwise_or x y = bitwise T.Or x y
let bitwise_xor x y = bitwise T.Xor x y

let left_shift x y =
  let a, b = p2 (promote_numeric [ x; y ]) in
  bind1 T.Shift_left [ a; b ]

let bitwise_left_shift = left_shift

let logical prim x y =
  let a, b = p2 (promote [ x; y ]) in
  bind1 prim [ to_bool a; to_bool b ]

let logical_and x y = logical T.And x y
let logical_or x y = logical T.Or x y
let logical_xor x y = logical T.Xor x y

let smallest_subnormal = function
  | D.F32 -> Float.ldexp 1.0 (-149)
  | _ -> Float.ldexp 1.0 (-1074)

let spacing x =
  let a = p1 (promote_inexact [ x ]) in
  let dt = dtype a and sh = shape a in
  let below = bind1 T.Lt [ a; zeros_like a ] in
  let pos_inf = const_full dt sh infinity in
  let neg_inf = const_full dt sh neg_infinity in
  let toward = bind1 T.Select_n [ below; pos_inf; neg_inf ] in
  let result = bind1 T.Sub [ bind1 T.Nextafter [ a; toward ]; a ] in
  let sss = smallest_subnormal dt in
  let filled =
    bind1 T.Select_n [ below; const_full dt sh sss; const_full dt sh (-.sss) ]
  in
  let is_zero = bind1 T.Eq [ result; zeros_like result ] in
  bind1 T.Select_n [ is_zero; result; filled ]
