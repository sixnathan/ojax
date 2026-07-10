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
  | D.I32 | D.Bool | D.Uint32 -> D.F32
  | D.I64 -> D.F64
  | (D.Complex64 | D.Complex128) as c -> c

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

let floor x =
  match dtype x with
  | D.I32 | D.I64 | D.Bool -> x
  | _ -> unary_inexact T.Floor x

let ceil x =
  match dtype x with D.I32 | D.I64 | D.Bool -> x | _ -> unary_inexact T.Ceil x

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

let is_integer = function D.I32 | D.I64 -> true | _ -> false
let is_inexact = function D.F32 | D.F64 -> true | _ -> false
let where c a b = bind1 T.Select_n [ c; b; a ]
let absolute x = if dtype x = D.Bool then x else bind1 T.Abs [ x ]
let abs = absolute
let acos = arccos
let acosh = arccosh
let asin = arcsin
let asinh = arcsinh
let atan = arctan
let atanh = arctanh
let atan2 = arctan2

let deg2rad x =
  let a = p1 (promote_inexact [ x ]) in
  bind1 T.Mul [ a; const_full (dtype a) (shape a) (Float.pi /. 180.0) ]

let rad2deg x =
  let a = p1 (promote_inexact [ x ]) in
  bind1 T.Mul [ a; const_full (dtype a) (shape a) (180.0 /. Float.pi) ]

let radians = deg2rad
let degrees = rad2deg
let exp2 x = unary_inexact T.Exp2 x

let log2 x =
  let a = p1 (promote_inexact [ x ]) in
  bind1 T.Div
    [ bind1 T.Log [ a ]; const_full (dtype a) (shape a) (Float.log 2.0) ]

let log10 x =
  let a = p1 (promote_inexact [ x ]) in
  bind1 T.Mul
    [ bind1 T.Log [ a ]; const_full (dtype a) (shape a) 0.4342944819032518 ]

let reciprocal x =
  let a = p1 (promote_inexact [ x ]) in
  bind1 (T.Integer_pow (-1)) [ a ]

let square x =
  let a = p1 (promote_numeric [ x ]) in
  bind1 T.Square [ a ]

let true_divide x y =
  let a, b = p2 (promote_inexact [ x; y ]) in
  bind1 T.Div [ a; b ]

let divide = true_divide
let less x y = comparison T.Lt x y
let less_equal x y = comparison T.Le x y

let right_shift x y =
  let a, b = p2 (promote_numeric [ x; y ]) in
  bind1 T.Shift_right_arithmetic [ a; b ]

let bitwise_right_shift = right_shift

let pow_int_int x1 x2 =
  let dt = dtype x1 and sh = shape x1 in
  let zero = const_full dt sh 0.0 and one = const_full dt sh 1.0 in
  let init = bind1 T.And [ bind1 T.Eq [ x1; zero ]; bind1 T.Ne [ x2; zero ] ] in
  let acc0 = where init zero one in
  let rec loop i acc b1 b2 =
    if i = 0 then acc
    else
      let is_odd = bind1 T.Ne [ bind1 T.And [ b2; one ]; zero ] in
      let acc' = where is_odd (bind1 T.Mul [ acc; b1 ]) acc in
      let b1' = bind1 T.Mul [ b1; b1 ] in
      let b2' = bind1 T.Shift_right_logical [ b2; one ] in
      loop (i - 1) acc' b1' b2'
  in
  loop 6 acc0 x1 x2

let power x1 x2 =
  let sh = broadcast_shapes (shape x1) (shape x2) in
  let x1 = broadcast_to x1 sh and x2 = broadcast_to x2 sh in
  let d1 = dtype x1 and d2 = dtype x2 in
  let a, b = p2 (promote_numeric [ x1; x2 ]) in
  if is_integer (dtype a) || dtype a = D.Bool then pow_int_int a b
  else if is_inexact d1 && is_integer d2 then bind1 T.Pow [ x1; x2 ]
  else bind1 T.Pow [ a; b ]

let pow = power

let float_divmod x1 x2 =
  let dt = dtype x1 and sh = shape x1 in
  let zero = const_full dt sh 0.0 and one = const_full dt sh 1.0 in
  let m = bind1 T.Rem [ x1; x2 ] in
  let x1c = where (bind1 T.Eq [ x2; zero ]) x1 (bind1 T.Sub [ x1; m ]) in
  let div = bind1 T.Div [ x1c; x2 ] in
  let ind =
    bind1 T.And
      [
        bind1 T.Ne [ m; zero ];
        bind1 T.Ne [ bind1 T.Sign [ x2 ]; bind1 T.Sign [ m ] ];
      ]
  in
  let m2 = where ind (bind1 T.Add [ m; x2 ]) m in
  let div2 = where ind (bind1 T.Sub [ div; one ]) div in
  (bind1 T.Round [ div2 ], m2)

let floor_divide x1 x2 =
  let a, b = p2 (promote_numeric [ x1; x2 ]) in
  let dt = dtype a and sh = shape a in
  if is_integer dt then begin
    let q = bind1 T.Div [ a; b ] in
    let sel =
      bind1 T.And
        [
          bind1 T.Ne [ bind1 T.Sign [ a ]; bind1 T.Sign [ b ] ];
          bind1 T.Ne [ bind1 T.Rem [ a; b ]; const_full dt sh 0.0 ];
        ]
    in
    where sel (bind1 T.Sub [ q; const_full dt sh 1.0 ]) q
  end
  else fst (float_divmod a b)

let remainder x1 x2 =
  let a, b0 = p2 (promote_numeric [ x1; x2 ]) in
  let dt = dtype a and sh = shape a in
  let zero = const_full dt sh 0.0 in
  let b =
    if is_integer dt then
      where (bind1 T.Eq [ b0; zero ]) (const_full dt sh 1.0) b0
    else b0
  in
  let trunc_mod = bind1 T.Rem [ a; b ] in
  let do_plus =
    bind1 T.And
      [
        bind1 T.Ne [ bind1 T.Lt [ trunc_mod; zero ]; bind1 T.Lt [ b; zero ] ];
        bind1 T.Ne [ trunc_mod; zero ];
      ]
  in
  where do_plus (bind1 T.Add [ trunc_mod; b ]) trunc_mod

let mod_ = remainder

let fmod x1 x2 =
  let rt, _ = Dtypes.result_type [ (dtype x1, weak x1); (dtype x2, weak x2) ] in
  let x2 =
    if is_integer rt then
      where
        (bind1 T.Eq [ x2; zeros_like x2 ])
        (const_full (dtype x2) (shape x2) 1.0)
        x2
    else x2
  in
  let a, b = p2 (promote_numeric [ x1; x2 ]) in
  bind1 T.Rem [ a; b ]

let divmod x1 x2 =
  let a, b = p2 (promote_numeric [ x1; x2 ]) in
  if is_integer (dtype a) then [ floor_divide a b; remainder a b ]
  else
    let d, m = float_divmod a b in
    [ d; m ]

let modf x =
  let a = p1 (promote_inexact [ x ]) in
  let dt = dtype a and sh = shape a in
  let whole =
    where
      (bind1 T.Ge [ a; const_full dt sh 0.0 ])
      (bind1 T.Floor [ a ]) (bind1 T.Ceil [ a ])
  in
  [ bind1 T.Sub [ a; whole ]; whole ]

let signbit x =
  let a = p1 (promote [ x ]) in
  let sh = shape a in
  match dtype a with
  | D.I32 | D.I64 -> bind1 T.Lt [ a; zeros_like a ]
  | D.Bool | D.Uint32 -> const_full D.Bool sh 0.0
  | D.Complex64 | D.Complex128 ->
      invalid_arg "signbit: complex dtype unsupported"
  | D.F32 ->
      let i = bind1 (T.Bitcast_convert_type D.I32) [ a ] in
      let s = bind1 T.Shift_right_arithmetic [ i; const_full D.I32 sh 31.0 ] in
      bind1 (T.Convert_element_type D.Bool) [ s ]
  | D.F64 ->
      let i = bind1 (T.Bitcast_convert_type D.I64) [ a ] in
      let s = bind1 T.Shift_right_arithmetic [ i; const_full D.I64 sh 63.0 ] in
      bind1 (T.Convert_element_type D.Bool) [ s ]

let copysign x1 x2 =
  let a, b = p2 (promote_inexact [ x1; x2 ]) in
  where (signbit b) (bind1 T.Neg [ bind1 T.Abs [ a ] ]) (bind1 T.Abs [ a ])

let isfinite x =
  match dtype x with
  | D.F32 | D.F64 -> bind1 T.Is_finite [ x ]
  | _ -> const_full D.Bool (shape x) 1.0

let isinf x =
  match dtype x with
  | D.F32 | D.F64 ->
      bind1 T.Eq [ bind1 T.Abs [ x ]; const_full (dtype x) (shape x) infinity ]
  | _ -> const_full D.Bool (shape x) 0.0

let isnan x = bind1 T.Ne [ x; x ]

let isposinf x =
  match dtype x with
  | D.F32 | D.F64 -> bind1 T.Eq [ x; const_full (dtype x) (shape x) infinity ]
  | _ -> const_full D.Bool (shape x) 0.0

let isneginf x =
  match dtype x with
  | D.F32 | D.F64 ->
      bind1 T.Eq [ x; const_full (dtype x) (shape x) neg_infinity ]
  | _ -> const_full D.Bool (shape x) 0.0

let heaviside x1 x2 =
  let a, b = p2 (promote_inexact [ x1; x2 ]) in
  let dt = dtype a and sh = shape a in
  let zero = const_full dt sh 0.0 and one = const_full dt sh 1.0 in
  let inner2 = where (bind1 T.Ne [ a; a ]) a b in
  let inner1 = where (bind1 T.Gt [ a; zero ]) one inner2 in
  where (bind1 T.Lt [ a; zero ]) zero inner1

let hypot x1 x2 =
  let a0, b0 = p2 (promote_inexact [ x1; x2 ]) in
  let dt = dtype a0 and sh = shape a0 in
  let a = bind1 T.Abs [ a0 ] and b = bind1 T.Abs [ b0 ] in
  let idx_inf = bind1 T.Or [ isposinf a; isposinf b ] in
  let x1m = bind1 T.Max [ a; b ] and x2m = bind1 T.Min [ a; b ] in
  let zero = const_full dt sh 0.0 and one = const_full dt sh 1.0 in
  let x1_is0 = bind1 T.Eq [ x1m; zero ] in
  let denom = where x1_is0 one x1m in
  let ratio = bind1 T.Div [ x2m; denom ] in
  let v =
    bind1 T.Mul
      [ x1m; bind1 T.Sqrt [ bind1 T.Add [ one; bind1 T.Square [ ratio ] ] ] ]
  in
  let x = where x1_is0 x1m v in
  where idx_inf (const_full dt sh infinity) x

let sinc x =
  let a = p1 (promote_inexact [ x ]) in
  let dt = dtype a and sh = shape a in
  let eq_zero = bind1 T.Eq [ a; const_full dt sh 0.0 ] in
  let pi_x = bind1 T.Mul [ const_full dt sh Float.pi; a ] in
  let safe = where eq_zero (const_full dt sh 1.0) pi_x in
  where eq_zero (const_full dt sh 1.0)
    (bind1 T.Div [ bind1 T.Sin [ safe ]; safe ])

let logaddexp x1 x2 =
  let a, b = p2 (promote_inexact [ x1; x2 ]) in
  let amax = bind1 T.Max [ a; b ] in
  let delta = bind1 T.Sub [ a; b ] in
  let normal =
    bind1 T.Add
      [
        amax;
        bind1 T.Log1p [ bind1 T.Exp [ bind1 T.Neg [ bind1 T.Abs [ delta ] ] ] ];
      ]
  in
  where (bind1 T.Ne [ delta; delta ]) (bind1 T.Add [ a; b ]) normal

let logaddexp2 x1 x2 =
  let a, b = p2 (promote_inexact [ x1; x2 ]) in
  let dt = dtype a and sh = shape a in
  let amax = bind1 T.Max [ a; b ] in
  let invln2 = const_full dt sh (1.0 /. Float.log 2.0) in
  let delta = bind1 T.Sub [ a; b ] in
  let normal =
    bind1 T.Add
      [
        amax;
        bind1 T.Mul
          [
            invln2;
            bind1 T.Log1p
              [ bind1 T.Exp2 [ bind1 T.Neg [ bind1 T.Abs [ delta ] ] ] ];
          ];
      ]
  in
  where (bind1 T.Ne [ delta; delta ]) (bind1 T.Add [ a; b ]) normal

let rint x =
  match dtype x with
  | D.I32 | D.I64 | D.Bool -> convert x (Dtypes.default_float_dtype ())
  | _ -> bind1 T.Round [ x ]

let is_complex_dtype = function
  | D.Complex64 | D.Complex128 -> true
  | _ -> false

let imag x =
  if is_complex_dtype (dtype x) then bind1 T.Imag [ x ] else zeros_like x

let real x = if is_complex_dtype (dtype x) then bind1 T.Real [ x ] else x
let conjugate x = if is_complex_dtype (dtype x) then bind1 T.Conj [ x ] else x
let conj = conjugate
