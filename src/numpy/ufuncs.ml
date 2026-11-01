module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let shape v = (get_aval v).T.shape
let weak v = (get_aval v).T.weak_type
let bind1 = C.bind1
let array_full dt sh x = Core.splat dt sh x
let const_full dt _sh x = Core.splat dt [||] x
let zeros_like v = const_full (dtype v) (shape v) 0.0

let to_inexact_dtype = function
  | D.F32 -> D.F32
  | D.F64 -> D.F64
  | ( D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn | D.F8_e3m4
    | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz | D.F8_e5m2
    | D.F8_e5m2fnuz | D.F8_e8m0fnu ) as narrow ->
      narrow
  | D.I32 | D.Bool | D.Uint32 | D.I8 | D.I16 | D.I4 | D.Uint8 | D.Uint16
  | D.Uint4 ->
      D.F32
  | (D.I2 | D.Uint2) as narrow -> narrow
  | D.Float0 -> invalid_arg "float0 has no inexact dtype"
  | D.I64 | D.Uint64 -> D.F64
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

let broadcast_to = Jnp_util.broadcast_to

let common_shape vs =
  List.fold_left (fun acc v -> broadcast_shapes acc (shape v)) [||] vs

let check_rank_promotion vs =
  let mode = Config.value Config.numpy_rank_promotion in
  if mode <> "allow" then begin
    let ranks =
      List.sort_uniq compare
        (List.filter_map
           (fun v ->
             match Array.length (shape v) with 0 -> None | r -> Some r)
           vs)
    in
    if List.length ranks > 1 then begin
      let text =
        "Operands could not be broadcast together: implicit rank promotion is \
         disabled by numpy_rank_promotion"
      in
      if mode = "raise" then invalid_arg text else prerr_endline text
    end
  end

let promote_shapes vs =
  match vs with
  | [] | [ _ ] -> vs
  | first :: rest ->
      let rank v = Array.length (shape v) in
      if List.for_all (fun v -> rank v = rank first) rest then vs
      else
        let nonscalar =
          List.sort_uniq compare
            (List.filter_map
               (fun v -> if rank v = 0 then None else Some (rank v))
               vs)
        in
        if List.length nonscalar < 2 then vs
        else
          let n = List.fold_left (fun acc v -> max acc (rank v)) 0 vs in
          List.map
            (fun v ->
              let s = shape v in
              let k = n - Array.length s in
              if k = 0 then v
              else
                bind1
                  (T.Broadcast_in_dim
                     {
                       shape = Array.append (Array.make k 1) s;
                       dims = Array.init (Array.length s) (fun i -> i + k);
                     })
                  [ v ])
            vs

let promote_to dt vs =
  check_rank_promotion vs;
  promote_shapes (List.map (fun v -> convert v dt) vs)

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

let fabs x =
  (match dtype x with
  | D.Complex64 | D.Complex128 ->
      raise
        (Errors.jax_type_error "ufunc 'fabs' does not support complex dtypes")
  | _ -> ());
  unary_inexact T.Abs x

let is_float_or_complex = function
  | D.F16 | D.Bf16 | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> true
  | _ -> false

let floor x =
  if is_float_or_complex (dtype x) then unary_inexact T.Floor x else x

let ceil x = if is_float_or_complex (dtype x) then unary_inexact T.Ceil x else x
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
  match (prim, dtype a) with
  | (T.Lt | T.Gt | T.Le | T.Ge), (D.Complex64 | D.Complex128) ->
      let re v = bind1 T.Real [ v ] and im v = bind1 T.Imag [ v ] in
      bind1 T.Select_n
        [
          bind1 T.Eq [ re a; re b ];
          bind1 prim [ re a; re b ];
          bind1 prim [ im a; im b ];
        ]
  | _ -> bind1 prim [ a; b ]

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
  if
    Dtypes.issubdtype (dtype a) `Floating && Dtypes.itemsize_bits (dtype a) < 16
  then failwith "jax.numpy.signbit only supports 16, 32, and 64-bit types.";
  let dt = dtype a and sh = shape a in
  if dt = D.F16 then
    bind1 T.Sub [ bind1 T.Nextafter [ a; const_full dt sh infinity ]; a ]
  else
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

let is_integer = function
  | D.I8 | D.I16 | D.I2 | D.I4 | D.I32 | D.I64 | D.Uint8 | D.Uint16 | D.Uint2
  | D.Uint4 | D.Uint32 | D.Uint64 ->
      true
  | _ -> false

let is_inexact = function
  | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn | D.F8_e3m4
  | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz | D.F8_e5m2
  | D.F8_e5m2fnuz | D.F8_e8m0fnu | D.F32 | D.F64 | D.Complex64 | D.Complex128 ->
      true
  | _ -> false

let where = Jnp_util.where

let absolute x =
  match dtype x with
  | D.Bool | D.Uint8 | D.Uint16 | D.Uint32 | D.Uint64 -> x
  | _ -> bind1 T.Abs [ x ]

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

let log_over base x =
  let a = p1 (promote_inexact [ x ]) in
  let log_x = bind1 T.Log [ a ] in
  let log_base part =
    bind1 T.Div [ part; const_full (dtype part) (shape part) (Float.log base) ]
  in
  match dtype a with
  | D.Complex64 | D.Complex128 ->
      bind1 T.Complex
        [ log_base (bind1 T.Real [ log_x ]); log_base (bind1 T.Imag [ log_x ]) ]
  | _ -> log_base log_x

let log2 x = log_over 2.0 x
let log10 x = log_over 10.0 x

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
  match dtype a with
  | D.Uint8 | D.Uint16 | D.Uint32 | D.Uint64 ->
      bind1 T.Shift_right_logical [ a; b ]
  | _ -> bind1 T.Shift_right_arithmetic [ a; b ]

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

let concrete_integer_exponent x2 =
  match x2 with
  | T.Concrete nd
    when Ndarray.shape nd = [||]
         && (is_integer (Ndarray.dtype nd) || Ndarray.dtype nd = D.Bool) ->
      Some (Int64.to_int (Ndarray.get_i64 nd [||]))
  | _ -> None

let power_general x1 x2 =
  let sh = broadcast_shapes (shape x1) (shape x2) in
  let x1 = broadcast_to x1 sh and x2 = broadcast_to x2 sh in
  let d1 = dtype x1 and d2 = dtype x2 in
  let a, b = p2 (promote_numeric [ x1; x2 ]) in
  if is_integer (dtype a) || dtype a = D.Bool then pow_int_int a b
  else if is_inexact d1 && is_integer d2 then bind1 T.Pow [ x1; x2 ]
  else bind1 T.Pow [ a; b ]

let power x1 x2 =
  match concrete_integer_exponent x2 with
  | Some y -> bind1 (T.Integer_pow y) [ p1 (promote_numeric [ x1 ]) ]
  | None -> power_general x1 x2

let pow = power

let float_divmod x1 x2 =
  let dt = dtype x1 and sh = shape x1 in
  let zero = const_full dt sh 0.0 and one = const_full dt sh 1.0 in
  let m = bind1 T.Rem [ x1; x2 ] in
  let x2_is_zero = bind1 T.Eq [ x2; zero ] in
  let corrected = bind1 T.Sub [ x1; m ] in
  let x1c = where x2_is_zero x1 corrected in
  let div = bind1 T.Div [ x1c; x2 ] in
  let m_nonzero = bind1 T.Ne [ m; zero ] in
  let sign_x2 = bind1 T.Sign [ x2 ] in
  let sign_m = bind1 T.Sign [ m ] in
  let sign_differs = bind1 T.Ne [ sign_x2; sign_m ] in
  let ind = bind1 T.And [ m_nonzero; sign_differs ] in
  let m_plus = bind1 T.Add [ m; x2 ] in
  let m2 = bind1 T.Select_n [ ind; m; m_plus ] in
  let div_minus = bind1 T.Sub [ div; one ] in
  let div2 = bind1 T.Select_n [ ind; div; div_minus ] in
  (bind1 T.Round [ div2 ], m2)

let is_unsigned dt = Dtypes.issubdtype dt `Unsigned

let floor_divide x1 x2 =
  Jnp_util.jit1 "floor_divide"
    (function
      | [ x1; x2 ] ->
          let a, b = p2 (promote_numeric [ x1; x2 ]) in
          let dt = dtype a and sh = shape a in
          if is_unsigned dt then bind1 T.Div [ a; b ]
          else if is_integer dt then begin
            let q = bind1 T.Div [ a; b ] in
            let sign_a = bind1 T.Sign [ a ] in
            let sign_b = bind1 T.Sign [ b ] in
            let differs = bind1 T.Ne [ sign_a; sign_b ] in
            let r = bind1 T.Rem [ a; b ] in
            let r_nonzero = bind1 T.Ne [ r; const_full dt sh 0.0 ] in
            let sel = bind1 T.And [ differs; r_nonzero ] in
            let q1 = bind1 T.Sub [ q; const_full dt sh 1.0 ] in
            where sel q1 q
          end
          else fst (float_divmod a b)
      | _ -> invalid_arg "floor_divide: arity")
    [ x1; x2 ]

let remainder x1 x2 =
  Jnp_util.jit1 "remainder"
    (function
      | [ x1; x2 ] ->
          let a, b0 = p2 (promote_numeric [ x1; x2 ]) in
          let dt = dtype a and sh = shape a in
          let zero = const_full dt sh 0.0 in
          let b =
            if is_integer dt then begin
              let is_zero = bind1 T.Eq [ b0; const_full dt sh 0.0 ] in
              where is_zero (array_full dt (shape b0) 1.0) b0
            end
            else b0
          in
          let trunc_mod = bind1 T.Rem [ a; b ] in
          let not_zero = bind1 T.Ne [ trunc_mod; zero ] in
          let mod_negative = bind1 T.Lt [ trunc_mod; zero ] in
          let b_negative = bind1 T.Lt [ b; zero ] in
          let differs = bind1 T.Ne [ mod_negative; b_negative ] in
          let do_plus = bind1 T.And [ differs; not_zero ] in
          let plus = bind1 T.Add [ trunc_mod; b ] in
          bind1 T.Select_n [ do_plus; trunc_mod; plus ]
      | _ -> invalid_arg "remainder: arity")
    [ x1; x2 ]

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

let signbit_body x =
  let a = p1 (promote [ x ]) in
  let sh = shape a in
  match dtype a with
  | D.I32 | D.I64 | D.I8 | D.I16 | D.I2 | D.I4 -> bind1 T.Lt [ a; zeros_like a ]
  | D.F16 | D.Bf16 ->
      let i = bind1 (T.Bitcast_convert_type D.I16) [ a ] in
      bind1 T.Lt [ i; zeros_like i ]
  | D.Float0 -> invalid_arg "signbit does not accept float0"
  | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn | D.F8_e3m4 | D.F8_e4m3
  | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz | D.F8_e5m2 | D.F8_e5m2fnuz
  | D.F8_e8m0fnu ->
      failwith "jax.numpy.signbit only supports 16, 32, and 64-bit types."
  | D.Bool | D.Uint32 | D.Uint8 | D.Uint16 | D.Uint2 | D.Uint4 | D.Uint64 ->
      array_full D.Bool sh 0.0
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

let signbit x =
  Jnp_util.jit1 "signbit"
    (function [ x ] -> signbit_body x | _ -> invalid_arg "signbit: arity")
    [ x ]

let copysign x1 x2 =
  let a, b = p2 (promote_inexact [ x1; x2 ]) in
  where (signbit b) (bind1 T.Neg [ bind1 T.Abs [ a ] ]) (bind1 T.Abs [ a ])

let isfinite x =
  match dtype x with
  | D.Complex64 | D.Complex128 ->
      bind1 T.And
        [
          bind1 T.Is_finite [ bind1 T.Real [ x ] ];
          bind1 T.Is_finite [ bind1 T.Imag [ x ] ];
        ]
  | d when Dtypes.issubdtype d `Floating -> bind1 T.Is_finite [ x ]
  | _ -> array_full D.Bool (shape x) 1.0

let isinf_body x =
  let of_real v =
    bind1 T.Eq [ bind1 T.Abs [ v ]; const_full (dtype v) (shape v) infinity ]
  in
  match dtype x with
  | D.Complex64 | D.Complex128 ->
      bind1 T.Or [ of_real (bind1 T.Real [ x ]); of_real (bind1 T.Imag [ x ]) ]
  | d when Dtypes.issubdtype d `Floating -> of_real x
  | _ -> array_full D.Bool (shape x) 0.0

let isinf x =
  Jnp_util.jit1 "isinf"
    (function [ x ] -> isinf_body x | _ -> invalid_arg "isinf: arity")
    [ x ]

let isnan x = bind1 T.Ne [ x; x ]

let isposinf x =
  match dtype x with
  | D.Complex64 | D.Complex128 ->
      invalid_arg "isposinf/isneginf are not well defined for complex types"
  | d when Dtypes.issubdtype d `Floating ->
      bind1 T.Eq [ x; const_full (dtype x) (shape x) infinity ]
  | _ -> array_full D.Bool (shape x) 0.0

let isneginf x =
  match dtype x with
  | D.Complex64 | D.Complex128 ->
      invalid_arg "isposinf/isneginf are not well defined for complex types"
  | d when Dtypes.issubdtype d `Floating ->
      bind1 T.Eq [ x; const_full (dtype x) (shape x) neg_infinity ]
  | _ -> array_full D.Bool (shape x) 0.0

let heaviside x1 x2 =
  let a, b = p2 (promote_inexact [ x1; x2 ]) in
  let dt = dtype a and sh = shape a in
  let zero = const_full dt sh 0.0 and one = const_full dt sh 1.0 in
  let inner2 = where (bind1 T.Ne [ a; a ]) a b in
  let inner1 = where (bind1 T.Gt [ a; zero ]) one inner2 in
  where (bind1 T.Lt [ a; zero ]) zero inner1

let hypot x1 x2 =
  let a0, b0 = p2 (promote_inexact [ x1; x2 ]) in
  (match dtype a0 with
  | D.Complex64 | D.Complex128 ->
      invalid_arg
        "jnp.hypot is not well defined for complex-valued inputs. Please \
         convert to real values first, such as by using abs(x)"
  | _ -> ());
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

let rec sinc_maclaurin_call k =
  Custom_derivatives.custom_jvp ~name:"_sinc_maclaurin"
    ~jvp_name:"_sinc_maclaurin_jvp"
    (function
      | [ x ] ->
          let zeroed = bind1 T.Mul [ x; Core.splat (dtype x) [||] 0.0 ] in
          if k mod 2 = 1 then [ zeroed ]
          else
            let sign = if k / 2 mod 2 = 0 then 1.0 else -1.0 in
            let c = sign /. float_of_int (k + 1) in
            [ bind1 T.Add [ zeroed; Core.splat (dtype x) [||] c ] ]
      | _ -> invalid_arg "sinc: arity")
    ~jvp:(fun primals tangents ->
      match (primals, tangents) with
      | [ x ], [ t ] ->
          let out = sinc_maclaurin_call k [ x ] in
          let next = List.hd (sinc_maclaurin_call (k + 1) [ x ]) in
          (out, [ bind1 T.Mul [ next; t ] ])
      | _ -> invalid_arg "sinc: arity")

let sinc x =
  Jnp_util.jit1 "sinc"
    (function
      | [ x ] ->
          let a = p1 (promote_inexact [ x ]) in
          let dt = dtype a and sh = shape a in
          ignore sh;
          let eq_zero = bind1 T.Eq [ a; Core.splat dt [||] 0.0 ] in
          let pi_x = bind1 T.Mul [ Core.splat dt [||] Float.pi; a ] in
          let safe = where eq_zero (Core.splat dt [||] 1.0) pi_x in
          let maclaurin = List.hd (sinc_maclaurin_call 0 [ pi_x ]) in
          let sin_safe = bind1 T.Sin [ safe ] in
          let ratio = bind1 T.Div [ sin_safe; safe ] in
          where eq_zero maclaurin ratio
      | _ -> invalid_arg "sinc: arity")
    [ x ]

let wrap_between x half =
  let sh = shape x in
  let two_half = const_full (dtype x) sh (2.0 *. half) in
  let half_v = const_full (dtype x) sh half in
  let rem = bind1 T.Rem [ bind1 T.Add [ x; half_v ]; two_half ] in
  let zero = const_full (dtype x) sh 0.0 in
  let rem =
    where (bind1 T.Lt [ rem; zero ]) (bind1 T.Add [ rem; two_half ]) rem
  in
  bind1 T.Sub [ rem; half_v ]

let is_complex v =
  match dtype v with D.Complex64 | D.Complex128 -> true | _ -> false

let logaddexp x1 x2 =
  let a, b = p2 (promote_inexact [ x1; x2 ]) in
  Lax_other.logaddexp a b

let logaddexp2 x1 x2 =
  let a, b = p2 (promote_inexact [ x1; x2 ]) in
  let dt = dtype a and sh = shape a in
  let amax = bind1 T.Max [ a; b ] in
  let invln2 = const_full dt sh (1.0 /. Float.log 2.0) in
  if is_complex a then begin
    let two = const_full dt sh 2.0 in
    let delta =
      bind1 T.Sub [ bind1 T.Add [ a; b ]; bind1 T.Mul [ amax; two ] ]
    in
    let out =
      bind1 T.Add
        [
          amax; bind1 T.Mul [ invln2; bind1 T.Log1p [ bind1 T.Exp2 [ delta ] ] ];
        ]
    in
    bind1 T.Complex
      [
        bind1 T.Real [ out ];
        wrap_between (bind1 T.Imag [ out ]) (Float.pi /. Float.log 2.0);
      ]
  end
  else
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
  | D.F16 | D.Bf16 | D.F32 | D.F64 -> bind1 T.Round_nearest_even [ x ]
  | D.Complex64 | D.Complex128 ->
      let part p = bind1 T.Round_nearest_even [ bind1 p [ x ] ] in
      bind1 T.Complex [ part T.Real; part T.Imag ]
  | _ -> convert x (Dtypes.default_float_dtype ())

let is_complex_dtype = function
  | D.Complex64 | D.Complex128 -> true
  | _ -> false

let imag x =
  if is_complex_dtype (dtype x) then bind1 T.Imag [ x ]
  else array_full (dtype x) (shape x) 0.0

let real x = if is_complex_dtype (dtype x) then bind1 T.Real [ x ] else x
let conjugate x = if is_complex_dtype (dtype x) then bind1 T.Conj [ x ] else x
let conj = conjugate

let arccosh p0 =
  Jnp_util.jit1 "arccosh"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> arccosh p0 | _ -> invalid_arg "arccosh: arity")
    [ p0 ]

let rint p0 =
  Jnp_util.jit1 "rint"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> rint p0 | _ -> invalid_arg "rint: arity")
    [ p0 ]

let copysign p0 p1 =
  Jnp_util.jit1 "copysign"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function
      | [ p0; p1 ] -> copysign p0 p1 | _ -> invalid_arg "copysign: arity")
    [ p0; p1 ]

let divmod p0 p1 =
  Jnp_util.jit "divmod"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> divmod p0 p1 | _ -> invalid_arg "divmod: arity")
    [ p0; p1 ]

let fmod p0 p1 =
  Jnp_util.jit1 "fmod"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> fmod p0 p1 | _ -> invalid_arg "fmod: arity")
    [ p0; p1 ]

let modf p0 =
  Jnp_util.jit "modf"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> modf p0 | _ -> invalid_arg "modf: arity")
    [ p0 ]

let heaviside p0 p1 =
  Jnp_util.jit1 "heaviside"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function
      | [ p0; p1 ] -> heaviside p0 p1 | _ -> invalid_arg "heaviside: arity")
    [ p0; p1 ]

let hypot p0 p1 =
  Jnp_util.jit1 "hypot"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> hypot p0 p1 | _ -> invalid_arg "hypot: arity")
    [ p0; p1 ]

let fabs a0 =
  Eager_jit.one "numpy.ufuncs.fabs" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> fabs a0 | _ -> assert false)

let bitwise_invert a0 =
  Eager_jit.one "numpy.ufuncs.bitwise_invert" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> bitwise_invert a0
      | _ -> assert false)

let bitwise_not a0 =
  Eager_jit.one "numpy.ufuncs.bitwise_not" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> bitwise_not a0 | _ -> assert false)

let invert a0 =
  Eager_jit.one "numpy.ufuncs.invert" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> invert a0 | _ -> assert false)

let positive a0 =
  Eager_jit.one "numpy.ufuncs.positive" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> positive a0 | _ -> assert false)

let sign a0 =
  Eager_jit.one "numpy.ufuncs.sign" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> sign a0 | _ -> assert false)

let floor a0 =
  Eager_jit.one "numpy.ufuncs.floor" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> floor a0 | _ -> assert false)

let ceil a0 =
  Eager_jit.one "numpy.ufuncs.ceil" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> ceil a0 | _ -> assert false)

let exp a0 =
  Eager_jit.one "numpy.ufuncs.exp" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> exp a0 | _ -> assert false)

let log a0 =
  Eager_jit.one "numpy.ufuncs.log" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> log a0 | _ -> assert false)

let expm1 a0 =
  Eager_jit.one "numpy.ufuncs.expm1" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> expm1 a0 | _ -> assert false)

let log1p a0 =
  Eager_jit.one "numpy.ufuncs.log1p" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> log1p a0 | _ -> assert false)

let sin a0 =
  Eager_jit.one "numpy.ufuncs.sin" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> sin a0 | _ -> assert false)

let cos a0 =
  Eager_jit.one "numpy.ufuncs.cos" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> cos a0 | _ -> assert false)

let tan a0 =
  Eager_jit.one "numpy.ufuncs.tan" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> tan a0 | _ -> assert false)

let arcsin a0 =
  Eager_jit.one "numpy.ufuncs.arcsin" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> arcsin a0 | _ -> assert false)

let arccos a0 =
  Eager_jit.one "numpy.ufuncs.arccos" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> arccos a0 | _ -> assert false)

let arctan a0 =
  Eager_jit.one "numpy.ufuncs.arctan" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> arctan a0 | _ -> assert false)

let sinh a0 =
  Eager_jit.one "numpy.ufuncs.sinh" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> sinh a0 | _ -> assert false)

let cosh a0 =
  Eager_jit.one "numpy.ufuncs.cosh" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> cosh a0 | _ -> assert false)

let arcsinh a0 =
  Eager_jit.one "numpy.ufuncs.arcsinh" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> arcsinh a0 | _ -> assert false)

let arccosh a0 =
  Eager_jit.one "numpy.ufuncs.arccosh" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> arccosh a0 | _ -> assert false)

let tanh a0 =
  Eager_jit.one "numpy.ufuncs.tanh" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> tanh a0 | _ -> assert false)

let arctanh a0 =
  Eager_jit.one "numpy.ufuncs.arctanh" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> arctanh a0 | _ -> assert false)

let sqrt a0 =
  Eager_jit.one "numpy.ufuncs.sqrt" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> sqrt a0 | _ -> assert false)

let cbrt a0 =
  Eager_jit.one "numpy.ufuncs.cbrt" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> cbrt a0 | _ -> assert false)

let left_shift a0 a1 =
  Eager_jit.one "numpy.ufuncs.left_shift" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> left_shift a0 a1
      | _ -> assert false)

let bitwise_left_shift a0 a1 =
  Eager_jit.one "numpy.ufuncs.bitwise_left_shift" () [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> bitwise_left_shift a0 a1
      | _ -> assert false)

let equal a0 a1 =
  Eager_jit.one "numpy.ufuncs.equal" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> equal a0 a1
      | _ -> assert false)

let not_equal a0 a1 =
  Eager_jit.one "numpy.ufuncs.not_equal" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> not_equal a0 a1
      | _ -> assert false)

let arctan2 a0 a1 =
  Eager_jit.one "numpy.ufuncs.arctan2" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> arctan2 a0 a1
      | _ -> assert false)

let float_power a0 a1 =
  Eager_jit.one "numpy.ufuncs.float_power" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> float_power a0 a1
      | _ -> assert false)

let nextafter a0 a1 =
  Eager_jit.one "numpy.ufuncs.nextafter" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> nextafter a0 a1
      | _ -> assert false)

let spacing a0 =
  Eager_jit.one "numpy.ufuncs.spacing" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> spacing a0 | _ -> assert false)

let logical_not a0 =
  Eager_jit.one "numpy.ufuncs.logical_not" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> logical_not a0 | _ -> assert false)

let greater_equal a0 a1 =
  Eager_jit.one "numpy.ufuncs.greater_equal" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> greater_equal a0 a1
      | _ -> assert false)

let greater a0 a1 =
  Eager_jit.one "numpy.ufuncs.greater" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> greater a0 a1
      | _ -> assert false)

let less_equal a0 a1 =
  Eager_jit.one "numpy.ufuncs.less_equal" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> less_equal a0 a1
      | _ -> assert false)

let less a0 a1 =
  Eager_jit.one "numpy.ufuncs.less" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with [], [ a0; a1 ] -> less a0 a1 | _ -> assert false)

let acos a0 =
  Eager_jit.one "numpy.ufuncs.acos" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> acos a0 | _ -> assert false)

let acosh a0 =
  Eager_jit.one "numpy.ufuncs.acosh" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> acosh a0 | _ -> assert false)

let asin a0 =
  Eager_jit.one "numpy.ufuncs.asin" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> asin a0 | _ -> assert false)

let asinh a0 =
  Eager_jit.one "numpy.ufuncs.asinh" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> asinh a0 | _ -> assert false)

let atan a0 =
  Eager_jit.one "numpy.ufuncs.atan" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> atan a0 | _ -> assert false)

let atanh a0 =
  Eager_jit.one "numpy.ufuncs.atanh" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> atanh a0 | _ -> assert false)

let atan2 a0 a1 =
  Eager_jit.one "numpy.ufuncs.atan2" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> atan2 a0 a1
      | _ -> assert false)

let right_shift a0 a1 =
  Eager_jit.one "numpy.ufuncs.right_shift" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> right_shift a0 a1
      | _ -> assert false)

let bitwise_right_shift a0 a1 =
  Eager_jit.one "numpy.ufuncs.bitwise_right_shift" () [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> bitwise_right_shift a0 a1
      | _ -> assert false)

let absolute a0 =
  Eager_jit.one "numpy.ufuncs.absolute" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> absolute a0 | _ -> assert false)

let abs a0 =
  Eager_jit.one "numpy.ufuncs.abs" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> abs a0 | _ -> assert false)

let rint a0 =
  Eager_jit.one "numpy.ufuncs.rint" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> rint a0 | _ -> assert false)

let copysign a0 a1 =
  Eager_jit.one "numpy.ufuncs.copysign" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> copysign a0 a1
      | _ -> assert false)

let true_divide a0 a1 =
  Eager_jit.one "numpy.ufuncs.true_divide" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> true_divide a0 a1
      | _ -> assert false)

let floor_divide a0 a1 =
  Eager_jit.one "numpy.ufuncs.floor_divide" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> floor_divide a0 a1
      | _ -> assert false)

let divmod a0 a1 =
  Eager_jit.many "numpy.ufuncs.divmod" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> divmod a0 a1
      | _ -> assert false)

let log2 a0 =
  Eager_jit.one "numpy.ufuncs.log2" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> log2 a0 | _ -> assert false)

let log10 a0 =
  Eager_jit.one "numpy.ufuncs.log10" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> log10 a0 | _ -> assert false)

let exp2 a0 =
  Eager_jit.one "numpy.ufuncs.exp2" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> exp2 a0 | _ -> assert false)

let signbit a0 =
  Eager_jit.one "numpy.ufuncs.signbit" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> signbit a0 | _ -> assert false)

let remainder a0 a1 =
  Eager_jit.one "numpy.ufuncs.remainder" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> remainder a0 a1
      | _ -> assert false)

let fmod a0 a1 =
  Eager_jit.one "numpy.ufuncs.fmod" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with [], [ a0; a1 ] -> fmod a0 a1 | _ -> assert false)

let square a0 =
  Eager_jit.one "numpy.ufuncs.square" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> square a0 | _ -> assert false)

let deg2rad a0 =
  Eager_jit.one "numpy.ufuncs.deg2rad" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> deg2rad a0 | _ -> assert false)

let rad2deg a0 =
  Eager_jit.one "numpy.ufuncs.rad2deg" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> rad2deg a0 | _ -> assert false)

let conjugate a0 =
  Eager_jit.one "numpy.ufuncs.conjugate" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> conjugate a0 | _ -> assert false)

let imag a0 =
  Eager_jit.one "numpy.ufuncs.imag" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> imag a0 | _ -> assert false)

let real a0 =
  Eager_jit.one "numpy.ufuncs.real" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> real a0 | _ -> assert false)

let modf a0 =
  Eager_jit.many "numpy.ufuncs.modf" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> modf a0 | _ -> assert false)

let isfinite a0 =
  Eager_jit.one "numpy.ufuncs.isfinite" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> isfinite a0 | _ -> assert false)

let isinf a0 =
  Eager_jit.one "numpy.ufuncs.isinf" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> isinf a0 | _ -> assert false)

let isnan a0 =
  Eager_jit.one "numpy.ufuncs.isnan" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> isnan a0 | _ -> assert false)

let heaviside a0 a1 =
  Eager_jit.one "numpy.ufuncs.heaviside" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> heaviside a0 a1
      | _ -> assert false)

let hypot a0 a1 =
  Eager_jit.one "numpy.ufuncs.hypot" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> hypot a0 a1
      | _ -> assert false)

let reciprocal a0 =
  Eager_jit.one "numpy.ufuncs.reciprocal" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> reciprocal a0 | _ -> assert false)

let sinc a0 =
  Eager_jit.one "numpy.ufuncs.sinc" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> sinc a0 | _ -> assert false)

let add a0 a1 =
  Eager_jit.one "numpy.ufuncs.add" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with [], [ a0; a1 ] -> add a0 a1 | _ -> assert false)

let subtract a0 a1 =
  Eager_jit.one "numpy.ufuncs.subtract" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> subtract a0 a1
      | _ -> assert false)

let multiply a0 a1 =
  Eager_jit.one "numpy.ufuncs.multiply" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> multiply a0 a1
      | _ -> assert false)

let divide a0 a1 =
  Eager_jit.one "numpy.ufuncs.divide" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> divide a0 a1
      | _ -> assert false)

let negative a0 =
  Eager_jit.one "numpy.ufuncs.negative" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> negative a0 | _ -> assert false)

let power a0 a1 =
  match concrete_integer_exponent a1 with
  | Some _ -> power a0 a1
  | None ->
      Eager_jit.one "numpy.ufuncs.power" () [] [ a0; a1 ] (fun opts vals ->
          match (opts, vals) with
          | [], [ a0; a1 ] -> power a0 a1
          | _ -> assert false)

let maximum a0 a1 =
  Eager_jit.one "numpy.ufuncs.maximum" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> maximum a0 a1
      | _ -> assert false)

let minimum a0 a1 =
  Eager_jit.one "numpy.ufuncs.minimum" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> minimum a0 a1
      | _ -> assert false)

let logaddexp a0 a1 =
  Eager_jit.one "numpy.ufuncs.logaddexp" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> logaddexp a0 a1
      | _ -> assert false)

let logaddexp2 a0 a1 =
  Eager_jit.one "numpy.ufuncs.logaddexp2" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> logaddexp2 a0 a1
      | _ -> assert false)

let bitwise_and a0 a1 =
  Eager_jit.one "numpy.ufuncs.bitwise_and" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> bitwise_and a0 a1
      | _ -> assert false)

let bitwise_or a0 a1 =
  Eager_jit.one "numpy.ufuncs.bitwise_or" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> bitwise_or a0 a1
      | _ -> assert false)

let bitwise_xor a0 a1 =
  Eager_jit.one "numpy.ufuncs.bitwise_xor" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> bitwise_xor a0 a1
      | _ -> assert false)

let logical_and a0 a1 =
  Eager_jit.one "numpy.ufuncs.logical_and" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> logical_and a0 a1
      | _ -> assert false)

let logical_or a0 a1 =
  Eager_jit.one "numpy.ufuncs.logical_or" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> logical_or a0 a1
      | _ -> assert false)

let logical_xor a0 a1 =
  Eager_jit.one "numpy.ufuncs.logical_xor" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> logical_xor a0 a1
      | _ -> assert false)
