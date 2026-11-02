open Types

let bind1 = Core.bind1
let dtype v = (Core.get_aval v).dtype
let shape v = (Core.get_aval v).shape
let is_complex v = Dtypes.issubdtype (dtype v) `Complex
let const_full dt sh x = Core.splat dt sh x
let where pred x y = bind1 Select_n [ pred; y; x ]

let lax_const x v =
  let a = Core.get_aval x in
  let default_float = Dtypes.default_float_dtype () in
  let default_complex =
    if default_float = Dtype.F64 then Dtype.Complex128 else Dtype.Complex64
  in
  if a.weak_type && Array.length a.shape = 0 then
    if Dtypes.issubdtype a.dtype `Floating && a.dtype = default_float then
      Core.py_scalar a.dtype v
    else if Dtypes.issubdtype a.dtype `Complex && a.dtype = default_complex then
      Concrete
        (Ndarray.as_weak
           (Ndarray.of_complex a.dtype [||] [| { Complex.re = v; im = 0.0 } |]))
    else Core.splat a.dtype [||] v
  else Core.splat a.dtype [||] v

let wrap_between x half =
  let half_v = lax_const x half in
  let two_half = lax_const x (2.0 *. half) in
  let zero = lax_const x 0.0 in
  let rem = bind1 Rem [ bind1 Add [ x; half_v ]; two_half ] in
  let rem = where (bind1 Lt [ rem; zero ]) (bind1 Add [ rem; two_half ]) rem in
  bind1 Sub [ rem; half_v ]

let logaddexp_body a b =
  let amax = bind1 Max [ a; b ] in
  if is_complex a then begin
    let two = lax_const amax 2.0 in
    let sum = bind1 Add [ a; b ] in
    let delta = bind1 Sub [ sum; bind1 Mul [ amax; two ] ] in
    let out = bind1 Add [ amax; bind1 Log1p [ bind1 Exp [ delta ] ] ] in
    bind1 Complex
      [ bind1 Real [ out ]; wrap_between (bind1 Imag [ out ]) Float.pi ]
  end
  else
    let delta = bind1 Sub [ a; b ] in
    let is_nan = bind1 Ne [ delta; delta ] in
    let sum = bind1 Add [ a; b ] in
    let magnitude = bind1 Abs [ delta ] in
    let negated = bind1 Neg [ magnitude ] in
    let exponent = bind1 Exp [ negated ] in
    let tail = bind1 Log1p [ exponent ] in
    let normal = bind1 Add [ amax; tail ] in
    bind1 Select_n [ is_nan; normal; sum ]

let replace_inf x =
  let re_x = if is_complex x then bind1 Real [ x ] else x in
  let inf = lax_const re_x Float.infinity in
  let pred = bind1 Eq [ re_x; inf ] in
  let zeros = Core.splat (dtype x) (shape x) 0.0 in
  bind1 Select_n [ pred; x; zeros ]

let logaddexp_custom =
  let self = ref (fun _ -> []) in
  let call =
    Custom_derivatives.custom_jvp ~name:"logaddexp" ~jvp_name:"_logaddexp_jvp"
      (function
        | [ a; b ] -> [ logaddexp_body a b ]
        | _ -> invalid_arg "logaddexp: arity")
      ~jvp:(fun primals tangents ->
        match (primals, tangents) with
        | [ x1; x2 ], [ t1; t2 ] ->
            let out = !self [ x1; x2 ] in
            let po = List.hd out in
            let term t x =
              let rx = replace_inf x in
              let rp = replace_inf po in
              let d = bind1 Sub [ rx; rp ] in
              let e = bind1 Exp [ d ] in
              bind1 Mul [ t; e ]
            in
            let first = term t1 x1 in
            let second = term t2 x2 in
            (out, [ bind1 Add [ first; second ] ])
        | _ -> invalid_arg "logaddexp: arity")
  in
  self := call;
  call

let logaddexp a b = List.hd (logaddexp_custom [ a; b ])

let logaddexp2_body a b =
  let amax = bind1 Max [ a; b ] in
  let invln2 = lax_const amax (1.0 /. Float.log 2.0) in
  if is_complex a then begin
    let sum = bind1 Add [ a; b ] in
    let delta = bind1 Sub [ sum; bind1 Mul [ amax; lax_const amax 2.0 ] ] in
    let out =
      bind1 Add
        [ amax; bind1 Mul [ invln2; bind1 Log1p [ bind1 Exp2 [ delta ] ] ] ]
    in
    bind1 Complex
      [
        bind1 Real [ out ];
        wrap_between (bind1 Imag [ out ]) (Float.pi /. Float.log 2.0);
      ]
  end
  else
    let delta = bind1 Sub [ a; b ] in
    let is_nan = bind1 Ne [ delta; delta ] in
    let sum = bind1 Add [ a; b ] in
    let magnitude = bind1 Abs [ delta ] in
    let negated = bind1 Neg [ magnitude ] in
    let exponent = bind1 Exp2 [ negated ] in
    let tail = bind1 Log1p [ exponent ] in
    let scaled = bind1 Mul [ invln2; tail ] in
    let normal = bind1 Add [ amax; scaled ] in
    bind1 Select_n [ is_nan; normal; sum ]

let logaddexp2_custom =
  let self = ref (fun _ -> []) in
  let call =
    Custom_derivatives.custom_jvp ~name:"logaddexp2" ~jvp_name:"_logaddexp2_jvp"
      (function
        | [ a; b ] -> [ logaddexp2_body a b ]
        | _ -> invalid_arg "logaddexp2: arity")
      ~jvp:(fun primals tangents ->
        match (primals, tangents) with
        | [ x1; x2 ], [ t1; t2 ] ->
            let out = !self [ x1; x2 ] in
            let po = List.hd out in
            let term t x =
              let rx = replace_inf x in
              let rp = replace_inf po in
              let d = bind1 Sub [ rx; rp ] in
              let e = bind1 Exp2 [ d ] in
              bind1 Mul [ t; e ]
            in
            let first = term t1 x1 in
            let second = term t2 x2 in
            (out, [ bind1 Add [ first; second ] ])
        | _ -> invalid_arg "logaddexp2: arity")
  in
  self := call;
  call

let logaddexp2 a b = List.hd (logaddexp2_custom [ a; b ])
