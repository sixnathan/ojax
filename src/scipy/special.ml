module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module UF = Numpy.Ufuncs
module RED = Numpy.Reductions
module NL = Numpy.Lax_numpy
module AC = Numpy.Array_creation
module Poly = Numpy.Polynomial
module NN = Nn.Functions
module TC = Numpy.Tensor_contractions
module Dts = Dtypes

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let shape v = (get_aval v).T.shape
let bind1 = C.bind1
let numel sh = Array.fold_left ( * ) 1 sh
let const_full dt sh x = Core.splat dt sh x
let scalar_of v x = const_full (dtype v) [||] x
let coeff_of v a = T.Concrete (Nd.of_floats (dtype v) [| Array.length a |] a)
let where_ = NL.where_

let to_inexact = function
  | D.I32 | D.Bool | D.Uint32 | D.I8 | D.I16 | D.I4 | D.Uint8 | D.Uint16
  | D.Uint4 ->
      D.F32
  | (D.I2 | D.Uint2) as narrow -> narrow
  | D.Float0 -> invalid_arg "float0 has no inexact dtype"
  | D.I64 | D.Uint64 -> D.F64
  | ( D.F32 | D.F64 | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn
    | D.F8_e3m4 | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz
    | D.F8_e5m2 | D.F8_e5m2fnuz | D.F8_e8m0fnu | D.Complex64 | D.Complex128 ) as
    d ->
      d

let promote_inexact vs =
  let dt = to_inexact (NL.result_type vs) in
  NL.broadcast_arrays
    (List.map (fun v -> if dtype v = dt then v else NL.astype v dt) vs)

let promote1 x =
  match promote_inexact [ x ] with [ y ] -> y | _ -> assert false

let promote2 a b =
  match promote_inexact [ a; b ] with [ a; b ] -> (a, b) | _ -> assert false

let promote3 a b c =
  match promote_inexact [ a; b; c ] with
  | [ a; b; c ] -> (a, b, c)
  | _ -> assert false

let is_integer_dtype = function
  | D.I32 | D.I64 | D.Uint32 | D.I8 | D.I16 | D.I2 | D.I4 | D.Uint8 | D.Uint16
  | D.Uint2 | D.Uint4 | D.Uint64 ->
      true
  | D.F32 | D.F64 | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn
  | D.F8_e3m4 | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz
  | D.F8_e5m2 | D.F8_e5m2fnuz | D.F8_e8m0fnu | D.Float0 | D.Bool | D.Complex64
  | D.Complex128 ->
      false

let is_complex_dtype = function
  | D.Complex64 | D.Complex128 -> true
  | _ -> false

let reject_complex name v =
  if is_complex_dtype (dtype v) then
    raise
      (Errors.jax_type_error
         (name ^ " does not support complex-valued inputs."))

let real_only name x =
  let x = promote1 x in
  reject_complex name x;
  x

let real_only2 name a b =
  let a, b = promote2 a b in
  reject_complex name a;
  (a, b)

let real_only3 name a b c =
  let a, b, c = promote3 a b c in
  reject_complex name a;
  (a, b, c)

let single_or_double x =
  match dtype x with
  | D.F32 | D.F64 -> ()
  | d ->
      raise
        (Errors.jax_type_error
           (Printf.sprintf
              "x.dtype=%s is not supported, see docstring for supported types."
              (D.short_name d)))

let lgamma_real name x = bind1 T.Lgamma [ real_only name x ]
let gammaln x = lgamma_real "gammaln" x

let gammasgn x =
  let x = real_only "gammasgn" x in
  let zero = scalar_of x 0.0 in
  let floor_x = UF.floor x in
  let x_neg = UF.less x zero in
  let is_nan = UF.isnan x in
  let two = scalar_of x 2.0 in
  let cond_nan =
    UF.logical_or (UF.logical_and x_neg (UF.equal x floor_x)) is_nan
  in
  let cond_neg =
    UF.logical_or
      (UF.logical_and x_neg (UF.not_equal (UF.remainder floor_x two) zero))
      (UF.logical_and (UF.equal x zero) (UF.signbit x))
  in
  where_ cond_nan (scalar_of x Float.nan)
    (where_ cond_neg (scalar_of x (-1.0)) (scalar_of x 1.0))

let lanczos_g = 7.0

let lanczos_coeffs =
  [|
    0.99999999999980993;
    676.5203681218851;
    -1259.1392167224028;
    771.32342877765313;
    -176.61502916214059;
    12.507343278686905;
    -0.13857109526572012;
    9.9843695780195716e-6;
    1.5056327351493116e-7;
  |]

let complex_of_parts z re im = NL.astype (bind1 T.Complex [ re; im ]) (dtype z)

let complex_nan z =
  let real_dt = if dtype z = D.Complex128 then D.F64 else D.F32 in
  let nan = const_full real_dt [||] Float.nan in
  complex_of_parts z nan nan

let complex_loggamma z =
  let s v = scalar_of z v in
  let needs_reflection = UF.less (UF.real z) (scalar_of (UF.real z) 0.5) in
  let z_lanczos = where_ needs_reflection (UF.subtract (s 1.0) z) z in
  let zz = UF.subtract z_lanczos (s 1.0) in
  let ag = ref (s lanczos_coeffs.(0)) in
  for k = 1 to 8 do
    ag :=
      UF.add !ag
        (UF.divide (s lanczos_coeffs.(k)) (UF.add zz (s (float_of_int k))))
  done;
  let t = UF.add zz (s (lanczos_g +. 0.5)) in
  let log_gamma_lanczos =
    UF.add
      (UF.subtract
         (UF.add
            (s (0.5 *. Float.log (2.0 *. Float.pi)))
            (UF.multiply (UF.add zz (s 0.5)) (UF.log t)))
         t)
      (UF.log !ag)
  in
  let z_safe = where_ needs_reflection z (AC.full_like z 0.5) in
  let reflected =
    UF.subtract
      (UF.subtract
         (UF.log (s Float.pi))
         (UF.log (UF.sin (UF.multiply (s Float.pi) z_safe))))
      log_gamma_lanczos
  in
  where_ needs_reflection reflected log_gamma_lanczos

let complex_pole_or_nan z =
  let re = UF.real z in
  let is_pole =
    UF.logical_and
      (UF.logical_and
         (UF.equal (UF.imag z) (scalar_of re 0.0))
         (UF.equal re (UF.floor re)))
      (UF.less_equal re (scalar_of re 0.0))
  in
  UF.logical_or is_pole (UF.isnan z)

let complex_gamma z =
  let safe = complex_pole_or_nan z in
  let z_safe = where_ safe (AC.ones_like z) z in
  where_ safe (complex_nan z) (UF.exp (complex_loggamma z_safe))

let complex_loggamma_scipy z =
  let safe = complex_pole_or_nan z in
  let z_safe = where_ safe (AC.ones_like z) z in
  let re = UF.real z_safe in
  let n =
    NL.astype
      (UF.maximum (scalar_of re 0.0)
         (UF.ceil (UF.subtract (scalar_of re 0.5) re)))
      (Dts.default_int_dtype ())
  in
  let idt = Dts.default_int_dtype () in
  let result =
    Lax_api.while_loop
      (fun state ->
        match state with
        | [ k; _ ] -> RED.any (UF.less k n)
        | _ -> failwith "loggamma: state")
      (fun state ->
        match state with
        | [ k; sum_log ] ->
            let mask = UF.less k n in
            let shifted = UF.add z_safe (NL.astype k (dtype z)) in
            let zk = where_ mask shifted (AC.ones_like z_safe) in
            [
              UF.add k (const_full idt [||] 1.0);
              UF.add sum_log (where_ mask (UF.log zk) (AC.zeros_like sum_log));
            ]
        | _ -> failwith "loggamma: state")
      [ const_full idt [||] 0.0; AC.zeros_like z_safe ]
  in
  let sum_log = List.nth result 1 in
  let res =
    UF.subtract
      (complex_loggamma (UF.add z_safe (NL.astype n (dtype z))))
      sum_log
  in
  where_ safe (complex_nan z) res

let loggamma x =
  let x = promote1 x in
  if is_complex_dtype (dtype x) then complex_loggamma_scipy x
  else
    let res = bind1 T.Lgamma [ x ] in
    where_ (UF.greater x (scalar_of x 0.0)) res (scalar_of x Float.nan)

let gamma x =
  let x = promote1 x in
  if is_complex_dtype (dtype x) then complex_gamma x
  else UF.multiply (gammasgn x) (UF.exp (bind1 T.Lgamma [ x ]))

let algdiv a b =
  let c0 = 0.833333333333333e-01
  and c1 = -0.277777777760991e-02
  and c2 = 0.793650666825390e-03
  and c3 = -0.595202931351870e-03
  and c4 = 0.837308034031215e-03
  and c5 = -0.165322962780713e-02 in
  let s v = scalar_of a v in
  let one = s 1.0 in
  let h = UF.divide a b in
  let c = UF.divide h (UF.add one h) in
  let x = UF.divide h (UF.add one h) in
  let d = UF.add b (UF.subtract a (s 0.5)) in
  let x2 = UF.multiply x x in
  let s3 = UF.add one (UF.add x x2) in
  let s5 = UF.add one (UF.add x (UF.multiply x2 s3)) in
  let s7 = UF.add one (UF.add x (UF.multiply x2 s5)) in
  let s9 = UF.add one (UF.add x (UF.multiply x2 s7)) in
  let s11 = UF.add one (UF.add x (UF.multiply x2 s9)) in
  let t = UF.square (UF.divide one b) in
  let w = UF.multiply (UF.multiply (s c5) s11) t in
  let w = UF.add w (UF.multiply (s c4) s9) in
  let w = UF.multiply w t in
  let w = UF.add w (UF.multiply (s c3) s7) in
  let w = UF.multiply w t in
  let w = UF.add w (UF.multiply (s c2) s5) in
  let w = UF.multiply w t in
  let w = UF.add w (UF.multiply (s c1) s3) in
  let w = UF.multiply w t in
  let w = UF.add w (s c0) in
  let w = UF.multiply w (UF.divide c b) in
  let u = UF.multiply d (UF.log1p (UF.divide a b)) in
  let v = UF.multiply a (UF.subtract (UF.log b) one) in
  where_ (UF.less_equal u v)
    (UF.subtract (UF.subtract w v) u)
    (UF.subtract (UF.subtract w u) v)

let betaln a b =
  let a, b = promote2 a b in
  let lo = UF.minimum a b and hi = UF.maximum a b in
  let a = lo and b = hi in
  let lg v = bind1 T.Lgamma [ v ] in
  let small_b = UF.add (lg a) (UF.subtract (lg b) (lg (UF.add a b))) in
  let large_b = UF.add (lg a) (algdiv a b) in
  where_ (UF.less b (scalar_of b 8.0)) small_b large_b

let factorial ?(exact = false) n =
  if exact then failwith "factorial with exact=True";
  let n = promote1 n in
  where_
    (UF.less n (scalar_of n 0.0))
    (scalar_of n 0.0)
    (UF.exp (bind1 T.Lgamma [ UF.add n (scalar_of n 1.0) ]))

let rec comb ?(repetition = false) n_ k_ =
  let n, k = promote2 n_ k_ in
  if repetition then
    let cond =
      UF.logical_and
        (UF.equal k (scalar_of k 0.0))
        (UF.greater_equal n (scalar_of n 0.0))
    in
    let result =
      comb ~repetition:false (UF.subtract (UF.add n k) (scalar_of n 1.0)) k
    in
    where_ cond (scalar_of n 1.0) result
  else
    let cond =
      UF.logical_and (UF.less_equal k n)
        (UF.logical_and
           (UF.greater_equal n (scalar_of n 0.0))
           (UF.greater_equal k (scalar_of k 0.0)))
    in
    let safe_n = where_ cond n (scalar_of n 0.0) in
    let safe_k = where_ cond k (scalar_of k 0.0) in
    let one_n = scalar_of safe_n 1.0 in
    let result =
      UF.exp
        (UF.subtract
           (UF.subtract
              (gammaln (UF.add safe_n one_n))
              (gammaln (UF.add safe_k one_n)))
           (gammaln (UF.subtract (UF.add safe_n one_n) safe_k)))
    in
    where_ cond result (scalar_of n 0.0)

let beta a b =
  let a, b = real_only2 "beta" a b in
  let sign =
    UF.multiply (UF.multiply (gammasgn a) (gammasgn b)) (gammasgn (UF.add a b))
  in
  UF.multiply sign (UF.exp (betaln a b))

let betainc a b x =
  let a, b, x = real_only3 "betainc" a b x in
  bind1 T.Regularized_incomplete_beta [ a; b; x ]

let digamma x = bind1 T.Digamma [ real_only "digamma" x ]

let gammainc a x =
  let a, x = promote2 a x in
  bind1 T.Igamma [ a; x ]

let gammaincc a x =
  let a, x = promote2 a x in
  bind1 T.Igammac [ a; x ]

let erf x = bind1 T.Erf [ promote1 x ]
let erfc x = bind1 T.Erfc [ promote1 x ]
let erfinv x = bind1 T.Erf_inv [ real_only "erfinv" x ]
let expit x = bind1 T.Logistic [ promote1 x ]

let logit x =
  let x = promote1 x in
  UF.log (UF.divide x (UF.subtract (scalar_of x 1.0) x))

let xlogy x y =
  let x, y = promote2 x y in
  let x_ok = UF.not_equal x (scalar_of x 0.0) in
  where_
    (UF.logical_or x_ok (UF.isnan y))
    (UF.multiply x (UF.log y))
    (AC.zeros_like x)

let xlog1py x y =
  let x, y = promote2 x y in
  let x_ok = UF.not_equal x (scalar_of x 0.0) in
  where_
    (UF.logical_or x_ok (UF.isnan y))
    (UF.multiply x (UF.log1p y))
    (AC.zeros_like x)

let xlogx x = xlogy x x

let entr x =
  let x = real_only "entr" x in
  where_
    (UF.less x (scalar_of x 0.0))
    (scalar_of x Float.neg_infinity)
    (UF.negative (xlogx x))

let boxcox x lmbda =
  let x, lmbda = promote2 x lmbda in
  let is_zero = UF.equal lmbda (scalar_of lmbda 0.0) in
  let safe = where_ is_zero (AC.ones_like lmbda) lmbda in
  let log_x = UF.log x in
  let power = UF.divide (UF.expm1 (UF.multiply safe log_x)) safe in
  where_ is_zero log_x power

let boxcox1p x lmbda =
  let x, lmbda = promote2 x lmbda in
  let is_zero = UF.equal lmbda (scalar_of lmbda 0.0) in
  let safe = where_ is_zero (AC.ones_like lmbda) lmbda in
  let log1p_x = UF.log1p x in
  let power = UF.divide (UF.expm1 (UF.multiply safe log1p_x)) safe in
  where_ is_zero log1p_x power

let multigammaln a d =
  let a = promote1 a in
  let dt = dtype a in
  let df = float_of_int d in
  let constant =
    UF.multiply
      (UF.multiply
         (UF.multiply (scalar_of a 0.25) (scalar_of a df))
         (UF.subtract (scalar_of a df) (scalar_of a 1.0)))
      (UF.log (scalar_of a Float.pi))
  in
  let b = UF.divide (NL.arange ~dtype:dt df) (scalar_of a 2.0) in
  let ndim = Array.length (shape a) in
  let a_exp = NL.expand_dims a [| ndim |] in
  let b_exp = NL.expand_dims b (Array.init ndim (fun i -> i)) in
  let res = RED.sum ~axis:[| -1 |] (gammaln (UF.subtract a_exp b_exp)) in
  UF.add res constant

let rel_entr p q =
  let p, q = real_only2 "rel_entr" p q in
  let zero = scalar_of p 0.0 in
  let both = UF.logical_and (UF.greater p zero) (UF.greater q zero) in
  let one_zero = UF.logical_and (UF.equal p zero) (UF.greater_equal q zero) in
  let safe_p = where_ both p (scalar_of p 1.0) in
  let safe_q = where_ both q (scalar_of q 1.0) in
  let log_val = UF.subtract (xlogx safe_p) (xlogy safe_p safe_q) in
  where_ both log_val (where_ one_zero zero (scalar_of p Float.infinity))

let kl_div p q =
  let p, q = real_only2 "kl_div" p q in
  UF.add (UF.subtract (rel_entr p q) p) q

let i0e x = bind1 T.Bessel_i0e [ promote1 x ]

let i0 x =
  let x = promote1 x in
  UF.multiply (UF.exp (UF.abs x)) (bind1 T.Bessel_i0e [ x ])

let i1e x = bind1 T.Bessel_i1e [ promote1 x ]

let i1 x =
  let x = promote1 x in
  UF.multiply (UF.exp (UF.abs x)) (bind1 T.Bessel_i1e [ x ])

let float32_max = 3.4028234663852886e38

let erfcx_asymptotic x nterms =
  let coeffs =
    [|
      7918.06640625;
      -1055.7421875;
      162.421875;
      -29.53125;
      6.5625;
      -1.875;
      0.75;
      -0.5;
      1.0;
    |]
  in
  let sel = Array.sub coeffs (Array.length coeffs - nterms) nterms in
  let p_coeffs = coeff_of x sel in
  let t = UF.divide (scalar_of x 1.0) (UF.square x) in
  let p = Poly.polyval p_coeffs t in
  UF.divide p (UF.multiply x (scalar_of x (sqrt Float.pi)))

let erfcx_impl x nterms =
  let fmax = match dtype x with D.F64 -> Float.max_float | _ -> float32_max in
  let threshold = sqrt (log fmax) in
  let large = UF.greater x (scalar_of x threshold) in
  let safe_x = where_ large (AC.ones_like x) x in
  let direct =
    UF.multiply (UF.exp (UF.square safe_x)) (bind1 T.Erfc [ safe_x ])
  in
  let asymp = erfcx_asymptotic x nterms in
  where_ large asymp direct

let dawsn_an =
  [|
    1.13681498971755972054e-11;
    8.49262267667473811108e-10;
    1.94434204175553054283e-8;
    9.53151741254484363489e-7;
    3.07828309874913200438e-6;
    3.52513368520288738649e-4;
    -8.50149846724410912031e-4;
    4.22618223005546594270e-2;
    -9.17480371773452345351e-2;
    9.99999999999999994612e-1;
  |]

let dawsn_ad =
  [|
    2.40372073066762605484e-11;
    1.48864681368493396752e-9;
    5.21265281010541664570e-8;
    1.27258478273186970203e-6;
    2.32490249820789513991e-5;
    3.25524741826057911661e-4;
    3.48805814657162590916e-3;
    2.79448531198828973716e-2;
    1.58874241960120565368e-1;
    5.74918629489320327824e-1;
    1.00000000000000000539e0;
  |]

let dawsn_bn =
  [|
    5.08955156417900903354e-1;
    -2.44754418142697847934e-1;
    9.41512335303534411857e-2;
    -2.18711255142039025206e-2;
    3.66207612329569181322e-3;
    -4.23209114460388756528e-4;
    3.59641304793896631888e-5;
    -2.14640351719968974225e-6;
    9.10010780076391431042e-8;
    -2.40274520828250956942e-9;
    3.59233385440928410398e-11;
  |]

let dawsn_bd =
  [|
    1.00000000000000000000e0;
    -6.31839869873368190192e-1;
    2.36706788228248691528e-1;
    -5.31806367003223277662e-2;
    8.48041718586295374409e-3;
    -9.47996768486665330168e-4;
    7.81025592944552338085e-5;
    -4.55875153252442634831e-6;
    1.89100358111421846170e-7;
    -4.91324691331920606875e-9;
    7.18466403235734541950e-11;
  |]

let dawsn_cn =
  [|
    -5.90592860534773254987e-1;
    6.29235242724368800674e-1;
    -1.72858975380388136411e-1;
    1.64837047825189632310e-2;
    -4.86827613020462700845e-4;
  |]

let dawsn_cd =
  [|
    1.00000000000000000000e0;
    -2.69820057197544900361e0;
    1.73270799045947845857e0;
    -3.93708582281939493482e-1;
    3.44278924041233391079e-2;
    -9.73655226040941223894e-4;
  |]

let dawsn x =
  let x = real_only "dawsn" x in
  let an = coeff_of x dawsn_an
  and ad = coeff_of x dawsn_ad
  and bn = coeff_of x dawsn_bn
  and bd = coeff_of x dawsn_bd
  and cn = coeff_of x dawsn_cn
  and cd = coeff_of x dawsn_cd in
  let c v = scalar_of x v in
  let one = c 1.0 in
  let sign = UF.sign x in
  let ax = UF.abs x in
  let ax2 = UF.square ax in
  let t = UF.divide one ax2 in
  let safe_ax_r1 = where_ (UF.less ax (c 3.25)) ax (AC.ones_like ax) in
  let safe_ax2_r1 = UF.square safe_ax_r1 in
  let val_r1 =
    UF.divide
      (UF.multiply safe_ax_r1 (Poly.polyval an safe_ax2_r1))
      (Poly.polyval ad safe_ax2_r1)
  in
  let m2 =
    UF.logical_and (UF.greater_equal ax (c 3.25)) (UF.less ax (c 6.25))
  in
  let safe_t_r2 = where_ m2 t (AC.ones_like t) in
  let safe_ax_r2 = where_ m2 ax (AC.ones_like ax) in
  let val_r2 =
    UF.multiply
      (UF.divide (c 0.5) safe_ax_r2)
      (UF.add one
         (UF.divide
            (UF.multiply safe_t_r2 (Poly.polyval bn safe_t_r2))
            (Poly.polyval bd safe_t_r2)))
  in
  let m3 = UF.greater_equal ax (c 6.25) in
  let safe_t_r3 = where_ m3 t (AC.ones_like t) in
  let safe_ax_r3 = where_ m3 ax (AC.ones_like ax) in
  let val_r3 =
    UF.multiply
      (UF.divide (c 0.5) safe_ax_r3)
      (UF.add one
         (UF.divide
            (UF.multiply safe_t_r3 (Poly.polyval cn safe_t_r3))
            (Poly.polyval cd safe_t_r3)))
  in
  let result =
    where_
      (UF.less ax (c 3.25))
      val_r1
      (where_ (UF.less ax (c 6.25)) val_r2 val_r3)
  in
  UF.multiply sign result

let ndtr_core x =
  let half_sqrt_2 = 0.5 *. sqrt 2.0 in
  let w = UF.multiply x (scalar_of x half_sqrt_2) in
  let z = UF.abs w in
  let y =
    where_
      (UF.less z (scalar_of x half_sqrt_2))
      (UF.add (scalar_of x 1.0) (bind1 T.Erf [ w ]))
      (where_
         (UF.greater w (scalar_of x 0.0))
         (UF.subtract (scalar_of x 2.0) (bind1 T.Erfc [ z ]))
         (bind1 T.Erfc [ z ]))
  in
  UF.multiply (scalar_of x 0.5) y

let ndtr x =
  single_or_double x;
  ndtr_core x

let ndtri_p0 =
  [|
    -5.99633501014107895267e1;
    9.80010754185999661536e1;
    -5.66762857469070293439e1;
    1.39312609387279679503e1;
    -1.23916583867381258016e0;
  |]

let ndtri_q0 =
  [|
    1.0;
    1.95448858338141759834e0;
    4.67627912898881538453e0;
    8.63602421390890590575e1;
    -2.25462687854119370527e2;
    2.00260212380060660359e2;
    -8.20372256168333339912e1;
    1.59056225126211695515e1;
    -1.18331621121330003142e0;
  |]

let ndtri_p1 =
  [|
    4.05544892305962419923e0;
    3.15251094599893866154e1;
    5.71628192246421288162e1;
    4.40805073893200834700e1;
    1.46849561928858024014e1;
    2.18663306850790267539e0;
    -1.40256079171354495875e-1;
    -3.50424626827848203418e-2;
    -8.57456785154685413611e-4;
  |]

let ndtri_q1 =
  [|
    1.0;
    1.57799883256466749731e1;
    4.53907635128879210584e1;
    4.13172038254672030440e1;
    1.50425385692907503408e1;
    2.50464946208309415979e0;
    -1.42182922854787788574e-1;
    -3.80806407691578277194e-2;
    -9.33259480895457427372e-4;
  |]

let ndtri_p2 =
  [|
    3.23774891776946035970e0;
    6.91522889068984211695e0;
    3.93881025292474443415e0;
    1.33303460815807542389e0;
    2.01485389549179081538e-1;
    1.23716634817820021358e-2;
    3.01581553508235416007e-4;
    2.65806974686737550832e-6;
    6.23974539184983293730e-9;
  |]

let ndtri_q2 =
  [|
    1.0;
    6.02427039364742014255e0;
    3.67983563856160859403e0;
    1.37702099489081330271e0;
    2.16236993594496635890e-1;
    1.34204006088543189037e-2;
    3.28014464682127739104e-4;
    2.89247864745380683936e-6;
    6.79019408009981274425e-9;
  |]

let ndtri p =
  single_or_double p;
  let p0 = coeff_of p ndtri_p0
  and q0 = coeff_of p ndtri_q0
  and p1 = coeff_of p ndtri_p1
  and q1 = coeff_of p ndtri_q1
  and p2 = coeff_of p ndtri_p2
  and q2 = coeff_of p ndtri_q2 in
  let c v = scalar_of p v in
  let maybe =
    where_ (UF.greater p (c (-.Float.expm1 (-2.0)))) (UF.subtract (c 1.0) p) p
  in
  let sanitized = where_ (UF.equal maybe (c 0.0)) (c 0.5) maybe in
  let w = UF.subtract sanitized (c 0.5) in
  let ww = UF.square w in
  let x_big =
    UF.add w
      (UF.multiply (UF.multiply w ww)
         (UF.divide (Poly.polyval p0 ww) (Poly.polyval q0 ww)))
  in
  let x_big = UF.multiply x_big (c (-.sqrt (2.0 *. Float.pi))) in
  let z = UF.sqrt (UF.multiply (c (-2.0)) (UF.log sanitized)) in
  let first = UF.subtract z (UF.divide (UF.log z) z) in
  let inv_z = UF.divide (c 1.0) z in
  let second_small =
    UF.divide (UF.divide (Poly.polyval p2 inv_z) (Poly.polyval q2 inv_z)) z
  in
  let second_other =
    UF.divide (UF.divide (Poly.polyval p1 inv_z) (Poly.polyval q1 inv_z)) z
  in
  let x_small = UF.subtract first second_small in
  let x_other = UF.subtract first second_other in
  let x =
    where_
      (UF.greater sanitized (c (exp (-2.0))))
      x_big
      (where_ (UF.greater_equal z (c 8.0)) x_small x_other)
  in
  let x = where_ (UF.greater p (c (1.0 -. exp (-2.0)))) x (UF.negative x) in
  let inf = c Float.infinity in
  where_
    (UF.equal p (c 0.0))
    (UF.negative inf)
    (where_ (UF.equal p (c 1.0)) inf x)

let double_factorial m =
  let rec loop k acc =
    if k <= 1 then acc else loop (k - 2) (acc *. float_of_int k)
  in
  loop m 1.0

let log_ndtr_asymptotic_series x series_order =
  if series_order <= 0 then scalar_of x 1.0
  else begin
    let x2 = UF.square x in
    let even = ref (AC.zeros_like x) in
    let odd = ref (AC.zeros_like x) in
    let x2n = ref x2 in
    for n = 1 to series_order do
      let y = UF.divide (scalar_of x (double_factorial ((2 * n) - 1))) !x2n in
      if n mod 2 = 1 then odd := UF.add !odd y else even := UF.add !even y;
      x2n := UF.multiply !x2n x2
    done;
    UF.subtract (UF.add (scalar_of x 1.0) !even) !odd
  end

let log_ndtr_lower x series_order =
  let x2 = UF.square x in
  let log_scale =
    UF.subtract
      (UF.subtract
         (UF.multiply (scalar_of x (-0.5)) x2)
         (UF.log (UF.negative x)))
      (scalar_of x (0.5 *. log (2.0 *. Float.pi)))
  in
  UF.add log_scale (UF.log (log_ndtr_asymptotic_series x series_order))

let log_ndtr ?(series_order = 3) x =
  if series_order < 0 then invalid_arg "series_order must be non-negative.";
  if series_order > 30 then invalid_arg "series_order must be <= 30.";
  single_or_double x;
  let lower, upper =
    match dtype x with D.F64 -> (-20.0, 8.0) | _ -> (-10.0, 5.0)
  in
  let gt_upper = UF.greater x (scalar_of x upper) in
  let ndtr_arg =
    where_ gt_upper (UF.negative x) (UF.maximum x (scalar_of x lower))
  in
  let ndtr_res = ndtr_core ndtr_arg in
  where_ gt_upper (UF.negative ndtr_res)
    (where_
       (UF.greater x (scalar_of x lower))
       (UF.log ndtr_res)
       (log_ndtr_lower (UF.minimum x (scalar_of x lower)) series_order))

let polygamma n x =
  if not (is_integer_dtype (dtype n)) then
    invalid_arg "Argument `n` to polygamma must be of integer type.";
  let n, x = real_only2 "polygamma" n x in
  bind1 T.Polygamma [ n; x ]

let bernoulli_coefs =
  [|
    12.0;
    -720.0;
    30240.0;
    -1209600.0;
    47900160.0;
    -1892437580.3183792;
    74724249600.0;
    -2950130727918.164;
    116467828143500.67;
    -4597978722407473.0;
    1.8152105401943546e+17;
    -7.166165256175667e+18;
    2.82908877253043e+20;
    -1.1168794925000445e+22;
    4.4092635141854666e+23;
    -1.7407074646225822e+25;
  |]

let zeta_series s a =
  let lit v = scalar_of a v in
  let count = if dtype a = D.F32 then 8 else 16 in
  let big_n = lit (float_of_int count) in
  let term k = UF.power (UF.add a (lit (float_of_int k))) (UF.negative s) in
  let series =
    List.fold_left
      (fun acc k -> UF.add acc (term k))
      (term 0)
      (List.init (count - 1) (fun k -> k + 1))
  in
  let shifted = UF.add a big_n in
  let integral =
    UF.divide
      (UF.power shifted (UF.subtract (lit 1.0) s))
      (UF.subtract s (lit 1.0))
  in
  let t0 = UF.power shifted (UF.negative s) in
  let ceiling =
    lit (if dtype a = D.F32 then 3.4028234663852886e38 else Float.max_float)
  in
  let total = ref (lit 0.0) and running = ref (lit 1.0) in
  for m = 0 to (2 * count) - 1 do
    running :=
      UF.multiply !running (UF.divide (UF.add s (lit (float_of_int m))) shifted);
    if m mod 2 = 0 then
      total :=
        UF.add !total
          (UF.divide
             (UF.minimum !running ceiling)
             (lit bernoulli_coefs.(m / 2)))
  done;
  UF.add (UF.add series integral) (UF.multiply t0 (UF.add (lit 0.5) !total))

let zeta_custom =
  lazy
    (Custom_derivatives.custom_jvp
       (fun args ->
         match args with
         | [ x; q ] -> [ bind1 T.Zeta [ x; q ] ]
         | _ -> failwith "zeta: arity")
       ~jvp:(fun primals tangents ->
         Interpreters.Ad.jvp
           (fun args ->
             match args with
             | [ x; q ] -> [ zeta_series x q ]
             | _ -> failwith "zeta: arity")
           primals tangents))

let zeta ?q x =
  match q with
  | None ->
      failwith
        "Riemann zeta function not implemented; pass q != None to compute the \
         Hurwitz Zeta function."
  | Some q ->
      let x, q = promote2 x q in
      List.hd ((Lazy.force zeta_custom) [ x; q ])

let wofz_terms = 64

let wofz_constants =
  lazy
    (let n = wofz_terms in
     let l = Float.sqrt (float_of_int n /. Float.sqrt 2.0) in
     let m = 2 * n in
     let m2 = 2 * m in
     let pi = 4.0 *. Float.atan 1.0 in
     let f k =
       let theta = float_of_int k *. pi /. float_of_int m in
       let t = l *. Float.tan (theta /. 2.0) in
       Float.exp (-.(t *. t)) *. ((l *. l) +. (t *. t))
     in
     let coefficient j =
       let total = ref 0.0 in
       for k = -m + 1 to m - 1 do
         let angle = 2.0 *. pi *. float_of_int (k * j) /. float_of_int m2 in
         total := !total +. (f k *. Float.cos angle)
       done;
       !total /. float_of_int m2
     in
     (l, Array.init n (fun i -> coefficient (i + 1))))

let wofz_upper (z : Complex.t) =
  let open Complex in
  let l, coefficients = Lazy.force wofz_constants in
  let pi = 4.0 *. Float.atan 1.0 in
  let iz = mul i z in
  let cl = { re = l; im = 0.0 } in
  let numerator = add cl iz and denominator = sub cl iz in
  let w = div numerator denominator in
  let poly = ref zero in
  for j = wofz_terms - 1 downto 0 do
    poly := add (mul !poly w) { re = coefficients.(j); im = 0.0 }
  done;
  add
    (div (mul { re = 2.0; im = 0.0 } !poly) (mul denominator denominator))
    (div { re = 1.0 /. Float.sqrt pi; im = 0.0 } denominator)

let wofz_scalar (z : Complex.t) =
  let open Complex in
  if z.im >= 0.0 then wofz_upper z
  else
    let reflected = wofz_upper (neg z) in
    sub (mul { re = 2.0; im = 0.0 } (exp (neg (mul z z)))) reflected

let wofz z =
  let a = Core.get_aval z in
  let out_dt =
    match a.Types.dtype with
    | D.F64 | D.Complex128 | D.I64 | D.Uint64 -> D.Complex128
    | _ -> D.Complex64
  in
  let host = function
    | [ v ] ->
        let nd =
          match v with
          | Types.Concrete n -> n
          | Types.Device b -> Pjrt.Buffer.to_host b
          | Types.Tracer _ -> invalid_arg "wofz: expected a concrete value"
        in
        let sh = Nd.shape nd in
        let n = Array.fold_left ( * ) 1 sh in
        let read i =
          let idx = Array.make (Array.length sh) 0 in
          let rem = ref i in
          for d = Array.length sh - 1 downto 0 do
            idx.(d) <- !rem mod sh.(d);
            rem := !rem / sh.(d)
          done;
          match Nd.dtype nd with
          | D.Complex64 | D.Complex128 -> Nd.get_c nd idx
          | _ -> { Complex.re = Nd.get_f nd idx; im = 0.0 }
        in
        [
          Types.Concrete
            (Nd.of_complex out_dt sh
               (Array.init n (fun i -> wofz_scalar (read i))));
        ]
    | _ -> invalid_arg "wofz: arity"
  in
  match z with
  | Types.Tracer _ ->
      List.hd
        (Custom_derivatives.pure_callback host
           ~result_avals:[ { a with Types.dtype = out_dt; weak_type = false } ]
           [ z ])
  | _ -> List.hd (host [ z ])

let erfcx x =
  let x = promote1 x in
  if is_complex_dtype (dtype x) then
    let re = UF.real x and im = UF.imag x in
    wofz (complex_of_parts x (UF.negative im) re)
  else
    let nterms = match dtype x with D.F64 -> 9 | _ -> 5 in
    erfcx_impl x nterms

let euler_gamma = 0.5772156649015329

let eps_of v =
  match dtype v with
  | D.F64 | D.Complex128 -> epsilon_float
  | D.F16 -> 0.0009765625
  | D.Bf16 -> 0.0078125
  | _ -> 1.1920928955078125e-07

let full_like v x = const_full (dtype v) (shape v) x

let masked_iter ~maxiter ~cond ~body state0 =
  let counter v = T.Concrete (Nd.of_floats D.I32 [||] [| float_of_int v |]) in
  let looped =
    Lax_api.while_loop
      (fun st ->
        match st with
        | i :: state ->
            UF.logical_and (UF.less i (counter maxiter)) (RED.any (cond state))
        | [] -> failwith "masked_iter: empty state")
      (fun st ->
        match st with
        | i :: state ->
            let m = cond state in
            let nxt = body state in
            UF.add i (counter 1)
            :: List.map2 (fun n o -> where_ m n o) nxt state
        | [] -> failwith "masked_iter: empty state")
      (counter 0 :: state0)
  in
  List.tl looped

let reject_bool x =
  if dtype x = D.Bool then
    raise (Errors.jax_type_error "sub does not accept dtype bool.")

let softmax ?axis x =
  reject_bool x;
  match axis with
  | Some ax -> NN.softmax ~axis:ax x
  | None ->
      let sh = shape x in
      NL.reshape (NN.softmax ~axis:0 (NL.reshape x [| numel sh |])) sh

let log_softmax ?axis x =
  reject_bool x;
  match axis with
  | Some ax -> NN.log_softmax ~axis:ax x
  | None ->
      let sh = shape x in
      NL.reshape (NN.log_softmax ~axis:0 (NL.reshape x [| numel sh |])) sh

let poch z m =
  let z, m = real_only2 "poch" z m in
  where_
    (UF.equal m (scalar_of m 0.0))
    (scalar_of z 1.0)
    (UF.divide (gamma (UF.add z m)) (gamma z))

let spence_a =
  [|
    4.65128586073990045278e-5;
    7.31589045238094711071e-3;
    1.33847639578309018650e-1;
    8.79691311754530315341e-1;
    2.71149851196553469920e0;
    4.25697156008121755724e0;
    3.29771340985225106936e0;
    1.00000000000000000126e0;
  |]

let spence_b =
  [|
    6.90990488912553276999e-4;
    2.54043763932544379113e-2;
    2.82974860602568089943e-1;
    1.41172597751831069617e0;
    3.63800533345137075418e0;
    5.03278880143316990390e0;
    3.54771340985225096217e0;
    9.99999999999999998740e-1;
  |]

let spence_poly w =
  let a = coeff_of w spence_a and b = coeff_of w spence_b in
  UF.divide (UF.multiply (UF.negative w) (Poly.polyval a w)) (Poly.polyval b w)

let spence_calc x =
  let s v = scalar_of x v in
  let one = s 1.0 in
  let x2_bool = UF.greater x (s 2.0) in
  let x = where_ x2_bool (UF.divide one x) x in
  let x1_5_bool = UF.greater x (s 1.5) in
  let x_5_bool = UF.less x (s 0.5) in
  let x2_bool = UF.logical_or x2_bool x1_5_bool in
  let w =
    where_ x1_5_bool
      (UF.subtract (UF.divide one x) one)
      (where_ x_5_bool (UF.negative x) (UF.subtract x one))
  in
  let y = spence_poly w in
  let pi2_6 = s (Float.pi *. Float.pi /. 6.0) in
  let y_flag_one =
    UF.subtract
      (UF.subtract pi2_6 (UF.multiply (UF.log x) (UF.log (UF.subtract one x))))
      y
  in
  let y = where_ x_5_bool y_flag_one y in
  let y_flag_two =
    UF.subtract (UF.multiply (s (-0.5)) (UF.square (UF.log x))) y
  in
  where_ x2_bool y_flag_two y

let spence x =
  (match dtype x with
  | D.F32 | D.F64 -> ()
  | _ ->
      invalid_arg "x.dtype is not supported, see docstring for supported types.");
  let s v = scalar_of x v in
  let pi2_6 = s (Float.pi *. Float.pi /. 6.0) in
  where_
    (UF.less x (s 0.0))
    (s Float.nan)
    (where_
       (UF.equal x (s 1.0))
       (s 0.0)
       (where_ (UF.equal x (s 0.0)) pi2_6 (spence_calc x)))

let sici_sn =
  [|
    -8.39167827910303881427e-11;
    4.62591714427012837309e-8;
    -9.75759303843632795789e-6;
    9.76945438170435310816e-4;
    -4.13470316229406538752e-2;
    1.00000000000000000302e0;
  |]

let sici_sd =
  [|
    2.03269266195951942049e-12;
    1.27997891179943299903e-9;
    4.41827842801218905784e-7;
    9.96412122043875552487e-5;
    1.42085239326149893930e-2;
    9.99999999999999996984e-1;
  |]

let sici_cn =
  [|
    2.02524002389102268789e-11;
    -1.35249504915790756375e-8;
    3.59325051419993077021e-6;
    -4.74007206873407909465e-4;
    2.89159652607555242092e-2;
    -1.00000000000000000080e0;
  |]

let sici_cd =
  [|
    4.07746040061880559506e-12;
    3.06780997581887812692e-9;
    1.23210355685883423679e-6;
    3.17442024775032769882e-4;
    5.10028056236446052392e-2;
    4.00000000000000000080e0;
  |]

let sici_fn4 =
  [|
    4.23612862892216586994e0;
    5.45937717161812843388e0;
    1.62083287701538329132e0;
    1.67006611831323023771e-1;
    6.81020132472518137426e-3;
    1.08936580650328664411e-4;
    5.48900223421373614008e-7;
  |]

let sici_fd4 =
  [|
    1.0;
    8.16496634205391016773e0;
    7.30828822505564552187e0;
    1.86792257950184183883e0;
    1.78792052963149907262e-1;
    7.01710668322789753610e-3;
    1.10034357153915731354e-4;
    5.48900252756255700982e-7;
  |]

let sici_gn4 =
  [|
    8.71001698973114191777e-2;
    6.11379109952219284151e-1;
    3.97180296392337498885e-1;
    7.48527737628469092119e-2;
    5.38868681462177273157e-3;
    1.61999794598934024525e-4;
    1.97963874140963632189e-6;
    7.82579040744090311069e-9;
  |]

let sici_gd4 =
  [|
    1.0;
    1.64402202413355338886e0;
    6.66296701268987968381e-1;
    9.88771761277688796203e-2;
    6.22396345441768420760e-3;
    1.73221081474177119497e-4;
    2.02659182086343991969e-6;
    7.82579218933534490868e-9;
  |]

let sici_fn8 =
  [|
    4.55880873470465315206e-1;
    7.13715274100146711374e-1;
    1.60300158222319456320e-1;
    1.16064229408124407915e-2;
    3.49556442447859055605e-4;
    4.86215430826454749482e-6;
    3.20092790091004902806e-8;
    9.41779576128512936592e-11;
    9.70507110881952024631e-14;
  |]

let sici_fd8 =
  [|
    1.0;
    9.17463611873684053703e-1;
    1.78685545332074536321e-1;
    1.22253594771971293032e-2;
    3.58696481881851580297e-4;
    4.92435064317881464393e-6;
    3.21956939101046018377e-8;
    9.43720590350276732376e-11;
    9.70507110881952025725e-14;
  |]

let sici_gn8 =
  [|
    6.97359953443276214934e-1;
    3.30410979305632063225e-1;
    3.84878767649974295920e-2;
    1.71718239052347903558e-3;
    3.48941165502279436777e-5;
    3.47131167084116673800e-7;
    1.70404452782044526189e-9;
    3.85945925430276600453e-12;
    3.14040098946363334640e-15;
  |]

let sici_gd8 =
  [|
    1.0;
    1.68548898811011640017e0;
    4.87852258695304967486e-1;
    4.67913194259625806320e-2;
    1.90284426674399523638e-3;
    3.68475504442561108162e-5;
    3.57043223443740838771e-7;
    1.72693748966316146736e-9;
    3.87830166023954706752e-12;
    3.14040098946363335242e-15;
  |]

let sici_series x =
  let s v = scalar_of x v in
  let t = UF.multiply x x in
  let si_s =
    UF.divide
      (UF.multiply x (Poly.polyval (coeff_of x sici_sn) t))
      (Poly.polyval (coeff_of x sici_sd) t)
  in
  let ci_s =
    UF.add
      (UF.add (s euler_gamma) (UF.log x))
      (UF.divide
         (UF.multiply t (Poly.polyval (coeff_of x sici_cn) t))
         (Poly.polyval (coeff_of x sici_cd) t))
  in
  let si = where_ (UF.equal x (s 0.0)) (s 0.0) si_s in
  let ci = where_ (UF.equal x (s 0.0)) (s Float.neg_infinity) ci_s in
  (si, ci)

let sici_asympt x =
  let s v = scalar_of x v in
  let sn = UF.sin x and cn = UF.cos x in
  let z = UF.divide (s 1.0) (UF.multiply x x) in
  let f4 =
    UF.divide
      (Poly.polyval (coeff_of x sici_fn4) z)
      (UF.multiply x (Poly.polyval (coeff_of x sici_fd4) z))
  in
  let g4 =
    UF.divide
      (UF.multiply z (Poly.polyval (coeff_of x sici_gn4) z))
      (Poly.polyval (coeff_of x sici_gd4) z)
  in
  let f8 =
    UF.divide
      (Poly.polyval (coeff_of x sici_fn8) z)
      (UF.multiply x (Poly.polyval (coeff_of x sici_fd8) z))
  in
  let g8 =
    UF.divide
      (UF.multiply z (Poly.polyval (coeff_of x sici_gn8) z))
      (Poly.polyval (coeff_of x sici_gd8) z)
  in
  let mask = UF.less x (s 8.0) in
  let f = where_ mask f4 f8 in
  let g = where_ mask g4 g8 in
  let si =
    UF.subtract
      (UF.subtract (s (Float.pi /. 2.0)) (UF.multiply f cn))
      (UF.multiply g sn)
  in
  let ci = UF.subtract (UF.multiply f sn) (UF.multiply g cn) in
  (si, ci)

let sici_approx x =
  let s v = scalar_of x v in
  let si = UF.subtract (s (Float.pi /. 2.0)) (UF.divide (UF.cos x) x) in
  let ci = UF.divide (UF.sin x) x in
  let si = where_ (UF.isposinf x) (s (Float.pi /. 2.0)) si in
  let ci = where_ (UF.isposinf x) (s 0.0) ci in
  (si, ci)

let sici x =
  let x = real_only "sici" x in
  let s v = scalar_of x v in
  let x_abs = UF.abs x in
  let si_series, ci_series = sici_series x_abs in
  let si_asymp, ci_asymp = sici_asympt x_abs in
  let si_approx, ci_approx = sici_approx x_abs in
  let cond1 = UF.less_equal x_abs (s 4.0) in
  let cond2 =
    UF.logical_and (UF.greater x_abs (s 4.0)) (UF.less_equal x_abs (s 1e9))
  in
  let si = where_ cond1 si_series (where_ cond2 si_asymp si_approx) in
  let ci = where_ cond1 ci_series (where_ cond2 ci_asymp ci_approx) in
  let si = UF.multiply (UF.sign x) si in
  let ci = where_ (UF.isneginf x) (s Float.nan) ci in
  [ si; ci ]

let owens_t_quad_pts =
  [|
    0.0035082039676451715;
    0.031279042338030754;
    0.085266826283219451;
    0.16245071730812277;
    0.25851196049125435;
    0.36807553840697534;
    0.48501092905604697;
    0.60277514152618577;
    0.71477884217753227;
    0.81475510988760099;
    0.89711029755948966;
    0.95723808085944262;
    0.99178832974629704;
  |]

let owens_t_quad_wts =
  [|
    0.018831438115323503;
    0.018567086243977649;
    0.018042093461223386;
    0.017263829606398753;
    0.016243219975989857;
    0.014994592034116705;
    0.013535474469662088;
    0.011886351605820165;
    0.010070377242777432;
    0.0081130545742299587;
    0.0060419009528470239;
    0.0038862217010742058;
    0.0016793031084546090;
  |]

let coeff_f64 a = T.Concrete (Nd.of_floats D.F64 [| Array.length a |] a)

let owens_t_quadrature h a =
  let ndim = Array.length (shape a) in
  let coeff_f64 v =
    T.Concrete
      (Nd.of_floats (Dts.default_float_dtype ()) [| Array.length v |] v)
  in
  let quad_pts =
    NL.expand_dims (coeff_f64 owens_t_quad_pts) (Array.init ndim (fun i -> i))
  in
  let r = UF.multiply (NL.expand_dims (UF.square a) [| ndim |]) quad_pts in
  let one = coeff_f64 [| 1.0 |] in
  let integrand =
    UF.divide
      (UF.exp
         (UF.multiply
            (UF.multiply (coeff_f64 [| -0.5 |])
               (NL.expand_dims (UF.square h) [| ndim |]))
            (UF.add one r)))
      (UF.add one r)
  in
  UF.multiply a (TC.matmul integrand (coeff_f64 owens_t_quad_wts))

let owens_t h a =
  let h, a = promote2 h a in
  let sign_a = UF.sign a in
  let h = UF.abs h in
  let abs_a = UF.abs a in
  let s v = scalar_of h v in
  let nan_mask = UF.logical_or (UF.isnan a) (UF.isnan h) in
  let root_2 = sqrt 2.0 in
  let h_normed = UF.divide h (s root_2) in
  let le1 = UF.less_equal abs_a (s 1.0) in
  let modified_a = where_ le1 abs_a (UF.divide (s 1.0) abs_a) in
  let modified_h = where_ le1 h (UF.multiply abs_a h) in
  let result = owens_t_quadrature modified_h modified_a in
  let result =
    where_
      (UF.equal modified_h (s 0.0))
      (UF.divide (UF.arctan modified_a) (s (2.0 *. Float.pi)))
      result
  in
  let result =
    where_
      (UF.equal modified_a (s 1.0))
      (UF.multiply
         (UF.multiply (s 0.125)
            (bind1 T.Erfc [ UF.divide (UF.negative modified_h) (s root_2) ]))
         (bind1 T.Erfc [ UF.divide modified_h (s root_2) ]))
      result
  in
  let normh = bind1 T.Erfc [ h_normed ] in
  let normah = bind1 T.Erfc [ UF.multiply abs_a h_normed ] in
  let branch_lo =
    UF.subtract
      (UF.subtract (s 0.25)
         (UF.multiply
            (UF.multiply (s 0.25) (bind1 T.Erf [ h_normed ]))
            (bind1 T.Erf [ UF.multiply abs_a h_normed ])))
      result
  in
  let branch_hi =
    UF.subtract
      (UF.multiply (s 0.25)
         (UF.subtract (UF.add normh normah) (UF.multiply normh normah)))
      result
  in
  let result =
    where_
      (UF.greater abs_a (s 1.0))
      (where_
         (UF.less_equal (UF.multiply abs_a h) (s 0.67))
         branch_lo branch_hi)
      result
  in
  let result = UF.multiply sign_a result in
  where_ nan_mask (full_like result Float.nan) result

let rec bernoulli n =
  if n < 0 then invalid_arg "n must be a non-negative integer.";
  let dt = Dts.default_float_dtype () in
  let b3 = [| 1.0; -0.5; 1.0 /. 6.0 |] in
  if n < 3 then
    T.Concrete (Nd.of_floats dt [| n + 1 |] (Array.sub b3 0 (n + 1)))
  else if n mod 2 <> 0 then
    NL.concatenate
      [ bernoulli (n - 1); T.Concrete (Nd.of_floats dt [| 1 |] [| 0.0 |]) ]
  else begin
    let num_even = (n - 2) / 2 in
    let s v = T.Concrete (Nd.of_floats dt [||] [| v |]) in
    let m = NL.arange ~start:4.0 ~step:2.0 ~dtype:dt (float_of_int (n + 1)) in
    let pi2 = Float.pi *. Float.pi in
    let term =
      UF.divide
        (UF.divide
           (UF.multiply (UF.negative (UF.subtract m (s 1.0))) m)
           (s 4.0))
        (s pi2)
    in
    let q1 = UF.multiply (s (1.0 /. pi2)) (RED.cumprod term) in
    let k = NL.arange ~start:2.0 ~dtype:dt 50.0 in
    let kk = NL.expand_dims k [| 1 |] in
    let mm = NL.expand_dims m [| 0 |] in
    let q2 = RED.sum ~axis:[| 0 |] (UF.power kk (UF.negative mm)) in
    let vals = UF.multiply q1 (UF.add (s 1.0) q2) in
    let zeros_e = AC.zeros_like vals in
    let body =
      NL.reshape (NL.stack ~axis:1 [ zeros_e; vals ]) [| 2 * num_even |]
    in
    let head = T.Concrete (Nd.of_floats dt [| 3 |] b3) in
    NL.concatenate [ head; body ]
  end

let bessel_jn ?(n_iter = 50) ~v z =
  let z = promote1 z in
  let dt = dtype z and sh = shape z in
  let s x = scalar_of z x in
  let cf x = const_full dt sh x in
  let f0 = ref (cf 0.0) in
  let f1 = ref (cf 1e-16) in
  let bs = ref (cf 0.0) in
  let out = Array.make (n_iter + 1) !f0 in
  for k = n_iter downto 0 do
    let kf = float_of_int k in
    let f =
      UF.subtract (UF.divide (UF.multiply (s (2.0 *. (kf +. 1.0))) !f1) z) !f0
    in
    if k mod 2 = 0 then bs := UF.add !bs (UF.multiply (s 2.0) f);
    out.(k) <- f;
    f0 := !f1;
    f1 := f
  done;
  let f_last = out.(0) in
  let denom = UF.subtract !bs f_last in
  let js =
    Array.to_list
      (Array.map (fun jv -> UF.divide jv denom) (Array.sub out 0 (v + 1)))
  in
  NL.stack ~axis:0 js

let hyp1f1_serie a b x =
  let s v = scalar_of x v in
  let prec = s (eps_of x) in
  let one = s 1.0 in
  let init = [ one; one; UF.multiply (UF.divide a b) x ] in
  let cond st =
    match st with
    | [ serie; k; term ] ->
        UF.logical_and
          (UF.less k (s 250.0))
          (UF.greater (UF.divide (UF.abs term) (UF.abs serie)) prec)
    | _ -> assert false
  in
  let body st =
    match st with
    | [ serie; k; term ] ->
        let serie' = UF.add serie term in
        let term' =
          UF.divide
            (UF.multiply
               (UF.multiply (UF.divide (UF.add a k) (UF.add b k)) x)
               term)
            (UF.add k one)
        in
        [ serie'; UF.add k one; term' ]
    | _ -> assert false
  in
  match masked_iter ~maxiter:250 ~cond ~body init with
  | [ serie; _; _ ] -> serie
  | _ -> assert false

let hyp1f1_asymptotic a b x =
  let s v = scalar_of x v in
  let prec = s (eps_of x) in
  let one = s 1.0 in
  let init =
    [
      one; one; UF.divide (UF.multiply (UF.subtract b a) (UF.subtract one a)) x;
    ]
  in
  let cond st =
    match st with
    | [ serie; k; term ] ->
        UF.logical_and
          (UF.less k (s 250.0))
          (UF.greater (UF.divide (UF.abs term) (UF.abs serie)) prec)
    | _ -> assert false
  in
  let body st =
    match st with
    | [ serie; k; term ] ->
        let serie' = UF.add serie term in
        let term' =
          UF.divide
            (UF.divide
               (UF.multiply
                  (UF.multiply
                     (UF.add (UF.subtract b a) k)
                     (UF.add (UF.subtract one a) k))
                  term)
               (UF.add k one))
            x
        in
        [ serie'; UF.add k one; term' ]
    | _ -> assert false
  in
  let serie =
    match masked_iter ~maxiter:250 ~cond ~body init with
    | [ serie; _; _ ] -> serie
    | _ -> assert false
  in
  UF.multiply
    (UF.multiply
       (UF.multiply (UF.divide (gamma b) (gamma a)) (UF.exp x))
       (UF.power x (UF.subtract a b)))
    serie

let hyp1f1_value a b x =
  let s v = scalar_of x v in
  let result =
    where_
      (UF.less (UF.abs x) (s 100.0))
      (hyp1f1_serie a b x) (hyp1f1_asymptotic a b x)
  in
  let a0 = UF.equal a (s 0.0) in
  let anz = UF.not_equal a (s 0.0) in
  let ab = UF.logical_and (UF.equal a b) anz in
  let b0 = UF.logical_and (UF.equal b (s 0.0)) anz in
  where_ a0 (s 1.0) (where_ ab (UF.exp x) (where_ b0 (s Float.infinity) result))

let hyp1f1_param_derivative shifted a b x =
  let s v = scalar_of x v in
  let precision = s (eps_of x) in
  let one = s 1.0 in
  let looped =
    Lax_api.while_loop
      (fun state ->
        match state with
        | [ serie; k; term ] ->
            UF.logical_and
              (UF.less k (s 250.0))
              (UF.greater
                 (UF.divide (UF.absolute term) (UF.absolute serie))
                 precision)
        | _ -> failwith "hyp1f1: arity")
      (fun state ->
        match state with
        | [ serie; k; term ] ->
            let serie = UF.add serie (UF.multiply term (shifted k)) in
            let term =
              UF.multiply term
                (UF.divide
                   (UF.multiply (UF.divide (UF.add a k) (UF.add b k)) x)
                   (UF.add k one))
            in
            [ serie; UF.add k one; term ]
        | _ -> failwith "hyp1f1: arity")
      [ s 0.0; one; UF.multiply (UF.divide a b) x ]
  in
  List.hd looped

let vectorize3 f args =
  Numpy.Vectorize.vectorize
    (fun scalars ->
      match scalars with
      | [ a; b; x ] -> f a b x
      | _ -> failwith "hyp1f1: arity")
    args

let rec hyp1f1_custom =
  lazy
    (Custom_derivatives.custom_jvp
       (fun args -> [ vectorize3 hyp1f1_value args ])
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ a; b; x ], [ da; db; dx ] ->
             let out = vectorize3 hyp1f1_value primals in
             let wrt_a =
               vectorize3
                 (fun a b x ->
                   hyp1f1_param_derivative
                     (fun k -> UF.subtract (digamma (UF.add a k)) (digamma a))
                     a b x)
                 primals
             in
             let wrt_b =
               vectorize3
                 (fun a b x ->
                   hyp1f1_param_derivative
                     (fun k -> UF.subtract (digamma b) (digamma (UF.add b k)))
                     a b x)
                 primals
             in
             let one = scalar_of x 1.0 in
             let wrt_x =
               UF.multiply (UF.divide a b)
                 (hyp1f1_apply (UF.add a one) (UF.add b one) x)
             in
             ( [ out ],
               [
                 UF.add
                   (UF.add (UF.multiply wrt_a da) (UF.multiply wrt_b db))
                   (UF.multiply wrt_x dx);
               ] )
         | _ -> failwith "hyp1f1: arity"))

and hyp1f1_apply a b x = List.hd ((Lazy.force hyp1f1_custom) [ a; b; x ])

let hyp1f1 a b x =
  let a, b, x = real_only3 "hyp1f1" a b x in
  hyp1f1_apply a b x

let expn2 x n =
  let s v = scalar_of x v in
  let big = s 1.44115188075855872e17 in
  let machep = s (eps_of x) in
  let one = s 1.0 and two = s 2.0 and zero = s 0.0 in
  let looped =
    Lax_api.while_loop
      (fun state ->
        match state with
        | k :: t :: _ ->
            UF.logical_and
              (UF.greater x (UF.multiply k zero))
              (UF.greater t machep)
        | _ -> failwith "expn: arity")
      (fun state ->
        match state with
        | [ k; _; r; ans; pkm2; qkm2; pkm1; qkm1 ] ->
            let k = UF.add k one in
            let odd = UF.equal (UF.remainder k two) one in
            let yk = where_ odd one x in
            let xk =
              where_ odd
                (UF.add n (UF.divide (UF.subtract k one) two))
                (UF.divide k two)
            in
            let pk = UF.add (UF.multiply pkm1 yk) (UF.multiply pkm2 xk) in
            let qk = UF.add (UF.multiply qkm1 yk) (UF.multiply qkm2 xk) in
            let nz = UF.not_equal qk zero in
            let r' = where_ nz (UF.divide pk qk) r in
            let t =
              where_ nz (UF.absolute (UF.divide (UF.subtract ans r') r')) one
            in
            let ans = where_ nz r' ans in
            let is_big = UF.greater (UF.absolute pk) big in
            let scale v = where_ is_big (UF.divide v big) v in
            [ k; t; r'; ans; scale pkm1; scale qkm1; scale pk; scale qk ]
        | _ -> failwith "expn: arity")
      [
        one;
        s Float.infinity;
        zero;
        UF.divide one (UF.add x n);
        one;
        x;
        one;
        UF.add x n;
      ]
  in
  UF.multiply (List.nth looped 3) (UF.exp (UF.negative x))

let expn1 x n =
  let s v = scalar_of x v in
  let one = s 1.0 and zero = s 0.0 in
  let machep = s (eps_of x) in
  let psi0 = UF.subtract (UF.negative (s 0.5772156649015329)) (UF.log x) in
  let psi_state =
    Lax_api.while_loop
      (fun state ->
        match state with [ i; _ ] -> UF.less i n | _ -> failwith "expn: arity")
      (fun state ->
        match state with
        | [ i; psi ] -> [ UF.add i one; UF.add psi (UF.divide one i) ]
        | _ -> failwith "expn: arity")
      [ one; psi0 ]
  in
  let psi = List.nth psi_state 1 in
  let n1 = where_ (UF.equal n one) (UF.add one one) n in
  let z = UF.negative x in
  let looped =
    Lax_api.while_loop
      (fun state ->
        match state with
        | [ _; _; _; _; t ] ->
            UF.logical_and (UF.greater x zero) (UF.greater t machep)
        | _ -> failwith "expn: arity")
      (fun state ->
        match state with
        | [ xk; yk; pk; ans; _ ] ->
            let xk = UF.add xk one in
            let yk = UF.multiply yk (UF.divide z xk) in
            let pk = UF.add pk one in
            let ans =
              UF.add ans (where_ (UF.not_equal pk zero) (UF.divide yk pk) zero)
            in
            let t =
              where_ (UF.not_equal ans zero)
                (UF.absolute (UF.divide yk ans))
                one
            in
            [ xk; yk; pk; ans; t ]
        | _ -> failwith "expn: arity")
      [
        zero;
        one;
        UF.subtract one n;
        where_ (UF.equal n one) zero (UF.divide one (UF.subtract one n1));
        s Float.infinity;
      ]
  in
  let ans = List.nth looped 3 in
  UF.subtract
    (UF.divide
       (UF.multiply (UF.power z (UF.subtract n one)) psi)
       (UF.exp (gammaln n)))
    ans

let expn3 x n =
  let s v = scalar_of x v in
  let one = s 1.0 in
  let xk = UF.add x n in
  let yk = UF.divide one (UF.multiply xk xk) in
  let t = n in
  let ans =
    UF.multiply yk
      (UF.multiply t
         (UF.add
            (UF.subtract
               (UF.multiply (UF.multiply (s 6.0) x) x)
               (UF.multiply (UF.multiply (s 8.0) t) x))
            (UF.multiply t t)))
  in
  let ans =
    UF.multiply yk
      (UF.add ans (UF.multiply t (UF.subtract t (UF.multiply (s 2.0) x))))
  in
  let ans = UF.multiply yk (UF.add ans t) in
  UF.divide (UF.multiply (UF.add ans one) (UF.exp (UF.negative x))) xk

let expn_scalar n x =
  let s v = scalar_of x v in
  let one = s 1.0 in
  let n1 = where_ (UF.equal n one) (UF.add n n) n in
  let cond0 = UF.logical_or (UF.less n (s 0.0)) (UF.less x (s 0.0)) in
  let cond1 = UF.logical_and (UF.equal x (s 0.0)) (UF.less n (s 2.0)) in
  let cond2 =
    UF.logical_and (UF.equal x (s 0.0)) (UF.greater_equal n (s 2.0))
  in
  let cond3 =
    UF.logical_and (UF.equal n (s 0.0)) (UF.greater_equal x (s 0.0))
  in
  let cond4 = UF.greater_equal n (s 5000.0) in
  let cond5 = UF.greater x one in
  let branch =
    where_ cond4 (expn3 x n) (where_ cond5 (expn2 x n) (expn1 x n))
  in
  where_ cond0 (s Float.nan)
    (where_ cond1 (s Float.infinity)
       (where_ cond2 (UF.divide one n1)
          (where_ cond3 (UF.divide (UF.exp (UF.negative x)) x) branch)))

let expn_array n x =
  Numpy.Vectorize.vectorize
    (fun args ->
      match args with
      | [ n; x ] -> expn_scalar n x
      | _ -> failwith "expn: arity")
    [ n; x ]

let rec expn_custom =
  lazy
    (Custom_derivatives.custom_jvp
       (fun args ->
         match args with
         | [ n; x ] -> [ expn_array n x ]
         | _ -> failwith "expn: arity")
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ n; x ], [ _; dx ] ->
             let lower = expn_apply (UF.subtract n (scalar_of n 1.0)) x in
             ([ expn_array n x ], [ UF.multiply (UF.negative dx) lower ])
         | _ -> failwith "expn: arity"))

and expn_apply n x = List.hd ((Lazy.force expn_custom) [ n; x ])

let expn n x =
  let n, x = promote2 n x in
  (match dtype x with
  | D.Complex64 | D.Complex128 ->
      invalid_arg "expn does not support complex-valued inputs."
  | _ -> ());
  expn_apply n x

let exp1 x =
  let x = real_only "exp1" x in
  expn (scalar_of x 1.0) x

module LA = Lax_api

let sp_eps v =
  match dtype v with
  | D.F64 | D.Complex128 -> epsilon_float
  | D.F16 -> 0.0009765625
  | D.Bf16 -> 0.0078125
  | d when Option.is_some (Float_format.of_dtype d) -> (Dtypes.finfo d).eps
  | _ -> 1.1920928955078125e-07

let sp_c v x = scalar_of v x
let ( +! ) = UF.add
let ( -! ) = UF.subtract
let ( *! ) = UF.multiply
let ( /! ) = UF.divide
let sp_and = UF.logical_and
let sp_or = UF.logical_or
let sp_not = UF.logical_not
let sp_lt = UF.less
let sp_le = UF.less_equal
let sp_gt = UF.greater
let sp_ge = UF.greater_equal
let sp_eq = UF.equal
let sp_ne = UF.not_equal
let sp_abs = UF.absolute
let sp_round v = NL.round v
let sp_is_integral c = sp_eq (UF.remainder c (sp_c c 1.0)) (sp_c c 0.0)

let hyp2f1_terminal a b c x =
  let eps = sp_eps x *. 50.0 in
  let ib = sp_round b in
  let mask =
    sp_and (sp_lt b a)
      (sp_and
         (sp_lt (sp_abs (b -! ib)) (sp_c x eps))
         (sp_not
            (sp_and (sp_is_integral c)
               (sp_and (sp_le c (sp_c c 0.0)) (sp_gt c b)))))
  in
  let a' = where_ mask b a and b' = where_ mask a b in
  let a' = sp_abs a' in
  let upper = a' +! sp_c a' 1.0 in
  let result =
    LA.while_loop
      (fun state ->
        match state with
        | [ i; _; _ ] -> sp_lt i upper
        | _ -> failwith "hyp2f1: arity")
      (fun state ->
        match state with
        | [ i; serie; term ] ->
            let term =
              term
              *! (UF.negative (a' -! i +! sp_c i 1.0)
                 /! (c +! i -! sp_c i 1.0)
                 *! (b' +! i -! sp_c i 1.0)
                 /! i *! x)
            in
            [ i +! sp_c i 1.0; serie +! term; term ]
        | _ -> failwith "hyp2f1: arity")
      [ sp_c a' 1.0; sp_c x 1.0; sp_c x 1.0 ]
  in
  List.nth result 1

let hyp2f1_serie a b c x =
  let rtol = sp_eps x in
  let result =
    LA.while_loop
      (fun state ->
        match state with
        | [ serie; k; term ] ->
            sp_and
              (sp_lt k (sp_c k 250.0))
              (sp_gt (sp_abs term) (sp_c x rtol *! sp_abs serie))
        | _ -> failwith "hyp2f1: arity")
      (fun state ->
        match state with
        | [ serie; k; term ] ->
            let serie = serie +! term in
            let one = sp_c k 1.0 in
            let term =
              term
              *! ((a +! k -! one)
                 *! (b +! k -! one)
                 /! (c +! k -! one)
                 /! k *! x)
            in
            [ serie; k +! one; term ]
        | _ -> failwith "hyp2f1: arity")
      [ sp_c x 0.0; sp_c x 1.0; sp_c x 1.0 ]
  in
  List.hd result

let hyp2f1_digamma_transform a b c x =
  let rtol = sp_eps x in
  let d = c -! a -! b in
  let one = sp_c x 1.0 and zero = sp_c x 0.0 and two = sp_c x 2.0 in
  let s = one -! x in
  let rd = sp_round d in
  let nonneg = sp_ge rd zero in
  let e = where_ nonneg d (UF.negative d) in
  let d1 = where_ nonneg d zero in
  let d2 = where_ nonneg zero d in
  let ard = NL.astype (where_ nonneg rd (UF.negative rd)) D.I32 in
  let ax = UF.log s in
  let y =
    digamma one
    +! digamma (one +! e)
    -! digamma (a +! d1)
    -! digamma (b +! d1)
    -! ax
  in
  let y = y /! gamma (e +! one) in
  let p = (a +! d1) *! (b +! d1) *! s /! gamma (e +! two) in
  let looped =
    LA.while_loop
      (fun state ->
        match state with
        | [ _; q; t; y ] ->
            sp_and
              (sp_lt t (sp_c t 250.0))
              (sp_ge (sp_abs q) (sp_c x rtol *! sp_abs y))
        | _ -> failwith "hyp2f1: arity")
      (fun state ->
        match state with
        | [ p; _; t; y ] ->
            let r =
              digamma (one +! t)
              +! digamma (one +! t +! e)
              -! digamma (a +! t +! d1)
              -! digamma (b +! t +! d1)
              -! ax
            in
            let q = p *! r in
            let y = y +! q in
            let p = p *! (s *! (a +! t +! d1) /! (t +! one)) in
            let p = p *! ((b +! t +! d1) /! (t +! one +! e)) in
            [ p; q; t +! one; y ]
        | _ -> failwith "hyp2f1: arity")
      [ p; y; one; y ]
  in
  let y = List.nth looped 3 in
  let compute_sum y =
    let int_one = NL.astype one D.I32 in
    let finite =
      LA.while_loop
        (fun state ->
          match state with
          | [ i; _; _; _ ] -> sp_lt i ard
          | _ -> failwith "hyp2f1: arity")
        (fun state ->
          match state with
          | [ i; p; t; y1 ] ->
              let r = one -! e +! t in
              let p = p *! (s *! (a +! t +! d2) *! (b +! t +! d2) /! r) in
              let t = t +! one in
              let p = p /! t in
              [ i +! int_one; p; t; y1 +! p ]
          | _ -> failwith "hyp2f1: arity")
        [ int_one; one; zero; one ]
    in
    let y1 = List.nth finite 3 in
    let pc = gamma c in
    let y1 = y1 *! (gamma e *! pc /! (gamma (a +! d1) *! gamma (b +! d1))) in
    let y = y *! (pc /! (gamma (a +! d2) *! gamma (b +! d2))) in
    let odd = sp_ne (UF.bitwise_and ard int_one) (NL.astype zero D.I32) in
    let y = where_ odd (UF.negative y) y in
    let q = UF.power s rd in
    where_ (sp_gt rd zero) ((y *! q) +! y1) (y +! (y1 *! q))
  in
  where_ (sp_eq rd zero) (y *! gamma c /! (gamma a *! gamma b)) (compute_sum y)

let hyp2f1_terminal_or_serie a b c x =
  let eps = sp_c x (sp_eps x *. 50.0) in
  let zero = sp_c x 0.0 in
  let d = c -! a -! b in
  let ia = sp_round a and ib = sp_round b and id = sp_round d in
  let neg_int v r = sp_and (sp_le v zero) (sp_lt (sp_abs (v -! r)) eps) in
  let either = sp_or (neg_int a ia) (neg_int b ib) in
  let idx v = T.Concrete (Nd.of_floats D.I32 [||] [| v |]) in
  let index =
    where_
      (sp_and (sp_gt x (sp_c x 0.9)) (sp_not either))
      (where_ (sp_ge (sp_abs (d -! id)) eps) (idx 0.0) (idx 1.0))
      (where_ either (idx 2.0) (idx 0.0))
  in
  LA.select_n index
    [
      hyp2f1_serie a b c x;
      hyp2f1_digamma_transform a b c x;
      hyp2f1_terminal a b c x;
    ]

let hyp2f1_scalar a b c x =
  let eps = sp_c x (sp_eps x *. 50.0) in
  let zero = sp_c x 0.0 and one = sp_c x 1.0 in
  let d = c -! a -! b in
  let s = one -! x in
  let ca = c -! a and cb = c -! b in
  let id = sp_round d in
  let idx v = T.Concrete (Nd.of_floats D.I32 [||] [| v |]) in
  let index =
    where_
      (sp_or (sp_eq x zero)
         (sp_and (sp_or (sp_eq a zero) (sp_eq b zero)) (sp_ne c zero)))
      (idx 0.0)
      (where_
         (sp_or (sp_eq c zero) (sp_and (sp_lt c zero) (sp_is_integral c)))
         (idx 1.0)
         (where_
            (sp_and
               (sp_le d (sp_c x (-1.0)))
               (sp_not (sp_and (sp_ge (sp_abs (d -! id)) eps) (sp_lt s zero))))
            (idx 2.0)
            (where_
               (sp_and (sp_le d zero) (sp_eq x one))
               (idx 1.0)
               (where_
                  (sp_and (sp_lt x one) (sp_eq b c))
                  (idx 3.0)
                  (where_
                     (sp_and (sp_lt x one) (sp_eq a c))
                     (idx 4.0)
                     (where_ (sp_gt x one) (idx 1.0)
                        (where_ (sp_eq x one) (idx 5.0) (idx 6.0))))))))
  in
  LA.select_n index
    [
      one;
      sp_c x Float.infinity;
      UF.power s d *! hyp2f1_terminal_or_serie ca cb c x;
      UF.power s (UF.negative a);
      UF.power s (UF.negative b);
      gamma c *! gamma d /! (gamma ca *! gamma cb);
      hyp2f1_terminal_or_serie a b c x;
    ]

let hyp2f1_promote args = promote_inexact args

let vectorize4 f args =
  Numpy.Vectorize.vectorize
    (fun scalars ->
      match scalars with
      | [ a; b; c; x ] -> f a b c x
      | _ -> failwith "hyp2f1: arity")
    args

let hyp2f1_param_derivative shifted a b c x =
  let precision = sp_c x (sp_eps x) in
  let result =
    LA.while_loop
      (fun state ->
        match state with
        | [ serie; k; term ] ->
            sp_and
              (sp_lt k (sp_c k 250.0))
              (sp_gt (sp_abs term /! sp_abs serie) precision)
        | _ -> failwith "hyp2f1: arity")
      (fun state ->
        match state with
        | [ serie; k; term ] ->
            let serie = serie +! (term *! shifted k) in
            let one = sp_c k 1.0 in
            let term =
              term *! ((a +! k) *! (b +! k) /! (c +! k) /! (k +! one) *! x)
            in
            [ serie; k +! one; term ]
        | _ -> failwith "hyp2f1: arity")
      [ sp_c x 0.0; sp_c x 1.0; a *! b /! c *! x ]
  in
  List.hd result

let rec hyp2f1_custom =
  lazy
    (Custom_derivatives.custom_jvp
       (fun args -> [ vectorize4 hyp2f1_scalar args ])
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ a; b; c; x ], [ da; db; dc; dx ] ->
             let out = vectorize4 hyp2f1_scalar primals in
             let wrt_a =
               vectorize4
                 (fun a b c x ->
                   hyp2f1_param_derivative
                     (fun k -> digamma (a +! k) -! digamma a)
                     a b c x)
                 primals
             in
             let wrt_b =
               vectorize4
                 (fun a b c x ->
                   hyp2f1_param_derivative
                     (fun k -> digamma (b +! k) -! digamma b)
                     a b c x)
                 primals
             in
             let wrt_c =
               vectorize4
                 (fun a b c x ->
                   hyp2f1_param_derivative
                     (fun k -> digamma c -! digamma (c +! k))
                     a b c x)
                 primals
             in
             let one = sp_c x 1.0 in
             let wrt_x =
               a *! b /! c *! hyp2f1_apply (a +! one) (b +! one) (c +! one) x
             in
             ( [ out ],
               [
                 (wrt_a *! da) +! (wrt_b *! db) +! (wrt_c *! dc) +! (wrt_x *! dx);
               ] )
         | _ -> failwith "hyp2f1: arity"))

and hyp2f1_apply a b c x = List.hd ((Lazy.force hyp2f1_custom) [ a; b; c; x ])

let hyp2f1 a b c x =
  match hyp2f1_promote [ a; b; c; x ] with
  | [ a; b; c; x ] -> hyp2f1_apply a b c x
  | _ -> failwith "hyp2f1: arity"

let expint_tables =
  [|
    ( [|
        -5.350447357812543;
        218.50491688166133;
        -4176.572384826693;
        55411.76756393557;
        -331338.13311781443;
        1592627.1633849454;
      |],
      [|
        1.0;
        -52.505479591128626;
        1259.6161867867906;
        -17565.495819735348;
        149306.2117002726;
        -729494.9239640528;
        1592627.1633849454;
      |] );
    ( [|
        0.019818085032596897;
        -1.2716456259849176;
        -2.0881603356812284;
        2.7555445091879367;
        -0.44095070487016;
        0.046656238059358916;
        -0.0015450426796734852;
        7.059980605299618e-05;
      |],
      [|
        1.0;
        1.4764986709149215;
        0.5629177174822436;
        0.16990178978793072;
        0.02291647179034212;
        0.0044501504397287525;
        0.00017274396122065214;
        3.953167195549672e-05;
      |] );
    ( [|
        -1.3732153758712087;
        -0.7084559133740839;
        1.580806855547941;
        -0.2601500427425623;
        0.029946746941137137;
        -0.001038086040188744;
        4.371064420753005e-05;
        2.141783679522603e-06;
      |],
      [|
        1.0;
        0.8585231423622028;
        0.4483285822873995;
        0.07687932158124476;
        0.024498682410218877;
        0.0008832165941927797;
        0.00045909522995113536;
        -4.729848351866523e-06;
        2.66519553739071e-06;
      |] );
    ( [|
        -2.1069346016919166;
        1.7327338696646881;
        -0.2423619178935842;
        0.02322724180937566;
        0.00023728804404931798;
        -8.343219561192553e-05;
        1.3634087956052505e-05;
        -3.655412321999254e-07;
        1.4649417339759612e-08;
        6.17640786371036e-10;
      |],
      [|
        1.0;
        -0.2298062239901678;
        0.11050770414740378;
        -0.015665429666307924;
        0.002761106850817353;
        -0.00020891480122840484;
        1.7085289388076754e-05;
        -4.4593117963566865e-07;
        1.3946349303538475e-08;
        6.150865933977338e-10;
      |] );
    ( [|
        -0.24581193676740204;
        -0.14833822533220778;
        0.07248291795735552;
        -0.013483156873809405;
        0.001342775069788637;
        -7.942465637159712e-05;
        2.644179518984236e-06;
        -4.239473659313765e-08;
      |],
      [|
        1.0;
        -0.10442259084438711;
        -0.26764531281014026;
        0.09695000254621984;
        -0.01601745692712991;
        0.001496414899205908;
        -8.462452563778485e-05;
        2.7289384034767266e-06;
        -4.239462431819542e-08;
      |] );
    ( [|
        0.12125611181054567;
        -0.5823133179043894;
        0.2348887314557017;
        -0.030400343181132484;
        0.0015100821468651906;
        -2.5231370954995714e-05;
      |],
      [|
        1.0;
        -1.002252150365854;
        0.2928709694872224;
        -0.03337004338674008;
        0.0015605448811273888;
        -2.5231370936032347e-05;
      |] );
    ( [|
        -0.7657847078286127;
        0.6886192415566705;
        -0.2132598113545206;
        0.033461075523841935;
        -0.003076541477344756;
        0.00017471193164549074;
        -6.1037116822741705e-06;
        1.218032765428652e-07;
        -1.0860761027932903e-09;
      |],
      [|
        1.0;
        -1.8888028686623086;
        1.066691687211409;
        -0.27519159823063805;
        0.039308526882338234;
        -0.003414684558602365;
        0.00018668443707035555;
        -6.345146083130515e-06;
        1.239754287483207e-07;
        -1.0860761027931265e-09;
      |] );
  |]

let expint_tail index x =
  let one = sp_c x 1.0 in
  let w = one /! x in
  let a, b = expint_tables.(index) in
  let f = Poly.polyval (coeff_of x a) w /! Poly.polyval (coeff_of x b) w in
  UF.exp x *! w *! ((w *! f) +! one)

let expint_head x =
  let a, b = expint_tables.(0) in
  let f = Poly.polyval (coeff_of x a) x /! Poly.polyval (coeff_of x b) x in
  (x *! f) +! sp_c x 0.5772156649015329 +! UF.log x

let expi_primal x =
  let in_range low high =
    sp_and (sp_lt (sp_c x low) x) (sp_le x (sp_c x high))
  in
  let positive =
    where_ (in_range 0.0 2.0) (expint_head x)
      (where_ (in_range 2.0 4.0) (expint_tail 1 x)
         (where_ (in_range 4.0 8.0) (expint_tail 2 x)
            (where_ (in_range 8.0 16.0) (expint_tail 3 x)
               (where_ (in_range 16.0 32.0) (expint_tail 4 x)
                  (where_ (in_range 32.0 64.0) (expint_tail 5 x)
                     (expint_tail 6 x))))))
  in
  where_ (sp_lt x (sp_c x 0.0)) (UF.negative (exp1 (UF.negative x))) positive

let expi_custom =
  lazy
    (Custom_derivatives.custom_jvp
       (fun args -> [ expi_primal (List.hd args) ])
       ~jvp:(fun primals tangents ->
         let x = List.hd primals and dx = List.hd tangents in
         ([ expi_primal x ], [ UF.exp x /! x *! dx ])))

let expi x =
  let x = real_only "expi" x in
  (match dtype x with
  | D.Complex64 | D.Complex128 ->
      invalid_arg "expi does not support complex-valued inputs."
  | _ -> ());
  List.hd ((Lazy.force expi_custom) [ x ])

let legendre_table ~normalized l_max x =
  let n = l_max + 1 in
  let zeros = AC.zeros_like x in
  let table = Array.make_matrix n n zeros in
  let lit v = sp_c x v in
  let q v =
    if dtype x = D.F32 then Int32.float_of_bits (Int32.bits_of_float v) else v
  in
  let wide = Config.x64_enabled () in
  let initial =
    if not normalized then 1.0
    else if wide then 0.5 /. Float.sqrt Float.pi
    else q (0.5 /. q (Float.sqrt Float.pi))
  in
  let initial_value =
    const_full (if wide then D.F64 else dtype x) [||] initial
  in
  table.(0).(0) <- NL.broadcast_to (lit initial) (shape x);
  let root = UF.sqrt (lit 1.0 -! (x *! x)) in
  let factor = ref 1.0 and power = ref (AC.ones_like x) in
  for l = 1 to l_max do
    let a = float_of_int l in
    let step =
      if normalized then -.q (Float.sqrt (q (1.0 +. q (0.5 /. a))))
      else 1.0 -. (2.0 *. a)
    in
    factor := q (!factor *. step);
    power := !power *! root;
    table.(l).(l) <-
      NL.astype (initial_value *! (lit !factor *! !power)) (dtype x)
  done;
  for m = 0 to l_max - 1 do
    let fm = float_of_int m in
    let coeff =
      if normalized then q (Float.sqrt ((2.0 *. fm) +. 3.0))
      else (2.0 *. fm) +. 1.0
    in
    table.(m).(m + 1) <- lit coeff *! x *! table.(m).(m)
  done;
  for offset = 2 to l_max do
    for m = 0 to l_max - offset do
      let l = m + offset in
      let fl = float_of_int l and fm = float_of_int m in
      let d0, d1 =
        if normalized then begin
          let c0 = fl *. fl and c1 = fm *. fm in
          let c2 = 2.0 *. fl and c3 = (fl -. 1.0) *. (fl -. 1.0) in
          ( q (Float.sqrt (q (((4.0 *. c0) -. 1.0) /. (c0 -. c1)))),
            q
              (Float.sqrt
                 (q
                    (q ((c2 +. 1.0) *. (c3 -. c1))
                    /. q ((c2 -. 3.0) *. (c0 -. c1))))) )
        end
        else
          (((2.0 *. fl) -. 1.0) /. (fl -. fm), (fl +. fm -. 1.0) /. (fl -. fm))
      in
      table.(m).(l) <-
        (lit d0 *! (table.(m).(l - 1) *! x)) -! (lit d1 *! table.(m).(l - 2))
    done
  done;
  table

let stack_table table =
  NL.stack
    (Array.to_list (Array.map (fun row -> NL.stack (Array.to_list row)) table))

let legendre_derivatives l_max x p =
  let n = l_max + 1 in
  let zeros = AC.zeros_like x in
  let lit v = sp_c x v in
  let at_l_minus m l = if l = 0 then zeros else p.(m).(l - 1) in
  let up2 m l = if m + 2 > l_max then zeros else at_l_minus (m + 2) l in
  let down2 =
    Array.init n (fun m ->
        Array.init n (fun l -> if m < 2 then zeros else at_l_minus (m - 2) l))
  in
  for l = 1 to n - 2 do
    let fl = float_of_int l in
    down2.(1).(l + 1) <- lit (-1.0 /. ((fl +. 1.0) *. fl)) *! p.(1).(l)
  done;
  for l = 2 to n - 2 do
    let fl = float_of_int l in
    down2.(0).(l + 1) <-
      lit (1.0 /. ((fl +. 2.0) *. (fl +. 1.0) *. fl *. (fl -. 1.0)))
      *! p.(2).(l)
  done;
  let derivative =
    Array.init n (fun m ->
        Array.init n (fun l ->
            if m > l then (
              let fl = float_of_int l and fm = float_of_int m in
              ignore (fl, fm);
              zeros)
            else begin
              let fl = float_of_int l and fm = float_of_int m in
              let b0 = fl +. fm in
              let minus_one =
                if m = 1 then zeros
                else begin
                  let a0 = -0.5 /. (fm -. 1.0) in
                  let c0 = a0 *. (b0 -. 2.0) *. (b0 -. 1.0) in
                  (lit a0 *! at_l_minus m l) +! (lit c0 *! down2.(m).(l))
                end
              in
              let d0 = -0.5 /. (fm +. 1.0) in
              let e0 = d0 *. b0 *. (b0 +. 1.0) in
              let plus_one =
                (lit d0 *! up2 m l) +! (lit e0 *! at_l_minus m l)
              in
              let f0 = b0 *. (fl -. fm +. 1.0) /. 2.0 in
              (lit f0 *! minus_one) -! (lit 0.5 *! plus_one)
            end))
  in
  if n > 1 then begin
    let scale = lit 0.5 /! UF.sqrt (lit 1.0 -! (x *! x)) in
    for l = 0 to n - 1 do
      let fl = float_of_int l in
      let g0 = lit ((fl +. 1.0) *. fl) *! p.(0).(l) in
      let g0 = if n > 2 then g0 -! p.(2).(l) else g0 in
      derivative.(1).(l) <- (if l = 0 then zeros else scale *! g0)
    done
  end;
  derivative

let check_lpmn m n z =
  (match dtype z with
  | D.F32 | D.F64 -> ()
  | d ->
      invalid_arg
        (Printf.sprintf
           "z.dtype=%s is not supported, see docstring for supported types."
           (D.short_name d)));
  if Array.length (shape z) <> 1 then invalid_arg "z must be a 1D array.";
  if m <> n then invalid_arg "Computations for m!=n are not yet supported."

let lpmn m n z =
  check_lpmn m n z;
  let p = legendre_table ~normalized:false n z in
  [ stack_table p; stack_table (legendre_derivatives n z p) ]

let lpmn_values m n z is_normalized =
  check_lpmn m n z;
  stack_table (legendre_table ~normalized:is_normalized n z)

let sph_harm_y ?diff_n ?n_max n m theta phi =
  if diff_n <> None then
    invalid_arg
      "The 'diff_n' argument to jax.scipy.special.sph_harm_y is not supported.";
  let theta = promote1 theta and phi = promote1 phi in
  let theta =
    if Array.length (shape theta) = 0 then NL.reshape theta [| 1 |] else theta
  in
  let n_max =
    match n_max with
    | Some v -> v
    | None -> (
        match RED.amax n with
        | T.Concrete nd -> int_of_float (Nd.get_f nd [||])
        | T.Device b -> int_of_float (Nd.get_f (Pjrt.Buffer.to_host b) [||])
        | T.Tracer _ ->
            invalid_arg
              "The `n_max` argument of `jnp.scipy.special.sph_harm` must be \
               statically specified to use `sph_harm` within JAX \
               transformations.")
  in
  let cube =
    stack_table (legendre_table ~normalized:true n_max (UF.cos theta))
  in
  let count = (shape theta).(0) in
  let abs_m = sp_abs m in
  let limit v =
    let top =
      T.Concrete (Nd.of_floats (dtype v) [||] [| float_of_int n_max |])
    in
    let low = T.Concrete (Nd.of_floats (dtype v) [||] [| 0.0 |]) in
    UF.minimum (UF.maximum v low) top
  in
  let rows = limit abs_m and cols = limit n in
  let lanes = NL.arange ~dtype:(dtype rows) (float_of_int count) in
  let side v =
    T.Concrete (Nd.of_floats (dtype rows) [||] [| float_of_int v |])
  in
  let linear =
    (((rows *! side (n_max + 1)) +! NL.astype cols (dtype rows)) *! side count)
    +! lanes
  in
  let value =
    Numpy.Indexing.take ~axis:0
      (NL.reshape cube [| (n_max + 1) * (n_max + 1) * count |])
      linear
  in
  let dt = dtype value in
  let angle = NL.astype abs_m dt *! phi in
  let harmonics =
    bind1 T.Complex [ value *! UF.cos angle; value *! UF.sin angle ]
  in
  let two = T.Concrete (Nd.of_floats (dtype abs_m) [||] [| 2.0 |]) in
  let odd =
    sp_eq (UF.remainder abs_m two)
      (T.Concrete (Nd.of_floats (dtype abs_m) [||] [| 1.0 |]))
  in
  let sign_dt = Dts.to_inexact_dtype (dtype abs_m) in
  let sign =
    where_ odd (const_full sign_dt [||] (-1.0)) (const_full sign_dt [||] 1.0)
  in
  let negative_order = sign *! UF.conj harmonics in
  where_
    (sp_lt m (T.Concrete (Nd.of_floats (dtype m) [||] [| 0.0 |])))
    negative_order harmonics

let fresnel_scalar x =
  let pi = 4.0 *. Float.atan 1.0 in
  let half_pi = pi /. 2.0 in
  let t = Float.abs x in
  let s, c =
    if t = 0.0 then (0.0, 0.0)
    else if Float.is_integer t && t > 1e15 then (0.5, 0.5)
    else if t = Float.infinity then (0.5, 0.5)
    else if t < 1.5 then begin
      let sum_c = ref t and sum_s = ref 0.0 in
      let sign = ref 1.0 and term = ref t in
      let fact = ref (half_pi *. t *. t) in
      let odd = ref true in
      let k = ref 1 in
      let continue_ = ref true in
      while !continue_ && !k < 200 do
        term := !term *. !fact /. float_of_int !k;
        let contribution = !term /. float_of_int ((2 * !k) + 1) in
        if !odd then sum_s := !sum_s +. (!sign *. contribution)
        else begin
          sign := -. !sign;
          sum_c := !sum_c +. (!sign *. contribution)
        end;
        if
          Float.abs contribution
          < 1e-17 *. (Float.abs !sum_c +. Float.abs !sum_s)
        then continue_ := false;
        odd := not !odd;
        incr k
      done;
      (!sum_s, !sum_c)
    end
    else begin
      let z = { Complex.re = 1.0; im = -.pi *. t *. t } in
      let b = ref z in
      let c_ = ref { Complex.re = 1.0 /. 1e-300; im = 0.0 } in
      let d = ref (Complex.div Complex.one !b) in
      let h = ref !d in
      let n = ref (-1.0) in
      let k = ref 2 in
      let continue_ = ref true in
      while !continue_ && !k < 500 do
        n := !n +. 2.0;
        let a = -. !n *. (!n +. 1.0) in
        b := Complex.add !b { Complex.re = 4.0; im = 0.0 };
        let ca = { Complex.re = a; im = 0.0 } in
        d := Complex.div Complex.one (Complex.add (Complex.mul ca !d) !b);
        c_ := Complex.add !b (Complex.div ca !c_);
        let del = Complex.mul !c_ !d in
        h := Complex.mul !h del;
        if Complex.norm (Complex.sub del Complex.one) < 1e-16 then
          continue_ := false;
        incr k
      done;
      let scaled = Complex.mul !h { Complex.re = t; im = -.t } in
      let phase = half_pi *. t *. t in
      let w = Complex.mul { Complex.re = cos phase; im = sin phase } scaled in
      let cs =
        Complex.mul { Complex.re = 0.5; im = 0.5 } (Complex.sub Complex.one w)
      in
      (cs.Complex.im, cs.Complex.re)
    end
  in
  if x < 0.0 then (-.s, -.c) else (s, c)

let fresnel_host = function
  | [ x ] ->
      let nd =
        match x with
        | Types.Concrete a -> a
        | Types.Device b -> Pjrt.Buffer.to_host b
        | Types.Tracer _ -> invalid_arg "fresnel: expected a concrete value"
      in
      let dt = Ndarray.dtype nd in
      [
        Types.Concrete (Ndarray.map dt (fun v -> fst (fresnel_scalar v)) nd);
        Types.Concrete (Ndarray.map dt (fun v -> snd (fresnel_scalar v)) nd);
      ]
  | _ -> invalid_arg "fresnel: arity"

let fresnel_primal args =
  match args with
  | [ Types.Tracer t ] ->
      Custom_derivatives.pure_callback fresnel_host
        ~result_avals:[ t.Types.aval; t.Types.aval ]
        args
  | _ -> fresnel_host args

let fresnel_rule =
  Custom_derivatives.custom_jvp fresnel_primal ~jvp:(fun primals tangents ->
      match (primals, tangents) with
      | [ x ], [ t ] ->
          let half_pi = 2.0 *. Float.atan 1.0 in
          let phase =
            Lax_api.mul (Lax_api.mul x x) (Lax_api.full_like x half_pi)
          in
          ( fresnel_primal primals,
            [
              Lax_api.mul (Lax_api.sin phase) t;
              Lax_api.mul (Lax_api.cos phase) t;
            ] )
      | _ -> invalid_arg "fresnel: arity")

let fresnel x =
  let x = promote1 x in
  if is_complex_dtype (dtype x) then
    failwith "Support for complex-valued inputs is not implemented yet.";
  match fresnel_rule [ x ] with
  | [ s; c ] -> (s, c)
  | _ -> invalid_arg "fresnel: arity"
