module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module UF = Numpy.Ufuncs
module RED = Numpy.Reductions
module NL = Numpy.Lax_numpy
module AC = Numpy.Array_creation
module Poly = Numpy.Polynomial

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let shape v = (get_aval v).T.shape
let bind1 = C.bind1
let numel sh = Array.fold_left ( * ) 1 sh

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (numel sh) x))

let scalar_of v x = const_full (dtype v) [||] x
let coeff_of v a = T.Concrete (Nd.of_floats (dtype v) [| Array.length a |] a)
let where_ = NL.where_

let to_inexact = function
  | D.I32 | D.Bool | D.Uint32 -> D.F32
  | D.I64 -> D.F64
  | (D.F32 | D.F64) as d -> d

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
  | D.I32 | D.I64 | D.Uint32 -> true
  | D.F32 | D.F64 | D.Bool -> false

let gammaln x = bind1 T.Lgamma [ promote1 x ]

let gammasgn x =
  let x = promote1 x in
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

let loggamma x =
  let x = promote1 x in
  let res = bind1 T.Lgamma [ x ] in
  where_ (UF.greater x (scalar_of x 0.0)) res (scalar_of x Float.nan)

let gamma x =
  let x = promote1 x in
  UF.multiply (gammasgn x) (UF.exp (bind1 T.Lgamma [ x ]))

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
  let a, b = promote2 a b in
  let sign =
    UF.multiply (UF.multiply (gammasgn a) (gammasgn b)) (gammasgn (UF.add a b))
  in
  UF.multiply sign (UF.exp (betaln a b))

let betainc a b x =
  let a, b, x = promote3 a b x in
  bind1 T.Regularized_incomplete_beta [ a; b; x ]

let digamma x = bind1 T.Digamma [ promote1 x ]

let gammainc a x =
  let a, x = promote2 a x in
  bind1 T.Igamma [ a; x ]

let gammaincc a x =
  let a, x = promote2 a x in
  bind1 T.Igammac [ a; x ]

let erf x = bind1 T.Erf [ promote1 x ]
let erfc x = bind1 T.Erfc [ promote1 x ]
let erfinv x = bind1 T.Erf_inv [ promote1 x ]
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
  let x = promote1 x in
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
  let p, q = promote2 p q in
  let zero = scalar_of p 0.0 in
  let both = UF.logical_and (UF.greater p zero) (UF.greater q zero) in
  let one_zero = UF.logical_and (UF.equal p zero) (UF.greater_equal q zero) in
  let safe_p = where_ both p (scalar_of p 1.0) in
  let safe_q = where_ both q (scalar_of q 1.0) in
  let log_val = UF.subtract (xlogx safe_p) (xlogy safe_p safe_q) in
  where_ both log_val (where_ one_zero zero (scalar_of p Float.infinity))

let kl_div p q =
  let p, q = promote2 p q in
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

let erfcx x =
  let x = promote1 x in
  let nterms = match dtype x with D.F64 -> 9 | _ -> 5 in
  erfcx_impl x nterms

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
  let x = promote1 x in
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

let ndtr x = ndtr_core (promote1 x)

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
  let p = promote1 p in
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
  let x = promote1 x in
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
  let n, x = promote2 n x in
  bind1 T.Polygamma [ n; x ]

let zeta ?q x =
  match q with
  | None ->
      failwith
        "Riemann zeta function not implemented; pass q != None to compute the \
         Hurwitz Zeta function."
  | Some q ->
      let x, q = promote2 x q in
      bind1 T.Zeta [ x; q ]

let wofz _z =
  failwith "wofz: complex-valued output unsupported (no complex dtype until M5)"
