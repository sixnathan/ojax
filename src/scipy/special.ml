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

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (numel sh) x))

let scalar_of v x = const_full (dtype v) [||] x
let coeff_of v a = T.Concrete (Nd.of_floats (dtype v) [| Array.length a |] a)
let where_ = NL.where_

let to_inexact = function
  | D.I32 | D.Bool | D.Uint32 -> D.F32
  | D.I64 -> D.F64
  | (D.F32 | D.F64 | D.Complex64 | D.Complex128) as d -> d

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
  | D.F32 | D.F64 | D.Bool | D.Complex64 | D.Complex128 -> false

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

let euler_gamma = 0.5772156649015329
let eps_of v = match dtype v with D.F64 -> epsilon_float | _ -> 1.1920929e-7
let full_like v x = const_full (dtype v) (shape v) x

let masked_iter ~maxiter ~cond ~body state0 =
  let state = ref state0 in
  for _ = 1 to maxiter do
    let m = cond !state in
    let nxt = body !state in
    state := List.map2 (fun n o -> where_ m n o) nxt !state
  done;
  !state

let softmax ?axis x =
  match axis with
  | Some ax -> NN.softmax ~axis:ax x
  | None ->
      let sh = shape x in
      NL.reshape (NN.softmax ~axis:0 (NL.reshape x [| numel sh |])) sh

let log_softmax ?axis x =
  match axis with
  | Some ax -> NN.log_softmax ~axis:ax x
  | None ->
      let sh = shape x in
      NL.reshape (NN.log_softmax ~axis:0 (NL.reshape x [| numel sh |])) sh

let poch z m =
  let z, m = promote2 z m in
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
  let x = promote1 x in
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
  let s v = scalar_of h v in
  let sign_a = UF.sign a in
  let nan_mask = UF.logical_or (UF.isnan a) (UF.isnan h) in
  let h = UF.abs h in
  let abs_a = UF.abs a in
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

let bernoulli n =
  if n < 0 then invalid_arg "n must be a non-negative integer.";
  let dt = Dts.default_float_dtype () in
  let b3 = [| 1.0; -0.5; 1.0 /. 6.0 |] in
  if n < 3 then
    T.Concrete (Nd.of_floats dt [| n + 1 |] (Array.sub b3 0 (n + 1)))
  else begin
    if n mod 2 <> 0 then
      failwith "bernoulli: only even n supported (odd-n assembly deferred)";
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

let hyp1f1 a b x =
  let a, b, x = promote3 a b x in
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

let expn2 x n =
  let s v = scalar_of x v in
  let big = s 1.44115188075855872e17 in
  let machep = s (eps_of x) in
  let one = s 1.0 in
  let two = s 2.0 in
  let init =
    [
      one;
      one;
      x;
      one;
      UF.add x n;
      UF.divide one (UF.add x n);
      s Float.infinity;
      s 0.0;
    ]
  in
  let cond st =
    match st with
    | [ _; _; _; _; _; _; t; _ ] ->
        UF.logical_and (UF.greater x (s 0.0)) (UF.greater t machep)
    | _ -> assert false
  in
  let body st =
    match st with
    | [ k; pkm2; qkm2; pkm1; qkm1; ans; _t; r ] ->
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
        let nz = UF.not_equal qk (s 0.0) in
        let r' = where_ nz (UF.divide pk qk) r in
        let t' = where_ nz (UF.abs (UF.divide (UF.subtract ans r') r')) one in
        let ans' = where_ nz r' ans in
        let big_m = UF.greater (UF.abs pk) big in
        let db u = where_ big_m (UF.divide u big) u in
        [ k; db pkm1; db qkm1; db pk; db qk; ans'; t'; r' ]
    | _ -> assert false
  in
  match masked_iter ~maxiter:500 ~cond ~body init with
  | [ _; _; _; _; _; ans; _; _ ] -> UF.multiply ans (UF.exp (UF.negative x))
  | _ -> assert false

let expn n x =
  let n, x = promote2 n x in
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
  let cond5 = UF.greater x one in
  where_ cond0 (s Float.nan)
    (where_ cond1 (s Float.infinity)
       (where_ cond2 (UF.divide one n1)
          (where_ cond3
             (UF.divide (UF.exp (UF.negative x)) x)
             (where_ cond5 (expn2 x n) (s Float.nan)))))

let exp1 x =
  let x = promote1 x in
  expn (scalar_of x 1.0) x

let hyp2f1 _a _b _c _x =
  failwith
    "hyp2f1: unsupported (digamma-transform + data-dependent masked fori \
     recurrence deferred)"

let expi _x =
  failwith
    "expi: unsupported (Cephes 7-interval piecewise coefficient tables \
     deferred)"

let lpmn _m _n _z =
  failwith
    "lpmn: unsupported (associated-Legendre recurrence needs fancy .at[] \
     scatter, deferred)"

let lpmn_values _m _n _z _is_normalized =
  failwith
    "lpmn_values: unsupported (associated-Legendre recurrence needs fancy \
     .at[] scatter, deferred)"

let sph_harm_y ?diff_n:_ ?n_max:_ _n _m _theta _phi =
  failwith
    "sph_harm_y: complex-valued output unsupported (no complex dtype until M5)"
