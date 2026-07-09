let pi = 4.0 *. Float.atan 1.0
let two_over_sqrt_pi = 2.0 /. Float.sqrt pi
let finfo_eps = function Dtype.F32 -> 1.1920929e-07 | _ -> epsilon_float
let finfo_max = function Dtype.F32 -> 3.4028235e+38 | _ -> max_float
let finfo_tiny = function Dtype.F32 -> 1.1754944e-38 | _ -> min_float

let cheb coeffs x =
  let b0 = ref 0.0 and b1 = ref 0.0 and b2 = ref 0.0 in
  Array.iter
    (fun c ->
      b2 := !b1;
      b1 := !b0;
      b0 := (x *. !b1) -. !b2 +. c)
    coeffs;
  0.5 *. (!b0 -. !b2)

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

let half_log_2pi = 0.5 *. Float.log (2.0 *. pi)

let lgamma_lanczos zz =
  let ag = ref lanczos_coeffs.(0) in
  for k = 1 to 8 do
    ag := !ag +. (lanczos_coeffs.(k) /. (zz +. float_of_int k))
  done;
  let t = zz +. lanczos_g +. 0.5 in
  half_log_2pi +. ((zz +. 0.5) *. Float.log t) -. t +. Float.log !ag

let lgamma x =
  if x < 0.5 then
    Float.log pi
    -. Float.log (Float.abs (Float.sin (pi *. x)))
    -. lgamma_lanczos (-.x)
  else lgamma_lanczos (x -. 1.0)

let rec digamma x =
  if x <= 0.0 && Float.equal (Float.rem x 1.0) 0.0 then Float.nan
  else if x < 0.0 then digamma (1.0 -. x) -. (pi /. Float.tan (pi *. x))
  else begin
    let acc = ref 0.0 in
    let y = ref x in
    while !y < 6.0 do
      acc := !acc -. (1.0 /. !y);
      y := !y +. 1.0
    done;
    let f = 1.0 /. (!y *. !y) in
    !acc +. Float.log !y
    -. (1.0 /. (2.0 *. !y))
    -. f
       *. ((1.0 /. 12.0)
          -. f
             *. ((1.0 /. 120.0)
                -. f
                   *. ((1.0 /. 252.0)
                      -. f
                         *. ((1.0 /. 240.0)
                            -. f
                               *. ((1.0 /. 132.0)
                                  -. f
                                     *. ((691.0 /. 32760.0)
                                        -. (f *. (1.0 /. 12.0))))))))
  end

let erf x = Float.erf x
let erfc x = Float.erfc x

let erf_inv x =
  let w = -.Float.log1p (-.(x *. x)) in
  if w < 5.0 then begin
    let w = w -. 2.5 in
    let p = 2.81022636e-08 in
    let p = 3.43273939e-07 +. (p *. w) in
    let p = -3.5233877e-06 +. (p *. w) in
    let p = -4.39150654e-06 +. (p *. w) in
    let p = 0.00021858087 +. (p *. w) in
    let p = -0.00125372503 +. (p *. w) in
    let p = -0.00417768164 +. (p *. w) in
    let p = 0.246640727 +. (p *. w) in
    let p = 1.50140941 +. (p *. w) in
    p *. x
  end
  else begin
    let w = Float.sqrt w -. 3.0 in
    let p = -0.000200214257 in
    let p = 0.000100950558 +. (p *. w) in
    let p = 0.00134934322 +. (p *. w) in
    let p = -0.00367342844 +. (p *. w) in
    let p = 0.00573950773 +. (p *. w) in
    let p = -0.0076224613 +. (p *. w) in
    let p = 0.00943887047 +. (p *. w) in
    let p = 1.00167406 +. (p *. w) in
    let p = 2.83297682 +. (p *. w) in
    p *. x
  end

let i0e_coeffs_a32 =
  [|
    -1.30002500998624804212e-8;
    6.04699502254191894932e-8;
    -2.67079385394061173391e-7;
    1.11738753912010371815e-6;
    -4.41673835845875056359e-6;
    1.64484480707288970893e-5;
    -5.75419501008210370398e-5;
    1.88502885095841655729e-4;
    -5.76375574538582365885e-4;
    1.63947561694133579842e-3;
    -4.32430999505057594430e-3;
    1.05464603945949983183e-2;
    -2.37374148058994688156e-2;
    4.93052842396707084878e-2;
    -9.49010970480476444210e-2;
    1.71620901522208775349e-1;
    -3.04682672343198398683e-1;
    6.76795274409476084995e-1;
  |]

let i0e_coeffs_b32 =
  [|
    3.39623202570838634515e-9;
    2.26666899049817806459e-8;
    2.04891858946906374183e-7;
    2.89137052083475648297e-6;
    6.88975834691682398426e-5;
    3.36911647825569408990e-3;
    8.04490411014108831608e-1;
  |]

let i0e_coeffs_a64 =
  [|
    -4.41534164647933937950e-18;
    3.33079451882223809783e-17;
    -2.43127984654795469359e-16;
    1.71539128555513303061e-15;
    -1.16853328779934516808e-14;
    7.67618549860493561688e-14;
    -4.85644678311192946090e-13;
    2.95505266312963983461e-12;
    -1.72682629144155570723e-11;
    9.67580903537323691224e-11;
    -5.18979560163526290666e-10;
    2.65982372468238665035e-9;
    -1.30002500998624804212e-8;
    6.04699502254191894932e-8;
    -2.67079385394061173391e-7;
    1.11738753912010371815e-6;
    -4.41673835845875056359e-6;
    1.64484480707288970893e-5;
    -5.75419501008210370398e-5;
    1.88502885095841655729e-4;
    -5.76375574538582365885e-4;
    1.63947561694133579842e-3;
    -4.32430999505057594430e-3;
    1.05464603945949983183e-2;
    -2.37374148058994688156e-2;
    4.93052842396707084878e-2;
    -9.49010970480476444210e-2;
    1.71620901522208775349e-1;
    -3.04682672343198398683e-1;
    6.76795274409476084995e-1;
  |]

let i0e_coeffs_b64 =
  [|
    -7.23318048787475395456e-18;
    -4.83050448594418207126e-18;
    4.46562142029675999901e-17;
    3.46122286769746109310e-17;
    -2.82762398051658348494e-16;
    -3.42548561967721913462e-16;
    1.77256013305652638360e-15;
    3.81168066935262242075e-15;
    -9.55484669882830764870e-15;
    -4.15056934728722208663e-14;
    1.54008621752140982691e-14;
    3.85277838274214270114e-13;
    7.18012445138366623367e-13;
    -1.79417853150680611778e-12;
    -1.32158118404477131188e-11;
    -3.14991652796324136454e-11;
    1.18891471078464383424e-11;
    4.94060238822496958910e-10;
    3.39623202570838634515e-9;
    2.26666899049817806459e-8;
    2.04891858946906374183e-7;
    2.89137052083475648297e-6;
    6.88975834691682398426e-5;
    3.36911647825569408990e-3;
    8.04490411014108831608e-1;
  |]

let i0e_with coeffs_a coeffs_b x =
  let x = Float.abs x in
  let le8 = cheb coeffs_a ((0.5 *. x) -. 2.0) in
  let gt8 = cheb coeffs_b ((32.0 /. x) -. 2.0) /. Float.sqrt x in
  if x <= 8.0 then le8 else gt8

let bessel_i0e dtype x =
  match dtype with
  | Dtype.F64 -> i0e_with i0e_coeffs_a64 i0e_coeffs_b64 x
  | _ -> i0e_with i0e_coeffs_a32 i0e_coeffs_b32 x

let i1e_coeffs_a =
  [|
    2.77791411276104639959e-18;
    -2.11142121435816608115e-17;
    1.55363195773620046921e-16;
    -1.10559694773538630805e-15;
    7.60068429473540693410e-15;
    -5.04218550472791168711e-14;
    3.22379336594557470981e-13;
    -1.98397439776494371520e-12;
    1.17361862988909016308e-11;
    -6.66348972350202774223e-11;
    3.62559028155211703701e-10;
    -1.88724975172282928790e-9;
    9.38153738649577178388e-9;
    -4.44505912879632808065e-8;
    2.00329475355213526229e-7;
    -8.56872026469545474066e-7;
    3.47025130813767847674e-6;
    -1.32731636560394358279e-5;
    4.78156510755005422638e-5;
    -1.61760815825896745588e-4;
    5.12285956168575772895e-4;
    -1.51357245063125314899e-3;
    4.15642294431288815669e-3;
    -1.05640848946261981558e-2;
    2.47264490306265168283e-2;
    -5.29459812080949914269e-2;
    1.02643658689847095384e-1;
    -1.76416518357834055153e-1;
    2.52587186443633654823e-1;
  |]

let i1e_coeffs_b =
  [|
    7.51729631084210481353e-18;
    4.41434832307170791151e-18;
    -4.65030536848935832153e-17;
    -3.20952592199342395980e-17;
    2.96262899764595013876e-16;
    3.30820231092092828324e-16;
    -1.88035477551078244854e-15;
    -3.81440307243700780478e-15;
    1.04202769841288027642e-14;
    4.27244001671195135429e-14;
    -2.10154184277266431302e-14;
    -4.08355111109219731823e-13;
    -7.19855177624590851209e-13;
    2.03562854414708950722e-12;
    1.41258074366137813316e-11;
    3.25260358301548823856e-11;
    -1.89749581235054123450e-11;
    -5.58974346219658380687e-10;
    -3.83538038596423702205e-9;
    -2.63146884688951950684e-8;
    -2.51223623787020892529e-7;
    -3.88256480887769039346e-6;
    -1.10588938762623716291e-4;
    -9.76109749136146840777e-3;
    7.78576235018280120474e-1;
  |]

let bessel_i1e _dtype x =
  let z = Float.abs x in
  let r =
    if z <= 8.0 then cheb i1e_coeffs_a ((z /. 2.0) -. 2.0) *. z
    else cheb i1e_coeffs_b ((32.0 /. z) -. 2.0) /. Float.sqrt z
  in
  if x < 0.0 then -.r else r

let igamma_series_value dtype ax x a =
  let eps = finfo_eps dtype in
  let r = ref a and c = ref 1.0 and ans = ref 1.0 in
  let continue = ref true and iter = ref 0 in
  while !continue && !iter < 2000 do
    r := !r +. 1.0;
    c := !c *. (x /. !r);
    ans := !ans +. !c;
    if !c /. !ans <= eps then continue := false;
    incr iter
  done;
  !ans *. ax /. a

let igamma_series_deriv dtype ax x a =
  let eps = finfo_eps dtype in
  let r = ref a and c = ref 1.0 and ans = ref 1.0 in
  let dc_da = ref 0.0 and dans_da = ref 0.0 in
  let continue = ref true and iter = ref 0 in
  while !continue && !iter < 2000 do
    r := !r +. 1.0;
    dc_da := (!dc_da *. (x /. !r)) -. (!c *. x /. (!r *. !r));
    dans_da := !dans_da +. !dc_da;
    c := !c *. (x /. !r);
    ans := !ans +. !c;
    if Float.abs (!dc_da /. !dans_da) <= eps then continue := false;
    incr iter
  done;
  let dlogax_da = Float.log x -. digamma (a +. 1.0) in
  ax *. ((!ans *. dlogax_da) +. !dans_da) /. a

let igammac_cf_value dtype ax x a =
  let eps = finfo_eps dtype in
  let y = ref (1.0 -. a) in
  let z = ref (x +. !y +. 1.0) in
  let c = ref 0.0 in
  let pkm2 = ref 1.0 and qkm2 = ref x in
  let pkm1 = ref (x +. 1.0) and qkm1 = ref (!z *. x) in
  let ans = ref (!pkm1 /. !qkm1) in
  let t = ref 1.0 in
  let continue = ref true in
  while !continue && !c < 2000.0 do
    c := !c +. 1.0;
    y := !y +. 1.0;
    z := !z +. 2.0;
    let yc = !y *. !c in
    let pk = (!pkm1 *. !z) -. (!pkm2 *. yc) in
    let qk = (!qkm1 *. !z) -. (!qkm2 *. yc) in
    if qk <> 0.0 then begin
      let r = pk /. qk in
      t := Float.abs ((!ans -. r) /. r);
      ans := r
    end
    else t := 1.0;
    pkm2 := !pkm1;
    pkm1 := pk;
    qkm2 := !qkm1;
    qkm1 := qk;
    if Float.abs pk > 1.0 /. eps then begin
      pkm2 := !pkm2 *. eps;
      pkm1 := !pkm1 *. eps;
      qkm2 := !qkm2 *. eps;
      qkm1 := !qkm1 *. eps
    end;
    if !t <= eps then continue := false
  done;
  !ans *. ax

let igammac_cf_deriv dtype ax x a =
  let eps = finfo_eps dtype in
  let y = ref (1.0 -. a) in
  let z = ref (x +. !y +. 1.0) in
  let c = ref 0.0 in
  let pkm2 = ref 1.0 and qkm2 = ref x in
  let pkm1 = ref (x +. 1.0) and qkm1 = ref (!z *. x) in
  let ans = ref (!pkm1 /. !qkm1) in
  let dpkm2_da = ref 0.0 and dqkm2_da = ref 0.0 in
  let dpkm1_da = ref 0.0 and dqkm1_da = ref (-.x) in
  let dans_da = ref ((!dpkm1_da -. (!ans *. !dqkm1_da)) /. !qkm1) in
  let continue = ref true in
  while !continue && !c < 2000.0 do
    c := !c +. 1.0;
    y := !y +. 1.0;
    z := !z +. 2.0;
    let yc = !y *. !c in
    let pk = (!pkm1 *. !z) -. (!pkm2 *. yc) in
    let qk = (!qkm1 *. !z) -. (!qkm2 *. yc) in
    let dpk_da =
      (!dpkm1_da *. !z) -. !pkm1 -. (!dpkm2_da *. yc) +. (!pkm2 *. !c)
    in
    let dqk_da =
      (!dqkm1_da *. !z) -. !qkm1 -. (!dqkm2_da *. yc) +. (!qkm2 *. !c)
    in
    let grad_conditional = ref 1.0 in
    if qk <> 0.0 then begin
      let r = pk /. qk in
      let dans_da_new = (dpk_da -. (!ans *. dqk_da)) /. qk in
      grad_conditional := Float.abs (dans_da_new -. !dans_da);
      ans := r;
      dans_da := dans_da_new
    end
    else grad_conditional := 1.0;
    pkm2 := !pkm1;
    pkm1 := pk;
    qkm2 := !qkm1;
    qkm1 := qk;
    dpkm2_da := !dpkm1_da;
    dqkm2_da := !dqkm1_da;
    dpkm1_da := dpk_da;
    dqkm1_da := dqk_da;
    if Float.abs pk > 1.0 /. eps then begin
      pkm2 := !pkm2 *. eps;
      pkm1 := !pkm1 *. eps;
      qkm2 := !qkm2 *. eps;
      qkm1 := !qkm1 *. eps;
      dpkm2_da := !dpkm2_da *. eps;
      dqkm2_da := !dqkm2_da *. eps;
      dpkm1_da := !dpkm1_da *. eps;
      dqkm1_da := !dqkm1_da *. eps
    end;
    if !grad_conditional <= eps then continue := false
  done;
  let dlogax_da = Float.log x -. digamma a in
  ax *. ((!ans *. dlogax_da) +. !dans_da)

let igamma dtype a x =
  let is_nan = Float.is_nan a || Float.is_nan x in
  let x_is_infinity = x = Float.infinity in
  let a_is_zero = a = 0.0 in
  let x_is_zero = x = 0.0 in
  let domain_error = x < 0.0 || a < 0.0 || (a_is_zero && x_is_zero) || is_nan in
  if domain_error then Float.nan
  else if x_is_zero then 0.0
  else if x_is_infinity then 1.0
  else begin
    let ax = (a *. Float.log x) -. x -. lgamma a in
    if ax < -.Float.log (finfo_max dtype) then 0.0
    else begin
      let ax = Float.exp ax in
      if x >= 1.0 && x > a then 1.0 -. igammac_cf_value dtype ax x a
      else igamma_series_value dtype ax x a
    end
  end

let igammac dtype a x =
  let is_nan = Float.is_nan a || Float.is_nan x in
  let a_is_zero = a = 0.0 in
  let x_is_zero = x = 0.0 in
  let x_is_infinity = x = Float.infinity in
  let domain_error = x < 0.0 || a < 0.0 || (a_is_zero && x_is_zero) || is_nan in
  if domain_error then Float.nan
  else if x_is_infinity || a_is_zero then 0.0
  else begin
    let use_igamma = x < 1.0 || x < a in
    let ax = (a *. Float.log x) -. x -. lgamma a in
    if ax < -.Float.log (finfo_max dtype) then if use_igamma then 1.0 else 0.0
    else begin
      let ax = Float.exp ax in
      if use_igamma then 1.0 -. igamma_series_value dtype ax x a
      else igammac_cf_value dtype ax x a
    end
  end

let igamma_grad_a dtype a x =
  let is_nan = Float.is_nan a || Float.is_nan x in
  let x_is_zero = x = 0.0 in
  let domain_error = x < 0.0 || a <= 0.0 in
  if domain_error || is_nan then Float.nan
  else if x_is_zero then 0.0
  else begin
    let ax = (a *. Float.log x) -. x -. lgamma a in
    if ax < -.Float.log (finfo_max dtype) then 0.0
    else begin
      let ax = Float.exp ax in
      if x > 1.0 && x > a then -.igammac_cf_deriv dtype ax x a
      else igamma_series_deriv dtype ax x a
    end
  end

let bernoulli_coefs =
  [|
    12.0;
    -720.0;
    30240.0;
    -1209600.0;
    47900160.0;
    -1307674368000.0 /. 691.0;
    74724249600.0;
    -10670622842880000.0 /. 3617.0;
    5109094217170944000.0 /. 43867.0;
    -802857662698291200000.0 /. 174611.0;
    14101100039391805440000.0 /. 77683.0;
    -1693824136731743669452800000.0 /. 236364091.0;
    186134520519971831808000000.0 /. 657931.0;
    -37893265687455865519472640000000.0 /. 3392780147.0;
    759790291646040068357842010112000000.0 /. 1723168255201.0;
    -134196726836183700385281186201600000000.0 /. 7709321041217.0;
  |]

let zeta dtype s a =
  let n = match dtype with Dtype.F32 -> 8 | _ -> 16 in
  let big = finfo_max dtype in
  let series = ref 0.0 in
  for k = 0 to n - 1 do
    series := !series +. Float.pow (a +. float_of_int k) (-.s)
  done;
  let an = a +. float_of_int n in
  let integral = Float.pow an (1.0 -. s) /. (s -. 1.0) in
  let t0 = Float.pow an (-.s) in
  let cp = ref 1.0 in
  let t1_sum = ref 0.0 in
  for j = 0 to n - 1 do
    let m0 = 2 * j and m1 = (2 * j) + 1 in
    cp := !cp *. ((s +. float_of_int m0) /. an);
    let even = !cp in
    cp := !cp *. ((s +. float_of_int m1) /. an);
    let clipped = if even > big then big else even in
    t1_sum := !t1_sum +. (clipped /. bernoulli_coefs.(j))
  done;
  let tail = t0 *. (0.5 +. !t1_sum) in
  let result = !series +. integral +. tail in
  if s < 1.0 then Float.nan else result

let polygamma dtype m x =
  let n = int_of_float (Float.round m) in
  if n = 0 then digamma x
  else
    let sign = if n mod 2 = 0 then -1.0 else 1.0 in
    let factorial = Float.exp (lgamma (m +. 1.0)) in
    sign *. factorial *. zeta dtype (m +. 1.0) x

let lentz dtype num_iterations partial_num partial_den =
  let small = finfo_eps dtype /. 2.0 in
  let threshold = finfo_eps dtype /. 2.0 in
  let pd0 = partial_den 0 in
  let h = ref (if Float.abs pd0 < small then small else pd0) in
  let c = ref !h and d = ref 0.0 in
  let iteration = ref 1 and converged = ref false in
  while !iteration < num_iterations && not !converged do
    let pn = partial_num !iteration in
    let pdn = partial_den !iteration in
    let cc = pdn +. (pn /. !c) in
    let cc = if Float.abs cc < small then small else cc in
    let dd = pdn +. (pn *. !d) in
    let dd = if Float.abs dd < small then small else dd in
    let dd = 1.0 /. dd in
    let delta = cc *. dd in
    h := !h *. delta;
    c := cc;
    d := dd;
    if Float.abs (delta -. 1.0) < threshold then converged := true;
    incr iteration
  done;
  !h

let regularized_incomplete_beta dtype a b x =
  let is_nan = Float.is_nan a || Float.is_nan b || Float.is_nan x in
  let a_is_zero = a = 0.0 || b = Float.infinity in
  let b_is_zero = b = 0.0 || a = Float.infinity in
  let x_is_zero = x = 0.0 in
  let x_is_one = x = 1.0 in
  let result_is_zero =
    (b_is_zero && not x_is_one) || (a_is_zero && x_is_zero)
  in
  let result_is_one = (a_is_zero && not x_is_zero) || (b_is_zero && x_is_one) in
  let result_is_nan =
    a < 0.0 || b < 0.0 || x < 0.0 || x > 1.0 || (a_is_zero && b_is_zero)
    || is_nan
  in
  if result_is_nan then Float.nan
  else if result_is_zero then 0.0
  else if result_is_one then 1.0
  else begin
    let converges_rapidly = x < (a +. 1.0) /. (a +. b +. 2.0) in
    let a_orig = a in
    let a = if converges_rapidly then a else b in
    let b = if converges_rapidly then b else a_orig in
    let x = if converges_rapidly then x else 1.0 -. x in
    let num_iterations = match dtype with Dtype.F32 -> 200 | _ -> 600 in
    let partial_num iteration =
      if iteration = 1 then 1.0
      else begin
        let iteration_is_even = iteration mod 2 = 0 in
        let m_int = (iteration - 1) / 2 in
        let m = float_of_int m_int in
        let one = 1.0 and two = 2.0 in
        if iteration_is_even then
          if m_int = 0 then -.(a +. b) *. x /. (a +. one)
          else
            -.(a +. m)
            *. (a +. b +. m)
            *. x
            /. ((a +. (two *. m)) *. (a +. (two *. m) +. one))
        else
          m *. (b -. m) *. x /. ((a +. (two *. m) -. one) *. (a +. (two *. m)))
      end
    in
    let partial_den iteration = if iteration = 0 then 0.0 else 1.0 in
    let continued_fraction =
      lentz dtype num_iterations partial_num partial_den
    in
    let very_small = finfo_tiny dtype *. 2.0 in
    let lbeta_ab_small_a = lgamma b -. lgamma (a +. b) in
    let lbeta_ab = lgamma a +. lbeta_ab_small_a in
    let factor =
      if a < very_small then
        Float.exp ((Float.log1p (-.x) *. b) -. lbeta_ab_small_a)
      else
        Float.exp ((Float.log x *. a) +. (Float.log1p (-.x) *. b) -. lbeta_ab)
        /. a
    in
    let result = continued_fraction *. factor in
    if converges_rapidly then result else 1.0 -. result
  end
