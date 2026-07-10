module T = Types
module C = Core
module D = Dtype
module Nd = Ndarray
module U = Numpy.Ufuncs
module LN = Numpy.Lax_numpy
module TC = Numpy.Tensor_contractions
module AC = Numpy.Array_creation
module API = Api

let concrete = function
  | T.Concrete nd -> nd
  | _ -> failwith "optimize: value not concrete"

let f_of v = Nd.get_f (concrete v) [||]
let b_of v = f_of v <> 0.0

let leaf = function
  | Tree_util.Leaf v -> v
  | _ -> failwith "optimize: expected leaf"

let aval v = C.get_aval v
let dtype_of v = (aval v).T.dtype

let is_inexact = function
  | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> true
  | _ -> false

let bits_of = function D.F64 | D.Complex128 -> 64 | _ -> 32
let konst dt x = AC.full ~dtype:dt [||] x
let k_like v x = konst (dtype_of v) x
let add = U.add
let sub = U.subtract
let mul = U.multiply
let div = U.divide
let neg = U.negative
let sqrtv = U.sqrt
let sq = U.square
let absv = U.abs
let sgn = U.sign
let realv = U.real
let gt = U.greater
let lt = U.less
let ge = U.greater_equal
let le = U.less_equal
let band = U.logical_and
let bor = U.logical_or
let bnot = U.logical_not
let dot a b = TC.dot a b

let to_inexact v =
  let dt = dtype_of v in
  if is_inexact dt then v else LN.astype v (Dtypes.default_float_dtype ())

let vg f x =
  let fv, gv =
    API.value_and_grad
      (fun xs -> Tree_util.Leaf (f (leaf (List.hd xs))))
      [ Tree_util.Leaf x ]
  in
  (leaf fv, leaf gv)

let cubicmin a fa fpa b fb c fc =
  let cc = fpa in
  let db = sub b a in
  let dc = sub c a in
  let denom = mul (sq (mul db dc)) (sub db dc) in
  let d2_0 = sub (sub fb fa) (mul cc db) in
  let d2_1 = sub (sub fc fa) (mul cc dc) in
  let aa = div (add (mul (sq dc) d2_0) (mul (neg (sq db)) d2_1)) denom in
  let bb =
    div
      (add (mul (neg (mul dc (sq dc))) d2_0) (mul (mul db (sq db)) d2_1))
      denom
  in
  let three = k_like a 3.0 in
  let radical = sub (mul bb bb) (mul (mul three aa) cc) in
  add a (div (add (neg bb) (sqrtv radical)) (mul three aa))

let quadmin a fa fpa b fb =
  let cc = fpa in
  let db = sub b a in
  let bb = div (sub (sub fb fa) (mul cc db)) (sq db) in
  sub a (div cc (mul (k_like a 2.0) bb))

type zoom_state = {
  z_done : bool;
  z_failed : bool;
  z_j : int;
  a_lo : T.value;
  phi_lo : T.value;
  dphi_lo : T.value;
  a_hi : T.value;
  phi_hi : T.value;
  dphi_hi : T.value;
  a_rec : T.value;
  phi_rec : T.value;
  z_a_star : T.value;
  z_phi_star : T.value;
  z_dphi_star : T.value;
  z_g_star : T.value;
  z_nfev : int;
  z_ngev : int;
}

let zoom restricted wolfe_one wolfe_two a_lo phi_lo dphi_lo a_hi phi_hi dphi_hi
    g_0 pass_through =
  let dt = dtype_of phi_lo in
  let delta1 = konst dt 0.2 in
  let delta2 = konst dt 0.1 in
  let threshold = if bits_of dt < 64 then konst dt 1e-5 else konst dt 1e-10 in
  let init =
    {
      z_done = false;
      z_failed = false;
      z_j = 0;
      a_lo;
      phi_lo;
      dphi_lo;
      a_hi;
      phi_hi;
      dphi_hi;
      a_rec = div (add a_lo a_hi) (konst dt 2.0);
      phi_rec = div (add phi_lo phi_hi) (konst dt 2.0);
      z_a_star = konst dt 1.0;
      z_phi_star = phi_lo;
      z_dphi_star = dphi_lo;
      z_g_star = g_0;
      z_nfev = 0;
      z_ngev = 0;
    }
  in
  let body st =
    let dalpha = sub st.a_hi st.a_lo in
    let a = U.minimum st.a_hi st.a_lo in
    let b = U.maximum st.a_hi st.a_lo in
    let cchk = mul delta1 dalpha in
    let qchk = mul delta2 dalpha in
    let failed = st.z_failed || b_of (le dalpha threshold) in
    let a_j_cubic =
      cubicmin st.a_lo st.phi_lo st.dphi_lo st.a_hi st.phi_hi st.a_rec
        st.phi_rec
    in
    let use_cubic =
      st.z_j > 0
      && b_of (gt a_j_cubic (add a cchk))
      && b_of (lt a_j_cubic (sub b cchk))
    in
    let a_j_quad = quadmin st.a_lo st.phi_lo st.dphi_lo st.a_hi st.phi_hi in
    let use_quad =
      (not use_cubic)
      && b_of (gt a_j_quad (add a qchk))
      && b_of (lt a_j_quad (sub b qchk))
    in
    let a_j_bisection = div (add st.a_lo st.a_hi) (konst dt 2.0) in
    let a_j =
      if use_cubic then a_j_cubic
      else if use_quad then a_j_quad
      else a_j_bisection
    in
    let phi_j, dphi_j, g_j = restricted a_j in
    let phi_j = LN.astype phi_j (dtype_of st.phi_lo) in
    let dphi_j = LN.astype dphi_j (dtype_of st.dphi_lo) in
    let g_j = LN.astype g_j (dtype_of st.z_g_star) in
    let nfev = st.z_nfev + 1 in
    let ngev = st.z_ngev + 1 in
    let hi_to_j = b_of (bor (wolfe_one a_j phi_j) (ge phi_j st.phi_lo)) in
    let star_to_j = (not hi_to_j) && b_of (wolfe_two dphi_j) in
    let lo_to_j = (not hi_to_j) && not star_to_j in
    let hi_to_lo =
      lo_to_j && b_of (ge (mul dphi_j (sub st.a_hi st.a_lo)) (konst dt 0.0))
    in
    let pre_a_hi = st.a_hi in
    let pre_phi_hi = st.phi_hi in
    let st = { st with z_nfev = nfev; z_ngev = ngev; z_j = st.z_j + 1 } in
    let st =
      if hi_to_j then
        {
          st with
          a_hi = a_j;
          phi_hi = phi_j;
          dphi_hi = dphi_j;
          a_rec = pre_a_hi;
          phi_rec = pre_phi_hi;
        }
      else if star_to_j then
        {
          st with
          z_done = true;
          z_a_star = a_j;
          z_phi_star = phi_j;
          z_dphi_star = dphi_j;
          z_g_star = g_j;
        }
      else if hi_to_lo then
        {
          st with
          a_hi = st.a_lo;
          phi_hi = st.phi_lo;
          dphi_hi = st.dphi_lo;
          a_rec = pre_a_hi;
          phi_rec = pre_phi_hi;
          a_lo = a_j;
          phi_lo = phi_j;
          dphi_lo = dphi_j;
        }
      else
        {
          st with
          a_rec = st.a_lo;
          phi_rec = st.phi_lo;
          a_lo = a_j;
          phi_lo = phi_j;
          dphi_lo = dphi_j;
        }
    in
    { st with z_failed = failed || st.z_j >= 30 }
  in
  let rec loop st =
    if st.z_done || pass_through || st.z_failed then st else loop (body st)
  in
  loop init

type line_search_results = {
  failed : bool;
  nit : int;
  nfev : int;
  ngev : int;
  k : int;
  a_k : T.value;
  f_k : T.value;
  g_k : T.value;
  status : int;
}

type ls_state = {
  s_done : bool;
  s_failed : bool;
  s_i : int;
  a_i1 : T.value;
  phi_i1 : T.value;
  dphi_i1 : T.value;
  s_nfev : int;
  s_ngev : int;
  s_a_star : T.value;
  s_phi_star : T.value;
  s_dphi_star : T.value;
  s_g_star : T.value;
}

let line_search f xk pk ?old_fval ?old_old_fval ?gfk ?(c1 = 1e-4) ?(c2 = 0.9)
    ?(maxiter = 20) () =
  let xk = to_inexact xk in
  let pk = to_inexact pk in
  let dt = dtype_of pk in
  let restricted t =
    let t = LN.astype t dt in
    let phi, g = vg f (add xk (mul t pk)) in
    let dphi = realv (dot g pk) in
    (phi, dphi, g)
  in
  let phi_0, dphi_0, gfk, base_ev =
    match (old_fval, gfk) with
    | Some fv, Some gk -> (fv, realv (dot gk pk), gk, 0)
    | _ ->
        let phi, dphi, g = restricted (konst dt 0.0) in
        (phi, dphi, g, 1)
  in
  let start_value =
    match old_old_fval with
    | Some oo ->
        let candidate = div (mul (konst dt 2.02) (sub phi_0 oo)) dphi_0 in
        if b_of (gt candidate (konst dt 1.0)) then konst dt 1.0 else candidate
    | None -> konst dt 1.0
  in
  let wolfe_one a_i phi_i =
    gt phi_i (add phi_0 (mul (mul (konst dt c1) a_i) dphi_0))
  in
  let wolfe_two dphi_i = le (absv dphi_i) (neg (mul (konst dt c2) dphi_0)) in
  let init =
    {
      s_done = false;
      s_failed = false;
      s_i = 1;
      a_i1 = konst dt 0.0;
      phi_i1 = phi_0;
      dphi_i1 = dphi_0;
      s_nfev = base_ev;
      s_ngev = base_ev;
      s_a_star = konst dt 0.0;
      s_phi_star = phi_0;
      s_dphi_star = dphi_0;
      s_g_star = gfk;
    }
  in
  let body st =
    let a_i = if st.s_i = 1 then start_value else mul st.a_i1 (konst dt 2.0) in
    let phi_i, dphi_i, g_i = restricted a_i in
    let nfev = st.s_nfev + 1 in
    let ngev = st.s_ngev + 1 in
    let star_to_zoom1 =
      b_of (wolfe_one a_i phi_i) || (b_of (ge phi_i st.phi_i1) && st.s_i > 1)
    in
    let star_to_i = (not star_to_zoom1) && b_of (wolfe_two dphi_i) in
    let star_to_zoom2 =
      (not star_to_zoom1) && (not star_to_i) && b_of (ge dphi_i (konst dt 0.0))
    in
    let zoom1 =
      zoom restricted wolfe_one wolfe_two st.a_i1 st.phi_i1 st.dphi_i1 a_i phi_i
        dphi_i gfk (not star_to_zoom1)
    in
    let nfev = nfev + zoom1.z_nfev in
    let ngev = ngev + zoom1.z_ngev in
    let zoom2 =
      zoom restricted wolfe_one wolfe_two a_i phi_i dphi_i st.a_i1 st.phi_i1
        st.dphi_i1 gfk (not star_to_zoom2)
    in
    let nfev = nfev + zoom2.z_nfev in
    let ngev = ngev + zoom2.z_ngev in
    let st = { st with s_nfev = nfev; s_ngev = ngev } in
    let st =
      if star_to_zoom1 then
        {
          st with
          s_done = true;
          s_failed = st.s_failed || zoom1.z_failed;
          s_a_star = zoom1.z_a_star;
          s_phi_star = zoom1.z_phi_star;
          s_dphi_star = zoom1.z_dphi_star;
          s_g_star = zoom1.z_g_star;
        }
      else if star_to_i then
        {
          st with
          s_done = true;
          s_a_star = a_i;
          s_phi_star = phi_i;
          s_dphi_star = dphi_i;
          s_g_star = g_i;
        }
      else if star_to_zoom2 then
        {
          st with
          s_done = true;
          s_failed = st.s_failed || zoom2.z_failed;
          s_a_star = zoom2.z_a_star;
          s_phi_star = zoom2.z_phi_star;
          s_dphi_star = zoom2.z_dphi_star;
          s_g_star = zoom2.z_g_star;
        }
      else st
    in
    { st with s_i = st.s_i + 1; a_i1 = a_i; phi_i1 = phi_i; dphi_i1 = dphi_i }
  in
  let rec loop st =
    if (not st.s_done) && st.s_i <= maxiter && not st.s_failed then
      loop (body st)
    else st
  in
  let st = loop init in
  let status = if st.s_failed then 1 else if st.s_i > maxiter then 3 else 0 in
  let alpha_k = st.s_a_star in
  let floor_needed =
    bits_of dt <> 64 && b_of (lt (absv alpha_k) (konst dt 1e-8))
  in
  let alpha_k =
    if floor_needed then mul (sgn alpha_k) (konst dt 1e-8) else alpha_k
  in
  {
    failed = st.s_failed || not st.s_done;
    nit = st.s_i - 1;
    nfev = st.s_nfev;
    ngev = st.s_ngev;
    k = st.s_i;
    a_k = alpha_k;
    f_k = st.s_phi_star;
    g_k = st.s_g_star;
    status;
  }
