module T = Types
module C = Core
module D = Dtype
module Nd = Ndarray
module U = Numpy.Ufuncs
module LN = Numpy.Lax_numpy
module TC = Numpy.Tensor_contractions
module AC = Numpy.Array_creation
module JU = Numpy.Jnp_util
module API = Api
module L = Lax_api

let concrete = function
  | T.Concrete nd -> nd
  | T.Device b -> Pjrt.Buffer.to_host b
  | T.Tracer _ -> failwith "optimize: value not concrete"

let highest =
  Lax_config.Precision
    (Lax_config.Precision_highest, Lax_config.Precision_highest)

let dot_ a b = TC.dot ~precision:highest a b
let dtype_of v = (C.get_aval v).T.dtype
let lit v x = T.Concrete (Nd.of_floats (dtype_of v) [||] [| x |])
let npf dt x = T.Concrete (Nd.of_floats dt [||] [| x |])
let py_int = JU.py_int
let conv v dt = if dtype_of v = dt then v else LN.astype v dt
let ( +! ) = U.add
let ( -! ) = U.subtract
let ( *! ) = U.multiply
let ( /! ) = U.true_divide
let band = U.bitwise_and
let bor = U.bitwise_or
let bnot = U.bitwise_not

let bool_v b =
  T.Concrete (Nd.of_floats D.Bool [||] [| (if b then 1.0 else 0.0) |])

let leaf = function
  | Tree_util.Leaf v -> v
  | _ -> failwith "optimize: expected leaf"

let value_and_grad f x =
  let fv, gv =
    API.value_and_grad
      (fun xs -> Tree_util.Leaf (f (leaf (List.hd xs))))
      [ Tree_util.Leaf x ]
  in
  (leaf fv, leaf gv)

let cubicmin ~dt a fa fpa b fb c fc =
  let cc = fpa in
  let db = b -! a in
  let dc = c -! a in
  let product = db *! dc in
  let squared = U.power product (py_int 2) in
  let denom = squared *! (db -! dc) in
  let p2 v = conv (U.power v (py_int 2)) dt in
  let p3 v = conv (U.power v (py_int 3)) dt in
  let e00 = p2 dc in
  let e01 = U.negative (conv (U.power db (py_int 2)) dt) in
  let e10 = U.negative (p3 dc) in
  let e11 = p3 db in
  let row0 = LN.stack ~axis:0 [ e00; e01 ] in
  let row1 = LN.stack ~axis:0 [ e10; e11 ] in
  let d1 = LN.stack ~axis:0 [ row0; row1 ] in
  let t1 = fb -! fa in
  let t1 = t1 -! (cc *! conv db dt) in
  let t2 = fc -! fa in
  let t2 = t2 -! (cc *! conv dc dt) in
  let d2 = LN.stack ~axis:0 [ t1; t2 ] in
  let solved = dot_ d1 d2 /! conv denom dt in
  let aa = L.index_in_dim ~keepdims:false solved 0 in
  let bb = L.index_in_dim ~keepdims:false solved 1 in
  let radical = (bb *! bb) -! (lit aa 3.0 *! aa *! cc) in
  let num = U.negative bb +! U.sqrt radical in
  let den = lit aa 3.0 *! aa in
  conv a dt +! (num /! den)

let quadmin ~dt a fa fpa b fb =
  let cc = fpa in
  let db = b -! a in
  let t = fb -! fa in
  let t = t -! (cc *! conv db dt) in
  let den = conv (U.power db (py_int 2)) dt in
  let bb = t /! den in
  conv a dt -! (cc /! (lit bb 2.0 *! bb))

type zoom_state = {
  z_done : T.value;
  z_failed : T.value;
  j : T.value;
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
  z_nfev : T.value;
  z_ngev : T.value;
}

let zoom_to_list s =
  [
    s.z_done;
    s.z_failed;
    s.j;
    s.a_lo;
    s.phi_lo;
    s.dphi_lo;
    s.a_hi;
    s.phi_hi;
    s.dphi_hi;
    s.a_rec;
    s.phi_rec;
    s.z_a_star;
    s.z_phi_star;
    s.z_dphi_star;
    s.z_g_star;
    s.z_nfev;
    s.z_ngev;
  ]

let zoom_of_list = function
  | [
      z_done;
      z_failed;
      j;
      a_lo;
      phi_lo;
      dphi_lo;
      a_hi;
      phi_hi;
      dphi_hi;
      a_rec;
      phi_rec;
      z_a_star;
      z_phi_star;
      z_dphi_star;
      z_g_star;
      z_nfev;
      z_ngev;
    ] ->
      {
        z_done;
        z_failed;
        j;
        a_lo;
        phi_lo;
        dphi_lo;
        a_hi;
        phi_hi;
        dphi_hi;
        a_rec;
        phi_rec;
        z_a_star;
        z_phi_star;
        z_dphi_star;
        z_g_star;
        z_nfev;
        z_ngev;
      }
  | _ -> assert false

let zoom restricted wolfe_one wolfe_two a_lo phi_lo dphi_lo a_hi phi_hi dphi_hi
    g_0 pass_through =
  let dt = dtype_of phi_lo in
  let a_sum = a_lo +! a_hi in
  let a_rec = conv (a_sum /! lit a_sum 2.0) dt in
  let phi_sum = phi_lo +! phi_hi in
  let phi_rec = phi_sum /! lit phi_sum 2.0 in
  let init =
    {
      z_done = bool_v false;
      z_failed = bool_v false;
      j = py_int 0;
      a_lo = conv a_lo dt;
      phi_lo;
      dphi_lo;
      a_hi = conv a_hi dt;
      phi_hi;
      dphi_hi;
      a_rec;
      phi_rec;
      z_a_star = npf dt 1.0;
      z_phi_star = phi_lo;
      z_dphi_star = dphi_lo;
      z_g_star = g_0;
      z_nfev = py_int 0;
      z_ngev = py_int 0;
    }
  in
  let delta1 = 0.2 and delta2 = 0.1 in
  let body vals =
    let s = zoom_of_list vals in
    let dalpha = s.a_hi -! s.a_lo in
    let a = U.minimum s.a_hi s.a_lo in
    let b = U.maximum s.a_hi s.a_lo in
    let cchk = lit dalpha delta1 *! dalpha in
    let qchk = lit dalpha delta2 *! dalpha in
    let threshold =
      lit dalpha
        (if (Dtypes.finfo (dtype_of dalpha)).bits < 64 then 1e-05 else 1e-10)
    in
    let failed = bor s.z_failed (U.less_equal dalpha threshold) in
    let a_j_cubic =
      cubicmin ~dt s.a_lo s.phi_lo s.dphi_lo s.a_hi s.phi_hi s.a_rec s.phi_rec
    in
    let u0 = U.greater s.j (py_int 0) in
    let u1 = U.greater a_j_cubic (a +! cchk) in
    let u2 = U.less a_j_cubic (b -! cchk) in
    let use_cubic = band (band u0 u1) u2 in
    let a_j_quad = quadmin ~dt s.a_lo s.phi_lo s.dphi_lo s.a_hi s.phi_hi in
    let v0 = bnot use_cubic in
    let v1 = U.greater a_j_quad (a +! qchk) in
    let v2 = U.less a_j_quad (b -! qchk) in
    let use_quad = band (band v0 v1) v2 in
    let bis = s.a_lo +! s.a_hi in
    let a_j_bisection = bis /! lit bis 2.0 in
    let use_bisection = band (bnot use_cubic) (bnot use_quad) in
    let a_j = LN.where_ use_cubic a_j_cubic s.a_rec in
    let a_j = LN.where_ use_quad a_j_quad a_j in
    let a_j = LN.where_ use_bisection a_j_bisection a_j in
    let phi_j, dphi_j, g_j = restricted a_j in
    let phi_j = conv phi_j (dtype_of s.phi_lo) in
    let dphi_j = conv dphi_j (dtype_of s.dphi_lo) in
    let g_j = conv g_j (dtype_of s.z_g_star) in
    let nfev = s.z_nfev +! py_int 1 in
    let ngev = s.z_ngev +! py_int 1 in
    let w1 = wolfe_one a_j phi_j in
    let hi_to_j = bor w1 (U.greater_equal phi_j s.phi_lo) in
    let w2 = wolfe_two dphi_j in
    let star_to_j = band w2 (bnot hi_to_j) in
    let t = dphi_j *! (s.a_hi -! s.a_lo) in
    let h0 = U.greater_equal t (lit t 0.0) in
    let h1 = band h0 (bnot hi_to_j) in
    let hi_to_lo = band h1 (bnot star_to_j) in
    let lo_to_j = band (bnot hi_to_j) (bnot star_to_j) in
    let w = LN.where_ in
    let s1 =
      {
        s with
        z_failed = failed;
        z_nfev = nfev;
        z_ngev = ngev;
        a_hi = w hi_to_j a_j s.a_hi;
        phi_hi = w hi_to_j phi_j s.phi_hi;
        dphi_hi = w hi_to_j dphi_j s.dphi_hi;
        a_rec = w hi_to_j s.a_hi s.a_rec;
        phi_rec = w hi_to_j s.phi_hi s.phi_rec;
      }
    in
    let s2 =
      {
        s1 with
        z_done = bor star_to_j s1.z_done;
        z_a_star = w star_to_j a_j s1.z_a_star;
        z_phi_star = w star_to_j phi_j s1.z_phi_star;
        z_dphi_star = w star_to_j dphi_j s1.z_dphi_star;
        z_g_star = w star_to_j g_j s1.z_g_star;
      }
    in
    let s3 =
      {
        s2 with
        a_hi = w hi_to_lo s2.a_lo s2.a_hi;
        phi_hi = w hi_to_lo s2.phi_lo s2.phi_hi;
        dphi_hi = w hi_to_lo s2.dphi_lo s2.dphi_hi;
        a_rec = w hi_to_lo s2.a_hi s2.a_rec;
        phi_rec = w hi_to_lo s2.phi_hi s2.phi_rec;
      }
    in
    let m4 = band lo_to_j (bnot hi_to_lo) in
    let s4 =
      {
        s3 with
        a_rec = w m4 s3.a_lo s3.a_rec;
        phi_rec = w m4 s3.phi_lo s3.phi_rec;
      }
    in
    let s5 =
      {
        s4 with
        a_lo = w lo_to_j a_j s4.a_lo;
        phi_lo = w lo_to_j phi_j s4.phi_lo;
        dphi_lo = w lo_to_j dphi_j s4.dphi_lo;
      }
    in
    let j = s5.j +! py_int 1 in
    let failed = bor s5.z_failed (U.greater_equal j (py_int 30)) in
    zoom_to_list { s5 with j; z_failed = failed }
  in
  let cond vals =
    let s = zoom_of_list vals in
    let c = band (bnot s.z_done) (bnot pass_through) in
    band c (bnot s.z_failed)
  in
  zoom_of_list (L.while_loop cond body (zoom_to_list init))

type line_search_values = {
  v_failed : T.value;
  v_nit : T.value;
  v_nfev : T.value;
  v_ngev : T.value;
  v_k : T.value;
  v_a_k : T.value;
  v_f_k : T.value;
  v_g_k : T.value;
  v_status : T.value;
}

let line_search_values f xk pk ?old_fval ?old_old_fval ?gfk ?(c1 = 0.0001)
    ?(c2 = 0.9) ?(maxiter = 20) () =
  let xk, pk =
    let dt = Dtypes.to_inexact_dtype (LN.result_type [ xk; pk ]) in
    (conv xk dt, conv pk dt)
  in
  let restricted t =
    let t = conv t (dtype_of pk) in
    let step = t *! pk in
    let phi, g = value_and_grad f (xk +! step) in
    let dphi = U.real (dot_ g pk) in
    (phi, dphi, g)
  in
  let fresh = old_fval = None || gfk = None in
  let phi_0, dphi_0, gfk =
    match (old_fval, gfk) with
    | Some phi_0, Some g ->
        let dphi_0 = U.real (dot_ g pk) in
        (phi_0, dphi_0, g)
    | _ -> restricted (npf (dtype_of pk) 0.0)
  in
  let dt = dtype_of phi_0 in
  let a_dt, start_value =
    match old_old_fval with
    | Some old ->
        let cand = lit phi_0 (1.01 *. 2.0) *! (phi_0 -! old) /! dphi_0 in
        let gt1 = U.greater cand (lit cand 1.0) in
        (dt, LN.where_ gt1 (lit cand 1.0) cand)
    | None ->
        let dflt = Dtypes.default_float_dtype () in
        (dflt, npf dflt 1.0)
  in
  let wolfe_one a_i phi_i =
    let scaled = npf (dtype_of a_i) c1 *! a_i in
    let rhs = phi_0 +! (conv scaled (dtype_of dphi_0) *! dphi_0) in
    U.greater phi_i rhs
  in
  let wolfe_two dphi_i =
    let rhs = lit dphi_0 (-.c2) *! dphi_0 in
    U.less_equal (U.abs dphi_i) rhs
  in
  let init0 = if fresh then 1 else 0 in
  let init =
    [
      bool_v false;
      bool_v false;
      py_int 1;
      npf a_dt 0.0;
      phi_0;
      dphi_0;
      py_int init0;
      py_int init0;
      npf dt 0.0;
      phi_0;
      dphi_0;
      gfk;
    ]
  in
  let body = function
    | [
        done_;
        failed;
        i;
        a_i1;
        phi_i1;
        dphi_i1;
        nfev;
        ngev;
        a_star;
        phi_star;
        dphi_star;
        g_star;
      ] ->
        let first = U.equal i (py_int 1) in
        let doubled = a_i1 *! lit a_i1 2.0 in
        let a_i = LN.where_ first start_value doubled in
        let phi_i, dphi_i, g_i = restricted a_i in
        let nfev = nfev +! py_int 1 in
        let ngev = ngev +! py_int 1 in
        let w1 = wolfe_one a_i phi_i in
        let p0 = U.greater_equal phi_i phi_i1 in
        let p1 = U.greater i (py_int 1) in
        let star_to_zoom1 = bor w1 (band p0 p1) in
        let w2 = wolfe_two dphi_i in
        let star_to_i = band w2 (bnot star_to_zoom1) in
        let d0 = U.greater_equal dphi_i (lit dphi_i 0.0) in
        let d1 = band d0 (bnot star_to_zoom1) in
        let star_to_zoom2 = band d1 (bnot star_to_i) in
        let zoom1 =
          zoom restricted wolfe_one wolfe_two a_i1 phi_i1 dphi_i1 a_i phi_i
            dphi_i gfk (bnot star_to_zoom1)
        in
        let nfev = nfev +! zoom1.z_nfev in
        let ngev = ngev +! zoom1.z_ngev in
        let zoom2 =
          zoom restricted wolfe_one wolfe_two a_i phi_i dphi_i a_i1 phi_i1
            dphi_i1 gfk (bnot star_to_zoom2)
        in
        let nfev = nfev +! zoom2.z_nfev in
        let ngev = ngev +! zoom2.z_ngev in
        let w = LN.where_ in
        let done_ = bor star_to_zoom1 done_ in
        let failed = bor (band star_to_zoom1 zoom1.z_failed) failed in
        let a_star = w star_to_zoom1 zoom1.z_a_star a_star in
        let phi_star = w star_to_zoom1 zoom1.z_phi_star phi_star in
        let dphi_star = w star_to_zoom1 zoom1.z_dphi_star dphi_star in
        let g_star = w star_to_zoom1 zoom1.z_g_star g_star in
        let done_ = bor star_to_i done_ in
        let a_star = w star_to_i (conv a_i dt) a_star in
        let phi_star = w star_to_i phi_i phi_star in
        let dphi_star = w star_to_i dphi_i dphi_star in
        let g_star = w star_to_i g_i g_star in
        let done_ = bor star_to_zoom2 done_ in
        let failed = bor (band star_to_zoom2 zoom2.z_failed) failed in
        let a_star = w star_to_zoom2 zoom2.z_a_star a_star in
        let phi_star = w star_to_zoom2 zoom2.z_phi_star phi_star in
        let dphi_star = w star_to_zoom2 zoom2.z_dphi_star dphi_star in
        let g_star = w star_to_zoom2 zoom2.z_g_star g_star in
        [
          done_;
          failed;
          i +! py_int 1;
          a_i;
          phi_i;
          dphi_i;
          nfev;
          ngev;
          a_star;
          phi_star;
          dphi_star;
          g_star;
        ]
    | _ -> assert false
  in
  let cond = function
    | done_ :: failed :: i :: _ ->
        let c = band (bnot done_) (U.less_equal i (py_int maxiter)) in
        band c (bnot failed)
    | _ -> assert false
  in
  match L.while_loop cond body init with
  | [ done_; failed; i; _; _; _; nfev; ngev; a_star; phi_star; _; g_star ] ->
      let status =
        LN.where_ failed (py_int 1)
          (LN.where_ (U.greater i (py_int maxiter)) (py_int 3) (py_int 0))
      in
      let alpha_k = a_star in
      let small = U.less (U.abs alpha_k) (lit alpha_k 1e-08) in
      let not64 = bool_v ((Dtypes.finfo (dtype_of alpha_k)).bits <> 64) in
      let alpha_k =
        LN.where_ (band not64 small)
          (U.sign alpha_k *! lit alpha_k 1e-08)
          alpha_k
      in
      {
        v_failed = bor failed (bnot done_);
        v_nit = i -! py_int 1;
        v_nfev = nfev;
        v_ngev = ngev;
        v_k = i;
        v_a_k = alpha_k;
        v_f_k = phi_star;
        v_g_k = g_star;
        v_status = status;
      }
  | _ -> assert false

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

let to_int v = int_of_float (Nd.get_f (concrete v) [||])
let to_bool v = Nd.get_f (concrete v) [||] <> 0.0

let line_search f xk pk ?old_fval ?old_old_fval ?gfk ?c1 ?c2 ?maxiter () =
  let r =
    line_search_values f xk pk ?old_fval ?old_old_fval ?gfk ?c1 ?c2 ?maxiter ()
  in
  {
    failed = to_bool r.v_failed;
    nit = to_int r.v_nit;
    nfev = to_int r.v_nfev;
    ngev = to_int r.v_ngev;
    k = to_int r.v_k;
    a_k = r.v_a_k;
    f_k = r.v_f_k;
    g_k = r.v_g_k;
    status = to_int r.v_status;
  }
