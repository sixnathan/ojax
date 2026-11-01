module T = Types
module C = Core
module D = Dtype
module Nd = Ndarray
module U = Numpy.Ufuncs
module LN = Numpy.Lax_numpy
module TC = Numpy.Tensor_contractions
module NLIN = Numpy.Linalg
module JU = Numpy.Jnp_util
module L = Lax_api
module LS = Line_search

let dtype_of v = (C.get_aval v).T.dtype
let shape_of v = (C.get_aval v).T.shape
let lit v x = T.Concrete (Nd.of_floats (dtype_of v) [||] [| x |])
let py_int = JU.py_int
let ( +! ) = U.add
let ( -! ) = U.subtract
let ( *! ) = U.multiply
let dot_ a b = TC.dot ~precision:LS.highest a b
let band = U.bitwise_and

let bool_v b =
  T.Concrete (Nd.of_floats D.Bool [||] [| (if b then 1.0 else 0.0) |])

type bfgs_results = {
  converged : bool;
  failed : bool;
  k : int;
  nfev : int;
  ngev : int;
  nhev : int;
  x_k : T.value;
  f_k : T.value;
  g_k : T.value;
  h_k : T.value;
  old_old_fval : T.value;
  status : int;
  line_search_status : int;
}

let norm_ord norm g =
  if norm = infinity then NLIN.norm ~ord:(NLIN.Onum infinity) g
  else NLIN.norm ~ord:(NLIN.Onum norm) g

let minimize_bfgs f x0 ?maxiter ?(norm = infinity) ?(gtol = 1e-5)
    ?(line_search_maxiter = 10) () =
  let size = Array.fold_left ( * ) 1 (shape_of x0) in
  let maxiter = match maxiter with Some m -> m | None -> size * 200 in
  let d = (shape_of x0).(0) in
  let initial_h = LN.eye ~dtype:(dtype_of x0) d in
  let f_0, g_0 = LS.value_and_grad f x0 in
  let n0 = norm_ord norm g_0 in
  let converged0 = U.less n0 (lit n0 gtol) in
  let n1 = NLIN.norm g_0 in
  let old_old_fval0 = f_0 +! U.true_divide n1 (lit n1 2.0) in
  let init =
    [
      converged0;
      bool_v false;
      py_int 0;
      py_int 1;
      py_int 1;
      py_int 0;
      x0;
      f_0;
      g_0;
      initial_h;
      old_old_fval0;
      py_int 0;
    ]
  in
  let cond = function
    | converged :: failed :: k :: _ ->
        let c = band (U.logical_not converged) (U.logical_not failed) in
        band c (U.less k (py_int maxiter))
    | _ -> assert false
  in
  let body = function
    | [ _; _; k; nfev; ngev; nhev; x_k; f_k; g_k; h_k; old_old_fval; _ ] ->
        let p_k = U.negative (dot_ h_k g_k) in
        let ls =
          LS.line_search_values f x_k p_k ~old_fval:f_k ~old_old_fval ~gfk:g_k
            ~maxiter:line_search_maxiter ()
        in
        let nfev = nfev +! ls.LS.v_nfev in
        let ngev = ngev +! ls.LS.v_ngev in
        let s_k = ls.LS.v_a_k *! p_k in
        let x_kp1 = x_k +! s_k in
        let f_kp1 = ls.LS.v_f_k in
        let g_kp1 = ls.LS.v_g_k in
        let y_k = g_kp1 -! g_k in
        let rho_k = U.reciprocal (dot_ y_k s_k) in
        let sc = L.expand_dims s_k [| 1 |] in
        let yc = L.expand_dims y_k [| 0 |] in
        let sy_k = sc *! yc in
        let eye = LN.eye ~dtype:(dtype_of rho_k) d in
        let w = eye -! (rho_k *! sy_k) in
        let e =
          Numpy.Einsum.einsum ~precision:LS.highest "ij,jk,lk" [ w; h_k; w ]
        in
        let t = rho_k *! L.expand_dims s_k [| 1 |] in
        let h_new = e +! (t *! L.expand_dims s_k [| 0 |]) in
        let h_kp1 = LN.where_ (U.isfinite rho_k) h_new h_k in
        let n = norm_ord norm g_kp1 in
        let converged = U.less n (lit n gtol) in
        [
          converged;
          ls.LS.v_failed;
          k +! py_int 1;
          nfev;
          ngev;
          nhev;
          x_kp1;
          f_kp1;
          g_kp1;
          h_kp1;
          f_k;
          ls.LS.v_status;
        ]
    | _ -> assert false
  in
  match L.while_loop cond body init with
  | [
   converged;
   failed;
   k;
   nfev;
   ngev;
   nhev;
   x_k;
   f_k;
   g_k;
   h_k;
   old_old_fval;
   ls_status;
  ] ->
      let converged = LS.to_bool converged
      and failed = LS.to_bool failed
      and k = LS.to_int k
      and line_search_status = LS.to_int ls_status in
      let status =
        if converged then 0
        else if k = maxiter then 1
        else if failed then 2 + line_search_status
        else -1
      in
      {
        converged;
        failed;
        k;
        nfev = LS.to_int nfev;
        ngev = LS.to_int ngev;
        nhev = LS.to_int nhev;
        x_k;
        f_k;
        g_k;
        h_k;
        old_old_fval;
        status;
        line_search_status;
      }
  | _ -> assert false
