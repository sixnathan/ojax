module T = Types
module C = Core
module D = Dtype
module U = Numpy.Ufuncs
module LN = Numpy.Lax_numpy
module TC = Numpy.Tensor_contractions
module NLIN = Numpy.Linalg
module API = Api
module LS = Line_search

let leaf = function
  | Tree_util.Leaf v -> v
  | _ -> failwith "optimize: expected leaf"

let aval v = C.get_aval v
let dtype_of v = (aval v).T.dtype
let shape_of v = (aval v).T.shape
let f_of v = Ndarray.get_f (LS.concrete v) [||]
let b_of v = f_of v <> 0.0
let konst dt x = Numpy.Array_creation.full ~dtype:dt [||] x

let vg f x =
  let fv, gv =
    API.value_and_grad
      (fun xs -> Tree_util.Leaf (f (leaf (List.hd xs))))
      [ Tree_util.Leaf x ]
  in
  (leaf fv, leaf gv)

let norm_ord ord g =
  match ord with
  | None -> NLIN.norm g
  | Some o -> NLIN.norm ~ord:(NLIN.Onum o) g

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

let minimize_bfgs f x0 ?maxiter ?(norm = infinity) ?(gtol = 1e-5)
    ?(line_search_maxiter = 10) () =
  let size = Array.fold_left ( * ) 1 (shape_of x0) in
  let maxiter = match maxiter with Some m -> m | None -> size * 200 in
  let d = (shape_of x0).(0) in
  let dt = dtype_of x0 in
  let initial_h = LN.eye ~dtype:dt d in
  let f_0, g_0 = vg f x0 in
  let ord = if norm = infinity then Some infinity else Some norm in
  let converged0 = b_of (U.less (norm_ord ord g_0) (konst dt gtol)) in
  let old_old_fval0 = U.add f_0 (U.divide (norm_ord None g_0) (konst dt 2.0)) in
  let init =
    {
      converged = converged0;
      failed = false;
      k = 0;
      nfev = 1;
      ngev = 1;
      nhev = 0;
      x_k = x0;
      f_k = f_0;
      g_k = g_0;
      h_k = initial_h;
      old_old_fval = old_old_fval0;
      status = 0;
      line_search_status = 0;
    }
  in
  let body st =
    let p_k = U.negative (TC.dot st.h_k st.g_k) in
    let ls =
      LS.line_search f st.x_k p_k ~old_fval:st.f_k ~old_old_fval:st.old_old_fval
        ~gfk:st.g_k ~maxiter:line_search_maxiter ()
    in
    let st =
      {
        st with
        nfev = st.nfev + ls.LS.nfev;
        ngev = st.ngev + ls.LS.ngev;
        failed = ls.LS.failed;
        line_search_status = ls.LS.status;
      }
    in
    let s_k = U.multiply ls.LS.a_k p_k in
    let x_kp1 = U.add st.x_k s_k in
    let f_kp1 = ls.LS.f_k in
    let g_kp1 = ls.LS.g_k in
    let y_k = U.subtract g_kp1 st.g_k in
    let rho_k = U.reciprocal (TC.dot y_k s_k) in
    let rdt = dtype_of rho_k in
    let sy_k = TC.outer s_k y_k in
    let w = U.subtract (LN.eye ~dtype:rdt d) (U.multiply rho_k sy_k) in
    let wt = LN.matrix_transpose w in
    let h_new =
      U.add
        (TC.matmul (TC.matmul w st.h_k) wt)
        (U.multiply rho_k (TC.outer s_k s_k))
    in
    let h_kp1 = LN.where_ (U.isfinite rho_k) h_new st.h_k in
    let converged = b_of (U.less (norm_ord ord g_kp1) (konst dt gtol)) in
    {
      st with
      converged;
      k = st.k + 1;
      x_k = x_kp1;
      f_k = f_kp1;
      g_k = g_kp1;
      h_k = h_kp1;
      old_old_fval = st.f_k;
    }
  in
  let rec loop st =
    if (not st.converged) && (not st.failed) && st.k < maxiter then
      loop (body st)
    else st
  in
  let st = loop init in
  let status =
    if st.converged then 0
    else if st.k = maxiter then 1
    else if st.failed then 2 + st.line_search_status
    else -1
  in
  { st with status }
