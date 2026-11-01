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
module AC = Numpy.Array_creation
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

let conv v dt = if dtype_of v = dt then v else LN.astype v dt

type lbfgs_results = {
  converged : bool;
  failed : bool;
  k : int;
  nfev : int;
  ngev : int;
  x_k : T.value;
  f_k : T.value;
  g_k : T.value;
  s_history : T.value;
  y_history : T.value;
  rho_history : T.value;
  gamma : T.value;
  status : int;
  ls_status : int;
}

let dyn_row mat i =
  let cols = (shape_of mat).(1) in
  let sl =
    C.bind1
      (T.Dynamic_slice { slice_sizes = [| 1; cols |] })
      [ mat; i; py_int 0 ]
  in
  LN.reshape sl [| cols |]

let dyn_elem v i =
  LN.reshape (C.bind1 (T.Dynamic_slice { slice_sizes = [| 1 |] }) [ v; i ]) [||]

let set_elem v i x =
  C.bind1 T.Dynamic_update_slice [ v; LN.reshape x [| 1 |]; i ]

let two_loop_recursion ~k ~g_k ~s_history ~y_history ~rho_history ~gamma =
  let dt = dtype_of rho_history in
  let his_size = (shape_of rho_history).(0) in
  let curr_size = LN.where_ (U.less k (py_int his_size)) k (py_int his_size) in
  let q = U.negative (U.conj g_k) in
  let a_his = AC.zeros_like rho_history in
  let looped =
    L.fori_loop (py_int 0) curr_size
      (fun j state ->
        match state with
        | [ q; a_his ] ->
            let i = py_int (his_size - 1) -! j in
            let rho = dyn_elem rho_history i in
            let d = dot_ (U.conj (dyn_row s_history i)) q in
            let a_i = rho *! conv (U.real d) dt in
            let a_his = set_elem a_his i a_i in
            let q = q -! (a_i *! U.conj (dyn_row y_history i)) in
            [ q; a_his ]
        | _ -> assert false)
      [ q; a_his ]
  in
  let q, a_his = match looped with [ q; a ] -> (q, a) | _ -> assert false in
  let q = gamma *! q in
  List.hd
    (L.fori_loop (py_int 0) curr_size
       (fun j state ->
         match state with
         | [ q ] ->
             let i = py_int his_size -! curr_size +! j in
             let rho = dyn_elem rho_history i in
             let d = dot_ (dyn_row y_history i) q in
             let b_i = rho *! conv (U.real d) dt in
             let c = dyn_elem a_his i -! b_i in
             [ q +! (c *! dyn_row s_history i) ]
         | _ -> assert false)
       [ q ])

let update_vectors hist fresh =
  let rolled = LN.roll ~axis:[| 0 |] hist [| -1 |] in
  let n = (shape_of hist).(0) in
  C.bind1 T.Dynamic_update_slice
    [ rolled; L.expand_dims fresh [| 0 |]; py_int (n - 1); py_int 0 ]

let update_scalars hist fresh =
  let rolled = LN.roll ~axis:[| 0 |] hist [| -1 |] in
  let n = (shape_of hist).(0) in
  set_elem rolled (py_int (n - 1)) fresh

let minimize_lbfgs f x0 ?maxiter ?(norm = infinity) ?(maxcor = 10)
    ?(ftol = 2.220446049250313e-09) ?(gtol = 1e-5) ?maxfun ?maxgrad
    ?(maxls = 20) () =
  let d = (shape_of x0).(0) in
  let dt = dtype_of x0 in
  let maxiter =
    match (maxiter, maxfun, maxgrad) with
    | None, None, None -> Some (d * 200)
    | m, _, _ -> m
  in
  let f_0, g_0 = LS.value_and_grad f x0 in
  let gdt = Dtypes.default_float_dtype () in
  let init =
    [
      bool_v false;
      bool_v false;
      py_int 0;
      py_int 1;
      py_int 1;
      x0;
      f_0;
      g_0;
      AC.zeros ~dtype:dt [| maxcor; d |];
      AC.zeros ~dtype:dt [| maxcor; d |];
      AC.zeros ~dtype:dt [| maxcor |];
      T.Concrete (Nd.of_floats gdt [||] [| 1.0 |]);
      py_int 0;
      py_int 0;
    ]
  in
  let init =
    List.mapi
      (fun i v -> if i = 11 then conv v (Dtypes.to_inexact_dtype dt) else v)
      init
  in
  let cond = function
    | converged :: failed :: _ ->
        band (U.bitwise_not converged) (U.bitwise_not failed)
    | _ -> assert false
  in
  let cap name_opt st_v status code =
    match name_opt with
    | None -> status
    | Some limit ->
        LN.where_ (U.greater_equal st_v (py_int limit)) (py_int code) status
  in
  let body = function
    | [
        _;
        _;
        k;
        nfev;
        ngev;
        x_k;
        f_k;
        g_k;
        s_history;
        y_history;
        rho_history;
        gamma;
        _;
        _;
      ] ->
        let p_k =
          two_loop_recursion ~k ~g_k ~s_history ~y_history ~rho_history ~gamma
        in
        let ls =
          LS.line_search_values f x_k p_k ~old_fval:f_k ~gfk:g_k ~maxiter:maxls
            ()
        in
        let s_k = conv ls.LS.v_a_k (dtype_of p_k) *! p_k in
        let x_kp1 = x_k +! s_k in
        let f_kp1 = ls.LS.v_f_k in
        let g_kp1 = ls.LS.v_g_k in
        let y_k = g_kp1 -! g_k in
        let rho_k_inv = U.real (dot_ y_k s_k) in
        let rho_k = conv (U.reciprocal rho_k_inv) (dtype_of y_k) in
        let denom = U.real (dot_ (U.conj y_k) y_k) in
        let new_gamma = U.true_divide rho_k_inv denom in
        let status = py_int 0 in
        let dec = f_k -! f_kp1 in
        let status = LN.where_ (U.less dec (lit dec ftol)) (py_int 4) status in
        let status = cap maxgrad ngev status 3 in
        let status = cap maxfun nfev status 2 in
        let status = cap maxiter k status 1 in
        let status = LN.where_ ls.LS.v_failed (py_int 5) status in
        let n = NLIN.norm ~ord:(NLIN.Onum norm) g_kp1 in
        let converged = U.less n (lit n gtol) in
        let failed =
          band (U.greater status (py_int 0)) (U.bitwise_not converged)
        in
        [
          converged;
          failed;
          k +! py_int 1;
          nfev +! ls.LS.v_nfev;
          ngev +! ls.LS.v_ngev;
          conv x_kp1 (dtype_of x_k);
          conv f_kp1 (dtype_of f_k);
          conv g_kp1 (dtype_of g_k);
          update_vectors s_history s_k;
          update_vectors y_history y_k;
          update_scalars rho_history rho_k;
          conv new_gamma (dtype_of gamma);
          LN.where_ converged (py_int 0) status;
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
   x_k;
   f_k;
   g_k;
   s_history;
   y_history;
   rho_history;
   gamma;
   status;
   ls_status;
  ] ->
      {
        converged = LS.to_bool converged;
        failed = LS.to_bool failed;
        k = LS.to_int k;
        nfev = LS.to_int nfev;
        ngev = LS.to_int ngev;
        x_k;
        f_k;
        g_k;
        s_history;
        y_history;
        rho_history;
        gamma;
        status = LS.to_int status;
        ls_status = LS.to_int ls_status;
      }
  | _ -> assert false
