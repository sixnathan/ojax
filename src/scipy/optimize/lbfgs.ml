module T = Types
module C = Core
module D = Dtype
module Nd = Ndarray
module U = Numpy.Ufuncs
module LN = Numpy.Lax_numpy
module TC = Numpy.Tensor_contractions
module NLIN = Numpy.Linalg
module AC = Numpy.Array_creation
module API = Api
module LS = Line_search

let leaf = function
  | Tree_util.Leaf v -> v
  | _ -> failwith "optimize: expected leaf"

let concrete = LS.concrete
let aval v = C.get_aval v
let dtype_of v = (aval v).T.dtype
let f_of v = Nd.get_f (concrete v) [||]
let b_of v = f_of v <> 0.0
let konst dt x = AC.full ~dtype:dt [||] x

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

let to_flat a =
  let total = Array.fold_left ( * ) 1 (Nd.shape a) in
  let buf = Array.make total 0.0 in
  let _ =
    Nd.fold
      (fun i x ->
        buf.(i) <- x;
        i + 1)
      0 a
  in
  buf

let get_row2 mat r =
  let a = concrete mat in
  let sh = Nd.shape a in
  T.Concrete
    (Nd.of_floats (Nd.dtype a)
       [| sh.(1) |]
       (Array.init sh.(1) (fun j -> Nd.get_f a [| r; j |])))

let set_row2 mat r row =
  let a = concrete mat and rw = concrete row in
  let sh = Nd.shape a in
  let flat = to_flat a in
  for j = 0 to sh.(1) - 1 do
    flat.((r * sh.(1)) + j) <- Nd.get_f rw [| j |]
  done;
  T.Concrete (Nd.of_floats (Nd.dtype a) sh flat)

let get_elem1 vec i =
  let a = concrete vec in
  T.Concrete (Nd.of_floats (Nd.dtype a) [||] [| Nd.get_f a [| i |] |])

let set_elem1 vec i x =
  let a = concrete vec in
  let flat = to_flat a in
  flat.(i) <- f_of x;
  T.Concrete (Nd.of_floats (Nd.dtype a) (Nd.shape a) flat)

let update_history_vectors history new_row =
  let rolled = LN.roll ~axis:[| 0 |] history [| -1 |] in
  let rows = (aval history).T.shape.(0) in
  set_row2 rolled (rows - 1) new_row

let update_history_scalars history new_val =
  let rolled = LN.roll ~axis:[| 0 |] history [| -1 |] in
  let n = (aval history).T.shape.(0) in
  set_elem1 rolled (n - 1) new_val

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

let two_loop_recursion st his_size =
  let dt = dtype_of st.rho_history in
  let curr_size = if st.k < his_size then st.k else his_size in
  let q = ref (U.negative (U.conj st.g_k)) in
  let a_his = ref (AC.zeros_like st.rho_history) in
  for j = 0 to curr_size - 1 do
    let i = his_size - 1 - j in
    let s_i = get_row2 st.s_history i in
    let y_i = get_row2 st.y_history i in
    let rho_i = get_elem1 st.rho_history i in
    let a_i =
      U.multiply rho_i (LN.astype (U.real (TC.dot (U.conj s_i) !q)) dt)
    in
    a_his := set_elem1 !a_his i a_i;
    q := U.subtract !q (U.multiply a_i (U.conj y_i))
  done;
  q := U.multiply st.gamma !q;
  for j = 0 to curr_size - 1 do
    let i = his_size - curr_size + j in
    let y_i = get_row2 st.y_history i in
    let s_i = get_row2 st.s_history i in
    let rho_i = get_elem1 st.rho_history i in
    let b_i = U.multiply rho_i (LN.astype (U.real (TC.dot y_i !q)) dt) in
    let a_i = get_elem1 !a_his i in
    q := U.add !q (U.multiply (U.subtract a_i b_i) s_i)
  done;
  !q

let minimize_lbfgs f x0 ?maxiter ?(norm = infinity) ?(maxcor = 10)
    ?(ftol = 2.220446049250313e-09) ?(gtol = 1e-05) ?maxfun ?maxgrad
    ?(maxls = 20) () =
  let d = (aval x0).T.shape.(0) in
  let dt = dtype_of x0 in
  let maxiter =
    match (maxiter, maxfun, maxgrad) with
    | None, None, None -> float_of_int (d * 200)
    | Some m, _, _ -> float_of_int m
    | None, _, _ -> infinity
  in
  let maxfun =
    match maxfun with Some m -> float_of_int m | None -> infinity
  in
  let maxgrad =
    match maxgrad with Some m -> float_of_int m | None -> infinity
  in
  let f_0, g_0 = vg f x0 in
  let ord = Some norm in
  let init =
    {
      converged = false;
      failed = false;
      k = 0;
      nfev = 1;
      ngev = 1;
      x_k = x0;
      f_k = f_0;
      g_k = g_0;
      s_history = AC.zeros ~dtype:dt [| maxcor; d |];
      y_history = AC.zeros ~dtype:dt [| maxcor; d |];
      rho_history = AC.zeros ~dtype:dt [| maxcor |];
      gamma = konst dt 1.0;
      status = 0;
      ls_status = 0;
    }
  in
  let body st =
    let p_k = two_loop_recursion st maxcor in
    let ls =
      LS.line_search f st.x_k p_k ~old_fval:st.f_k ~gfk:st.g_k ~maxiter:maxls ()
    in
    let s_k = U.multiply (LN.astype ls.LS.a_k (dtype_of p_k)) p_k in
    let x_kp1 = U.add st.x_k s_k in
    let f_kp1 = ls.LS.f_k in
    let g_kp1 = ls.LS.g_k in
    let y_k = U.subtract g_kp1 st.g_k in
    let rho_k_inv = U.real (TC.dot y_k s_k) in
    let rho_k = LN.astype (U.reciprocal rho_k_inv) (dtype_of y_k) in
    let gamma = U.divide rho_k_inv (U.real (TC.dot (U.conj y_k) y_k)) in
    let status = ref 0 in
    if b_of (U.less (U.subtract st.f_k f_kp1) (konst dt ftol)) then status := 4;
    if float_of_int st.ngev >= maxgrad then status := 3;
    if float_of_int st.nfev >= maxfun then status := 2;
    if float_of_int st.k >= maxiter then status := 1;
    if ls.LS.failed then status := 5;
    let converged = b_of (U.less (norm_ord ord g_kp1) (konst dt gtol)) in
    {
      converged;
      failed = !status > 0 && not converged;
      k = st.k + 1;
      nfev = st.nfev + ls.LS.nfev;
      ngev = st.ngev + ls.LS.ngev;
      x_k = LN.astype x_kp1 (dtype_of st.x_k);
      f_k = LN.astype f_kp1 (dtype_of st.f_k);
      g_k = LN.astype g_kp1 (dtype_of st.g_k);
      s_history = update_history_vectors st.s_history s_k;
      y_history = update_history_vectors st.y_history y_k;
      rho_history = update_history_scalars st.rho_history rho_k;
      gamma = LN.astype gamma (dtype_of st.g_k);
      status = (if converged then 0 else !status);
      ls_status = ls.LS.status;
    }
  in
  let rec loop st =
    if (not st.converged) && not st.failed then loop (body st) else st
  in
  loop init
