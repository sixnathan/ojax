module T = Types
module C = Core
module D = Dtype
module Nd = Ndarray
module U = Numpy.Ufuncs
module LN = Numpy.Lax_numpy
module TC = Numpy.Tensor_contractions
module AC = Numpy.Array_creation
module IDX = Numpy.Indexing
module LL = Lax.Linalg
module NLIN = Numpy.Linalg

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let shape v = (get_aval v).T.shape

let is_inexact = function
  | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> true
  | _ -> false

let to_inexact_dtype dt =
  if is_inexact dt then dt else Dtypes.default_float_dtype ()

let float_scalar dt x = AC.full ~dtype:dt [||] x
let int_scalar i = AC.full ~dtype:D.I32 [||] (float_of_int i)
let identity_op x = x

let concrete = function
  | T.Concrete nd -> nd
  | _ -> failwith "sparse.linalg: value not concrete"

let scalar_float v = Nd.get_f (concrete v) [||]
let eps_of = function D.F64 | D.Complex128 -> 0x1p-52 | _ -> 0x1p-23

let normalize_matvec f =
  let a = concrete f in
  let sh = Nd.shape a in
  if Array.length sh <> 2 || sh.(0) <> sh.(1) then
    invalid_arg
      (Printf.sprintf
         "linear operator must be a square matrix, but has shape: %s"
         (String.concat "x" (Array.to_list (Array.map string_of_int sh))));
  fun x -> TC.matmul f x

let vdot_real x y =
  let base = TC.vdot (U.real x) (U.real y) in
  if LN.iscomplexobj x || LN.iscomplexobj y then
    U.add base (TC.vdot (U.imag x) (U.imag y))
  else base

let vdot_ x y = TC.vdot x y
let mul s t = U.multiply s t
let add_ a b = U.add a b
let sub_ a b = U.subtract a b
let norm_ x = U.sqrt (vdot_real x x)

let fori_loop lower upper body init =
  let cond vs =
    match vs with k :: _ -> U.less k (int_scalar upper) | [] -> assert false
  in
  let step vs =
    match vs with
    | k :: carry -> U.add k (int_scalar 1) :: body k carry
    | [] -> assert false
  in
  match Lax.while_loop cond step (int_scalar lower :: init) with
  | _ :: carry -> carry
  | [] -> assert false

let cg_solve matvec b x0 ~maxiter ~tol ~atol m_op =
  let bs = vdot_real b b in
  let atol2 =
    U.maximum
      (mul (U.square (float_scalar (dtype b) tol)) bs)
      (U.square (float_scalar (dtype b) atol))
  in
  let r0 = sub_ b (matvec x0) in
  let z0 = m_op r0 in
  let p0 = z0 in
  let dt = dtype p0 in
  let gamma0 = LN.astype (vdot_real r0 z0) dt in
  let maxiter_v = int_scalar maxiter in
  let cond vs =
    match vs with
    | [ _x; r; gamma; _p; k ] ->
        let rs = if m_op == identity_op then U.real gamma else vdot_real r r in
        U.logical_and (U.greater rs atol2) (U.less k maxiter_v)
    | _ -> assert false
  in
  let body vs =
    match vs with
    | [ x; r; gamma; p; k ] ->
        let ap = matvec p in
        let alpha = U.divide gamma (LN.astype (vdot_real p ap) dt) in
        let x_ = add_ x (mul alpha p) in
        let r_ = sub_ r (mul alpha ap) in
        let z_ = m_op r_ in
        let gamma_ = LN.astype (vdot_real r_ z_) dt in
        let beta_ = U.divide gamma_ gamma in
        let p_ = add_ z_ (mul beta_ p) in
        [ x_; r_; gamma_; p_; U.add k (int_scalar 1) ]
    | _ -> assert false
  in
  match Lax.while_loop cond body [ x0; r0; gamma0; p0; int_scalar 0 ] with
  | x_final :: _ -> x_final
  | [] -> assert false

let bicgstab_solve matvec b x0 ~maxiter ~tol ~atol m_op =
  let bs = vdot_real b b in
  let atol2 =
    U.maximum
      (mul (U.square (float_scalar (dtype b) tol)) bs)
      (U.square (float_scalar (dtype b) atol))
  in
  let dt = to_inexact_dtype (dtype b) in
  let one = float_scalar dt 1.0 in
  let zero = float_scalar dt 0.0 in
  let r0 = sub_ b (matvec x0) in
  let maxiter_v = int_scalar maxiter in
  let cond vs =
    match vs with
    | [ _x; r; _rhat; _alpha; _omega; _rho; _p; _q; k ] ->
        let rs = vdot_real r r in
        U.logical_and
          (U.logical_and (U.greater rs atol2) (U.less k maxiter_v))
          (U.greater_equal k (int_scalar 0))
    | _ -> assert false
  in
  let body vs =
    match vs with
    | [ x; r; rhat; alpha; omega; rho; p; q; k ] ->
        let rho_ = vdot_ rhat r in
        let beta = U.multiply (U.divide (U.divide rho_ rho) omega) alpha in
        let p_ = add_ r (mul beta (sub_ p (mul omega q))) in
        let phat = m_op p_ in
        let q_ = matvec phat in
        let alpha_ = U.divide rho_ (vdot_ rhat q_) in
        let s = sub_ r (mul alpha_ q_) in
        let exit_early = U.less (vdot_real s s) atol2 in
        let shat = m_op s in
        let t = matvec shat in
        let omega_ = U.divide (vdot_ t s) (vdot_ t t) in
        let x_ =
          LN.where_ exit_early
            (add_ x (mul alpha_ phat))
            (add_ x (add_ (mul alpha_ phat) (mul omega_ shat)))
        in
        let r_ = LN.where_ exit_early s (sub_ s (mul omega_ t)) in
        let k_ =
          LN.where_
            (U.logical_or (U.equal omega_ zero) (U.equal alpha_ zero))
            (int_scalar (-11))
            (U.add k (int_scalar 1))
        in
        let k_ = LN.where_ (U.equal rho_ zero) (int_scalar (-10)) k_ in
        [ x_; r_; rhat; alpha_; omega_; rho_; p_; q_; k_ ]
    | _ -> assert false
  in
  let init = [ x0; r0; r0; one; one; one; r0; r0; int_scalar 0 ] in
  match Lax.while_loop cond body init with
  | x_final :: _ -> x_final
  | [] -> assert false

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

let set_col mat k col =
  let a = concrete mat and c = concrete col in
  let sh = Nd.shape a in
  let flat = to_flat a in
  for i = 0 to sh.(0) - 1 do
    flat.((i * sh.(1)) + k) <- Nd.get_f c [| i |]
  done;
  T.Concrete (Nd.of_floats (Nd.dtype a) sh flat)

let set_row mat k row =
  let a = concrete mat and r = concrete row in
  let sh = Nd.shape a in
  let flat = to_flat a in
  for j = 0 to sh.(1) - 1 do
    flat.((k * sh.(1)) + j) <- Nd.get_f r [| j |]
  done;
  T.Concrete (Nd.of_floats (Nd.dtype a) sh flat)

let set_elem vec k x =
  let a = concrete vec in
  let flat = to_flat a in
  flat.(k) <- scalar_float x;
  T.Concrete (Nd.of_floats (Nd.dtype a) (Nd.shape a) flat)

let get_col mat k =
  let a = concrete mat in
  let sh = Nd.shape a in
  T.Concrete
    (Nd.of_floats (Nd.dtype a)
       [| sh.(0) |]
       (Array.init sh.(0) (fun i -> Nd.get_f a [| i; k |])))

let get_row mat k =
  let a = concrete mat in
  let sh = Nd.shape a in
  T.Concrete
    (Nd.of_floats (Nd.dtype a)
       [| sh.(1) |]
       (Array.init sh.(1) (fun j -> Nd.get_f a [| k; j |])))

let first_cols mat ncol =
  IDX.take ~axis:1 mat (LN.arange ~dtype:D.I32 (float_of_int ncol))

let first_elems vec ncol =
  IDX.take vec (LN.arange ~dtype:D.I32 (float_of_int ncol))

let dyn_index vec i =
  LN.reshape
    (C.bind1 (T.Dynamic_slice { slice_sizes = [| 1 |] }) [ vec; i ])
    [||]

let dyn_set vec i x =
  C.bind1 T.Dynamic_update_slice [ vec; LN.reshape x [| 1 |]; i ]

let dyn_row mat i =
  let cols = (shape mat).(1) in
  LN.reshape
    (C.bind1
       (T.Dynamic_slice { slice_sizes = [| 1; cols |] })
       [ mat; i; int_scalar 0 ])
    [| cols |]

let safe_normalize ?thresh x =
  let dt = dtype x in
  let norm = norm_ x in
  let thr =
    match thresh with Some t -> t | None -> float_scalar dt (eps_of dt)
  in
  let thr = U.real (LN.astype thr dt) in
  let use_norm = U.greater norm thr in
  let norm_cast = LN.astype norm dt in
  let zero = float_scalar dt 0.0 in
  let normalized = LN.where_ use_norm (U.divide x norm_cast) zero in
  let norm = LN.where_ use_norm norm zero in
  (normalized, norm)

let project_on_columns q x = TC.matmul (U.conj (LN.matrix_transpose q)) x

let gram_schmidt q_mat x =
  let h = project_on_columns q_mat x in
  let qh = TC.matmul q_mat h in
  (sub_ x qh, h)

let kth_arnoldi k a_op m_op vmat hmat =
  let dt = dtype vmat in
  let v = get_col vmat k in
  let v = m_op (a_op v) in
  let _, v_norm_0 = safe_normalize v in
  let v, h = gram_schmidt vmat v in
  let tol = mul (float_scalar dt (eps_of dt)) v_norm_0 in
  let unit_v, v_norm_1 = safe_normalize ~thresh:tol v in
  let vmat = set_col vmat (k + 1) unit_v in
  let h = set_elem h (k + 1) (LN.astype v_norm_1 dt) in
  let hmat = set_row hmat k h in
  (vmat, hmat, scalar_float v_norm_1 = 0.0)

let lstsq amat bvec =
  let at = U.conj (LN.matrix_transpose amat) in
  NLIN.solve (TC.matmul at amat) (TC.matmul at bvec)

let init_krylov dt n restart unit_residual =
  set_col (AC.zeros ~dtype:dt [| n; restart + 1 |]) 0 unit_residual

let arnoldi_all a_op m_op vmat hmat restart =
  let rec go vmat hmat k =
    if k >= restart then (vmat, hmat)
    else
      let vmat, hmat, breakdown = kth_arnoldi k a_op m_op vmat hmat in
      if breakdown then (vmat, hmat) else go vmat hmat (k + 1)
  in
  go vmat hmat 0

let gmres_batched a_op m_op b x0 unit_residual residual_norm _ptol restart =
  let dt = dtype b in
  let n = (shape b).(0) in
  let vmat = init_krylov dt n restart unit_residual in
  let hmat = LN.eye ~m:(restart + 1) ~dtype:dt restart in
  let vmat, hmat = arnoldi_all a_op m_op vmat hmat restart in
  let beta =
    set_elem
      (AC.zeros ~dtype:dt [| restart + 1 |])
      0
      (LN.astype residual_norm dt)
  in
  let y = lstsq (LN.matrix_transpose hmat) beta in
  let dx = TC.matmul (first_cols vmat restart) y in
  let x = add_ x0 dx in
  let residual = m_op (sub_ b (a_op x)) in
  let unit_residual, residual_norm = safe_normalize residual in
  (x, unit_residual, residual_norm)

let givens_rotation a b =
  let dt = dtype a in
  let b_zero = U.equal (U.abs b) (float_scalar dt 0.0) in
  let a_lt_b = U.less (U.abs a) (U.abs b) in
  let t = U.divide (U.negative (LN.where_ a_lt_b a b)) (LN.where_ a_lt_b b a) in
  let r =
    LN.astype
      (U.divide (float_scalar dt 1.0)
         (U.sqrt (add_ (float_scalar dt 1.0) (U.square (U.abs t)))))
      dt
  in
  let one = float_scalar dt 1.0 and zero = float_scalar dt 0.0 in
  let cs = LN.where_ b_zero one (LN.where_ a_lt_b (mul r t) r) in
  let sn = LN.where_ b_zero zero (LN.where_ a_lt_b r (mul r t)) in
  (cs, sn)

let rotate_vectors hvec i cs sn =
  let one = int_scalar 1 in
  let x1 = dyn_index hvec i and y1 = dyn_index hvec (U.add i one) in
  let x2 = sub_ (mul (U.conj cs) x1) (mul (U.conj sn) y1) in
  let y2 = add_ (mul sn x1) (mul cs y1) in
  dyn_set (dyn_set hvec i x2) (U.add i one) y2

let apply_givens_rotations h_row givens k =
  let body i carry =
    match carry with
    | [ hrow ] ->
        let row = dyn_row givens i in
        let cs = dyn_index row (int_scalar 0)
        and sn = dyn_index row (int_scalar 1) in
        [ rotate_vectors hrow i cs sn ]
    | _ -> assert false
  in
  let r_row = List.hd (fori_loop 0 k body [ h_row ]) in
  let cs, sn =
    givens_rotation
      (dyn_index r_row (int_scalar k))
      (dyn_index r_row (int_scalar (k + 1)))
  in
  let givens = set_row givens k (LN.stack [ cs; sn ]) in
  (rotate_vectors r_row (int_scalar k) cs sn, givens)

let gmres_incremental a_op m_op b x0 unit_residual residual_norm ptol restart =
  let dt = dtype b in
  let n = (shape b).(0) in
  let vmat = init_krylov dt n restart unit_residual in
  let rmat = LN.eye ~m:(restart + 1) ~dtype:dt restart in
  let givens = AC.zeros ~dtype:dt [| restart; 2 |] in
  let beta_vec =
    set_elem
      (AC.zeros ~dtype:dt [| restart + 1 |])
      0
      (LN.astype residual_norm dt)
  in
  let rec go vmat rmat beta_vec givens k err =
    if k >= restart || not (scalar_float err > scalar_float ptol) then
      (vmat, rmat, beta_vec)
    else
      let vmat, hmat, _ = kth_arnoldi k a_op m_op vmat rmat in
      let r_row, givens = apply_givens_rotations (get_row hmat k) givens k in
      let rmat = set_row hmat k r_row in
      let grow = get_row givens k in
      let beta_vec =
        rotate_vectors beta_vec (int_scalar k)
          (dyn_index grow (int_scalar 0))
          (dyn_index grow (int_scalar 1))
      in
      let err = U.abs (dyn_index beta_vec (int_scalar (k + 1))) in
      go vmat rmat beta_vec givens (k + 1) err
  in
  let vmat, rmat, beta_vec = go vmat rmat beta_vec givens 0 residual_norm in
  let rt = LN.matrix_transpose (first_cols rmat restart) in
  let bcol = LN.reshape (first_elems beta_vec restart) [| restart; 1 |] in
  let y =
    LN.reshape
      (LL.triangular_solve ~left_side:true ~lower:false rt bcol)
      [| restart |]
  in
  let dx = TC.matmul (first_cols vmat restart) y in
  let x = add_ x0 dx in
  let residual = m_op (sub_ b (a_op x)) in
  let unit_residual, residual_norm = safe_normalize residual in
  (x, unit_residual, residual_norm)

let gmres_solve a_op m_op b x0 ~atol ~ptol ~restart ~maxiter gmres_func =
  let residual = m_op (sub_ b (a_op x0)) in
  let unit_residual, residual_norm = safe_normalize residual in
  let rec go x unit_residual residual_norm k =
    if k >= maxiter || not (scalar_float residual_norm > scalar_float atol) then
      x
    else
      let x, unit_residual, residual_norm =
        gmres_func a_op m_op b x unit_residual residual_norm ptol restart
      in
      go x unit_residual residual_norm (k + 1)
  in
  go x0 unit_residual residual_norm 0

let gmres ?(tol = 1e-5) ?(atol = 0.0) ?(restart = 20) ?maxiter ?m
    ?(solve_method = "batched") ?x0 a b =
  let x0 = match x0 with Some x -> x | None -> AC.zeros_like b in
  let m_op =
    match m with Some mm -> normalize_matvec mm | None -> identity_op
  in
  let a_op = normalize_matvec a in
  let size = Array.fold_left ( * ) 1 (shape b) in
  let maxiter = match maxiter with Some m -> m | None -> 10 * size in
  let restart = min restart size in
  if shape x0 <> shape b then invalid_arg "x0 and b must have matching shapes";
  let dt = dtype b in
  let b_norm = norm_ b in
  let atol_v =
    U.maximum (mul (float_scalar dt tol) b_norm) (float_scalar dt atol)
  in
  let mb_norm = norm_ (m_op b) in
  let ptol =
    mul mb_norm (U.minimum (float_scalar dt 1.0) (U.divide atol_v b_norm))
  in
  let gmres_func =
    match solve_method with
    | "incremental" -> gmres_incremental
    | "batched" -> gmres_batched
    | s -> invalid_arg ("invalid solve_method " ^ s)
  in
  let x =
    gmres_solve a_op m_op b x0 ~atol:atol_v ~ptol ~restart ~maxiter gmres_func
  in
  let info = LN.where_ (U.isnan (norm_ x)) (int_scalar (-1)) (int_scalar 0) in
  (x, Some info)

let isolve solver a b x0 ~tol ~atol ~maxiter m =
  let x0 = match x0 with Some x -> x | None -> AC.zeros_like b in
  let size = Array.fold_left ( * ) 1 (shape b) in
  let maxiter = match maxiter with Some m -> m | None -> 10 * size in
  let m_op =
    match m with Some mm -> normalize_matvec mm | None -> identity_op
  in
  let matvec = normalize_matvec a in
  if shape x0 <> shape b then
    invalid_arg "arrays in x0 and b must have matching shapes";
  solver matvec b x0 ~maxiter ~tol ~atol m_op

let cg ?(tol = 1e-5) ?(atol = 0.0) ?maxiter ?m ?x0 a b =
  (isolve cg_solve a b x0 ~tol ~atol ~maxiter m, None)

let bicgstab ?(tol = 1e-5) ?(atol = 0.0) ?maxiter ?m ?x0 a b =
  (isolve bicgstab_solve a b x0 ~tol ~atol ~maxiter m, None)
