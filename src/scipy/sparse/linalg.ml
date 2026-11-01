module T = Types
module C = Core
module D = Dtype
module Nd = Ndarray
module U = Numpy.Ufuncs
module LN = Numpy.Lax_numpy
module TC = Numpy.Tensor_contractions
module AC = Numpy.Array_creation
module JU = Numpy.Jnp_util
module L = Lax_api

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let shape v = (get_aval v).T.shape

let highest =
  Lax_config.Precision
    (Lax_config.Precision_highest, Lax_config.Precision_highest)

let dot_ a b = TC.dot ~precision:highest a b
let vdot_ a b = TC.vdot ~precision:highest a b
let lit v x = T.Concrete (Nd.of_floats (dtype v) [||] [| x |])
let py_int = JU.py_int
let py_float = JU.py_float
let identity_op x = x

let is_complex v =
  match dtype v with D.Complex64 | D.Complex128 -> true | _ -> false

let real_part v = if is_complex v then U.real v else v
let eps_of dt = (Dtypes.finfo dt).eps

let normalize_matvec f =
  let sh = shape f in
  if Array.length sh <> 2 || sh.(0) <> sh.(1) then
    invalid_arg
      (Printf.sprintf
         "linear operator must be a square matrix, but has shape: %s"
         (String.concat "x" (Array.to_list (Array.map string_of_int sh))));
  fun x -> dot_ f x

let vdot_real_part x y =
  let base = vdot_ (real_part x) (real_part y) in
  if LN.iscomplexobj x || LN.iscomplexobj y then
    U.add base (vdot_ (U.imag x) (U.imag y))
  else base

let leaf_sum v = U.add (lit v 0.0) v
let vdot_real_tree x y = leaf_sum (vdot_real_part x y)
let vdot_tree x y = leaf_sum (vdot_ x y)
let norm_ x = U.sqrt (leaf_sum (vdot_real_part x x))
let mul s t = U.multiply s t
let add_ a b = U.add a b
let sub_ a b = U.subtract a b

let cg_solve matvec b x0 ~maxiter ~tol ~atol m_op =
  let bs = vdot_real_tree b b in
  let t2 = U.square (py_float tol) in
  let t2 = LN.astype t2 (dtype bs) in
  let atol2 =
    let left = mul t2 bs in
    U.maximum left (LN.astype (U.square (py_float atol)) (dtype bs))
  in
  let r0 = sub_ b (matvec x0) in
  let z0 = m_op r0 in
  let p0 = z0 in
  let dt = dtype p0 in
  let gamma0 = LN.astype (vdot_real_tree r0 z0) dt in
  let maxiter_v = py_int maxiter in
  let cond vs =
    match vs with
    | [ _x; r; gamma; _p; k ] ->
        let rs =
          if m_op == identity_op then real_part gamma else vdot_real_tree r r
        in
        let left = U.greater rs atol2 in
        U.bitwise_and left (U.less k maxiter_v)
    | _ -> assert false
  in
  let body vs =
    match vs with
    | [ x; r; gamma; p; k ] ->
        let ap = matvec p in
        let denom = LN.astype (vdot_real_tree p ap) dt in
        let alpha = U.true_divide gamma denom in
        let x_ = add_ x (mul alpha p) in
        let r_ = sub_ r (mul alpha ap) in
        let z_ = m_op r_ in
        let gamma_ = LN.astype (vdot_real_tree r_ z_) dt in
        let beta_ = U.true_divide gamma_ gamma in
        let p_ = add_ z_ (mul beta_ p) in
        [ x_; r_; gamma_; p_; U.add k (py_int 1) ]
    | _ -> assert false
  in
  List.hd (L.while_loop cond body [ x0; r0; gamma0; p0; py_int 0 ])

let bicgstab_solve matvec b x0 ~maxiter ~tol ~atol m_op =
  let bs = vdot_real_tree b b in
  let t2 = LN.astype (U.square (py_float tol)) (dtype bs) in
  let atol2 =
    let left = mul t2 bs in
    U.maximum left (LN.astype (U.square (py_float atol)) (dtype bs))
  in
  let dt = Dtypes.to_inexact_dtype (dtype b) in
  let one = T.Concrete (Nd.of_floats dt [||] [| 1.0 |]) in
  let r0 = sub_ b (matvec x0) in
  let maxiter_v = py_int maxiter in
  let cond vs =
    match vs with
    | [ _x; r; _rhat; _alpha; _omega; _rho; _p; _q; k ] ->
        let rs = vdot_real_tree r r in
        let c1 = U.greater rs atol2 in
        let c2 = U.less k maxiter_v in
        let c12 = U.bitwise_and c1 c2 in
        U.bitwise_and c12 (U.greater_equal k (py_int 0))
    | _ -> assert false
  in
  let body vs =
    match vs with
    | [ x; r; rhat; alpha; omega; rho; p; q; k ] ->
        let rho_ = vdot_tree rhat r in
        let beta = U.true_divide rho_ rho in
        let beta = U.multiply beta alpha in
        let beta = U.true_divide beta omega in
        let omq = mul omega q in
        let p_ = add_ r (mul beta (sub_ p omq)) in
        let phat = m_op p_ in
        let q_ = matvec phat in
        let alpha_ = U.true_divide rho_ (vdot_tree rhat q_) in
        let s = sub_ r (mul alpha_ q_) in
        let exit_early = U.less (vdot_real_tree s s) atol2 in
        let shat = m_op s in
        let t = matvec shat in
        let tvs = vdot_tree t s in
        let omega_ = U.true_divide tvs (vdot_tree t t) in
        let x1 = add_ x (mul alpha_ phat) in
        let ap = mul alpha_ phat in
        let x2 = add_ x (add_ ap (mul omega_ shat)) in
        let x_ = LN.where_ exit_early x1 x2 in
        let r2 = sub_ s (mul omega_ t) in
        let r_ = LN.where_ exit_early s r2 in
        let c1 = U.equal omega_ (lit omega_ 0.0) in
        let c2 = U.equal alpha_ (lit alpha_ 0.0) in
        let k_ =
          LN.where_ (U.bitwise_or c1 c2) (py_int (-11)) (U.add k (py_int 1))
        in
        let k_ = LN.where_ (U.equal rho_ (lit rho_ 0.0)) (py_int (-10)) k_ in
        [ x_; r_; rhat; alpha_; omega_; rho_; p_; q_; k_ ]
    | _ -> assert false
  in
  let init = [ x0; r0; r0; one; one; one; r0; r0; py_int 0 ] in
  List.hd (L.while_loop cond body init)

let safe_normalize ?thresh x =
  let norm = norm_ x in
  let dt = dtype x in
  let thr =
    match thresh with
    | Some t -> real_part (LN.astype t dt)
    | None ->
        let rd = dtype norm in
        real_part (T.Concrete (Nd.of_floats dt [||] [| eps_of rd |]))
  in
  let use_norm = U.greater norm thr in
  let norm_cast = LN.astype norm dt in
  let normalized =
    JU.where_weak ~y_weak:true use_norm
      (U.true_divide x norm_cast)
      (py_float 0.0)
  in
  let norm = JU.where_weak ~y_weak:true use_norm norm (py_float 0.0) in
  (normalized, norm)

let project_on_columns q v =
  Numpy.Einsum.einsum ~precision:highest "...n,...->n" [ U.conj q; v ]

let gram_schmidt q_mat x xnorm ~max_iterations =
  let q0 = q_mat in
  let r = AC.zeros ~dtype:(dtype q0) [| (shape q0).(1) |] in
  let sqrt2 () = U.sqrt (py_float 2.0) in
  let scaled v = U.true_divide v (LN.astype (sqrt2 ()) (dtype v)) in
  let xnorm_scaled = scaled xnorm in
  let body = function
    | [ k; q; r; qnorm_scaled ] ->
        let h = project_on_columns q_mat q in
        let qh = dot_ q_mat h in
        let q = sub_ q qh in
        let r = add_ r h in
        let inner =
          L.while_loop
            (function
              | [ k; not_done; _; _ ] ->
                  U.logical_and not_done
                    (U.less k (py_int (max_iterations - 1)))
              | _ -> assert false)
            (function
              | [ k; _; q; _ ] ->
                  let _, qnorm = safe_normalize q in
                  [
                    k;
                    T.Concrete (Nd.of_floats D.Bool [||] [| 0.0 |]);
                    q;
                    scaled qnorm;
                  ]
              | _ -> assert false)
            [
              k;
              T.Concrete (Nd.of_floats D.Bool [||] [| 1.0 |]);
              q;
              qnorm_scaled;
            ]
        in
        let q = List.nth inner 2 and qnorm_scaled = List.nth inner 3 in
        [ U.add k (py_int 1); q; r; qnorm_scaled ]
    | _ -> assert false
  in
  let cond = function
    | [ k; _; r; qnorm_scaled ] ->
        let _, rnorm = safe_normalize r in
        U.logical_and
          (U.less k (py_int (max_iterations - 1)))
          (U.less rnorm qnorm_scaled)
    | _ -> assert false
  in
  match body [ py_int 0; x; r; xnorm_scaled ] with
  | init -> (
      match L.while_loop cond body init with
      | [ _; q; r; _ ] -> (q, r)
      | _ -> assert false)

let norm_index i n =
  let i32 = LN.astype i D.I32 in
  let neg = U.less i32 (T.Concrete (Nd.of_floats D.I32 [||] [| 0.0 |])) in
  let shifted =
    U.add i32 (T.Concrete (Nd.of_floats D.I32 [||] [| float_of_int n |]))
  in
  L.select neg shifted i32

let scatter_at mat k update ~window ~inserted =
  let idx = L.broadcast_in_dim (LN.astype k D.I32) [| 1 |] [||] in
  L.scatter ~mode:Lax_config.Fill_or_drop ~unique_indices:true mat idx update
    {
      T.update_window_dims = window;
      inserted_window_dims = [| inserted |];
      scatter_dims_to_operand_dims = [| inserted |];
      s_operand_batching_dims = [||];
      s_scatter_indices_batching_dims = [||];
      s_sorted_unique = false;
    }

let i32_lit v = T.Concrete (Nd.of_floats D.I32 [||] [| float_of_int v |])

let dyn_col mat k =
  let rows = (shape mat).(0) and cols = (shape mat).(1) in
  let k = norm_index k cols in
  let sl =
    C.bind1
      (T.Dynamic_slice { slice_sizes = [| rows; 1 |] })
      [ mat; i32_lit 0; k ]
  in
  L.squeeze sl [| 1 |]

let set_dyn_col mat k col =
  scatter_at mat (norm_index k (shape mat).(1)) col ~window:[| 0 |] ~inserted:1

let dyn_row mat k =
  let rows = (shape mat).(0) and cols = (shape mat).(1) in
  let k = norm_index k rows in
  let sl =
    C.bind1
      (T.Dynamic_slice { slice_sizes = [| 1; cols |] })
      [ mat; k; i32_lit 0 ]
  in
  L.squeeze sl [| 0 |]

let set_dyn_row mat k row =
  scatter_at mat (norm_index k (shape mat).(0)) row ~window:[| 0 |] ~inserted:0

let dyn_elem vec i =
  let i = norm_index i (shape vec).(0) in
  L.squeeze
    (C.bind1 (T.Dynamic_slice { slice_sizes = [| 1 |] }) [ vec; i ])
    [| 0 |]

let set_dyn_elem vec i x =
  scatter_at vec (norm_index i (shape vec).(0)) x ~window:[||] ~inserted:0

let kth_arnoldi k a_op m_op vmat hmat =
  let dt = dtype vmat in
  let eps = eps_of dt in
  let v = dyn_col vmat k in
  let v = m_op (a_op v) in
  let _, v_norm_0 = safe_normalize v in
  let v, h = gram_schmidt vmat v v_norm_0 ~max_iterations:2 in
  let tol = mul (lit v_norm_0 eps) v_norm_0 in
  let unit_v, v_norm_1 = safe_normalize ~thresh:tol v in
  let k1 = U.add k (py_int 1) in
  let vmat = set_dyn_col vmat k1 unit_v in
  let h = set_dyn_elem h k1 (LN.astype v_norm_1 dt) in
  let hmat = set_dyn_row hmat k h in
  let breakdown = U.equal v_norm_1 (lit v_norm_1 0.0) in
  (vmat, hmat, breakdown)

let rotate_vectors h i cs sn =
  let x1 = dyn_elem h i in
  let i1 = U.add i (py_int 1) in
  let y1 = dyn_elem h i1 in
  let t1 = mul (U.conj cs) x1 in
  let x2 = sub_ t1 (mul (U.conj sn) y1) in
  let t2 = mul sn x1 in
  let y2 = add_ t2 (mul cs y1) in
  let h = set_dyn_elem h i x2 in
  set_dyn_elem h i1 y2

let givens_rotation a b =
  let absb = U.abs b in
  let b_zero = U.equal absb (lit absb 0.0) in
  let a_lt_b =
    let aa = U.abs a in
    U.less aa (U.abs b)
  in
  let numer = LN.where_ a_lt_b a b in
  let t = U.true_divide (U.negative numer) (LN.where_ a_lt_b b a) in
  let abst = U.abs t in
  let sq = U.power abst (py_int 2) in
  let r = LN.astype (L.rsqrt (U.add (lit sq 1.0) sq)) (dtype t) in
  let rt = mul r t in
  let cs = LN.where_ b_zero (py_int 1) (LN.where_ a_lt_b rt r) in
  let rt2 = mul r t in
  let sn = LN.where_ b_zero (py_int 0) (LN.where_ a_lt_b r rt2) in
  (cs, sn)

let apply_givens h_row givens k =
  let r_row =
    List.hd
      (L.fori_loop (py_int 0) k
         (fun i state ->
           match state with
           | [ row ] ->
               let g = dyn_row givens i in
               let cs = dyn_elem g (py_int 0) in
               let sn = dyn_elem g (py_int 1) in
               [ rotate_vectors row i cs sn ]
           | _ -> assert false)
         [ h_row ])
  in
  let k1 = U.add k (py_int 1) in
  let a = dyn_elem r_row k in
  let b = dyn_elem r_row k1 in
  let cs, sn = givens_rotation a b in
  let pair = LN.stack ~axis:0 [ cs; sn ] in
  let givens = set_dyn_row givens k pair in
  (rotate_vectors r_row k cs sn, givens)

let init_krylov unit_residual restart =
  let expanded = L.expand_dims unit_residual [| 1 |] in
  LN.pad expanded [| (0, 0); (0, restart) |] 0.0

let finish_cycle a_op m_op b x0 vmat y restart =
  let rows = (shape vmat).(0) in
  let first = L.slice vmat [| 0; 0 |] [| rows; restart |] in
  let dx = dot_ first y in
  let x = add_ x0 dx in
  let ax = a_op x in
  let residual = m_op (sub_ b ax) in
  let unit_residual, residual_norm = safe_normalize residual in
  (x, unit_residual, residual_norm)

let gmres_incremental a_op m_op b x0 unit_residual residual_norm ptol restart =
  let vmat = init_krylov unit_residual restart in
  let dt = dtype b in
  let rmat = LN.eye ~m:(restart + 1) ~dtype:dt restart in
  let givens = AC.zeros ~dtype:dt [| restart; 2 |] in
  let beta_vec = AC.zeros ~dtype:dt [| restart + 1 |] in
  let beta_vec =
    set_dyn_elem beta_vec (py_int 0) (LN.astype residual_norm dt)
  in
  let cond = function
    | k :: err :: _ ->
        U.logical_and (U.less k (py_int restart)) (U.greater err ptol)
    | _ -> assert false
  in
  let body = function
    | [ k; _; vmat; rmat; beta_vec; givens ] ->
        let vmat, hmat, _ = kth_arnoldi k a_op m_op vmat rmat in
        let row = dyn_row hmat k in
        let r_row, givens = apply_givens row givens k in
        let rmat = set_dyn_row hmat k r_row in
        let g = dyn_row givens k in
        let cs = dyn_elem g (py_int 0) in
        let sn = dyn_elem g (py_int 1) in
        let beta_vec = rotate_vectors beta_vec k cs sn in
        let err = U.abs (dyn_elem beta_vec (U.add k (py_int 1))) in
        [ U.add k (py_int 1); err; vmat; rmat; beta_vec; givens ]
    | _ -> assert false
  in
  match
    L.while_loop cond body
      [ py_int 0; residual_norm; vmat; rmat; beta_vec; givens ]
  with
  | [ _; _; vmat; rmat; beta_vec; _ ] ->
      let rows = (shape rmat).(0) in
      let cols = L.slice rmat [| 0; 0 |] [| rows; restart |] in
      let rt = LN.transpose cols in
      let beta = L.slice_in_dim beta_vec 0 restart in
      let y = Dense_solvers.solve_triangular rt beta in
      finish_cycle a_op m_op b x0 vmat y restart
  | _ -> assert false

let lstsq a b =
  let at = U.conj (LN.transpose a) in
  let a2 = dot_ at a in
  let at2 = U.conj (LN.transpose a) in
  let b2 = dot_ at2 b in
  Dense_solvers.solve ~assume_a:"pos" a2 b2

let gmres_batched a_op m_op b x0 unit_residual residual_norm _ptol restart =
  let vmat = init_krylov unit_residual restart in
  let dt = dtype b in
  let hmat = LN.eye ~m:(restart + 1) ~dtype:dt restart in
  let cond = function
    | [ _; _; breakdown; k ] ->
        U.logical_and (U.less k (py_int restart)) (U.logical_not breakdown)
    | _ -> assert false
  in
  let body = function
    | [ vmat; hmat; _; k ] ->
        let vmat, hmat, breakdown = kth_arnoldi k a_op m_op vmat hmat in
        [ vmat; hmat; breakdown; U.add k (py_int 1) ]
    | _ -> assert false
  in
  let init =
    [ vmat; hmat; T.Concrete (Nd.of_floats D.Bool [||] [| 0.0 |]); py_int 0 ]
  in
  match L.while_loop cond body init with
  | vmat :: hmat :: _ ->
      let beta_vec = AC.zeros ~dtype:dt [| restart + 1 |] in
      let beta_vec =
        set_dyn_elem beta_vec (py_int 0) (LN.astype residual_norm dt)
      in
      let y = lstsq (LN.transpose hmat) beta_vec in
      finish_cycle a_op m_op b x0 vmat y restart
  | _ -> assert false

let gmres_solve a_mat m_op b x0 ~atol ~ptol ~restart ~maxiter gmres_func =
  let ax0 = dot_ a_mat x0 in
  let residual = m_op (sub_ b ax0) in
  let unit_residual, residual_norm = safe_normalize residual in
  let cond = function
    | [ _; _; atol; _; _; k; _; residual_norm ] ->
        U.logical_and (U.less k (py_int maxiter)) (U.greater residual_norm atol)
    | _ -> assert false
  in
  let body = function
    | [ a_c; b_c; atol_c; ptol_c; x; k; unit_residual; residual_norm ] ->
        let x, unit_residual, residual_norm =
          gmres_func (dot_ a_c) m_op b_c x unit_residual residual_norm ptol_c
            restart
        in
        [
          a_c;
          b_c;
          atol_c;
          ptol_c;
          x;
          U.add k (py_int 1);
          unit_residual;
          residual_norm;
        ]
    | _ -> assert false
  in
  List.nth
    (L.while_loop cond body
       [ a_mat; b; atol; ptol; x0; py_int 0; unit_residual; residual_norm ])
    4

let solve_op matvec solver = fun _ bs -> [ solver matvec (List.hd bs) ]

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
  let b_norm = norm_ b in
  let tolb = mul (lit b_norm tol) b_norm in
  let atol_v = U.maximum tolb (lit b_norm atol) in
  let mb = m_op b in
  let mb_norm = norm_ mb in
  let ratio = U.true_divide atol_v b_norm in
  let ptol = mul mb_norm (U.minimum (lit ratio 1.0) ratio) in
  let gmres_func =
    match solve_method with
    | "incremental" -> gmres_incremental
    | "batched" -> gmres_batched
    | s -> invalid_arg ("invalid solve_method " ^ s)
  in
  let solve _ bs =
    match bs with
    | [ b ] ->
        [
          gmres_solve a m_op b x0 ~atol:atol_v ~ptol ~restart ~maxiter
            gmres_func;
        ]
    | _ -> assert false
  in
  let x =
    List.hd
      (L.custom_linear_solve ~transpose_solve:solve
         (function [ v ] -> [ a_op v ] | _ -> assert false)
         [ b ] solve)
  in
  let failed = U.isnan (norm_ x) in
  let info = LN.where_ failed (py_int (-1)) (py_int 0) in
  (x, Some info)

let isolve solver ?(symmetric = false) a b x0 ~tol ~atol ~maxiter m =
  let x0 = match x0 with Some x -> x | None -> AC.zeros_like b in
  let size = Array.fold_left ( * ) 1 (shape b) in
  let maxiter = match maxiter with Some m -> m | None -> 10 * size in
  let m_op =
    match m with Some mm -> normalize_matvec mm | None -> identity_op
  in
  let matvec = normalize_matvec a in
  if shape x0 <> shape b then
    invalid_arg "arrays in x0 and b must have matching shapes";
  let solve op bs =
    match (op, bs) with
    | op, [ b ] ->
        [ solver (fun v -> List.hd (op [ v ])) b x0 ~maxiter ~tol ~atol m_op ]
    | _ -> assert false
  in
  List.hd
    (L.custom_linear_solve ~symmetric ~transpose_solve:solve
       (function [ v ] -> [ matvec v ] | _ -> assert false)
       [ b ] solve)

let real_valued b = not (is_complex b)

let cg ?(tol = 1e-5) ?(atol = 0.0) ?maxiter ?m ?x0 a b =
  (isolve cg_solve ~symmetric:(real_valued b) a b x0 ~tol ~atol ~maxiter m, None)

let bicgstab ?(tol = 1e-5) ?(atol = 0.0) ?maxiter ?m ?x0 a b =
  (isolve bicgstab_solve a b x0 ~tol ~atol ~maxiter m, None)
