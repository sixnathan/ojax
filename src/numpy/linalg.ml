module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Lax_numpy
module UF = Ufuncs
module RED = Reductions
module TC = Tensor_contractions
module IDX = Indexing
module LL = Lax.Linalg

type ord = Onone | Ofro | Onuc | Onum of float
type norm_axis = Anone | Aint of int | Apair of int * int

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let ndim v = Array.length (shape v)

let canon_axis ax n =
  let a = if ax < 0 then ax + n else ax in
  if a < 0 || a >= n then
    invalid_arg
      (Printf.sprintf "axis %d out of bounds for array of ndim %d" ax n);
  a

let is_inexact = function
  | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> true
  | _ -> false

let default_float () = Dtypes.default_float_dtype ()

let to_inexact v =
  if is_inexact (dtype v) then v else NL.astype v (default_float ())

let const dt x = T.Concrete (Nd.of_floats dt [||] [| x |])

let eps_of = function
  | D.F64 | D.Complex128 -> 2.220446049250313e-16
  | _ -> 1.1920928955078125e-07

let matrix_transpose_v x =
  let nd = ndim x in
  if nd < 2 then
    invalid_arg
      (Printf.sprintf "matrix_transpose requires at least 2 dimensions; got %d"
         nd);
  NL.matrix_transpose x

let hconj x = UF.conj (matrix_transpose_v x)
let symmetrize x = UF.divide (UF.add x (hconj x)) (const (dtype x) 2.0)

let rec cholesky ?(upper = false) ?(symmetrize_input = true) a =
  let a = to_inexact a in
  let a = if symmetrize_input then symmetrize a else a in
  let l = LL.cholesky a in
  if upper then hconj l else l

and svd_uv ~full_matrices a =
  match LL.svd ~full_matrices ~compute_uv:true a with
  | [ u; s; vh ] -> (u, s, vh)
  | _ -> assert false

and svd_values a =
  match LL.svd ~full_matrices:false ~compute_uv:false a with
  | [ s ] -> s
  | _ -> assert false

and svd ?(full_matrices = true) ?(compute_uv = true) ?(hermitian = false) a =
  let a = to_inexact a in
  if hermitian then
    invalid_arg "jnp.linalg.svd: hermitian=True not supported in ojax";
  if compute_uv then
    let u, s, vh = svd_uv ~full_matrices a in
    [ u; s; vh ]
  else
    [
      (match LL.svd ~full_matrices ~compute_uv:false a with
      | [ s ] -> s
      | _ -> assert false);
    ]

and svdvals x = svd_values (to_inexact x)

and solve a b =
  let a = to_inexact a and b = to_inexact b in
  let sa = shape a in
  let na = Array.length sa in
  if na < 2 then invalid_arg "left hand array must be at least two dimensional";
  let m = sa.(na - 1) in
  let b_vec = ndim b = 1 in
  let bmat = if b_vec then NL.reshape b [| m; 1 |] else b in
  let lu, _piv, perm = LL.lu a in
  let x0 = IDX.take ~axis:0 bmat perm in
  let x1 =
    LL.triangular_solve ~left_side:true ~lower:true ~unit_diagonal:true lu x0
  in
  let x2 = LL.triangular_solve ~left_side:true ~lower:false lu x1 in
  if b_vec then NL.reshape x2 [| m |] else x2

and inv a =
  let sa = shape a in
  let na = Array.length sa in
  if na < 2 || sa.(na - 1) <> sa.(na - 2) then
    invalid_arg "Argument to inv must have shape [..., n, n]";
  let n = sa.(na - 1) in
  solve a (NL.eye ~dtype:(dtype a) n)

and slogdet ?(method_ = "lu") a =
  let a = to_inexact a in
  if method_ <> "lu" then
    invalid_arg "jnp.linalg.slogdet: only method='lu' supported in ojax";
  let sa = shape a in
  let na = Array.length sa in
  let n = sa.(na - 1) in
  let dt = dtype a in
  let lu, piv, _perm = LL.lu a in
  let diag = NL.diagonal ~axis1:(-2) ~axis2:(-1) lu in
  let zero = const dt 0.0 in
  let is_zero = RED.any ~axis:[| -1 |] (UF.equal diag zero) in
  let iota = NL.arange ~dtype:(dtype piv) (float_of_int n) in
  let swaps = RED.count_nonzero ~axis:[| -1 |] (UF.not_equal piv iota) in
  let neg = RED.count_nonzero ~axis:[| -1 |] (UF.less diag zero) in
  let parity = UF.add (NL.astype swaps dt) (NL.astype neg dt) in
  let two = const dt 2.0 in
  let half = UF.floor (UF.divide parity two) in
  let pmod = UF.subtract parity (UF.multiply two half) in
  let sgn = UF.subtract (const dt 1.0) (UF.multiply two pmod) in
  let logdet = RED.sum ~axis:[| -1 |] (UF.log (UF.abs diag)) in
  let sign = NL.where_ is_zero (const dt 0.0) sgn in
  let logabsdet = NL.where_ is_zero (const dt neg_infinity) logdet in
  (sign, logabsdet)

and det a =
  let a = to_inexact a in
  let sign, logdet = slogdet a in
  UF.multiply sign (NL.astype (UF.exp logdet) (dtype sign))

and eig a =
  let a = to_inexact a in
  match
    LL.eig ~compute_left_eigenvectors:false ~compute_right_eigenvectors:true a
  with
  | [ w; v ] -> (w, v)
  | _ -> assert false

and eigvals a =
  let a = to_inexact a in
  match
    LL.eig ~compute_left_eigenvectors:false ~compute_right_eigenvectors:false a
  with
  | [ w ] -> w
  | _ -> assert false

and eigh ?(uplo = "L") ?(symmetrize_input = true) a =
  let lower =
    match uplo with
    | "L" -> true
    | "U" -> false
    | _ -> invalid_arg "UPLO must be one of None, 'L', or 'U'"
  in
  let a = to_inexact a in
  let a = if symmetrize_input then symmetrize a else a in
  let v, w = LL.eigh ~lower a in
  (w, v)

and eigvalsh ?(uplo = "L") ?(symmetrize_input = true) a =
  let w, _ = eigh ~uplo ~symmetrize_input a in
  w

and pinv ?rtol ?(hermitian = false) a =
  let a = to_inexact a in
  if hermitian then
    invalid_arg "jnp.linalg.pinv: hermitian=True not supported in ojax";
  let arr = UF.conj a in
  let dt = dtype arr in
  let sa = shape arr in
  let na = Array.length sa in
  let m = sa.(na - 2) and n = sa.(na - 1) in
  let maxrc = max m n in
  let rtol =
    match rtol with
    | Some r -> r
    | None -> const dt (10.0 *. float_of_int maxrc *. eps_of dt)
  in
  let u, s, vh = svd_uv ~full_matrices:false arr in
  let rtolc = if ndim rtol = 0 then NL.expand_dims rtol [| 0 |] else rtol in
  let s0 = RED.amax ~axis:[| -1 |] ~keepdims:true s in
  let cutoff = UF.multiply rtolc s0 in
  let s = NL.where_ (UF.greater s cutoff) s (const dt infinity) in
  let s = NL.astype s (dtype u) in
  let s_col = NL.expand_dims s [| ndim s |] in
  let scaled = UF.divide (matrix_transpose_v u) s_col in
  let res = TC.matmul (matrix_transpose_v vh) scaled in
  NL.astype res dt

and matrix_power a p =
  let sa = shape a in
  let na = Array.length sa in
  if na < 2 then
    invalid_arg "matrix_power: array must be at least two-dimensional";
  if sa.(na - 1) <> sa.(na - 2) then
    invalid_arg "matrix_power: last 2 dimensions of the array must be square";
  let m = sa.(na - 1) in
  if p = 0 then NL.broadcast_to (NL.eye ~dtype:(dtype a) m) sa
  else begin
    let a, p = if p < 0 then (inv a, -p) else (a, p) in
    if p = 1 then a
    else if p = 2 then TC.matmul a a
    else if p = 3 then TC.matmul (TC.matmul a a) a
    else begin
      let z = ref None and result = ref None and nn = ref p in
      while !nn > 0 do
        (match !z with
        | None -> z := Some a
        | Some zz -> z := Some (TC.matmul zz zz));
        let bit = !nn land 1 in
        nn := !nn asr 1;
        if bit = 1 then
          match !result with
          | None -> result := Some (Option.get !z)
          | Some r -> result := Some (TC.matmul r (Option.get !z))
      done;
      Option.get !result
    end
  end

and matrix_rank ?rtol ?(hermitian = false) a =
  let a = to_inexact a in
  let s = if hermitian then UF.abs (svdvals a) else svdvals a in
  let sa = shape a in
  let na = Array.length sa in
  let maxdim = max sa.(na - 1) sa.(na - 2) in
  let dt = dtype s in
  let rtol =
    match rtol with
    | Some r -> r
    | None ->
        UF.multiply
          (RED.amax ~axis:[| -1 |] s)
          (const dt (float_of_int maxdim *. eps_of dt))
  in
  let rtol = NL.expand_dims rtol [| ndim rtol |] in
  RED.sum ~axis:[| -1 |] (UF.greater s rtol)

and vector_norm ?(axis = None) ?(keepdims = false) ?(ord = Onum 2.0) x =
  let x = to_inexact x in
  let sumf v =
    match axis with
    | None -> RED.sum ~keepdims v
    | Some ax -> RED.sum ~axis:ax ~keepdims v
  in
  let amaxf v =
    match axis with
    | None -> RED.amax ~keepdims v
    | Some ax -> RED.amax ~axis:ax ~keepdims v
  in
  let aminf v =
    match axis with
    | None -> RED.amin ~keepdims v
    | Some ax -> RED.amin ~axis:ax ~keepdims v
  in
  let two_norm () = UF.sqrt (sumf (UF.real (UF.multiply x (UF.conj x)))) in
  match ord with
  | Onone -> two_norm ()
  | Ofro | Onuc -> invalid_arg "Invalid order for vector norm."
  | Onum o ->
      if o = 2.0 then two_norm ()
      else if o = infinity then amaxf (UF.abs x)
      else if o = neg_infinity then aminf (UF.abs x)
      else if o = 0.0 then
        sumf (NL.astype (UF.not_equal x (const (dtype x) 0.0)) (dtype x))
      else if o = 1.0 then sumf (UF.abs x)
      else
        let ax = UF.abs x in
        let dt = dtype ax in
        UF.power (sumf (UF.power ax (const dt o))) (const dt (1.0 /. o))

and matrix_norm_impl x ord r c keepdims =
  let nd = ndim x in
  let r = canon_axis r nd and c = canon_axis c nd in
  let abs_x () = UF.abs x in
  match ord with
  | Onone | Ofro ->
      UF.sqrt
        (RED.sum ~axis:[| r; c |] ~keepdims
           (UF.real (UF.multiply x (UF.conj x))))
  | Onuc ->
      let x2 = NL.moveaxis [| r; c |] [| nd - 2; nd - 1 |] x in
      let s = svdvals x2 in
      let y = RED.sum ~axis:[| -1 |] s in
      if keepdims then NL.expand_dims y [| r; c |] else y
  | Onum o ->
      if o = 1.0 then begin
        let c = if (not keepdims) && c > r then c - 1 else c in
        RED.amax ~axis:[| c |] ~keepdims
          (RED.sum ~axis:[| r |] ~keepdims (abs_x ()))
      end
      else if o = -1.0 then begin
        let c = if (not keepdims) && c > r then c - 1 else c in
        RED.amin ~axis:[| c |] ~keepdims
          (RED.sum ~axis:[| r |] ~keepdims (abs_x ()))
      end
      else if o = infinity then begin
        let r = if (not keepdims) && r > c then r - 1 else r in
        RED.amax ~axis:[| r |] ~keepdims
          (RED.sum ~axis:[| c |] ~keepdims (abs_x ()))
      end
      else if o = neg_infinity then begin
        let r = if (not keepdims) && r > c then r - 1 else r in
        RED.amin ~axis:[| r |] ~keepdims
          (RED.sum ~axis:[| c |] ~keepdims (abs_x ()))
      end
      else if o = 2.0 || o = -2.0 then begin
        let x2 = NL.moveaxis [| r; c |] [| nd - 2; nd - 1 |] x in
        let s = svdvals x2 in
        let y =
          if o = 2.0 then RED.amax ~axis:[| -1 |] s
          else RED.amin ~axis:[| -1 |] s
        in
        if keepdims then NL.expand_dims y [| r; c |] else y
      end
      else invalid_arg "Invalid order for matrix norm."

and norm ?(ord = Onone) ?(axis = Anone) ?(keepdims = false) x =
  let x = to_inexact x in
  let nd = ndim x in
  match axis with
  | Anone -> (
      match ord with
      | Onone ->
          UF.sqrt (RED.sum ~keepdims (UF.real (UF.multiply x (UF.conj x))))
      | _ ->
          if nd = 1 then vector_norm ~axis:(Some [| 0 |]) ~keepdims ~ord x
          else if nd = 2 then matrix_norm_impl x ord 0 1 keepdims
          else invalid_arg "Improper number of axes for norm")
  | Aint a ->
      let ord = match ord with Onone -> Onum 2.0 | o -> o in
      vector_norm ~axis:(Some [| canon_axis a nd |]) ~keepdims ~ord x
  | Apair (r, c) -> matrix_norm_impl x ord r c keepdims

and matrix_norm ?(keepdims = false) ?(ord = Ofro) x =
  norm ~ord ~axis:(Apair (-2, -1)) ~keepdims (to_inexact x)

and matrix_transpose x = matrix_transpose_v x

and qr ?(mode = "reduced") a =
  let a = to_inexact a in
  if mode = "raw" then
    invalid_arg "jnp.linalg.qr: mode 'raw' not supported in ojax";
  let full_matrices =
    match mode with
    | "reduced" | "r" | "full" -> false
    | "complete" -> true
    | _ ->
        invalid_arg
          (Printf.sprintf "Unsupported QR decomposition mode '%s'" mode)
  in
  let q, r = LL.qr ~full_matrices a in
  if mode = "r" then [ r ] else [ q; r ]

and lstsq ?rcond a b =
  let a = to_inexact a and b = to_inexact b in
  let b_vec = ndim b = 1 in
  let bm = if b_vec then NL.reshape b [| (shape b).(0); 1 |] else b in
  let sa = shape a in
  let m = sa.(0) and n = sa.(1) in
  let dt = dtype a in
  let rcond =
    match rcond with
    | Some r -> const dt r
    | None -> const dt (eps_of dt *. float_of_int (max n m))
  in
  let u, s, vt = svd_uv ~full_matrices:false a in
  let s0 = RED.amax ~axis:[| -1 |] ~keepdims:true s in
  let mask =
    UF.logical_and
      (UF.greater s (const dt 0.0))
      (UF.greater_equal s (UF.multiply rcond s0))
  in
  let rank = RED.sum mask in
  let safe_s = NL.where_ mask s (const dt 1.0) in
  let s_inv = NL.where_ mask (UF.divide (const dt 1.0) safe_s) (const dt 0.0) in
  let s_inv_col = NL.expand_dims s_inv [| ndim s_inv |] in
  let utb = TC.matmul (UF.conj (matrix_transpose_v u)) bm in
  let x =
    TC.matmul (UF.conj (matrix_transpose_v vt)) (UF.multiply s_inv_col utb)
  in
  let b_est = TC.matmul a x in
  let resid =
    UF.power (norm ~axis:(Aint 0) (UF.subtract bm b_est)) (const dt 2.0)
  in
  let x = if b_vec then NL.ravel x else x in
  (x, resid, rank, s)

and cross ?(axis = -1) x1 x2 = NL.cross ~axis x1 x2
and outer x1 x2 = TC.outer x1 x2
and matmul ?preferred x1 x2 = TC.matmul ?preferred x1 x2
and vecdot ?(axis = -1) ?preferred x1 x2 = TC.vecdot ~axis ?preferred x1 x2

and tensordot ?preferred ?(axes = TC.Ax_int 2) x1 x2 =
  TC.tensordot ?preferred ~axes x1 x2

and diagonal ?(offset = 0) x = NL.diagonal ~offset ~axis1:(-2) ~axis2:(-1) x

and trace ?(offset = 0) ?dtype x =
  NL.trace ~offset ~axis1:(-2) ~axis2:(-1) ?dtype x

and tensorinv ?(ind = 2) a =
  if ind <= 0 then invalid_arg "ind must be a positive integer";
  let sa = shape a in
  let nd = Array.length sa in
  let prod lo hi =
    let r = ref 1 in
    for i = lo to hi - 1 do
      r := !r * sa.(i)
    done;
    !r
  in
  let p0 = prod 0 ind and p1 = prod ind nd in
  if p0 <> p1 then
    invalid_arg "tensorinv: product of first ind dims must equal the rest";
  let flat = NL.reshape a [| p0; p1 |] in
  let out_shape =
    Array.append (Array.sub sa ind (nd - ind)) (Array.sub sa 0 ind)
  in
  NL.reshape (inv flat) out_shape

and tensorsolve ?axes a b =
  let a =
    match axes with
    | None -> a
    | Some ax ->
        let nd = ndim a in
        NL.moveaxis ax (Array.map (fun _ -> nd - 1) ax) a
  in
  let sb = shape b in
  let bnd = Array.length sb in
  let sa = shape a in
  let out_shape = Array.sub sa bnd (Array.length sa - bnd) in
  let bsize = Array.fold_left ( * ) 1 sb in
  let prod_out = Array.fold_left ( * ) 1 out_shape in
  let a2 = NL.reshape a [| bsize; prod_out |] in
  let x = solve a2 (NL.ravel b) in
  NL.reshape x out_shape

and multi_dot arrs =
  match arrs with
  | [] | [ _ ] -> invalid_arg "multi_dot requires at least two arrays"
  | first :: rest -> List.fold_left (fun acc m -> TC.matmul acc m) first rest

and cond ?(p = Onone) x =
  let arr = to_inexact x in
  let nd = ndim arr in
  if nd < 2 then invalid_arg "jnp.linalg.cond: input array must be at least 2D";
  let is_two = match p with Onone -> true | Onum o -> o = 2.0 | _ -> false in
  let is_neg_two = match p with Onum o -> o = -2.0 | _ -> false in
  if is_two then
    let s = svdvals arr in
    UF.divide (RED.amax ~axis:[| -1 |] s) (RED.amin ~axis:[| -1 |] s)
  else if is_neg_two then
    let s = svdvals arr in
    UF.divide (RED.amin ~axis:[| -1 |] s) (RED.amax ~axis:[| -1 |] s)
  else
    let dt = dtype arr in
    let r =
      UF.multiply
        (norm ~ord:p ~axis:(Apair (-2, -1)) arr)
        (norm ~ord:p ~axis:(Apair (-2, -1)) (inv arr))
    in
    let bad =
      UF.logical_and (UF.isnan r)
        (UF.logical_not (RED.any ~axis:[| -2; -1 |] (UF.isnan arr)))
    in
    NL.where_ bad (const dt infinity) r
