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
  if is_inexact (dtype v) then v
  else NL.astype v (Dtypes.to_inexact_dtype (dtype v))

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
  let l = LL.cholesky ~symmetrize_input a in
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
  if hermitian then svd_hermitian ~compute_uv a
  else if compute_uv then
    let u, s, vh = svd_uv ~full_matrices a in
    [ u; s; vh ]
  else
    [
      (match LL.svd ~full_matrices ~compute_uv:false a with
      | [ s ] -> s
      | _ -> assert false);
    ]

and svdvals x = svd_values (to_inexact x)

and svd_hermitian ~compute_uv a =
  let w, v = eigh a in
  let rank = ndim a in
  let magnitudes = UF.absolute w in
  let order =
    Sorting.argsort ~axis:(Some (rank - 2)) ~descending:true magnitudes
  in
  let sorted_s = IDX.take_along_axis ~axis:(rank - 2) magnitudes order in
  if not compute_uv then [ sorted_s ]
  else
    let sorted_w = IDX.take_along_axis ~axis:(rank - 2) w order in
    let expanded = NL.expand_dims order [| rank - 2 |] in
    let columns = NL.broadcast_to expanded (shape v) in
    let u = IDX.take_along_axis ~axis:(rank - 1) v columns in
    let signs = UF.sign sorted_w in
    let scaled =
      UF.multiply u
        (NL.broadcast_to (NL.expand_dims signs [| rank - 2 |]) (shape u))
    in
    [ u; sorted_s; hconj scaled ]

and solve a b =
  let a = to_inexact a and b = to_inexact b in
  let sa = shape a in
  let na = Array.length sa in
  if na < 2 then invalid_arg "left hand array must be at least two dimensional";
  let m = sa.(na - 1) in
  let b_vec = ndim b = 1 in
  let bmat = if b_vec then NL.reshape b [| m; 1 |] else b in
  let sb = shape bmat in
  let nb = Array.length sb in
  let batch =
    Lax_api.broadcast_shapes
      [ Array.sub sa 0 (na - 2); Array.sub sb 0 (nb - 2) ]
  in
  let k = sb.(nb - 1) in
  let a = NL.broadcast_to a (Array.append batch [| m; m |]) in
  let bmat = NL.broadcast_to bmat (Array.append batch [| m; k |]) in
  let lu, _piv, perm = LL.lu a in
  let rank = Array.length batch + 2 in
  let x0 =
    if rank = 2 then IDX.take ~axis:0 bmat perm
    else
      let expanded = NL.expand_dims perm [| -1 |] in
      IDX.take_along_axis ~axis:(rank - 2) bmat
        (NL.broadcast_to expanded (Array.append batch [| m; k |]))
  in
  let x1 =
    LL.triangular_solve ~left_side:true ~lower:true ~unit_diagonal:true lu x0
  in
  let x2 = LL.triangular_solve ~left_side:true ~lower:false lu x1 in
  if b_vec then NL.reshape x2 (Array.append batch [| m |]) else x2

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
  if n = 1 || n = 2 then begin
    let d = if n = 1 then matrix_cell a 0 0 else det_2x2 a in
    let magnitude = UF.abs d in
    let log_abs = UF.log magnitude in
    if dt = D.Complex64 || dt = D.Complex128 then
      let rdt = dtype magnitude in
      let zero = UF.equal magnitude (const rdt 0.0) in
      let safe = NL.where_ zero (const rdt 1.0) magnitude in
      (NL.where_ zero (const dt 0.0) (UF.divide d (NL.astype safe dt)), log_abs)
    else (UF.sign d, log_abs)
  end
  else
    let lu, piv, _perm = LL.lu a in
    let diag = NL.diagonal ~axis1:(-2) ~axis2:(-1) lu in
    let zero = const dt 0.0 in
    let is_zero = RED.any ~axis:[| -1 |] (UF.equal diag zero) in
    let iota = NL.arange ~dtype:(dtype piv) (float_of_int n) in
    let swaps = RED.count_nonzero ~axis:[| -1 |] (UF.not_equal piv iota) in
    if dt = D.Complex64 || dt = D.Complex128 then begin
      let magnitude = UF.abs diag in
      let rdt = dtype magnitude in
      let safe =
        NL.where_ (UF.equal magnitude (const rdt 0.0)) (const rdt 1.0) magnitude
      in
      let unit = UF.divide diag (NL.astype safe dt) in
      let swap_count = NL.astype swaps rdt in
      let half = UF.floor (UF.divide swap_count (const rdt 2.0)) in
      let odd = UF.subtract swap_count (UF.multiply (const rdt 2.0) half) in
      let parity_sign =
        UF.subtract (const rdt 1.0) (UF.multiply (const rdt 2.0) odd)
      in
      let phase =
        UF.multiply (RED.prod ~axis:[| -1 |] unit) (NL.astype parity_sign dt)
      in
      let logdet = RED.sum ~axis:[| -1 |] (UF.log magnitude) in
      ( NL.where_ is_zero (const dt 0.0) phase,
        NL.where_ is_zero (const rdt neg_infinity) logdet )
    end
    else
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
  let sa = shape a in
  let na = Array.length sa in
  if na >= 2 && sa.(na - 1) = 2 && sa.(na - 2) = 2 then det_2x2 a
  else if na >= 2 && sa.(na - 1) = 3 && sa.(na - 2) = 3 then begin
    let e i j = matrix_cell a i j in
    let product x y z = UF.multiply (UF.multiply x y) z in
    let plus =
      UF.add
        (UF.add
           (product (e 0 0) (e 1 1) (e 2 2))
           (product (e 0 1) (e 1 2) (e 2 0)))
        (product (e 0 2) (e 1 0) (e 2 1))
    in
    UF.subtract
      (UF.subtract
         (UF.subtract plus (product (e 0 2) (e 1 1) (e 2 0)))
         (product (e 0 0) (e 1 2) (e 2 1)))
      (product (e 0 1) (e 1 0) (e 2 2))
  end
  else
    let sign, logdet = slogdet a in
    UF.multiply sign (NL.astype (UF.exp logdet) (dtype sign))

and matrix_cell a i j =
  let na = ndim a in
  Lax_api.index_in_dim ~axis:(na - 2) ~keepdims:false
    (Lax_api.index_in_dim ~axis:(na - 1) ~keepdims:false a j)
    i

and det_2x2 a =
  let e i j = matrix_cell a i j in
  UF.subtract (UF.multiply (e 0 0) (e 1 1)) (UF.multiply (e 0 1) (e 1 0))

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
  let v, w = LL.eigh ~lower ~symmetrize_input a in
  (w, v)

and eigvalsh ?(uplo = "L") ?(symmetrize_input = true) a =
  let w, _ = eigh ~uplo ~symmetrize_input a in
  w

and pinv ?rtol ?(hermitian = false) ?rcond a =
  let rtol =
    match (rcond, rtol) with
    | Some _, Some _ ->
        invalid_arg "pinv: only one of rtol and rcond may be specified."
    | Some r, None -> Some r
    | None, r -> r
  in
  let a = to_inexact a in
  ignore hermitian;
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

and matrix_rank ?rtol ?(hermitian = false) ?tol a =
  let rtol =
    match (tol, rtol) with
    | Some _, Some _ ->
        invalid_arg "matrix_rank: only one of tol or rtol may be specified."
    | Some t, None -> Some t
    | None, r -> r
  in
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

and raw_qr a =
  let sh = shape a in
  let nd = Array.length sh in
  if nd > 2 then
    let outs =
      Interpreters.Batching.vmap
        (fun args -> raw_qr (List.hd args))
        [ Some 0 ] [ a ]
    in
    outs
  else
    let k = min sh.(0) sh.(1) in
    let aval = Core.get_aval a in
    match
      Custom_derivatives.pure_callback
        (fun args ->
          match args with
          | [ T.Concrete nd ] ->
              List.map (fun x -> T.Concrete x) (Lax.Linalg.geqrf_impl nd)
          | [ T.Device b ] ->
              List.map
                (fun x -> T.Concrete x)
                (Lax.Linalg.geqrf_impl (Pjrt.Buffer.to_host b))
          | _ -> failwith "jnp.linalg.qr: raw mode arity")
        ~result_avals:[ aval; { aval with T.shape = [| k |] } ]
        [ a ]
    with
    | [ h; t ] -> [ matrix_transpose_v h; t ]
    | _ -> failwith "jnp.linalg.qr: raw mode arity"

and qr ?(mode = "reduced") a =
  let a = to_inexact a in
  if mode = "raw" then raw_qr a
  else
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

and lstsq ?rcond ?(numpy_resid = false) a b =
  let a = to_inexact a and b = to_inexact b in
  let b_vec = ndim b = 1 in
  let bm = if b_vec then NL.reshape b [| (shape b).(0); 1 |] else b in
  let sa = shape a in
  let m = sa.(0) and n = sa.(1) in
  let dt = dtype a in
  let rcond =
    match rcond with
    | Some r when r < 0.0 -> const dt (eps_of dt)
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
  let full_resid () =
    UF.power (norm ~axis:(Aint 0) (UF.subtract bm b_est)) (const dt 2.0)
  in
  let resid =
    if numpy_resid then
      let concrete_rank =
        match rank with
        | T.Concrete nd -> Nd.get_i64 nd [||]
        | T.Device b -> Nd.get_i64 (Pjrt.Buffer.to_host b) [||]
        | T.Tracer _ -> invalid_arg "lstsq: numpy_resid needs a concrete rank"
      in
      if Int64.to_int concrete_rank < n || m <= n then
        T.Concrete (Nd.of_floats dt [| 0 |] [||])
      else full_resid ()
    else full_resid ()
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

let cholesky ?upper ?symmetrize_input p0 =
  Jnp_util.jit1 "cholesky"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (upper, symmetrize_input) [])))
    (function
      | [ p0 ] -> cholesky ?upper ?symmetrize_input p0
      | _ -> invalid_arg "cholesky: arity")
    [ p0 ]

let svd ?full_matrices ?compute_uv ?hermitian p0 =
  Jnp_util.jit "svd"
    ~key:
      (Digest.to_hex
         (Digest.string
            (Marshal.to_string (full_matrices, compute_uv, hermitian) [])))
    (function
      | [ p0 ] -> svd ?full_matrices ?compute_uv ?hermitian p0
      | _ -> invalid_arg "svd: arity")
    [ p0 ]

let matrix_power p0 s1 =
  Jnp_util.jit1 "matrix_power"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string s1 [])))
    (function
      | [ p0 ] -> matrix_power p0 s1 | _ -> invalid_arg "matrix_power: arity")
    [ p0 ]

let matrix_rank ?rtol ?hermitian ?tol p0 =
  let rtol_given = rtol <> None in
  let tol_given = tol <> None in
  if rtol_given || tol_given then matrix_rank ?rtol ?hermitian ?tol p0
  else
    Jnp_util.jit1 "matrix_rank"
      ~key:(Digest.to_hex (Digest.string (Marshal.to_string hermitian [])))
      (function
        | [ p0 ] -> matrix_rank ?rtol ?hermitian ?tol p0
        | _ -> invalid_arg "matrix_rank: arity")
      [ p0 ]

let slogdet ?method_ p0 =
  match
    Jnp_util.jit "slogdet"
      ~key:(Digest.to_hex (Digest.string (Marshal.to_string method_ [])))
      (function
        | [ p0 ] ->
            let a, b = slogdet ?method_ p0 in
            [ a; b ]
        | _ -> invalid_arg "slogdet: arity")
      [ p0 ]
  with
  | [ a; b ] -> (a, b)
  | _ -> invalid_arg "slogdet: arity"

let det p0 =
  Jnp_util.jit1 "det"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> det p0 | _ -> invalid_arg "det: arity")
    [ p0 ]

let eigvals p0 =
  Jnp_util.jit1 "eigvals"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> eigvals p0 | _ -> invalid_arg "eigvals: arity")
    [ p0 ]

let eigh ?uplo ?symmetrize_input p0 =
  match
    Jnp_util.jit "eigh"
      ~key:
        (Digest.to_hex
           (Digest.string (Marshal.to_string (uplo, symmetrize_input) [])))
      (function
        | [ p0 ] ->
            let a, b = eigh ?uplo ?symmetrize_input p0 in
            [ a; b ]
        | _ -> invalid_arg "eigh: arity")
      [ p0 ]
  with
  | [ a; b ] -> (a, b)
  | _ -> invalid_arg "eigh: arity"

let eigvalsh ?uplo ?symmetrize_input p0 =
  Jnp_util.jit1 "eigvalsh"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (uplo, symmetrize_input) [])))
    (function
      | [ p0 ] -> eigvalsh ?uplo ?symmetrize_input p0
      | _ -> invalid_arg "eigvalsh: arity")
    [ p0 ]

let inv p0 =
  Jnp_util.jit1 "inv"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> inv p0 | _ -> invalid_arg "inv: arity")
    [ p0 ]

let norm ?ord ?axis ?keepdims p0 =
  Jnp_util.jit1 "norm"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (ord, axis, keepdims) [])))
    (function
      | [ p0 ] -> norm ?ord ?axis ?keepdims p0 | _ -> invalid_arg "norm: arity")
    [ p0 ]

let qr ?mode p0 =
  Jnp_util.jit "qr"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string mode [])))
    (function [ p0 ] -> qr ?mode p0 | _ -> invalid_arg "qr: arity")
    [ p0 ]

let solve p0 p1 =
  Jnp_util.jit1 "solve"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> solve p0 p1 | _ -> invalid_arg "solve: arity")
    [ p0; p1 ]

let cond ?p p0 =
  Jnp_util.jit1 "cond"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string p [])))
    (function [ p0 ] -> cond ?p p0 | _ -> invalid_arg "cond: arity")
    [ p0 ]

let cholesky ?upper ?symmetrize_input a0 =
  Eager_jit.one "numpy.linalg.cholesky" (upper, symmetrize_input) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> cholesky ?upper ?symmetrize_input a0
      | _ -> assert false)

let matrix_power a0 a1 =
  Eager_jit.one "numpy.linalg.matrix_power" a1 [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> matrix_power a0 a1
      | _ -> assert false)

let matrix_rank ?rtol ?hermitian ?tol a0 =
  Eager_jit.one "numpy.linalg.matrix_rank" hermitian [ rtol; tol ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ rtol; tol ], [ a0 ] -> matrix_rank ?rtol ?hermitian ?tol a0
      | _ -> assert false)

let slogdet ?method_ a0 =
  Eager_jit.pair "numpy.linalg.slogdet" method_ [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> slogdet ?method_ a0
      | _ -> assert false)

let det a0 =
  Eager_jit.one "numpy.linalg.det" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> det a0 | _ -> assert false)

let eigvals a0 =
  Eager_jit.one "numpy.linalg.eigvals" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> eigvals a0 | _ -> assert false)

let eigh ?uplo ?symmetrize_input a0 =
  Eager_jit.pair "numpy.linalg.eigh" (uplo, symmetrize_input) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> eigh ?uplo ?symmetrize_input a0
      | _ -> assert false)

let eigvalsh ?uplo ?symmetrize_input a0 =
  Eager_jit.one "numpy.linalg.eigvalsh" (uplo, symmetrize_input) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> eigvalsh ?uplo ?symmetrize_input a0
      | _ -> assert false)

let inv a0 =
  Eager_jit.one "numpy.linalg.inv" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> inv a0 | _ -> assert false)

let norm ?ord ?axis ?keepdims a0 =
  Eager_jit.one "numpy.linalg.norm" (ord, axis, keepdims) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> norm ?ord ?axis ?keepdims a0
      | _ -> assert false)

let qr ?mode a0 =
  Eager_jit.many "numpy.linalg.qr" mode [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> qr ?mode a0 | _ -> assert false)

let solve a0 a1 =
  Eager_jit.one "numpy.linalg.solve" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> solve a0 a1
      | _ -> assert false)

let cond ?p a0 =
  Eager_jit.one "numpy.linalg.cond" p [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> cond ?p a0 | _ -> assert false)

let pinv ?rtol ?hermitian ?rcond a0 =
  Eager_jit.one "numpy.linalg.pinv" hermitian [ rtol; rcond ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ rtol; rcond ], [ a0 ] -> pinv ?rtol ?hermitian ?rcond a0
      | _ -> assert false)

let svd ?full_matrices ?compute_uv ?hermitian a0 =
  Eager_jit.many "numpy.linalg.svd" (full_matrices, compute_uv, hermitian)
    [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> svd ?full_matrices ?compute_uv ?hermitian a0
      | _ -> assert false)
