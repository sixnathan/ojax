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

let highest = Lax_config.precision_of_level Lax_config.Precision_highest
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

let slogdet_lu_call : (T.value list -> T.value list) ref =
  ref (fun _ -> invalid_arg "slogdet_lu: not installed")

let rec cholesky_body ?(upper = false) ?(symmetrize_input = true) a =
  let a = to_inexact a in
  let l = LL.cholesky ~symmetrize_input a in
  if upper then hconj l else l

and cholesky ?upper ?symmetrize_input p0 =
  Jnp_util.jit1 "cholesky"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (upper, symmetrize_input) [])))
    (function
      | [ p0 ] -> cholesky_body ?upper ?symmetrize_input p0
      | _ -> invalid_arg "cholesky: arity")
    [ p0 ]

and svd ?full_matrices ?compute_uv ?hermitian p0 =
  Jnp_util.jit "svd"
    ~key:
      (Digest.to_hex
         (Digest.string
            (Marshal.to_string (full_matrices, compute_uv, hermitian) [])))
    (function
      | [ p0 ] -> svd_body ?full_matrices ?compute_uv ?hermitian p0
      | _ -> invalid_arg "svd: arity")
    [ p0 ]

and matrix_power p0 s1 =
  Jnp_util.jit1 "matrix_power"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string s1 [])))
    (function
      | [ p0 ] -> matrix_power_body p0 s1
      | _ -> invalid_arg "matrix_power: arity")
    [ p0 ]

and matrix_rank ?rtol ?hermitian ?tol p0 =
  let rtol_given = rtol <> None in
  let tol_given = tol <> None in
  if rtol_given || tol_given then matrix_rank_body ?rtol ?hermitian ?tol p0
  else
    Jnp_util.jit1 "matrix_rank"
      ~key:(Digest.to_hex (Digest.string (Marshal.to_string hermitian [])))
      (function
        | [ p0 ] -> matrix_rank_body ?rtol ?hermitian ?tol p0
        | _ -> invalid_arg "matrix_rank: arity")
      [ p0 ]

and slogdet ?method_ p0 =
  match
    Jnp_util.jit "slogdet"
      ~key:(Digest.to_hex (Digest.string (Marshal.to_string method_ [])))
      (function
        | [ p0 ] ->
            let a, b = slogdet_body ?method_ p0 in
            [ a; b ]
        | _ -> invalid_arg "slogdet: arity")
      [ p0 ]
  with
  | [ a; b ] -> (a, b)
  | _ -> invalid_arg "slogdet: arity"

and det p0 =
  Jnp_util.jit1 "det"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> det_body p0 | _ -> invalid_arg "det: arity")
    [ p0 ]

and eigvals p0 =
  Jnp_util.jit1 "eigvals"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> eigvals_body p0 | _ -> invalid_arg "eigvals: arity")
    [ p0 ]

and eigh ?uplo ?symmetrize_input p0 =
  match
    Jnp_util.jit "eigh"
      ~key:
        (Digest.to_hex
           (Digest.string (Marshal.to_string (uplo, symmetrize_input) [])))
      (function
        | [ p0 ] ->
            let a, b = eigh_body ?uplo ?symmetrize_input p0 in
            [ a; b ]
        | _ -> invalid_arg "eigh: arity")
      [ p0 ]
  with
  | [ a; b ] -> (a, b)
  | _ -> invalid_arg "eigh: arity"

and eigvalsh ?uplo ?symmetrize_input p0 =
  Jnp_util.jit1 "eigvalsh"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (uplo, symmetrize_input) [])))
    (function
      | [ p0 ] -> eigvalsh_body ?uplo ?symmetrize_input p0
      | _ -> invalid_arg "eigvalsh: arity")
    [ p0 ]

and inv p0 =
  Jnp_util.jit1 "inv"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> inv_body p0 | _ -> invalid_arg "inv: arity")
    [ p0 ]

and norm ?ord ?axis ?keepdims p0 =
  Jnp_util.jit1 "norm"
    ~key:
      (Digest.to_hex
         (Digest.string (Marshal.to_string (ord, axis, keepdims) [])))
    (function
      | [ p0 ] -> norm_body ?ord ?axis ?keepdims p0
      | _ -> invalid_arg "norm: arity")
    [ p0 ]

and qr ?mode p0 =
  Jnp_util.jit "qr"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string mode [])))
    (function [ p0 ] -> qr_body ?mode p0 | _ -> invalid_arg "qr: arity")
    [ p0 ]

and solve p0 p1 =
  Jnp_util.jit1 "solve"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function
      | [ p0; p1 ] -> solve_body p0 p1 | _ -> invalid_arg "solve: arity")
    [ p0; p1 ]

and cond ?p p0 =
  Jnp_util.jit1 "cond"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string p [])))
    (function [ p0 ] -> cond_body ?p p0 | _ -> invalid_arg "cond: arity")
    [ p0 ]

and svd_uv ~full_matrices a =
  match LL.svd ~full_matrices ~compute_uv:true a with
  | [ u; s; vh ] -> (u, s, vh)
  | _ -> assert false

and svd_values a =
  match LL.svd ~full_matrices:false ~compute_uv:false a with
  | [ s ] -> s
  | _ -> assert false

and svd_body ?(full_matrices = true) ?(compute_uv = true) ?(hermitian = false) a
    =
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

and svdvals x = List.hd (svd ~compute_uv:false ~hermitian:false x)

and svd_hermitian ~compute_uv a =
  let module L = Lax_api in
  let w, v = LL.eigh a in
  let magnitudes = L.abs v in
  let last = ndim v - 1 in
  if not compute_uv then
    [ L.rev (List.hd (L.sort ~dimension:last [ magnitudes ])) [| last |] ]
  else
    let sign = L.sign v in
    let idx_dtype =
      if (shape magnitudes).(last) > 4294967295 then D.Uint64 else D.Uint32
    in
    let idxs = L.broadcasted_iota idx_dtype (shape magnitudes) last in
    match L.sort ~dimension:last ~num_keys:1 [ magnitudes; idxs; sign ] with
    | [ s; idxs; sign ] ->
        let s = L.rev s [| last |] in
        let idxs = L.rev idxs [| last |] in
        let sign = L.rev sign [| last |] in
        let u =
          IDX.take_along_axis ~axis:(-1) w (NL.expand_dims idxs [| last |])
        in
        let vh =
          hconj
            (UF.multiply u
               (NL.astype (NL.expand_dims sign [| last |]) (dtype u)))
        in
        [ u; s; vh ]
    | _ -> invalid_arg "svd: arity"

and solve_body a b =
  let a = to_inexact a and b = to_inexact b in
  let dt = Dtypes.promote_types (dtype a) (dtype b) in
  let a = NL.astype a dt and b = NL.astype b dt in
  let sa = shape a and sb = shape b in
  let na = Array.length sa and nb = Array.length sb in
  if na < 2 then invalid_arg "left hand array must be at least two dimensional";
  if nb < 1 then invalid_arg "right hand array must be at least one dimensional";
  if
    nb > 1
    && na = nb + 1
    && sa.(na - 1) = sb.(nb - 1)
    && sa.(na - 1) <> sb.(nb - 2)
  then
    invalid_arg
      "Invalid shapes for solve: Prior to JAX v0.5.0, this would have been \
       treated as a batched 1-dimensional solve. To recover this behavior, use \
       solve(a, b[..., None]).squeeze(-1).";
  let signature = if nb = 1 then "(m,m),(m)->(m)" else "(m,m),(m,n)->(m,n)" in
  Vectorize.vectorize ~signature
    (function
      | [ a; b ] -> Lax.solve_with_lu a b | _ -> invalid_arg "solve: arity")
    [ a; b ]

and inv_body a =
  let sa = shape a in
  let na = Array.length sa in
  if na < 2 || sa.(na - 1) <> sa.(na - 2) then
    invalid_arg "Argument to inv must have shape [..., n, n]";
  let n = sa.(na - 1) in
  let eye = NL.eye ~dtype:(dtype a) n in
  let batched =
    if na = 2 then eye
    else
      Core.bind1
        (T.Broadcast_in_dim
           { shape = sa; dims = Array.init 2 (fun i -> na - 2 + i) })
        [ eye ]
  in
  solve a batched

and slogdet_body ?(method_ = "lu") a =
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
    match !slogdet_lu_call [ a ] with
    | [ sign; logdet ] -> (sign, logdet)
    | _ -> assert false

and det_body a =
  let a = to_inexact a in
  let sa = shape a in
  let na = Array.length sa in
  if na >= 2 && sa.(na - 1) = 2 && sa.(na - 2) = 2 then det_2x2 a
  else if na >= 2 && sa.(na - 1) = 3 && sa.(na - 2) = 3 then begin
    let product (i0, j0) (i1, j1) (i2, j2) =
      let x = matrix_cell a i0 j0 in
      let y = matrix_cell a i1 j1 in
      let xy = UF.multiply x y in
      let z = matrix_cell a i2 j2 in
      UF.multiply xy z
    in
    let t0 = product (0, 0) (1, 1) (2, 2) in
    let t1 = product (0, 1) (1, 2) (2, 0) in
    let s1 = UF.add t0 t1 in
    let t2 = product (0, 2) (1, 0) (2, 1) in
    let s2 = UF.add s1 t2 in
    let t3 = product (0, 2) (1, 1) (2, 0) in
    let s3 = UF.subtract s2 t3 in
    let t4 = product (0, 0) (1, 2) (2, 1) in
    let s4 = UF.subtract s3 t4 in
    let t5 = product (0, 1) (1, 0) (2, 2) in
    UF.subtract s4 t5
  end
  else
    let sign, logdet = slogdet a in
    UF.multiply sign (NL.astype (UF.exp logdet) (dtype sign))

and matrix_cell a i j =
  let na = ndim a in
  let sh = shape a in
  let start_indices = Array.make na 0 and limit_indices = Array.copy sh in
  start_indices.(na - 2) <- i;
  limit_indices.(na - 2) <- i + 1;
  start_indices.(na - 1) <- j;
  limit_indices.(na - 1) <- j + 1;
  let sliced =
    C.bind1 (T.Slice { start_indices; limit_indices; strides = None }) [ a ]
  in
  C.bind1 (T.Squeeze [| na - 2; na - 1 |]) [ sliced ]

and det_2x2 a =
  let x = matrix_cell a 0 0 in
  let y = matrix_cell a 1 1 in
  let xy = UF.multiply x y in
  let u = matrix_cell a 0 1 in
  let v = matrix_cell a 1 0 in
  let uv = UF.multiply u v in
  UF.subtract xy uv

and eig a =
  let a = to_inexact a in
  match
    LL.eig ~compute_left_eigenvectors:false ~compute_right_eigenvectors:true a
  with
  | [ w; v ] -> (w, v)
  | _ -> assert false

and eigvals_body a =
  let a = to_inexact a in
  match
    LL.eig ~compute_left_eigenvectors:false ~compute_right_eigenvectors:false a
  with
  | [ w ] -> w
  | _ -> assert false

and eigh_body ?(uplo = "L") ?(symmetrize_input = true) a =
  let lower =
    match uplo with
    | "L" -> true
    | "U" -> false
    | _ -> invalid_arg "UPLO must be one of None, 'L', or 'U'"
  in
  let a = to_inexact a in
  let v, w = LL.eigh ~lower ~symmetrize_input a in
  (w, v)

and eigvalsh_body ?(uplo = "L") ?(symmetrize_input = true) a =
  let w, _ = eigh ~uplo ~symmetrize_input a in
  w

and pinv_body ~hermitian ~rtol a =
  let arr = to_inexact a in
  let sa = shape arr in
  let na = Array.length sa in
  let m = sa.(na - 2) and n = sa.(na - 1) in
  if m = 0 || n = 0 then
    Array_creation.empty ~dtype:(dtype arr)
      (Array.append (Array.sub sa 0 (na - 2)) [| n; m |])
  else
    let arr = UF.conj arr in
    let dt = dtype arr in
    let rtol =
      match rtol with
      | Some r -> r
      | None ->
          UF.multiply
            (Jnp_util.py_float (10.0 *. float_of_int (max m n)))
            (const
               (match dt with
               | D.Complex128 -> D.F64
               | D.Complex64 -> D.F32
               | d -> d)
               (eps_of dt))
    in
    let u, s, vh =
      match svd ~full_matrices:false ~hermitian arr with
      | [ u; s; vh ] -> (u, s, vh)
      | _ -> invalid_arg "pinv: svd arity"
    in
    let expanded = NL.expand_dims rtol [| ndim rtol |] in
    let missing = ndim s - ndim rtol - 1 in
    let rtol =
      if missing > 0 then
        Lax_api.expand_dims expanded (Array.init missing Fun.id)
      else expanded
    in
    let last = ndim s - 1 in
    let s0 =
      Core.bind1
        (Types.Slice
           {
             start_indices = Array.make (last + 1) 0;
             limit_indices =
               Array.mapi (fun i d -> if i = last then 1 else d) (shape s);
             strides = None;
           })
        [ s ]
    in
    let cutoff = UF.multiply rtol s0 in
    let s =
      Jnp_util.where_weak ~y_weak:true (UF.greater s cutoff) s
        (Jnp_util.py_float infinity)
    in
    let s = NL.astype s (dtype u) in
    let scaled =
      let vh_t = matrix_transpose_v vh in
      let u_t = matrix_transpose_v u in
      (vh_t, UF.divide u_t (NL.expand_dims s [| ndim s |]))
    in
    let res = TC.matmul ~precision:highest (fst scaled) (snd scaled) in
    NL.astype res dt

and pinv_jvp ~hermitian ~rtol a a_dot =
  let p = pinv_custom ~hermitian ~rtol a in
  let symmetrize x = UF.divide (UF.add x (hconj x)) (Jnp_util.py_int 2) in
  let a, a_dot =
    if hermitian then (symmetrize a, symmetrize a_dot) else (a, a_dot)
  in
  let sa = shape a in
  let na = Array.length sa in
  let m x y = TC.matmul x y in
  let ( +! ) = UF.add and ( -! ) = UF.subtract in
  let h = hconj in
  let p_dot =
    if sa.(na - 2) >= sa.(na - 1) then
      let s = m (m p (h p)) (h a_dot) in
      let t = m (m (h a_dot) (h p)) p in
      m (UF.negative (m p a_dot)) p +! s -! m (m s a) p +! t -! m (m p a) t
    else
      let s = m p (m (h p) (h a_dot)) in
      let t = m (h a_dot) (m (h p) p) in
      m (UF.negative p) (m a_dot p) +! s -! m s (m a p) +! t -! m p (m a t)
  in
  ([ p ], [ p_dot ])

and pinv_custom ~hermitian ~rtol a =
  let extra = match rtol with Some r -> [ r ] | None -> [] in
  let key =
    Digest.to_hex (Digest.string (Marshal.to_string (hermitian, extra = []) []))
  in
  let call =
    Custom_derivatives.custom_jvp ~name:"_pinv" ~jvp_name:"_pinv_jvp"
      (function
        | [ a ] ->
            [
              Jnp_util.jit1 "_pinv" ~key
                (function
                  | a :: rest ->
                      pinv_body ~hermitian
                        ~rtol:(match rest with r :: _ -> Some r | [] -> None)
                        a
                  | [] -> invalid_arg "pinv: arity")
                (a :: extra);
            ]
        | _ -> invalid_arg "pinv: arity")
      ~jvp:(fun primals tangents ->
        match (primals, tangents) with
        | [ a ], [ a_dot ] ->
            Config.with_value Config.default_matmul_precision (Some "float32")
              (fun () -> pinv_jvp ~hermitian ~rtol a a_dot)
        | _ -> invalid_arg "pinv: arity")
  in
  List.hd (call [ a ])

and pinv ?rtol ?(hermitian = false) ?rcond a =
  let rtol =
    match (rcond, rtol) with
    | Some _, Some _ ->
        invalid_arg "pinv: only one of rtol and rcond may be specified."
    | Some r, None -> Some r
    | None, r -> r
  in
  pinv_custom ~hermitian ~rtol a

and matrix_power_body a p =
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

and matrix_rank_body ?rtol ?(hermitian = false) ?tol a =
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
      else if o = infinity then
        match axis with
        | None -> RED.amax ~keepdims ~initial:(Jnp_util.py_int 0) (UF.abs x)
        | Some ax ->
            RED.amax ~axis:ax ~keepdims ~initial:(Jnp_util.py_int 0) (UF.abs x)
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
        RED.amax ~axis:[| c |] ~keepdims ~initial:(Jnp_util.py_int 0)
          (RED.sum ~axis:[| r |] ~keepdims (abs_x ()))
      end
      else if o = -1.0 then begin
        let c = if (not keepdims) && c > r then c - 1 else c in
        RED.amin ~axis:[| c |] ~keepdims
          (RED.sum ~axis:[| r |] ~keepdims (abs_x ()))
      end
      else if o = infinity then begin
        let r = if (not keepdims) && r > c then r - 1 else r in
        RED.amax ~axis:[| r |] ~keepdims ~initial:(Jnp_util.py_int 0)
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
          if o = 2.0 then RED.amax ~axis:[| -1 |] ~initial:(Jnp_util.py_int 0) s
          else RED.amin ~axis:[| -1 |] s
        in
        if keepdims then NL.expand_dims y [| r; c |] else y
      end
      else invalid_arg "Invalid order for matrix norm."

and norm_body ?(ord = Onone) ?(axis = Anone) ?(keepdims = false) x =
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

and matrix_transpose x = matrix_transpose_v x

and raw_qr a =
  match Core.bind T.Geqrf [ a ] with
  | [ h; t ] -> [ matrix_transpose_v h; t ]
  | _ -> failwith "jnp.linalg.qr: raw mode arity"

and qr_body ?(mode = "reduced") a =
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
  if numpy_resid then
    lstsq_body ~rcond:(Option.map (fun r -> `Python r) rcond) ~numpy_resid a b
  else
    let call key args body =
      match
        Jnp_util.jit "_lstsq" ~key
          (fun args ->
            let x, resid, rank, s = body args in
            [ x; resid; rank; s ])
          args
      with
      | [ x; resid; rank; s ] -> (x, resid, rank, s)
      | _ -> invalid_arg "_lstsq: arity"
    in
    match rcond with
    | None ->
        call "default" [ a; b ] (function
          | [ a; b ] -> lstsq_body ~rcond:None ~numpy_resid a b
          | _ -> invalid_arg "_lstsq: arity")
    | Some r ->
        call "traced"
          [ a; b; Jnp_util.py_float r ]
          (function
            | [ a; b; r ] ->
                lstsq_body ~rcond:(Some (`Traced r)) ~numpy_resid a b
            | _ -> invalid_arg "_lstsq: arity")

and lstsq_body ~rcond ~numpy_resid a b =
  let a = to_inexact a and b = to_inexact b in
  let b_vec = ndim b = 1 in
  let bm = if b_vec then NL.expand_dims b [| 1 |] else b in
  let sa = shape a in
  let m = sa.(0) and n = sa.(1) in
  let dt = dtype a in
  let rcond =
    match rcond with
    | Some (`Python r) when r < 0.0 -> const dt (eps_of dt)
    | Some (`Python r) -> const dt r
    | Some (`Traced r) ->
        let negative = UF.less r (Jnp_util.py_int 0) in
        Jnp_util.where_weak ~x_weak:false negative (const dt (eps_of dt)) r
    | None -> const dt (eps_of dt *. float_of_int (max n m))
  in
  let u, s, vt =
    match
      Jnp_util.jit "svd" ~key:"reduced"
        (function
          | [ a ] ->
              let u, s, vh = svd_uv ~full_matrices:false a in
              [ u; s; vh ]
          | _ -> invalid_arg "svd: arity")
        [ a ]
    with
    | [ u; s; vh ] -> (u, s, vh)
    | _ -> invalid_arg "svd: arity"
  in
  let positive = UF.greater s (const dt 0.0) in
  let rcond =
    match rcond with
    | T.Tracer _ -> Core.bind1 T.Copy [ NL.astype rcond (dtype s) ]
    | _ -> NL.astype rcond (dtype s)
  in
  let s0 =
    if ndim s = 1 then
      Core.bind1 (T.Squeeze [| 0 |])
        [
          Core.bind1
            (T.Slice
               {
                 start_indices = [| 0 |];
                 limit_indices = [| 1 |];
                 strides = None;
               })
            [ s ];
        ]
    else RED.amax ~axis:[| -1 |] ~keepdims:true s
  in
  let mask =
    UF.logical_and positive (UF.greater_equal s (UF.multiply rcond s0))
  in
  let rank = RED.sum mask in
  let safe_s = NL.astype (NL.where_ mask s (Jnp_util.py_int 1)) dt in
  let s_inv =
    NL.where_ mask (UF.divide (Jnp_util.py_int 1) safe_s) (Jnp_util.py_int 0)
  in
  let s_inv_col = NL.expand_dims s_inv [| ndim s_inv |] in
  let utb = TC.matmul ~precision:highest (UF.conj (matrix_transpose_v u)) bm in
  let x =
    TC.matmul ~precision:highest
      (UF.conj (matrix_transpose_v vt))
      (UF.multiply s_inv_col utb)
  in
  let b_est = TC.matmul ~precision:highest a x in
  let full_resid () =
    Core.bind1 (T.Integer_pow 2) [ norm ~axis:(Aint 0) (UF.subtract bm b_est) ]
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

and cond_body ?(p = Onone) x =
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

let slogdet_lu a =
  let dt = dtype a in
  let n = (shape a).(ndim a - 1) in
  let lu, pivot, _ = LL.lu a in
  let diag = NL.diagonal ~axis1:(-2) ~axis2:(-1) lu in
  let eq0 = UF.equal diag (const dt 0.0) in
  let is_zero = RED.any ~axis:[| -1 |] eq0 in
  let iota = NL.arange ~dtype:(dtype pivot) (float_of_int n) in
  let ne = UF.not_equal pivot iota in
  let parity = RED.count_nonzero ~axis:[| -1 |] ne in
  let is_complex = dt = D.Complex64 || dt = D.Complex128 in
  let sign, parity =
    if is_complex then begin
      let ab = UF.abs diag in
      let conv = NL.astype ab dt in
      let unit = UF.divide diag conv in
      (RED.prod ~axis:[| -1 |] unit, parity)
    end
    else begin
      let lt = UF.less diag (const dt 0.0) in
      let c = RED.count_nonzero ~axis:[| -1 |] lt in
      (const dt 1.0, UF.add parity c)
    end
  in
  let pdt = dtype parity in
  let rem = UF.remainder parity (Core.py_scalar pdt 2.0) in
  let m = UF.multiply (const pdt (-2.0)) rem in
  let s = UF.add m (const pdt 1.0) in
  let copied = C.bind1 T.Copy [ s ] in
  let conv = NL.astype copied dt in
  let product = UF.multiply sign conv in
  let sign = NL.where_ is_zero (const dt 0.0) product in
  let ab = UF.abs diag in
  let lg = UF.log ab in
  let lg = NL.astype lg dt in
  let total = RED.sum ~axis:[| -1 |] lg in
  let logdet = NL.where_ is_zero (const dt neg_infinity) total in
  (sign, UF.real logdet)

let () =
  let self = ref (fun _ -> []) in
  let call =
    Custom_derivatives.custom_jvp ~name:"_slogdet_lu" ~jvp_name:"_slogdet_jvp"
      (function
        | [ a ] ->
            let sign, logdet = slogdet_lu a in
            [ sign; logdet ]
        | _ -> invalid_arg "slogdet: arity")
      ~jvp:(fun primals tangents ->
        match (primals, tangents) with
        | [ x ], [ g ] -> (
            match !self [ x ] with
            | [ sign; ans ] as out ->
                let solved = solve x g in
                let ans_dot = NL.trace ~axis1:(-1) ~axis2:(-2) solved in
                if dtype x = D.Complex64 || dtype x = D.Complex128 then begin
                  let re = UF.real ans_dot in
                  let conv = NL.astype re (dtype ans_dot) in
                  let d = UF.subtract ans_dot conv in
                  let sign_dot = UF.multiply d sign in
                  let re2 = UF.real ans_dot in
                  (out, [ sign_dot; re2 ])
                end
                else (out, [ C.splat (dtype sign) (shape sign) 0.0; ans_dot ])
            | _ -> invalid_arg "slogdet: arity")
        | _ -> invalid_arg "slogdet: arity")
  in
  self := call;
  slogdet_lu_call := call

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

let matrix_norm ?(keepdims = false) ?(ord = Ofro) x =
  norm ~ord ~axis:(Apair (-2, -1)) ~keepdims x

let multi_dot arrs =
  (match arrs with
  | [] | [ _ ] -> invalid_arg "multi_dot requires at least two arrays"
  | _ -> ());
  let letter i = String.make 1 (Char.chr (Char.code 'a' + i)) in
  let term i arr =
    let left = letter i and right = letter (i + 1) in
    if ndim arr = 1 then if i = 0 then right else left else left ^ right
  in
  Einsum.einsum (String.concat "," (List.mapi term arrs)) arrs

let () = Tree_util.register_attribute_keys "QRResult" [ "Q"; "R" ]

let qr_result_def =
  Tree_util.Custom_def
    ("QRResult", "", [ Tree_util.Leaf_def; Tree_util.Leaf_def ])

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
