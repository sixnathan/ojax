module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module LL = Lax.Linalg
module LN = Numpy.Linalg
module NL = Numpy.Lax_numpy
module UF = Numpy.Ufuncs
module TC = Numpy.Tensor_contractions
module IDX = Numpy.Indexing
module SORT = Numpy.Sorting
module RED = Numpy.Reductions
module AC = Numpy.Array_creation

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let ndim v = Array.length (shape v)

let is_inexact = function
  | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> true
  | _ -> false

let default_float () = Dtypes.default_float_dtype ()

let to_inexact v =
  if is_inexact (dtype v) then v else NL.astype v (default_float ())

let matrix_transpose_v x = NL.matrix_transpose x
let hconj x = UF.conj (matrix_transpose_v x)
let iota_idx n = NL.arange ~dtype:D.I32 (float_of_int n)

let cholesky ?(lower = false) a =
  let a = to_inexact a in
  let a = if lower then a else hconj a in
  let l = LL.cholesky a in
  if lower then l else hconj l

let cho_factor ?(lower = false) a = (cholesky ~lower a, lower)

let cho_solve (c, lower) b =
  let c = to_inexact c and b = to_inexact b in
  let sc = shape c in
  let n = sc.(Array.length sc - 1) in
  let bvec = ndim b = 1 in
  let bm = if bvec then NL.reshape b [| n; 1 |] else b in
  let x1 =
    LL.triangular_solve ~left_side:true ~lower ~transpose_a:(not lower)
      ~conjugate_a:(not lower) c bm
  in
  let x2 =
    LL.triangular_solve ~left_side:true ~lower ~transpose_a:lower
      ~conjugate_a:lower c x1
  in
  if bvec then NL.reshape x2 [| n |] else x2

let det a = LN.det a
let inv a = LN.inv a

let lu ?(permute_l = false) a =
  let a = to_inexact a in
  let dt = dtype a in
  let sa = shape a in
  let m = sa.(0) and n = sa.(1) in
  let k = min m n in
  let lu_, _piv, permutation = LL.lu a in
  let perm_row = NL.expand_dims permutation [| 0 |] in
  let iota = NL.arange ~dtype:(dtype permutation) (float_of_int m) in
  let iota_col = NL.expand_dims iota [| 1 |] in
  let p = NL.astype (UF.equal perm_row iota_col) dt in
  let kcols = iota_idx k in
  let l_lower = IDX.take ~axis:1 (NL.tril ~k:(-1) lu_) kcols in
  let l = UF.add l_lower (NL.eye ~m:k ~dtype:dt m) in
  let u = IDX.take ~axis:0 (NL.triu lu_) kcols in
  if permute_l then [ TC.matmul p l; u ] else [ p; l; u ]

let lu_factor a =
  let a = to_inexact a in
  let lu_, piv, _ = LL.lu a in
  (lu_, piv)

let lu_solve ?(trans = 0) (lu_, piv) b =
  let lu_ = to_inexact lu_ and b = to_inexact b in
  let sa = shape lu_ in
  let m = sa.(Array.length sa - 2) in
  let perm = LL.lu_pivots_to_permutation ~permutation_size:m piv in
  let bvec = ndim b = 1 in
  let x = if bvec then NL.reshape b [| m; 1 |] else b in
  let out =
    if trans = 0 then begin
      let x = IDX.take ~axis:0 x perm in
      let x =
        LL.triangular_solve ~left_side:true ~lower:true ~unit_diagonal:true lu_
          x
      in
      LL.triangular_solve ~left_side:true ~lower:false lu_ x
    end
    else if trans = 1 || trans = 2 then begin
      let conj = trans = 2 in
      let x =
        LL.triangular_solve ~left_side:true ~lower:false ~transpose_a:true
          ~conjugate_a:conj lu_ x
      in
      let x =
        LL.triangular_solve ~left_side:true ~lower:true ~unit_diagonal:true
          ~transpose_a:true ~conjugate_a:conj lu_ x
      in
      IDX.take ~axis:0 x (SORT.argsort perm)
    end
    else invalid_arg (Printf.sprintf "lu_solve: invalid trans value %d" trans)
  in
  if bvec then NL.reshape out [| m |] else out

let qr ?(mode = "full") ?(pivoting = false) a =
  if pivoting then
    invalid_arg "scipy.linalg.qr: pivoting=True not supported in ojax";
  let full_matrices =
    match mode with
    | "full" | "r" -> true
    | "economic" -> false
    | _ ->
        invalid_arg
          (Printf.sprintf "Unsupported QR decomposition mode '%s'" mode)
  in
  let a = to_inexact a in
  let q, r = LL.qr ~full_matrices a in
  if mode = "r" then [ r ] else [ q; r ]

let solve ?(lower = false) ?(assume_a = "gen") a b =
  (match assume_a with
  | "gen" | "sym" | "her" | "pos" -> ()
  | _ ->
      invalid_arg
        (Printf.sprintf
           "Expected assume_a to be one of ['gen', 'sym', 'her', 'pos']; got \
            '%s'"
           assume_a));
  if assume_a <> "pos" then LN.solve a b else cho_solve (cho_factor ~lower a) b

let solve_triangular ?(trans = 0) ?(lower = false) ?(unit_diagonal = false) a b
    =
  let transpose_a, conjugate_a =
    match trans with
    | 0 -> (false, false)
    | 1 -> (true, false)
    | 2 -> (true, true)
    | _ -> invalid_arg (Printf.sprintf "Invalid 'trans' value %d" trans)
  in
  let a = to_inexact a and b = to_inexact b in
  let sa = shape a in
  let n = sa.(Array.length sa - 1) in
  let bvec = ndim b = 1 in
  let bm = if bvec then NL.reshape b [| n; 1 |] else b in
  let x =
    LL.triangular_solve ~left_side:true ~lower ~transpose_a ~conjugate_a
      ~unit_diagonal a bm
  in
  if bvec then NL.reshape x [| n |] else x

let host_nd = function
  | T.Concrete nd -> nd
  | _ ->
      failwith "scipy/linalg: forward-eval only (concrete host inputs required)"

let host_float v = Nd.get_f (host_nd v) [||]
let scalar_of v x = T.Concrete (Nd.of_floats (dtype v) [||] [| x |])
let i32_scalar x = T.Concrete (Nd.of_floats D.I32 [||] [| float_of_int x |])

let svd ?(full_matrices = true) ?(compute_uv = true) a =
  let a = to_inexact a in
  LL.svd ~full_matrices ~compute_uv a

let eigh ?(lower = true) ?(eigvals_only = false) a =
  let a = to_inexact a in
  let v, w = LL.eigh ~lower a in
  if eigvals_only then [ w ] else [ w; v ]

let schur ?(output = "real") a =
  let a = to_inexact a in
  (match output with
  | "real" -> ()
  | "complex" ->
      failwith
        "scipy.linalg.schur: output='complex' unsupported in ojax (needs the \
         complex Schur decomposition / LAPACK zgees; the seam is real-double, \
         row 99)"
  | _ ->
      invalid_arg
        (Printf.sprintf "argument must be 'real' or 'complex', got %s" output));
  LL.schur a

let block_diag arrs =
  let arrs =
    match arrs with
    | [] -> [ AC.zeros ~dtype:(default_float ()) [| 1; 0 |] ]
    | l -> l
  in
  List.iteri
    (fun i a ->
      if ndim a > 2 then
        invalid_arg
          (Printf.sprintf
             "Arguments to jax.scipy.linalg.block_diag must have at most 2 \
              dimensions, got %d dimensions at argument %d."
             (ndim a) i))
    arrs;
  let dt = NL.result_type arrs in
  let arrs = List.map (fun a -> NL.atleast_2d (NL.astype a dt)) arrs in
  let total_cols = List.fold_left (fun s a -> s + (shape a).(1)) 0 arrs in
  let cur = ref 0 in
  let padded =
    List.map
      (fun a ->
        let cols = (shape a).(1) in
        let p = NL.pad a [| (0, 0); (!cur, total_cols - cols - !cur) |] 0.0 in
        cur := !cur + cols;
        p)
      arrs
  in
  NL.concatenate ~axis:0 padded

let toeplitz ?r c =
  let c = NL.atleast_1d c in
  let r = match r with None -> UF.conj c | Some r -> NL.atleast_1d r in
  let m = (shape c).(0) and n = (shape r).(0) in
  if m = 0 || n = 0 then
    AC.zeros ~dtype:(Dtypes.promote_types (dtype c) (dtype r)) [| m; n |]
  else begin
    let r_tail =
      if n = 1 then NL.reshape r [| 0 |]
      else IDX.take r (UF.add (iota_idx (n - 1)) (i32_scalar 1))
    in
    let elems = NL.concatenate [ NL.flip c; r_tail ] in
    let ii = NL.reshape (iota_idx m) [| m; 1 |] in
    let jj = NL.reshape (iota_idx n) [| 1; n |] in
    let idx = UF.add (UF.subtract (i32_scalar (m - 1)) ii) jj in
    let flat = NL.reshape idx [| m * n |] in
    NL.reshape (IDX.take elems flat) [| m; n |]
  end

let hessenberg ?(calc_q = false) a =
  let a = to_inexact a in
  let sh = shape a in
  let n = sh.(Array.length sh - 1) in
  if n = 0 then
    if calc_q then [ AC.zeros_like a; AC.zeros_like a ] else [ AC.zeros_like a ]
  else begin
    let a_out, taus = LL.hessenberg a in
    let h = NL.triu ~k:(-1) a_out in
    if not calc_q then [ h ]
    else begin
      let rows = UF.add (iota_idx (n - 1)) (i32_scalar 1) in
      let cols = iota_idx (n - 1) in
      let sub = IDX.take ~axis:1 (IDX.take ~axis:0 a_out rows) cols in
      let q_inner = LL.householder_product sub taus in
      let dt = dtype a in
      let top =
        NL.concatenate ~axis:1
          [ AC.ones ~dtype:dt [| 1; 1 |]; AC.zeros ~dtype:dt [| 1; n - 1 |] ]
      in
      let bot =
        NL.concatenate ~axis:1 [ AC.zeros ~dtype:dt [| n - 1; 1 |]; q_inner ]
      in
      [ h; NL.concatenate ~axis:0 [ top; bot ] ]
    end
  end

let expm ?(upper_triangular = false) a =
  let a = to_inexact a in
  let sh = shape a in
  let nd = Array.length sh in
  if nd < 2 || sh.(nd - 1) <> sh.(nd - 2) then
    invalid_arg "expm: expected A to be a (batched) square matrix";
  if nd > 2 then
    failwith "scipy.linalg.expm: batched inputs (ndim>2) unsupported in ojax";
  let n = sh.(0) in
  let dt = dtype a in
  let ident = NL.eye ~dtype:dt n in
  let sc co m = UF.multiply (scalar_of a co) m in
  let dotp = TC.matmul in
  let l1f = host_float (RED.amax (RED.sum ~axis:[| 0 |] (UF.abs a))) in
  let is64 = dt = D.F64 || dt = D.Complex128 in
  let maxnorm = if is64 then 5.371920351148152 else 3.925724783138660 in
  let nsq =
    if l1f <= 0.0 then 0
    else max 0 (int_of_float (Float.floor (Float.log2 (l1f /. maxnorm))))
  in
  let max_squarings = 16 in
  if nsq > max_squarings then AC.full_like a Float.nan
  else begin
    let a =
      if nsq = 0 then a else UF.divide a (scalar_of a (2.0 ** float_of_int nsq))
    in
    let conds =
      if is64 then
        [|
          1.495585217958292e-2;
          2.539398330063230e-1;
          9.504178996162932e-1;
          2.097847961257068;
        |]
      else [| 4.258730016922831e-1; 1.880152677804762 |]
    in
    let idx =
      Array.fold_left (fun acc c -> if c <= l1f then acc + 1 else acc) 0 conds
    in
    let pade3 () =
      let a2 = dotp a a in
      let u = dotp a (UF.add (sc 1.0 a2) (sc 60.0 ident)) in
      let v = UF.add (sc 12.0 a2) (sc 120.0 ident) in
      (u, v)
    in
    let pade5 () =
      let a2 = dotp a a in
      let a4 = dotp a2 a2 in
      let u =
        dotp a (UF.add (sc 1.0 a4) (UF.add (sc 420.0 a2) (sc 15120.0 ident)))
      in
      let v = UF.add (sc 30.0 a4) (UF.add (sc 3360.0 a2) (sc 30240.0 ident)) in
      (u, v)
    in
    let pade7 () =
      let a2 = dotp a a in
      let a4 = dotp a2 a2 in
      let a6 = dotp a4 a2 in
      let u =
        dotp a
          (UF.add (sc 1.0 a6)
             (UF.add (sc 1512.0 a4)
                (UF.add (sc 277200.0 a2) (sc 8648640.0 ident))))
      in
      let v =
        UF.add (sc 56.0 a6)
          (UF.add (sc 25200.0 a4)
             (UF.add (sc 1995840.0 a2) (sc 17297280.0 ident)))
      in
      (u, v)
    in
    let pade9 () =
      let a2 = dotp a a in
      let a4 = dotp a2 a2 in
      let a6 = dotp a4 a2 in
      let a8 = dotp a6 a2 in
      let u =
        dotp a
          (UF.add (sc 1.0 a8)
             (UF.add (sc 3960.0 a6)
                (UF.add (sc 2162160.0 a4)
                   (UF.add (sc 302702400.0 a2) (sc 8821612800.0 ident)))))
      in
      let v =
        UF.add (sc 90.0 a8)
          (UF.add (sc 110880.0 a6)
             (UF.add (sc 30270240.0 a4)
                (UF.add (sc 2075673600.0 a2) (sc 17643225600.0 ident))))
      in
      (u, v)
    in
    let pade13 () =
      let a2 = dotp a a in
      let a4 = dotp a2 a2 in
      let a6 = dotp a4 a2 in
      let u =
        dotp a
          (UF.add
             (dotp a6
                (UF.add (sc 1.0 a6) (UF.add (sc 16380.0 a4) (sc 40840800.0 a2))))
             (UF.add (sc 33522128640.0 a6)
                (UF.add (sc 10559470521600.0 a4)
                   (UF.add (sc 1187353796428800.0 a2)
                      (sc 32382376266240000.0 ident)))))
      in
      let v =
        UF.add
          (dotp a6
             (UF.add (sc 182.0 a6)
                (UF.add (sc 960960.0 a4) (sc 1323241920.0 a2))))
          (UF.add (sc 670442572800.0 a6)
             (UF.add (sc 129060195264000.0 a4)
                (UF.add (sc 7771770303897600.0 a2)
                   (sc 64764752532480000.0 ident))))
      in
      (u, v)
    in
    let pades =
      if is64 then [| pade3; pade5; pade7; pade9; pade13 |]
      else [| pade3; pade5; pade7 |]
    in
    let u, v = pades.(idx) () in
    let p = UF.add u v and q = UF.add (UF.negative u) v in
    let r = if upper_triangular then solve_triangular q p else LN.solve q p in
    let rec sq r k = if k = 0 then r else sq (dotp r r) (k - 1) in
    sq r nsq
  end

let expm_frechet ?(compute_expm = true) a e =
  ignore compute_expm;
  ignore a;
  ignore e;
  failwith
    "scipy.linalg.expm_frechet: unsupported in ojax (its Frechet derivative is \
     jvp(expm), which requires forward-mode AD through a linear solve; linalg \
     jvp rules are an M5 gap, rows 98/99)"

let polar ?(side = "right") ?(method_ = "qdwh") a =
  let a = to_inexact a in
  if ndim a < 2 then invalid_arg "The input `a` must be at least a 2-D array.";
  (match side with
  | "right" | "left" -> ()
  | _ -> invalid_arg "The argument `side` must be either 'right' or 'left'.");
  match method_ with
  | "qdwh" ->
      failwith
        "scipy.linalg.polar: method='qdwh' unsupported in ojax (qdwh lives in \
         the deferred jax._src.tpu.linalg tree); use method=\"svd\""
  | "svd" ->
      let u, s, vh =
        match LL.svd ~full_matrices:false a with
        | [ u; s; vh ] -> (u, s, vh)
        | _ -> failwith "polar: svd arity"
      in
      let s = NL.astype s (dtype u) in
      let unitary = TC.matmul u vh in
      let s_row = NL.expand_dims s [| 0 |] in
      let posdef =
        if side = "right" then TC.matmul (UF.multiply (hconj vh) s_row) vh
        else TC.matmul (UF.multiply u s_row) (hconj u)
      in
      [ unitary; posdef ]
  | _ ->
      invalid_arg
        (Printf.sprintf "Unknown polar decomposition method %s." method_)

let sqrtm ?(blocksize = 1) a =
  if blocksize > 1 then
    failwith "scipy.linalg.sqrtm: blocked version (blocksize>1) not implemented";
  ignore a;
  failwith
    "scipy.linalg.sqrtm: unsupported in ojax (requires the complex Schur \
     decomposition / LAPACK zgees, out of scope for the real-double seam, row \
     99)"

let funm ?(disp = true) a _func =
  ignore disp;
  ignore a;
  failwith
    "scipy.linalg.funm: unsupported in ojax (requires rsf2csf real->complex \
     Schur-form conversion over the complex Schur machinery deferred with row \
     99)"

let eigh_tridiagonal ?(eigvals_only = false) ?(select = "a") ?select_range d e =
  ignore select_range;
  if not eigvals_only then
    failwith
      "scipy.linalg.eigh_tridiagonal: eigenvector computation \
       (eigvals_only=False) unsupported in ojax (needs random-init inverse \
       iteration + static-size nonzero; bounded, M5)";
  (match select with
  | "a" -> ()
  | _ ->
      failwith
        "scipy.linalg.eigh_tridiagonal: only select='a' supported in ojax");
  let d = to_inexact d and e = to_inexact e in
  let n = (shape d).(0) in
  if n <= 1 then UF.real d
  else begin
    let t = UF.add (NL.diag d) (UF.add (NL.diag ~k:1 e) (NL.diag ~k:(-1) e)) in
    let _, w = LL.eigh ~lower:true t in
    w
  end
