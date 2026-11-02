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
let i32_scalar x = T.Concrete (Nd.of_floats D.I32 [||] [| float_of_int x |])
let is_complex_dt = function D.Complex64 | D.Complex128 -> true | _ -> false
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let ndim v = Array.length (shape v)

let is_inexact = function
  | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> true
  | _ -> false

let default_float () = Dtypes.default_float_dtype ()

let to_inexact v =
  if is_inexact (dtype v) then v
  else NL.astype v (Dtypes.to_inexact_dtype (dtype v))

let matrix_transpose_v x = NL.matrix_transpose x
let hconj x = UF.conj (matrix_transpose_v x)
let iota_idx n = NL.arange ~dtype:D.I32 (float_of_int n)

let cholesky ?(lower = false) a =
  let a = to_inexact a in
  let a = if lower then a else hconj a in
  let l = LL.cholesky ~symmetrize_input:false a in
  if lower then l else hconj l

let cholesky ?lower a0 =
  Eager_jit.one ~jit:"_cholesky" "scipy.linalg.cholesky" lower [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> cholesky ?lower a0
      | _ -> assert false)

let cho_factor ?(lower = false) a = (cholesky ~lower a, lower)

let triangular_signature ?(extra = "") a b =
  let vector = Printf.sprintf "(n,n)%s,(n)->(n)" extra in
  if ndim b = 1 then vector
  else if
    ndim a = ndim b + 1
    && Validate.tuple_index (shape a) (-1) = Validate.tuple_index (shape b) (-1)
  then vector
  else Printf.sprintf "(n,n)%s,(n,k)->(n,k)" extra

let cho_solve (c, lower) b =
  let c = to_inexact c and b = to_inexact b in
  let signature = triangular_signature c b in
  Numpy.Vectorize.vectorize ~signature
    (function
      | [ c; b ] ->
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
      | _ -> assert false)
    [ c; b ]

let det a = LN.det a
let inv a = LN.inv a

let lu ?(permute_l = false) a =
  let body a =
    let a = to_inexact a in
    let dt = dtype a in
    let lu_, _piv, permutation = LL.lu a in
    let m, n = Validate.unpack_exact_two (shape a) in
    let k = min m n in
    let perm_row = NL.expand_dims permutation [| 0 |] in
    let iota = NL.arange ~dtype:(dtype permutation) (float_of_int m) in
    let iota_col = NL.expand_dims iota [| 1 |] in
    let p = UF.real (NL.astype (UF.equal perm_row iota_col) dt) in
    let cut axis size v =
      if (shape v).(axis) = size then v else Lax_api.slice_in_dim ~axis v 0 size
    in
    let l_lower = cut 1 k (NL.tril ~k:(-1) lu_) in
    let l = UF.add l_lower (NL.eye ~m:k ~dtype:dt m) in
    let u = cut 0 k (NL.triu lu_) in
    if permute_l then
      [
        TC.matmul
          ~precision:
            (Lax_config.precision_of_level Lax_config.Precision_highest)
          p l;
        u;
      ]
    else [ p; l; u ]
  in
  Numpy.Jnp_util.jit "_lu" ~key:(string_of_bool permute_l)
    (function [ a ] -> body a | _ -> assert false)
    [ a ]

let lu ?permute_l a0 =
  Eager_jit.many ~jit:"lu" "scipy.linalg.lu" permute_l [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> lu ?permute_l a0
      | _ -> assert false)

let lu_factor a =
  let a = to_inexact a in
  let lu_, piv, _ = LL.lu a in
  (lu_, piv)

let lu_factor a0 =
  Eager_jit.pair ~jit:"lu_factor" "scipy.linalg.lu_factor" () [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> lu_factor a0 | _ -> assert false)

let lu_solve ?(trans = 0) (lu_, piv) b =
  let lu_ = to_inexact lu_ and b = to_inexact b in
  let m, _ = Validate.unpack_last_two (shape lu_) in
  let perm = LL.lu_pivots_to_permutation ~permutation_size:m piv in
  let signature = triangular_signature ~extra:",(n)" lu_ b in
  Numpy.Vectorize.vectorize ~signature
    (function
      | [ lu_; perm; b ] ->
          let m = (shape lu_).(ndim lu_ - 2) in
          let bvec = ndim b = 1 in
          let x = if bvec then NL.reshape b [| m; 1 |] else b in
          let out =
            if trans = 0 then begin
              let x = IDX.take_index ~axis:0 x perm in
              let x =
                LL.triangular_solve ~left_side:true ~lower:true
                  ~unit_diagonal:true lu_ x
              in
              LL.triangular_solve ~left_side:true ~lower:false lu_ x
            end
            else if trans = 1 || trans = 2 then begin
              let conj = trans = 2 in
              let x =
                LL.triangular_solve ~left_side:true ~lower:false
                  ~transpose_a:true ~conjugate_a:conj lu_ x
              in
              let x =
                LL.triangular_solve ~left_side:true ~lower:true
                  ~unit_diagonal:true ~transpose_a:true ~conjugate_a:conj lu_ x
              in
              IDX.take_index ~axis:0 x (SORT.argsort perm)
            end
            else
              Py_exn.value_error
                (Printf.sprintf "'trans' value must be 0, 1, or 2, got %d" trans)
          in
          if bvec then NL.reshape out [| m |] else out
      | _ -> assert false)
    [ lu_; perm; b ]

let qr ?(mode = "full") ?(pivoting = false) a =
  let full_matrices =
    match mode with
    | "full" | "r" -> true
    | "economic" -> false
    | _ ->
        Py_exn.value_error
          (Printf.sprintf "Unsupported QR decomposition mode '%s'" mode)
  in
  Numpy.Jnp_util.jit "_qr"
    ~key:(mode ^ if pivoting then "/p" else "")
    (function
      | [ a ] -> (
          let a = to_inexact a in
          if not pivoting then
            let q, r = LL.qr ~full_matrices a in
            if mode = "r" then [ r ] else [ q; r ]
          else
            match Core.bind (T.Qr_pivoted full_matrices) [ a ] with
            | [ q; r; p ] -> if mode = "r" then [ r; p ] else [ q; r; p ]
            | _ -> assert false)
      | _ -> assert false)
    [ a ]

let rec qr_multiply ?(mode = "right") ?(pivoting = false) ?(conjugate = false) a
    c =
  let a = to_inexact a and c = to_inexact c in
  let dt =
    if is_complex_dt (dtype a) || is_complex_dt (dtype c) then
      if
        dtype a = D.Complex128
        || dtype c = D.Complex128
        || dtype a = D.F64
        || dtype c = D.F64
      then D.Complex128
      else D.Complex64
    else if dtype a = D.F64 || dtype c = D.F64 then D.F64
    else D.F32
  in
  let a = NL.astype a dt and c = NL.astype c dt in
  if mode <> "right" && mode <> "left" then
    Py_exn.value_error
      (Printf.sprintf "mode must be 'right' or 'left', got '%s'" mode);
  let onedim = Array.length (shape c) = 1 in
  let c =
    if not onedim then c
    else
      let l = (shape c).(0) in
      NL.reshape c (if mode = "left" then [| l; 1 |] else [| 1; l |])
  in
  let sh = shape a in
  let rank = Array.length sh in
  let m, n = Validate.unpack_last_two sh in
  let k = min m n in
  let cs = shape c in
  let crank = Array.length cs in
  ignore (Validate.tuple_index cs (if mode = "left" then -2 else -1));
  let shape_text s =
    match s with
    | [| x |] -> Printf.sprintf "(%d,)" x
    | _ ->
        "("
        ^ String.concat ", " (Array.to_list (Array.map string_of_int s))
        ^ ")"
  in
  if mode = "left" && cs.(crank - 2) <> k then
    Py_exn.value_error
      (Printf.sprintf
         "Array shapes are not compatible for Q @ c operation: a has shape %s \
          so Q has %d columns, but c has %d rows (expected %d)."
         (shape_text sh) k
         cs.(crank - 2)
         k);
  if mode = "right" && cs.(crank - 1) <> m then
    Py_exn.value_error
      (Printf.sprintf
         "Array shapes are not compatible for c @ Q operation: a has shape %s \
          so Q has %d rows, but c has %d columns (expected %d)."
         (shape_text sh) m
         cs.(crank - 1)
         m);
  let batch =
    NL.broadcast_shapes_n
      [ Array.sub sh 0 (rank - 2); Array.sub cs 0 (crank - 2) ]
  in
  let a = NL.broadcast_to a (Array.append batch [| m; n |]) in
  let c =
    NL.broadcast_to c (Array.append batch [| cs.(crank - 2); cs.(crank - 1) |])
  in
  let h, perm, taus =
    if pivoting then
      let jpvt = AC.zeros ~dtype:D.I32 (Array.append batch [| n |]) in
      let h, p, t = Linalg_extras.geqp3 a jpvt in
      (h, Some (UF.subtract p (i32_scalar 1)), t)
    else
      let h, t = Linalg_extras.geqrf a in
      (h, None, t)
  in
  let c =
    if m > n && mode = "left" then
      let cs = shape c in
      let r = Array.length cs in
      let zeros =
        AC.zeros ~dtype:(dtype c)
          (Array.append (Array.sub cs 0 (r - 2)) [| m - k; cs.(r - 1) |])
      in
      NL.concatenate ~axis:(-2) [ c; zeros ]
    else c
  in
  let c = if conjugate then NL.swapaxes (-1) (-2) c else c in
  let left = mode = "left" <> conjugate in
  let cq = Linalg_extras.ormqr ~left ~transpose:conjugate h taus c in
  let cq = if conjugate then NL.swapaxes (-1) (-2) cq else cq in
  let cq =
    if mode = "right" then (
      let r = Array.length (shape cq) in
      let limits = Array.copy (shape cq) in
      limits.(r - 1) <- k;
      if limits = shape cq then cq else Lax_api.slice cq (Array.make r 0) limits)
    else cq
  in
  let cq =
    if onedim then NL.reshape cq [| Array.fold_left ( * ) 1 (shape cq) |]
    else cq
  in
  let starts = Array.make rank 0 in
  let limits = Array.append batch [| k; n |] in
  let r =
    NL.triu (if limits = shape h then h else Lax_api.slice h starts limits)
  in
  match perm with None -> [ cq; r ] | Some p -> [ cq; r; p ]

let rsf2csf_2d t z =
  let module L = Lax_api in
  let widest =
    dtype t = D.F64
    || dtype z = D.F64
    || dtype t = D.Complex128
    || dtype z = D.Complex128
  in
  let cdt = if widest then D.Complex128 else D.Complex64 in
  let t = NL.astype t cdt and z = NL.astype z cdt in
  let rdt = if widest then D.F64 else D.F32 in
  let eps = if widest then epsilon_float else 1.1920928955078125e-07 in
  let n = (shape t).(0) in
  if n = 1 then (t, z)
  else
    let index v =
      T.Concrete
        (Ndarray.as_weak
           (Nd.of_floats
              (Dtypes.default_int_dtype ())
              [||]
              [| float_of_int v |]))
    in
    let pred m = L.sub m (index 1) in
    let update m t z =
      let start_row = pred m in
      let start_col = pred m in
      let vals =
        LN.eigvals (L.dynamic_slice t [ start_row; start_col ] [| 2; 2 |])
      in
      let mu = UF.subtract vals (IDX.dynamic_at2 t m m) in
      let first () =
        C.bind1 (T.Squeeze [| 0 |])
          [
            C.bind1
              (T.Slice
                 {
                   start_indices = [| 0 |];
                   limit_indices = [| 1 |];
                   strides = None;
                 })
              [ mu ];
          ]
      in
      let array_of a b =
        let a = L.broadcast_in_dim a [| 1 |] [||] in
        let b = L.broadcast_in_dim b [| 1 |] [||] in
        L.concatenate [ a; b ] 0
      in
      let mu0 = first () in
      let sub = IDX.dynamic_at2 t m (pred m) in
      let pair = array_of mu0 sub in
      let r = NL.astype (LN.norm pair) cdt in
      let c = UF.divide (first ()) r in
      let s = UF.divide (IDX.dynamic_at2 t m (pred m)) r in
      let c_conj = UF.conj c in
      let s_neg = UF.negative s in
      let row v = L.broadcast_in_dim v [| 1; 2 |] [| 1 |] in
      let row0 = row (array_of c_conj s) in
      let row1 = row (array_of s_neg c) in
      let g = L.concatenate [ row0; row1 ] 0 in
      let g_h () = L.transpose (UF.conj g) [| 1; 0 |] in
      let zero = Numpy.Jnp_util.py_int 0 in
      let rows = L.dynamic_slice_in_dim ~axis:0 t (pred m) 2 in
      let col_mask =
        UF.greater_equal
          (NL.arange ~dtype:(Dtypes.default_int_dtype ()) (float_of_int n))
          (pred m)
      in
      let masked = NL.where_ col_mask rows zero in
      let g_dot = L.dot_typed g masked in
      let rows_new = NL.where_ (UF.logical_not col_mask) rows g_dot in
      let t = L.dynamic_update_slice_in_dim ~axis:0 t rows_new (pred m) in
      let cols = L.dynamic_slice_in_dim ~axis:1 t (pred m) 2 in
      let row_mask =
        UF.less
          (NL.expand_dims
             (NL.arange ~dtype:(Dtypes.default_int_dtype ()) (float_of_int n))
             [| 1 |])
          (L.add m (index 1))
      in
      let masked = NL.where_ row_mask cols zero in
      let dot_gh = L.dot_typed masked (g_h ()) in
      let cols_new = NL.where_ (UF.logical_not row_mask) cols dot_gh in
      let t = L.dynamic_update_slice_in_dim ~axis:1 t cols_new (pred m) in
      let zcols = L.dynamic_slice_in_dim ~axis:1 z (pred m) 2 in
      let z_new = L.dot_typed zcols (g_h ()) in
      let z = L.dynamic_update_slice_in_dim ~axis:1 z z_new (pred m) in
      [ t; z ]
    in
    let iter i = function
      | [ t; z ] -> (
          let m = L.sub (index n) i in
          let lhs = UF.abs (IDX.dynamic_at2 t m (pred m)) in
          let left = pred m in
          let right = pred m in
          let d0 = UF.abs (IDX.dynamic_at2 t left right) in
          let d1 = UF.abs (IDX.dynamic_at2 t m m) in
          let threshold = UF.multiply (C.splat rdt [||] eps) (UF.add d0 d1) in
          match
            L.cond (UF.greater lhs threshold)
              (function [ m; t; z ] -> update m t z | _ -> assert false)
              (function [ _; t; z ] -> [ t; z ] | _ -> assert false)
              [ m; t; z ]
          with
          | [ t; z ] ->
              let col = pred m in
              [ IDX.set_at2 t m col (C.splat cdt [||] 0.0); z ]
          | _ -> assert false)
      | _ -> assert false
    in
    match L.fori_loop (index 1) (index n) iter [ t; z ] with
    | [ t; z ] -> (t, z)
    | _ -> assert false

let rec rsf2csf t z =
  let st = shape t and sz = shape z in
  let nd = Array.length st in
  if nd < 2 || st.(nd - 1) <> st.(nd - 2) then
    Py_exn.value_error "Input 'T' must be square.";
  if Array.length sz < 2 || sz.(Array.length sz - 1) <> sz.(Array.length sz - 2)
  then Py_exn.value_error "Input 'Z' must be square.";
  if st.(nd - 1) <> sz.(Array.length sz - 1) then
    Py_exn.value_error
      (Printf.sprintf "Input array shapes must match: Z: %s vs. T: %s"
         (Validate.python_tuple sz) (Validate.python_tuple st));
  if nd > 2 then
    let outs =
      Interpreters.Batching.vmap
        (fun args ->
          match args with
          | [ t; z ] ->
              let a, b = rsf2csf t z in
              [ a; b ]
          | _ -> assert false)
        [ Some 0; Some 0 ] [ t; z ]
    in
    (List.nth outs 0, List.nth outs 1)
  else
    match
      Numpy.Jnp_util.jit "_rsf2csf_2d"
        (function
          | [ t; z ] ->
              let a, b = rsf2csf_2d t z in
              [ a; b ]
          | _ -> assert false)
        [ t; z ]
    with
    | [ a; b ] -> (a, b)
    | _ -> assert false

let rsf2csf a0 a1 =
  Eager_jit.pair ~jit:"rsf2csf" "scipy.linalg.rsf2csf" () [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> rsf2csf a0 a1
      | _ -> assert false)

let solve ?(lower = false) ?(assume_a = "gen") a b =
  (match assume_a with
  | "gen" | "sym" | "her" | "pos" -> ()
  | _ ->
      Py_exn.value_error
        (Printf.sprintf
           "Expected assume_a to be one of ['gen', 'sym', 'her', 'pos']; got \
            '%s'"
           assume_a));
  if assume_a <> "pos" then LN.solve a b else cho_solve (cho_factor ~lower a) b

let solve ?lower ?assume_a a0 a1 =
  Eager_jit.one ~jit:"_solve" "scipy.linalg.solve" (lower, assume_a) []
    [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> solve ?lower ?assume_a a0 a1
      | _ -> assert false)

let trans_of_string = function
  | "N" -> 0
  | "T" -> 1
  | "C" -> 2
  | other -> Py_exn.value_error ("Invalid 'trans' value " ^ other)

let solve_triangular ?(trans = 0) ?(lower = false) ?(unit_diagonal = false) a b
    =
  let transpose_a, conjugate_a =
    match trans with
    | 0 -> (false, false)
    | 1 -> (true, false)
    | 2 -> (true, true)
    | _ -> Py_exn.value_error (Printf.sprintf "Invalid 'trans' value %d" trans)
  in
  let a = to_inexact a and b = to_inexact b in
  let signature = triangular_signature a b in
  Numpy.Vectorize.vectorize ~signature
    (function
      | [ a; b ] ->
          let sa = shape a in
          let n = sa.(Array.length sa - 1) in
          let bvec = ndim b = 1 in
          let bm = if bvec then NL.reshape b [| n; 1 |] else b in
          let x =
            LL.triangular_solve ~left_side:true ~lower ~transpose_a ~conjugate_a
              ~unit_diagonal a bm
          in
          if bvec then NL.reshape x [| n |] else x
      | _ -> assert false)
    [ a; b ]

let solve_triangular ?trans ?lower ?unit_diagonal a0 a1 =
  Eager_jit.one ~jit:"_solve_triangular" "scipy.linalg.solve_triangular"
    (trans, lower, unit_diagonal) [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> solve_triangular ?trans ?lower ?unit_diagonal a0 a1
      | _ -> assert false)

let svd ?(full_matrices = true) ?(compute_uv = true) a =
  let a = to_inexact a in
  LL.svd ~full_matrices ~compute_uv a

let svd ?full_matrices ?compute_uv ?(lapack_driver : string option) a0 =
  ignore lapack_driver;
  Eager_jit.many ~jit:"_svd" "scipy.linalg.svd" (full_matrices, compute_uv)
    [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> svd ?full_matrices ?compute_uv a0
      | _ -> assert false)

let eigh ?(lower = true) ?(eigvals_only = false) a =
  let a = to_inexact a in
  let v, w = LL.eigh ~lower a in
  if eigvals_only then [ w ] else [ w; v ]

let eigh ?b ?lower ?eigvals_only ?(eigvals : (int * int) option) ?(type_ = 1) a0
    =
  if b <> None then
    raise
      (Py_exn.Not_implemented_error
         "Only the b=None case of eigh is implemented");
  if type_ <> 1 then
    raise
      (Py_exn.Not_implemented_error
         "Only the type=1 case of eigh is implemented.");
  if eigvals <> None then
    raise
      (Py_exn.Not_implemented_error
         "Only the eigvals=None case of eigh is implemented.");
  Eager_jit.many ~jit:"_eigh" "scipy.linalg.eigh" (lower, eigvals_only) []
    [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> eigh ?lower ?eigvals_only a0
      | _ -> assert false)

let schur ?(output = "real") a =
  let a =
    match output with
    | "real" -> a
    | "complex" ->
        if is_complex_dt (dtype a) then a
        else NL.astype a (if dtype a = D.F64 then D.Complex128 else D.Complex64)
    | _ ->
        Py_exn.value_error
          (Printf.sprintf
             "Expected 'output' to be either 'real' or 'complex', got \
              output='%s'."
             output)
  in
  LL.schur a

let schur ?output a0 =
  Eager_jit.many ~jit:"_schur" "scipy.linalg.schur" output [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> schur ?output a0
      | _ -> assert false)

let block_diag arrs =
  let arrs =
    match arrs with
    | [] -> [ AC.zeros ~dtype:(default_float ()) [| 1; 0 |] ]
    | l -> l
  in
  List.iteri
    (fun i a ->
      if ndim a > 2 then
        Py_exn.value_error
          (Printf.sprintf
             "Arguments to jax.scipy.linalg.block_diag must have at most 2 \
              dimensions, got JitTracer(%s[%s]) at argument %d."
             (D.numpy_name (dtype a))
             (String.concat ","
                (Array.to_list (Array.map string_of_int (shape a))))
             i))
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

let toeplitz = Linalg_special.toeplitz

let hessenberg ?(calc_q = false) a =
  let sh = shape a in
  let n = Validate.tuple_index sh (-1) in
  if n = 0 then
    if calc_q then [ AC.zeros_like a; AC.zeros_like a ] else [ AC.zeros_like a ]
  else begin
    let a_out, taus = LL.hessenberg a in
    let h = NL.triu ~k:(-1) a_out in
    if not calc_q then [ h ]
    else begin
      let r = Array.length sh in
      let sub =
        C.bind1
          (T.Slice
             {
               start_indices =
                 Array.init r (fun d -> if d = r - 2 then 1 else 0);
               limit_indices =
                 Array.mapi
                   (fun d v -> if d = r - 1 then v - 1 else v)
                   (shape a_out);
               strides = None;
             })
          [ a_out ]
      in
      let q_inner = LL.householder_product sub taus in
      let dt = dtype a in
      let batch = Array.sub sh 0 (r - 2) in
      let block_of a b = Array.append batch [| a; b |] in
      let open Numpy.Extras in
      let tl = AC.ones ~dtype:dt (block_of 1 1) in
      let tr = AC.zeros ~dtype:dt (block_of 1 (n - 1)) in
      let bl = AC.zeros ~dtype:dt (block_of (n - 1) 1) in
      let q =
        block
          (Group [ Group [ Item tl; Item tr ]; Group [ Item bl; Item q_inner ] ])
      in
      [ h; q ]
    end
  end

let hessenberg ?calc_q a0 =
  Eager_jit.many ~jit:"hessenberg" "scipy.linalg.hessenberg" calc_q [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> hessenberg ?calc_q a0
      | _ -> assert false)

let expm_coefficient m c = C.bind1 T.Mul [ C.splat (dtype m) [||] c; m ]

let expm_sum terms =
  match terms with
  | [] -> assert false
  | first :: rest ->
      List.fold_left
        (fun acc (c, m) -> C.bind1 T.Add [ acc; expm_coefficient m c ])
        (expm_coefficient (snd first) (fst first))
        rest

let precise_dot a b =
  TC.dot
    ~precision:(Lax_config.precision_of_level Lax_config.Precision_highest)
    a b

let pade3 a =
  let ident = NL.eye ~dtype:(dtype a) (shape a).(0) in
  let a2 = precise_dot a a in
  let u = precise_dot a (expm_sum [ (1.0, a2); (60.0, ident) ]) in
  let v = expm_sum [ (12.0, a2); (120.0, ident) ] in
  [ u; v ]

let pade5 a =
  let ident = NL.eye ~dtype:(dtype a) (shape a).(0) in
  let a2 = precise_dot a a in
  let a4 = precise_dot a2 a2 in
  let u =
    precise_dot a (expm_sum [ (1.0, a4); (420.0, a2); (15120.0, ident) ])
  in
  let v = expm_sum [ (30.0, a4); (3360.0, a2); (30240.0, ident) ] in
  [ u; v ]

let pade7 a =
  let ident = NL.eye ~dtype:(dtype a) (shape a).(0) in
  let a2 = precise_dot a a in
  let a4 = precise_dot a2 a2 in
  let a6 = precise_dot a4 a2 in
  let u =
    precise_dot a
      (expm_sum [ (1.0, a6); (1512.0, a4); (277200.0, a2); (8648640.0, ident) ])
  in
  let v =
    expm_sum [ (56.0, a6); (25200.0, a4); (1995840.0, a2); (17297280.0, ident) ]
  in
  [ u; v ]

let pade9 a =
  let ident = NL.eye ~dtype:(dtype a) (shape a).(0) in
  let a2 = precise_dot a a in
  let a4 = precise_dot a2 a2 in
  let a6 = precise_dot a4 a2 in
  let a8 = precise_dot a6 a2 in
  let u =
    precise_dot a
      (expm_sum
         [
           (1.0, a8);
           (3960.0, a6);
           (2162160.0, a4);
           (302702400.0, a2);
           (8821612800.0, ident);
         ])
  in
  let v =
    expm_sum
      [
        (90.0, a8);
        (110880.0, a6);
        (30270240.0, a4);
        (2075673600.0, a2);
        (17643225600.0, ident);
      ]
  in
  [ u; v ]

let pade13 a =
  let ident = NL.eye ~dtype:(dtype a) (shape a).(0) in
  let a2 = precise_dot a a in
  let a4 = precise_dot a2 a2 in
  let a6 = precise_dot a4 a2 in
  let u_inner =
    precise_dot a6 (expm_sum [ (1.0, a6); (16380.0, a4); (40840800.0, a2) ])
  in
  let u_tail =
    List.fold_left
      (fun acc (c, m) -> C.bind1 T.Add [ acc; expm_coefficient m c ])
      u_inner
      [
        (33522128640.0, a6);
        (10559470521600.0, a4);
        (1187353796428800.0, a2);
        (32382376266240000.0, ident);
      ]
  in
  let u = precise_dot a u_tail in
  let v_inner =
    precise_dot a6
      (expm_sum [ (182.0, a6); (960960.0, a4); (1323241920.0, a2) ])
  in
  let v =
    List.fold_left
      (fun acc (c, m) -> C.bind1 T.Add [ acc; expm_coefficient m c ])
      v_inner
      [
        (670442572800.0, a6);
        (129060195264000.0, a4);
        (7771770303897600.0, a2);
        (64764752532480000.0, ident);
      ]
  in
  [ u; v ]

let calc_p_q a =
  Numpy.Jnp_util.jit "_calc_P_Q"
    (function
      | [ a ] -> (
          let is64 = dtype a = D.F64 || dtype a = D.Complex128 in
          let a_l1 = LN.norm ~ord:(LN.Onum 1.0) a in
          let rdt = dtype a_l1 in
          let maxnorm = if is64 then 5.371920351148152 else 3.925724783138660 in
          let scaled_norm = C.bind1 T.Div [ a_l1; C.splat rdt [||] maxnorm ] in
          let log2 = UF.log2 scaled_norm in
          let n_squarings =
            C.bind1 T.Max [ C.splat rdt [||] 0.0; UF.floor log2 ]
          in
          let exponent = NL.astype n_squarings (dtype a) in
          let divisor =
            C.bind1 T.Pow [ C.splat (dtype a) [||] 2.0; exponent ]
          in
          let a = C.bind1 T.Div [ a; divisor ] in
          let conds, branches =
            if is64 then
              ( [|
                  1.495585217958292e-2;
                  2.539398330063230e-1;
                  9.504178996162932e-1;
                  2.097847961257068;
                |],
                [ pade3; pade5; pade7; pade9; pade13 ] )
            else
              ( [| 4.258730016922831e-1; 1.880152677804762 |],
                [ pade3; pade5; pade7 ] )
          in
          let bins =
            T.Concrete (Nd.of_floats rdt [| Array.length conds |] conds)
          in
          let idx = NL.digitize a_l1 bins in
          let wrap f = function [ a ] -> f a | _ -> assert false in
          match Lax_api.switch idx (List.map wrap branches) [ a ] with
          | [ u; v ] ->
              let p = C.bind1 T.Add [ u; v ] in
              let q = C.bind1 T.Add [ C.bind1 T.Neg [ u ]; v ] in
              [ p; q; n_squarings ]
          | _ -> assert false)
      | _ -> assert false)
    [ a ]

let expm_squaring r n_squarings max_squarings =
  Numpy.Jnp_util.jit1 "_squaring"
    ~key:(string_of_int max_squarings)
    (function
      | [ r; n_squarings ] -> (
          let steps =
            C.bind1
              (T.Iota
                 {
                   dtype = dtype n_squarings;
                   shape = [| Stdlib.max max_squarings 0 |];
                   dimension = 0;
                 })
              []
          in
          let body = function
            | [ c; i ] ->
                Lax_api.cond
                  (C.bind1 T.Lt [ i; n_squarings ])
                  (function [ x ] -> [ precise_dot x x ] | _ -> assert false)
                  Fun.id [ c ]
            | _ -> assert false
          in
          match Lax_api.scan body [ r ] [ steps ] with
          | [ res ] -> res
          | _ -> assert false)
      | _ -> assert false)
    [ r; n_squarings ]

let rec expm ?(upper_triangular = false) ?(max_squarings = 16) a =
  let a = to_inexact a in
  let sh = shape a in
  let nd = Array.length sh in
  if nd < 2 || sh.(nd - 1) <> sh.(nd - 2) then
    Py_exn.value_error
      (Printf.sprintf
         "Expected A to be a (batched) square matrix, got A.shape=%s."
         (Validate.python_tuple sh));
  if nd > 2 then
    Numpy.Vectorize.vectorize ~signature:"(n,n)->(n,n)"
      (function
        | [ x ] -> expm_jit ~upper_triangular ~max_squarings x
        | _ -> assert false)
      [ a ]
  else
    match calc_p_q a with
    | [ p; q; n_squarings ] -> (
        let limit =
          C.splat (dtype n_squarings) [||] (float_of_int max_squarings)
        in
        let compute = function
          | [ _; p; q ] ->
              let r =
                if upper_triangular then solve_triangular q p else LN.solve q p
              in
              [ expm_squaring r n_squarings max_squarings ]
          | _ -> assert false
        in
        let nan = function
          | a :: _ -> [ AC.full_like a Float_format.nan ]
          | [] -> assert false
        in
        match
          Lax_api.cond
            (C.bind1 T.Gt [ n_squarings; limit ])
            nan compute [ a; p; q ]
        with
        | [ r ] -> r
        | _ -> assert false)
    | _ -> assert false

and expm_jit ?upper_triangular ?max_squarings a0 =
  Eager_jit.one ~jit:"expm" "scipy.linalg.expm"
    (upper_triangular, max_squarings) [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> expm ?upper_triangular ?max_squarings a0
      | _ -> assert false)

let expm = expm_jit

let expm_frechet ?(compute_expm = true) a0 e0 =
  let a = to_inexact a0 and e = to_inexact e0 in
  let sa = shape a and se = shape e in
  let batched_square s =
    let n = Array.length s in
    n >= 2 && s.(n - 1) = s.(n - 2)
  in
  if Array.length sa < 2 || sa.(Array.length sa - 2) <> sa.(1) then
    Py_exn.value_error
      ("expected A to be a (batched) square matrix, got A.shape="
     ^ Validate.python_tuple sa);
  if not (batched_square se) then
    Py_exn.value_error
      ("expected E to be a (batched) square matrix, got E.shape="
     ^ Validate.python_tuple se);
  if sa <> se then
    Py_exn.value_error
      (Printf.sprintf
         "expected A and E to be the same shape, got A.shape=%s E.shape=%s"
         (Validate.python_tuple sa) (Validate.python_tuple se));
  let primal_dtype = dtype a0 and tangent_dtype = dtype e0 in
  let integral =
    primal_dtype = Dtype.Bool || Dtypes.issubdtype primal_dtype `Integer
  in
  if integral || tangent_dtype <> primal_dtype then
    Py_exn.type_error
      (Printf.sprintf
         "primal and tangent arguments to jax.jvp do not match; dtypes must be \
          equal, or in case of int/bool primal dtype the tangent dtype must be \
          float0.Got primal dtype %s and so expected tangent dtype %s, but got \
          tangent dtype %s instead."
         (Dtype.numpy_name primal_dtype)
         (if integral then "[('float0', 'V')]"
          else Dtype.numpy_name primal_dtype)
         (Dtype.numpy_name tangent_dtype));
  let primal, tangent =
    Interpreters.Ad.jvp
      (fun args -> [ expm (List.hd args) ])
      [ a ]
      [ NL.astype e (dtype a) ]
  in
  if compute_expm then [ List.hd primal; List.hd tangent ]
  else [ List.hd tangent ]

let expm_frechet ?(method_ : string option) ?compute_expm a0 a1 =
  ignore method_;
  Eager_jit.many ~jit:"expm_frechet" "scipy.linalg.expm_frechet" compute_expm []
    [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> expm_frechet ?compute_expm a0 a1
      | _ -> assert false)

let traced_static_eps eps =
  match eps with
  | None -> ()
  | Some _ ->
      Py_exn.value_error
        (Printf.sprintf
           "Non-hashable static arguments are not supported. An error occurred \
            while trying to hash an object of type <class \
            'jax._src.interpreters.partial_eval.DynamicJaxprTracer'>, \
            JitTracer(~%s[]). The error was:\n\
            TypeError: unhashable type: 'DynamicJaxprTracer'\n"
           (Dtype.numpy_name (Dtypes.default_float_dtype ())))

let polar_2d ~side ~method_ ?eps a =
  match method_ with
  | "qdwh" ->
      let sh = shape a in
      let m = sh.(0) and n = sh.(1) in
      if m >= n && side = "right" then
        let () = traced_static_eps eps in
        let unitary, posdef, _, _ = Linalg_extras.qdwh ?eps a in
        [ unitary; posdef ]
      else if m < n && side = "left" then
        let () = traced_static_eps eps in
        let unitary, posdef, _, _ = Linalg_extras.qdwh ?eps (hconj a) in
        let posdef = hconj posdef in
        let unitary = hconj unitary in
        [ unitary; posdef ]
      else
        raise
          (Py_exn.Not_implemented_error
             (Printf.sprintf
                "method='qdwh' only supports mxn matrices where m < n where \
                 side='right' and m >= n side='left', got %s with side='%s'"
                (Validate.python_tuple (shape a))
                side))
  | "svd" ->
      let u, s, vh =
        match LL.svd ~full_matrices:false a with
        | [ u; s; vh ] -> (u, s, vh)
        | _ -> assert false
      in
      let s = NL.astype s (dtype u) in
      let unitary = TC.matmul u vh in
      let s_row = NL.expand_dims s [| 0 |] in
      let posdef =
        if side = "right" then TC.matmul (UF.multiply (hconj vh) s_row) vh
        else TC.matmul (UF.multiply u s_row) (hconj u)
      in
      [ unitary; posdef ]
  | _ -> assert false

let polar ?(side = "right") ?(method_ = "qdwh") ?eps a =
  if ndim a < 2 then
    Py_exn.value_error "The input `a` must be at least a 2-D array.";
  let signature =
    match side with
    | "right" -> "(m,n)->(m,n),(n,n)"
    | "left" -> "(m,n)->(m,n),(m,m)"
    | _ ->
        Py_exn.value_error
          "The argument `side` must be either 'right' or 'left'."
  in
  Numpy.Vectorize.vectorize_many ~signature
    (fun args ->
      match method_ with
      | "qdwh" | "svd" ->
          Numpy.Jnp_util.jit "_polar_2d"
            ~key:(Printf.sprintf "%s,%s,%b" side method_ (eps <> None))
            (function
              | [ a ] -> polar_2d ~side ~method_ ?eps a
              | _ -> invalid_arg "polar: arity")
            args
      | _ ->
          Py_exn.value_error
            (Printf.sprintf "Unknown polar decomposition method %s." method_))
    [ a ]

let polar ?side ?method_ ?eps ?(max_iterations : int option) a0 =
  ignore max_iterations;
  Eager_jit.many ~jit:"polar" "scipy.linalg.polar" (side, method_, eps) []
    [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> polar ?side ?method_ ?eps a0
      | _ -> assert false)

let sqrtm_triu t =
  Numpy.Jnp_util.jit1 "_sqrtm_triu"
    (function
      | [ t ] ->
          let module L = Lax_api in
          let diag = UF.sqrt (NL.diag t) in
          let n = (shape diag).(0) in
          let u0 = NL.diag diag in
          let dt = dtype t in
          let index v =
            T.Concrete
              (Ndarray.as_weak
                 (Nd.of_floats
                    (Dtypes.default_int_dtype ())
                    [||]
                    [| float_of_int v |]))
          in
          let one () = index 1 in
          let i_loop j l = function
            | [ u ] ->
                let i = L.sub (L.sub j (one ())) l in
                let lo = L.add i (one ()) in
                let inner k = function
                  | [ acc ] ->
                      [
                        UF.add acc
                          (UF.multiply (IDX.dynamic_at2 u i k)
                             (IDX.dynamic_at2 u k j));
                      ]
                  | _ -> assert false
                in
                let s =
                  List.hd (L.fori_loop lo j inner [ C.splat dt [||] 0.0 ])
                in
                let same = UF.equal (IDX.dynamic_at2 t i j) s in
                let quotient =
                  UF.divide
                    (UF.subtract (IDX.dynamic_at2 t i j) s)
                    (UF.add (IDX.dynamic_at diag i) (IDX.dynamic_at diag j))
                in
                let value =
                  Numpy.Jnp_util.where_weak ~x_weak:true same
                    (Numpy.Jnp_util.py_float 0.0)
                    quotient
                in
                [ IDX.set_at2 u i j value ]
            | _ -> assert false
          in
          let j_loop j = function
            | [ u ] -> L.fori_loop (index 0) j (i_loop j) [ u ]
            | _ -> assert false
          in
          List.hd (L.fori_loop (index 0) (index n) j_loop [ u0 ])
      | _ -> assert false)
    [ t ]

let sqrtm ?(blocksize = 1) a =
  if blocksize > 1 then
    raise
      (Py_exn.Not_implemented_error "Blocked version is not implemented yet.");
  let a = to_inexact a in
  match schur ~output:"complex" a with
  | [ t; z ] ->
      if ndim t > 2 then Py_exn.value_error "diag input must be 1d or 2d";
      let root = sqrtm_triu t in
      let highest =
        Lax_config.precision_of_level Lax_config.Precision_highest
      in
      let product = TC.matmul ~precision:highest z root in
      TC.matmul ~precision:highest product
        (UF.conj (C.bind1 (T.Transpose [| 1; 0 |]) [ z ]))
  | _ -> assert false

let sqrtm ?blocksize a0 =
  Eager_jit.one ~jit:"_sqrtm" "scipy.linalg.sqrtm" blocksize [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> sqrtm ?blocksize a0
      | _ -> assert false)

let funm ?disp:_ _a _func =
  raise
    (Py_exn.Not_implemented_error
       "module 'jax.scipy.linalg' has no attribute 'funm'")

let tridiagonal_vectors ~key ~base_dtype ~solve_dtype alpha beta eigvals =
  let module L = Lax_api in
  let py_int = Numpy.Jnp_util.py_int in
  let n = (shape alpha).(0) and k = (shape eigvals).(0) in
  let idt = Dtypes.default_int_dtype () in
  let real_const dt v = T.Concrete (Nd.of_floats dt [||] [| v |]) in
  let eps =
    if base_dtype = D.F32 then 1.1920928955078125e-07 else epsilon_float
  in
  let as_solve v = NL.astype v solve_dtype in
  let pad_low v = NL.pad v [| (1, 0) |] 0.0 in
  let pad_high v = NL.pad v [| (0, 1) |] 0.0 in
  let dl =
    pad_low (if is_complex_dt (dtype beta) then UF.conj beta else beta)
  in
  let du = pad_high beta in
  let static_slice v lo hi =
    C.bind1
      (T.Slice
         { start_indices = [| lo |]; limit_indices = [| hi |]; strides = None })
      [ v ]
  in
  let static_at v i =
    C.bind1 (T.Squeeze [| 0 |]) [ static_slice v i (i + 1) ]
  in
  let tail = static_slice eigvals 1 k in
  let head = static_slice eigvals 0 (k - 1) in
  let gap = UF.subtract tail head in
  let first = UF.abs (static_at eigvals 0) in
  let last = UF.abs (static_at eigvals (k - 1)) in
  let t_norm = UF.maximum first last in
  let root_eps = UF.sqrt (real_const base_dtype eps) in
  let gaptol = UF.multiply root_eps t_norm in
  let close = UF.less gap gaptol in
  let left_close = NL.pad ~bool_literal:true close [| (1, 0) |] 0.0 in
  let right_close = NL.pad ~bool_literal:true close [| (0, 1) |] 0.0 in
  let not_left = UF.logical_not left_close in
  let start_mask = UF.logical_and not_left right_close in
  let not_right = UF.logical_not right_close in
  let end_mask = UF.logical_and left_close not_right in
  let max_clusters = (k / 2) + 1 in
  let positions mask = List.hd (Numpy.Extras.nonzero ~size:max_clusters mask) in
  let starts = positions start_mask in
  let ends = UF.add (positions end_mask) (py_int 1) in
  let num_clusters = RED.sum start_mask in
  let arange_k = NL.arange ~dtype:idt (float_of_int k) in
  let raw =
    Random.Core.with_dtypes ~float_dtype:base_dtype (fun () ->
        Random.Core.normal key ~shape:[| k; n |])
  in
  let v0 = as_solve raw in
  let norm_keep v = LN.norm ~axis:(LN.Aint (-1)) ~keepdims:true v in
  let norm0 = as_solve (norm_keep v0) in
  let v0 = UF.divide v0 norm0 in
  let solve_dt_of v = as_solve v in
  let orthogonalize ev =
    let cluster i ev =
      let start = IDX.dynamic_at starts i in
      let stop = IDX.dynamic_at ends i in
      let c = L.sub stop start in
      let padded = NL.pad ev [| (0, k); (0, 0) |] 0.0 in
      let zero_i = T.Concrete (Nd.of_floats (dtype start) [||] [| 0.0 |]) in
      let wrapped extent =
        let negative = C.bind1 T.Lt [ start; py_int 0 ] in
        let shifted = C.bind1 T.Add [ start; py_int extent ] in
        C.bind1 T.Select_n [ negative; start; shifted ]
      in
      let start_at = wrapped (2 * k) in
      let block =
        L.dynamic_slice_normalized padded [ start_at; zero_i ] [| k; n |]
      in
      let mask = NL.astype (UF.less arange_k c) (dtype ev) in
      let column () = L.broadcast_in_dim mask [| k; 1 |] [| 0 |] in
      let mask_col = column () in
      let masked = UF.multiply block mask_col in
      let transposed = NL.transpose masked in
      let q = List.hd (LN.qr transposed) in
      let qt = NL.transpose q in
      let mask_col = column () in
      let qt_masked = UF.multiply qt mask_col in
      let big = AC.zeros ~dtype:(dtype ev) [| 2 * k; n |] in
      let start_at = wrapped (2 * k) in
      let updated = L.dynamic_update_slice big qt_masked [ start_at; zero_i ] in
      let update_full =
        C.bind1
          (T.Slice
             {
               start_indices = [| 0; 0 |];
               limit_indices = [| k; n |];
               strides = None;
             })
          [ updated ]
      in
      let above = UF.greater_equal arange_k start in
      let below = UF.less arange_k stop in
      let in_cluster = C.bind1 T.And [ above; below ] in
      let in_cluster = L.broadcast_in_dim in_cluster [| k; 1 |] [| 0 |] in
      NL.where_ in_cluster update_full ev
    in
    match
      L.fori_loop (py_int 0) num_clusters
        (fun i state -> [ cluster i (List.hd state) ])
        [ ev ]
    with
    | [ out ] -> out
    | _ -> assert false
  in
  let batched v =
    let row = L.broadcast_in_dim v [| 1; n |] [| 1 |] in
    L.broadcast_in_dim row [| k; n |] [| 0; 1 |]
  in
  let alpha_row = L.broadcast_in_dim alpha [| 1; n |] [| 1 |] in
  let eigvals_col = L.broadcast_in_dim eigvals [| k; 1 |] [| 0 |] in
  let alpha_shifted = UF.subtract alpha_row eigvals_col in
  let dl_batched = batched dl in
  let du_batched = batched du in
  let step state =
    match state with
    | [ i; v; nrm; _ ] ->
        let a0 = solve_dt_of dl_batched in
        let a1 = solve_dt_of alpha_shifted in
        let a2 = solve_dt_of du_batched in
        let rhs = L.broadcast_in_dim v [| k; n; 1 |] [| 0; 1 |] in
        let v_new =
          LL.tridiagonal_solve ~perturb_singular:true a0 a1 a2 (solve_dt_of rhs)
        in
        let v_new = NL.squeeze ~axis:[| -1 |] v_new in
        let nrm_new = norm_keep v_new in
        let v_new = UF.divide v_new (as_solve nrm_new) in
        let nrm_new = NL.squeeze ~axis:[| -1 |] nrm_new in
        let v_new = orthogonalize v_new in
        [ UF.add i (py_int 1); v_new; nrm_new; nrm ]
    | _ -> assert false
  in
  let keep_going state =
    match state with
    | [ i; _; nrm; old ] ->
        let more = UF.less i (py_int 5) in
        let grown = UF.multiply (Numpy.Jnp_util.py_float (1.0 +. 0.1)) old in
        let any_grew = RED.any (UF.greater_equal nrm grown) in
        UF.logical_and more any_grew
    | _ -> assert false
  in
  let ones = AC.ones ~dtype:base_dtype [| k |] in
  let zeros = AC.zeros ~dtype:base_dtype [| k |] in
  let result = L.while_loop keep_going step [ py_int 0; v0; ones; zeros ] in
  List.nth result 1

let eigh_tridiagonal ?(eigvals_only = false) ?(select = "a") ?select_range ?tol
    ?key d e =
  let module L = Lax_api in
  let py_int = Numpy.Jnp_util.py_int in
  let py_float = Numpy.Jnp_util.py_float in
  let key =
    if eigvals_only then key
    else
      match key with
      | Some _ -> key
      | None ->
          Some
            (Random.Core.key
               (T.Concrete
                  (Nd.of_floats (Dtypes.default_int_dtype ()) [||] [| 42.0 |])))
  in
  let alpha = d and beta = e in
  if dtype alpha <> dtype beta then
    raise
      (Py_exn.Type_error
         (Printf.sprintf
            "diagonal and off-diagonal values must have same dtype, got %s and \
             %s"
            (D.numpy_name (dtype alpha))
            (D.numpy_name (dtype beta))));
  (match dtype alpha with
  | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> ()
  | dt ->
      raise
        (Py_exn.Type_error
           ("Only float32 and float64 inputs are supported as inputs to \
             jax.scipy.linalg.eigh_tridiagonal, got " ^ D.numpy_name dt
          ^ " and " ^ D.numpy_name dt)));
  let n = Validate.tuple_index (shape alpha) 0 in
  let solve_dtype = dtype alpha in
  let is_complex = is_complex_dt solve_dtype in
  if n <= 1 then begin
    let real_alpha = UF.real alpha in
    if eigvals_only then [ real_alpha ]
    else [ real_alpha; NL.eye ~dtype:solve_dtype n ]
  end
  else begin
    let alpha_real = if is_complex then UF.real alpha else alpha in
    let beta_sq, beta_abs =
      if is_complex then begin
        let sq = UF.real (UF.multiply beta (UF.conj beta)) in
        let ab = UF.sqrt sq in
        (sq, ab)
      end
      else begin
        let ab = UF.abs beta in
        let sq = UF.square beta in
        (sq, ab)
      end
    in
    let base_dtype = dtype alpha_real in
    let single = base_dtype = D.F32 in
    let eps, tiny, largest, nmant =
      if single then
        ( 1.1920928955078125e-07,
          1.1754943508222875e-38,
          3.4028234663852886e38,
          23 )
      else (epsilon_float, 2.2250738585072014e-308, Float.max_float, 52)
    in
    let round v =
      if single then Int32.float_of_bits (Int32.bits_of_float v) else v
    in
    let c v = T.Concrete (Nd.of_floats base_dtype [||] [| v |]) in
    let static_slice v lo hi =
      let sv = shape v in
      if Array.length sv = 0 then
        Py_exn.index_error
          "Too many indices: array is 0-dimensional, but 1 were indexed";
      let lo, hi = Validate.slice_indices ~start:lo ~stop:hi sv.(0) in
      let start_indices = Array.make (Array.length sv) 0 in
      start_indices.(0) <- lo;
      let limit_indices = Array.copy sv in
      limit_indices.(0) <- max lo hi;
      C.bind1 (T.Slice { start_indices; limit_indices; strides = None }) [ v ]
    in
    let static_at v i =
      C.bind1 (T.Squeeze [| 0 |]) [ static_slice v i (i + 1) ]
    in
    let row_sum =
      let head = static_slice beta_abs 0 1 in
      let left = static_slice beta_abs 0 (-1) in
      let right = static_slice beta_abs 1 Stdlib.max_int in
      let middle = UF.add left right in
      let tail = static_slice beta_abs (-1) Stdlib.max_int in
      NL.concatenate [ head; middle; tail ]
    in
    let lambda_max = RED.amax (UF.add alpha_real row_sum) in
    let lambda_min = RED.amin (UF.subtract alpha_real row_sum) in
    let t_norm =
      let lo = UF.abs lambda_min in
      let hi = UF.abs lambda_max in
      UF.maximum lo hi
    in
    let safemin =
      Float.max (round (1.0 /. largest)) (round (round (1.0 +. eps) *. tiny))
    in
    let pivmin =
      UF.multiply (c safemin) (UF.maximum (c 1.0) (RED.amax beta_sq))
    in
    let alpha0_perturbation =
      UF.square (UF.multiply (c eps) (static_at beta_abs 0))
    in
    let abs_tol = UF.multiply (c eps) t_norm in
    let abs_tol =
      match tol with Some t -> UF.maximum (c t) abs_tol | None -> abs_tol
    in
    let max_it = nmant + 1 in
    let target_counts =
      match (select, select_range) with
      | "a", _ -> NL.arange ~dtype:D.I32 (float_of_int n)
      | "i", None ->
          Py_exn.value_error "for select='i', select_range must be specified."
      | "i", Some (l, h) ->
          if l > h then
            Py_exn.value_error "Got empty index range in select_range.";
          NL.arange ~dtype:D.I32 ~start:l (h +. 1.0)
      | "v", _ ->
          raise
            (Py_exn.Not_implemented_error
               "eigh_tridiagonal(..., select='v') is not implemented")
      | _ -> Py_exn.value_error "'select must have a value in {'a', 'i', 'v'}."
    in
    let k = (shape target_counts).(0) in
    let fudge = 2.1 in
    let norm_slack =
      let scaled = UF.multiply (c (float_of_int n)) (py_float fudge) in
      let scaled = UF.multiply scaled (c eps) in
      UF.multiply scaled t_norm
    in
    let lower =
      let shifted = UF.subtract lambda_min norm_slack in
      let margin = UF.multiply (py_float (2.0 *. fudge)) pivmin in
      UF.subtract shifted margin
    in
    let upper =
      let shifted = UF.add lambda_max norm_slack in
      let margin = UF.multiply (py_float fudge) pivmin in
      UF.add shifted margin
    in
    let spread v = L.broadcast_in_dim v [| k |] [||] in
    let lower = spread lower in
    let upper = spread upper in
    let mid = UF.multiply (py_float 0.5) (UF.add upper lower) in
    let pivmin = spread pivmin in
    let alpha0_perturbation = spread alpha0_perturbation in
    let sturm x =
      let zeros = AC.zeros ~dtype:D.I32 (shape x) in
      let ones = AC.ones ~dtype:D.I32 (shape x) in
      let q = UF.subtract (static_at alpha_real 0) x in
      let count = NL.where_ (UF.less q (py_int 0)) ones zeros in
      let at_x = UF.equal (static_at alpha_real 0) x in
      let q = NL.where_ at_x alpha0_perturbation q in
      let sturm_step alpha_i beta_i q count =
        let quotient = UF.divide beta_i q in
        let q = UF.subtract alpha_i quotient in
        let q = UF.subtract q x in
        let low = UF.less_equal q pivmin in
        let bumped = UF.add count (py_int 1) in
        let count = NL.where_ low bumped count in
        let low = UF.less_equal q pivmin in
        let floor = UF.negative pivmin in
        let clamped = UF.minimum q floor in
        let q = NL.where_ low clamped q in
        (q, count)
      in
      let blocksize = 16 in
      let peel = (n - 1) mod blocksize in
      let state = ref (q, count) in
      for j = 0 to peel - 1 do
        let q, count = !state in
        let i = 1 + j in
        let alpha_i = static_at alpha_real i in
        let beta_i = static_at beta_sq (i - 1) in
        state := sturm_step alpha_i beta_i q count
      done;
      let q, count = !state in
      let cond = function
        | [ i; _; _ ] -> UF.less i (py_int n)
        | _ -> assert false
      in
      let body = function
        | [ start; q; count ] ->
            let state = ref (q, count) in
            for j = 0 to blocksize - 1 do
              let q, count = !state in
              let i = UF.add start (py_int j) in
              let alpha_i = IDX.dynamic_at alpha_real i in
              let prev = UF.subtract i (py_int 1) in
              let beta_i = IDX.dynamic_at beta_sq prev in
              state := sturm_step alpha_i beta_i q count
            done;
            let q, count = !state in
            [ UF.add start (py_int blocksize); q; count ]
        | _ -> assert false
      in
      List.nth (L.while_loop cond body [ py_int (1 + peel); q; count ]) 2
    in
    let cond state =
      match state with
      | [ i; lo; _; hi ] ->
          let more = UF.less i (py_int max_it) in
          let wide = RED.amax (UF.subtract hi lo) in
          UF.logical_and more (UF.less abs_tol wide)
      | _ -> assert false
    in
    let body state =
      match state with
      | [ i; lo; mid; hi ] ->
          let counts = sturm mid in
          let lo = NL.where_ (UF.less_equal counts target_counts) mid lo in
          let hi = NL.where_ (UF.greater counts target_counts) mid hi in
          let mid = UF.multiply (py_float 0.5) (UF.add lo hi) in
          [ UF.add i (py_int 1); lo; mid; hi ]
      | _ -> assert false
    in
    let final = L.while_loop cond body [ py_int 0; lower; mid; upper ] in
    let eigvals = List.nth final 2 in
    if eigvals_only then [ eigvals ]
    else
      let key = Option.get key in
      let vectors =
        tridiagonal_vectors ~key ~base_dtype ~solve_dtype alpha_real beta
          eigvals
      in
      [ eigvals; NL.transpose vectors ]
  end

let eigh_tridiagonal ?eigvals_only ?select ?select_range ?tol ?key a0 a1 =
  Eager_jit.many ~jit:"eigh_tridiagonal" "scipy.linalg.eigh_tridiagonal"
    (eigvals_only, select, select_range, tol) [ key ] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ key ], [ a0; a1 ] ->
          eigh_tridiagonal ?eigvals_only ?select ?select_range ?tol ?key a0 a1
      | _ -> assert false)
