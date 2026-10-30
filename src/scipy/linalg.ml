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

let rec qr ?(mode = "full") ?(pivoting = false) a =
  let full_matrices =
    match mode with
    | "full" | "r" -> true
    | "economic" -> false
    | _ ->
        invalid_arg
          (Printf.sprintf "Unsupported QR decomposition mode '%s'" mode)
  in
  let a = to_inexact a in
  if not pivoting then
    let q, r = LL.qr ~full_matrices a in
    if mode = "r" then [ r ] else [ q; r ]
  else if Array.length (shape a) > 2 then
    Interpreters.Batching.vmap
      (fun args -> qr ~mode ~pivoting (List.hd args))
      [ Some 0 ] [ a ]
  else begin
    let sh = shape a in
    let m = sh.(0) and n = sh.(1) in
    let k = min m n in
    let cols = if full_matrices then m else k in
    let h, jpvt, taus = Linalg_extras.geqp3 a in
    let perm = UF.subtract jpvt (i32_scalar 1) in
    let r = NL.triu (Lax_api.slice h [| 0; 0 |] [| cols; n |]) in
    if mode = "r" then [ r; perm ]
    else
      let dt = dtype a in
      let base =
        if cols <= n then Lax_api.slice h [| 0; 0 |] [| m; cols |]
        else NL.concatenate ~axis:1 [ h; AC.zeros ~dtype:dt [| m; cols - n |] ]
      in
      [ LL.householder_product base taus; r; perm ]
  end

let rec qr_multiply ?(mode = "right") ?(pivoting = false) ?(conjugate = false) a
    c =
  if mode <> "right" && mode <> "left" then
    invalid_arg (Printf.sprintf "mode must be 'right' or 'left', got '%s'" mode);
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
  if Array.length (shape a) > 2 then
    Interpreters.Batching.vmap
      (fun args ->
        match args with
        | [ a; c ] -> qr_multiply ~mode ~pivoting ~conjugate a c
        | _ -> failwith "qr_multiply: arity")
      [ Some 0; Some 0 ] [ a; c ]
  else begin
    let onedim = Array.length (shape c) = 1 in
    let c =
      if not onedim then c
      else
        let l = (shape c).(0) in
        NL.reshape c (if mode = "left" then [| l; 1 |] else [| 1; l |])
    in
    let sh = shape a in
    let m = sh.(0) and n = sh.(1) in
    let k = min m n in
    let cs = shape c in
    if mode = "left" && cs.(0) <> k then
      invalid_arg "Array shapes are not compatible for Q @ c operation";
    if mode = "right" && cs.(1) <> m then
      invalid_arg "Array shapes are not compatible for c @ Q operation";
    let h, perm, taus =
      if pivoting then
        let h, p, t = Linalg_extras.geqp3 a in
        (h, Some (UF.subtract p (i32_scalar 1)), t)
      else
        let h, t = Linalg_extras.geqrf a in
        (h, None, t)
    in
    let c =
      if m > n && mode = "left" then
        NL.concatenate ~axis:0 [ c; AC.zeros ~dtype:dt [| m - k; cs.(1) |] ]
      else c
    in
    let c = if conjugate then NL.matrix_transpose c else c in
    let left = mode = "left" <> conjugate in
    let cq = Linalg_extras.ormqr ~left ~transpose:conjugate h taus c in
    let cq = if conjugate then NL.matrix_transpose cq else cq in
    let cq =
      if mode = "right" then Lax_api.slice_in_dim ~axis:1 cq 0 k else cq
    in
    let cq =
      if onedim then NL.reshape cq [| Array.fold_left ( * ) 1 (shape cq) |]
      else cq
    in
    let r = NL.triu (Lax_api.slice h [| 0; 0 |] [| k; n |]) in
    match perm with None -> [ cq; r ] | Some p -> [ cq; r; p ]
  end

let rec rsf2csf t z =
  let st = shape t and sz = shape z in
  let nd = Array.length st in
  if nd < 2 || st.(nd - 1) <> st.(nd - 2) then
    invalid_arg "Input 'T' must be square.";
  if Array.length sz < 2 || sz.(Array.length sz - 1) <> sz.(Array.length sz - 2)
  then invalid_arg "Input 'Z' must be square.";
  if st.(nd - 1) <> sz.(Array.length sz - 1) then
    invalid_arg "Input array shapes must match";
  if nd > 2 then
    let outs =
      Interpreters.Batching.vmap
        (fun args ->
          match args with
          | [ t; z ] ->
              let a, b = rsf2csf t z in
              [ a; b ]
          | _ -> failwith "rsf2csf: arity")
        [ Some 0; Some 0 ] [ t; z ]
    in
    (List.nth outs 0, List.nth outs 1)
  else begin
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
    let n = st.(0) in
    let idx v = T.Concrete (Nd.of_floats D.I32 [||] [| float_of_int v |]) in
    let at x i j = L.reshape (L.slice x [| i; j |] [| i + 1; j + 1 |]) [||] in
    let update m tv zv =
      let mu0 =
        let vals =
          LN.eigvals (L.slice tv [| m - 1; m - 1 |] [| m + 1; m + 1 |])
        in
        UF.subtract (L.index_in_dim ~axis:0 ~keepdims:false vals 0) (at tv m m)
      in
      let sub = at tv m (m - 1) in
      let magnitude v =
        let a = UF.abs v in
        UF.multiply a a
      in
      let r = NL.astype (L.sqrt (UF.add (magnitude mu0) (magnitude sub))) cdt in
      let c = UF.divide mu0 r and s = UF.divide sub r in
      let entry v = L.reshape v [| 1 |] in
      let g =
        L.reshape
          (L.concatenate
             [ entry (L.conj c); entry s; entry (L.neg s); entry c ]
             0)
          [| 2; 2 |]
      in
      let g_h = L.conj (NL.matrix_transpose g) in
      let rows = L.slice_in_dim ~axis:0 tv (m - 1) (m + 1) in
      let kept = L.slice_in_dim ~axis:1 rows 0 (m - 1) in
      let moved = L.dot g (L.slice_in_dim ~axis:1 rows (m - 1) n) in
      let tv =
        L.dynamic_update_slice tv
          (L.concatenate [ kept; moved ] 1)
          [ idx (m - 1); idx 0 ]
      in
      let cols = L.slice_in_dim ~axis:1 tv (m - 1) (m + 1) in
      let moved = L.dot (L.slice_in_dim ~axis:0 cols 0 (m + 1)) g_h in
      let kept = L.slice_in_dim ~axis:0 cols (m + 1) n in
      let tv =
        L.dynamic_update_slice tv
          (L.concatenate [ moved; kept ] 0)
          [ idx 0; idx (m - 1) ]
      in
      let zcols = L.slice_in_dim ~axis:1 zv (m - 1) (m + 1) in
      let zv =
        L.dynamic_update_slice zv (L.dot zcols g_h) [ idx 0; idx (m - 1) ]
      in
      [ tv; zv ]
    in
    let rec go i tv zv =
      if i >= n then (tv, zv)
      else
        let m = n - i in
        let threshold =
          UF.multiply (L.full [||] eps rdt)
            (UF.add (UF.abs (at tv (m - 1) (m - 1))) (UF.abs (at tv m m)))
        in
        match
          L.cond
            (L.gt (UF.abs (at tv m (m - 1))) threshold)
            (fun args ->
              match args with
              | [ a; b ] -> update m a b
              | _ -> failwith "rsf2csf: arity")
            Fun.id [ tv; zv ]
        with
        | [ tv; zv ] ->
            let zero = L.full [| 1; 1 |] 0.0 cdt in
            go (i + 1)
              (L.dynamic_update_slice tv zero [ idx m; idx (m - 1) ])
              zv
        | _ -> failwith "rsf2csf: arity"
    in
    if n = 1 then (t, z) else go 1 t z
  end

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

let svd ?(full_matrices = true) ?(compute_uv = true) a =
  let a = to_inexact a in
  LL.svd ~full_matrices ~compute_uv a

let eigh ?(lower = true) ?(eigvals_only = false) a =
  let a = to_inexact a in
  let v, w = LL.eigh ~lower a in
  if eigvals_only then [ w ] else [ w; v ]

let schur ?(output = "real") a =
  let a = to_inexact a in
  let a =
    match output with
    | "real" -> a
    | "complex" ->
        if is_complex_dt (dtype a) then a
        else NL.astype a (if dtype a = D.F64 then D.Complex128 else D.Complex64)
    | _ ->
        invalid_arg
          (Printf.sprintf "argument must be 'real' or 'complex', got %s" output)
  in
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

let rec expm ?(upper_triangular = false) a =
  let a = to_inexact a in
  let sh = shape a in
  let nd = Array.length sh in
  if nd < 2 || sh.(nd - 1) <> sh.(nd - 2) then
    invalid_arg "expm: expected A to be a (batched) square matrix";
  if nd > 2 then
    List.hd
      (Interpreters.Batching.vmap
         (fun args -> [ expm ~upper_triangular (List.hd args) ])
         [ Some 0 ] [ a ])
  else
    let n = sh.(0) in
    let dt = dtype a in
    let ident = NL.eye ~dtype:dt n in
    let sc co m = UF.multiply (scalar_of a co) m in
    let dotp = TC.matmul in
    let is64 = dt = D.F64 || dt = D.Complex128 in
    let maxnorm = if is64 then 5.371920351148152 else 3.925724783138660 in
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
    let pades_for a =
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
        let v =
          UF.add (sc 30.0 a4) (UF.add (sc 3360.0 a2) (sc 30240.0 ident))
        in
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
                  (UF.add (sc 1.0 a6)
                     (UF.add (sc 16380.0 a4) (sc 40840800.0 a2))))
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
      pades
    in
    let is_traced = match a with T.Tracer _ -> true | _ -> false in
    if is_traced then begin
      let module L = Lax_api in
      let l1 = RED.amax (RED.sum ~axis:[| 0 |] (UF.abs a)) in
      let real_dt = (Core.get_aval l1).T.dtype in
      let const x = L.full [||] x real_dt in
      let exact = L.floor (UF.log2 (L.div l1 (const maxnorm))) in
      let nsq_v =
        L.select (L.gt l1 (const 0.0)) (L.max exact (const 0.0)) (const 0.0)
      in
      let factor = L.pow (const 2.0) nsq_v in
      let widen v = NL.astype (L.broadcast_in_dim v sh [||]) dt in
      let scaled = UF.divide a (widen factor) in
      let results = Array.map (fun f -> f ()) (pades_for scaled) in
      let pick select_part =
        let last = Array.length results - 1 in
        let rec go i =
          if i = last then select_part results.(i)
          else
            NL.where_
              (widen_bool (L.lt l1 (const conds.(i))))
              (select_part results.(i))
              (go (i + 1))
        and widen_bool c = L.broadcast_in_dim c sh [||] in
        go 0
      in
      let u = pick fst and v = pick snd in
      let p = UF.add u v and q = UF.add (UF.negative u) v in
      let r = if upper_triangular then solve_triangular q p else LN.solve q p in
      let max_squarings = 16 in
      let rec loop r k =
        if k = max_squarings then r
        else
          let now = L.lt (const (float_of_int k)) nsq_v in
          loop (NL.where_ (L.broadcast_in_dim now sh [||]) (dotp r r) r) (k + 1)
      in
      let out = loop r 0 in
      let too_many = L.gt nsq_v (const (float_of_int max_squarings)) in
      NL.where_
        (L.broadcast_in_dim too_many sh [||])
        (AC.full_like out Float.nan)
        out
    end
    else
      let l1f = host_float (RED.amax (RED.sum ~axis:[| 0 |] (UF.abs a))) in
      let nsq =
        if l1f <= 0.0 then 0
        else max 0 (int_of_float (Float.floor (Float.log2 (l1f /. maxnorm))))
      in
      let max_squarings = 16 in
      if nsq > max_squarings then AC.full_like a Float.nan
      else begin
        let a =
          if nsq = 0 then a
          else UF.divide a (scalar_of a (2.0 ** float_of_int nsq))
        in
        let idx =
          Array.fold_left
            (fun acc c -> if c <= l1f then acc + 1 else acc)
            0 conds
        in
        let pades = pades_for a in
        let u, v = pades.(idx) () in
        let p = UF.add u v and q = UF.add (UF.negative u) v in
        let r =
          if upper_triangular then solve_triangular q p else LN.solve q p
        in
        let rec sq r k = if k = 0 then r else sq (dotp r r) (k - 1) in
        sq r nsq
      end

let expm_frechet ?(compute_expm = true) a e =
  let a = to_inexact a and e = to_inexact e in
  let sa = shape a and se = shape e in
  if Array.length sa < 2 || sa.(Array.length sa - 1) <> sa.(Array.length sa - 2)
  then invalid_arg "expected A to be a (batched) square matrix";
  if sa <> se then invalid_arg "expected A and E to be the same shape";
  let primal, tangent =
    Interpreters.Ad.jvp
      (fun args -> [ expm (List.hd args) ])
      [ a ]
      [ NL.astype e (dtype a) ]
  in
  if compute_expm then [ List.hd primal; List.hd tangent ]
  else [ List.hd tangent ]

let rec polar ?(side = "right") ?(method_ = "qdwh") ?eps a =
  let a = to_inexact a in
  if ndim a < 2 then invalid_arg "The input `a` must be at least a 2-D array.";
  (match side with
  | "right" | "left" -> ()
  | _ -> invalid_arg "The argument `side` must be either 'right' or 'left'.");
  match method_ with
  | "qdwh" when ndim a > 2 ->
      Interpreters.Batching.vmap
        (fun args -> polar ~side ~method_ ?eps (List.hd args))
        [ Some 0 ] [ a ]
  | "qdwh" ->
      let sh = shape a in
      let m = sh.(0) and n = sh.(1) in
      if m >= n && side = "right" then
        let unitary, posdef, _, _ = Linalg_extras.qdwh ?eps a in
        [ unitary; posdef ]
      else if m < n && side = "left" then
        let unitary, posdef, _, _ = Linalg_extras.qdwh ?eps (hconj a) in
        [ hconj unitary; hconj posdef ]
      else
        invalid_arg
          "method='qdwh' only supports mxn matrices where m < n where \
           side='right' and m >= n side='left'"
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

let rec sqrtm ?(blocksize = 1) a =
  if blocksize > 1 then
    failwith "scipy.linalg.sqrtm: Blocked version is not implemented yet.";
  let a = to_inexact a in
  if ndim a > 2 then
    List.hd
      (Interpreters.Batching.vmap
         (fun args -> [ sqrtm (List.hd args) ])
         [ Some 0 ] [ a ])
  else
    match schur ~output:"complex" a with
    | [ t; z ] ->
        let module L = Lax_api in
        let n = (shape t).(0) in
        let dt = dtype t in
        let zero = L.full [||] 0.0 dt in
        let at x i j =
          L.reshape (L.slice x [| i; j |] [| i + 1; j + 1 |]) [||]
        in
        let diag = Array.init n (fun i -> L.sqrt (at t i i)) in
        let u = Array.make_matrix n n zero in
        for i = 0 to n - 1 do
          u.(i).(i) <- diag.(i)
        done;
        for j = 0 to n - 1 do
          for l = 0 to j - 1 do
            let i = j - 1 - l in
            let sum = ref zero in
            for k = i + 1 to j - 1 do
              sum := L.add !sum (L.mul u.(i).(k) u.(k).(j))
            done;
            let tij = at t i j in
            u.(i).(j) <-
              NL.where_ (L.eq tij !sum) zero
                (L.div (L.sub tij !sum) (L.add diag.(i) diag.(j)))
          done
        done;
        let cells =
          List.concat_map
            (fun row ->
              List.map (fun x -> L.reshape x [| 1 |]) (Array.to_list row))
            (Array.to_list u)
        in
        let root = L.reshape (L.concatenate cells 0) [| n; n |] in
        L.dot (L.dot z root) (L.conj (NL.matrix_transpose z))
    | _ -> failwith "sqrtm: schur arity"

let funm ?(disp = true) a _func =
  ignore disp;
  ignore a;
  failwith
    "scipy.linalg.funm: unsupported in ojax (requires rsf2csf real->complex \
     Schur-form conversion over the complex Schur machinery deferred with row \
     99)"

let eigh_tridiagonal ?(eigvals_only = false) ?(select = "a") ?select_range d e =
  let d = to_inexact d and e = to_inexact e in
  let n = (shape d).(0) in
  let lo, hi =
    match (select, select_range) with
    | "a", _ -> (0, n - 1)
    | "i", Some (l, h) -> (int_of_float l, int_of_float h)
    | "i", None -> invalid_arg "eigh_tridiagonal: select='i' needs select_range"
    | _ ->
        invalid_arg
          "eigh_tridiagonal: select must be 'a' or 'i'; 'v' has a data \
           dependent output size"
  in
  let t =
    if n <= 1 then NL.diag d
    else UF.add (NL.diag d) (UF.add (NL.diag ~k:1 e) (NL.diag ~k:(-1) e))
  in
  let v, w = LL.eigh ~lower:true t in
  let w = Lax_api.slice_in_dim ~axis:0 w lo (hi + 1) in
  if eigvals_only then [ w ]
  else [ w; Lax_api.slice_in_dim ~axis:1 v lo (hi + 1) ]
