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
  let real_dt =
    match dt with D.Complex64 -> D.F32 | D.Complex128 -> D.F64 | d -> d
  in
  let p = NL.astype (UF.equal perm_row iota_col) real_dt in
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
  let full_matrices =
    match mode with
    | "full" | "r" -> true
    | "economic" -> false
    | _ ->
        invalid_arg
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
            | _ -> failwith "qr: arity")
      | _ -> failwith "qr: arity")
    [ a ]

let qr_multiply ?(mode = "right") ?(pivoting = false) ?(conjugate = false) a c =
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
    invalid_arg (Printf.sprintf "mode must be 'right' or 'left', got '%s'" mode);
  let onedim = Array.length (shape c) = 1 in
  let c =
    if not onedim then c
    else
      let l = (shape c).(0) in
      NL.reshape c (if mode = "left" then [| l; 1 |] else [| 1; l |])
  in
  let sh = shape a in
  let rank = Array.length sh in
  let m = sh.(rank - 2) and n = sh.(rank - 1) in
  let k = min m n in
  let cs = shape c in
  let crank = Array.length cs in
  let shape_text s =
    match s with
    | [| x |] -> Printf.sprintf "(%d,)" x
    | _ ->
        "("
        ^ String.concat ", " (Array.to_list (Array.map string_of_int s))
        ^ ")"
  in
  if mode = "left" && cs.(crank - 2) <> k then
    invalid_arg
      (Printf.sprintf
         "Array shapes are not compatible for Q @ c operation: a has shape %s \
          so Q has %d columns, but c has %d rows (expected %d)."
         (shape_text sh) k
         cs.(crank - 2)
         k);
  if mode = "right" && cs.(crank - 1) <> m then
    invalid_arg
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
      Lax_api.slice cq (Array.make r 0) limits)
    else cq
  in
  let cq =
    if onedim then NL.reshape cq [| Array.fold_left ( * ) 1 (shape cq) |]
    else cq
  in
  let starts = Array.make rank 0 in
  let limits = Array.append batch [| k; n |] in
  let r = NL.triu (Lax_api.slice h starts limits) in
  match perm with None -> [ cq; r ] | Some p -> [ cq; r; p ]

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

let expm_coefficient m c = C.bind1 T.Mul [ C.splat (dtype m) [||] c; m ]

let expm_sum terms =
  match terms with
  | [] -> invalid_arg "expm: empty sum"
  | first :: rest ->
      List.fold_left
        (fun acc (c, m) -> C.bind1 T.Add [ acc; expm_coefficient m c ])
        (expm_coefficient (snd first) (fst first))
        rest

let pade3 a =
  let ident = NL.eye ~dtype:(dtype a) (shape a).(0) in
  let a2 = TC.dot a a in
  let u = TC.dot a (expm_sum [ (1.0, a2); (60.0, ident) ]) in
  let v = expm_sum [ (12.0, a2); (120.0, ident) ] in
  [ u; v ]

let pade5 a =
  let ident = NL.eye ~dtype:(dtype a) (shape a).(0) in
  let a2 = TC.dot a a in
  let a4 = TC.dot a2 a2 in
  let u = TC.dot a (expm_sum [ (1.0, a4); (420.0, a2); (15120.0, ident) ]) in
  let v = expm_sum [ (30.0, a4); (3360.0, a2); (30240.0, ident) ] in
  [ u; v ]

let pade7 a =
  let ident = NL.eye ~dtype:(dtype a) (shape a).(0) in
  let a2 = TC.dot a a in
  let a4 = TC.dot a2 a2 in
  let a6 = TC.dot a4 a2 in
  let u =
    TC.dot a
      (expm_sum [ (1.0, a6); (1512.0, a4); (277200.0, a2); (8648640.0, ident) ])
  in
  let v =
    expm_sum [ (56.0, a6); (25200.0, a4); (1995840.0, a2); (17297280.0, ident) ]
  in
  [ u; v ]

let pade9 a =
  let ident = NL.eye ~dtype:(dtype a) (shape a).(0) in
  let a2 = TC.dot a a in
  let a4 = TC.dot a2 a2 in
  let a6 = TC.dot a4 a2 in
  let a8 = TC.dot a6 a2 in
  let u =
    TC.dot a
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
  let a2 = TC.dot a a in
  let a4 = TC.dot a2 a2 in
  let a6 = TC.dot a4 a2 in
  let u_inner =
    TC.dot a6 (expm_sum [ (1.0, a6); (16380.0, a4); (40840800.0, a2) ])
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
  let u = TC.dot a u_tail in
  let v_inner =
    TC.dot a6 (expm_sum [ (182.0, a6); (960960.0, a4); (1323241920.0, a2) ])
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
          let wrap f = function
            | [ a ] -> f a
            | _ -> invalid_arg "expm: arity"
          in
          match Lax_api.switch idx (List.map wrap branches) [ a ] with
          | [ u; v ] ->
              let p = C.bind1 T.Add [ u; v ] in
              let q = C.bind1 T.Add [ C.bind1 T.Neg [ u ]; v ] in
              [ p; q; n_squarings ]
          | _ -> invalid_arg "expm: arity")
      | _ -> invalid_arg "expm: arity")
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
                   shape = [| max_squarings |];
                   dimension = 0;
                 })
              []
          in
          let body = function
            | [ c; i ] ->
                Lax_api.cond
                  (C.bind1 T.Lt [ i; n_squarings ])
                  (function
                    | [ x ] -> [ TC.dot x x ] | _ -> invalid_arg "expm: arity")
                  Fun.id [ c ]
            | _ -> invalid_arg "expm: arity"
          in
          match Lax_api.scan body [ r ] [ steps ] with
          | [ res ] -> res
          | _ -> invalid_arg "expm: arity")
      | _ -> invalid_arg "expm: arity")
    [ r; n_squarings ]

let rec expm ?(upper_triangular = false) ?(max_squarings = 16) a =
  let a = to_inexact a in
  let sh = shape a in
  let nd = Array.length sh in
  if nd < 2 || sh.(nd - 1) <> sh.(nd - 2) then
    invalid_arg "expm: expected A to be a (batched) square matrix";
  if nd > 2 then
    List.hd
      (Interpreters.Batching.vmap
         (fun args -> [ expm ~upper_triangular ~max_squarings (List.hd args) ])
         [ Some 0 ] [ a ])
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
          | _ -> invalid_arg "expm: arity"
        in
        let nan = function
          | a :: _ -> [ AC.full_like a Float.nan ]
          | [] -> invalid_arg "expm: arity"
        in
        match
          Lax_api.cond
            (C.bind1 T.Gt [ n_squarings; limit ])
            nan compute [ a; p; q ]
        with
        | [ r ] -> r
        | _ -> invalid_arg "expm: arity")
    | _ -> invalid_arg "expm: arity"

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

let tridiagonal_vectors ~key ~base_dtype ~solve_dtype alpha beta eigvals =
  let module L = Lax_api in
  let n = (shape alpha).(0) and k = (shape eigvals).(0) in
  let idt = D.I32 in
  let int_const sh v = L.full sh (float_of_int v) idt in
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
  let gap =
    UF.subtract (L.slice_in_dim eigvals 1 k) (L.slice_in_dim eigvals 0 (k - 1))
  in
  let first = L.index_in_dim ~keepdims:false eigvals 0 in
  let last = L.index_in_dim ~keepdims:false eigvals (k - 1) in
  let t_norm = UF.maximum (UF.abs first) (UF.abs last) in
  let gaptol = UF.multiply (UF.sqrt (real_const base_dtype eps)) t_norm in
  let close = UF.less gap gaptol in
  let left_close = pad_low close in
  let right_close = pad_high close in
  let start_mask = UF.logical_and (UF.logical_not left_close) right_close in
  let end_mask = UF.logical_and left_close (UF.logical_not right_close) in
  let max_clusters = (k / 2) + 1 in
  let positions mask =
    let values =
      Numpy.Extras.flatnonzero ~size:max_clusters ~fill_value:0 mask
    in
    NL.astype values idt
  in
  let starts = positions start_mask in
  let ends = L.add (positions end_mask) (int_const [| max_clusters |] 1) in
  let num_clusters = NL.astype (RED.sum start_mask) idt in
  let arange_k = NL.arange ~dtype:idt (float_of_int k) in
  let raw =
    Random.Core.with_dtypes ~float_dtype:base_dtype (fun () ->
        Random.Core.normal key ~shape:[| k; n |])
  in
  let v0 = as_solve raw in
  let v0 =
    UF.divide v0
      (as_solve (LN.vector_norm ~axis:(Some [| -1 |]) ~keepdims:true v0))
  in
  let solve_dt_of v = as_solve v in
  let orthogonalize ev =
    let cluster i ev =
      let index arr = L.dynamic_index_in_dim ~keepdims:false arr i in
      let start = index starts and stop = index ends in
      let c = L.sub stop start in
      let padded =
        L.pad ev
          (T.Concrete (Nd.of_floats (dtype ev) [||] [| 0.0 |]))
          [| (0, k, 0); (0, 0, 0) |]
      in
      let block =
        L.dynamic_slice padded [ start; int_const [||] 0 ] [| k; n |]
      in
      let mask =
        NL.astype
          (UF.less arange_k (L.broadcast_in_dim c [| k |] [||]))
          (dtype ev)
      in
      let column = NL.reshape mask [| k; 1 |] in
      let masked = UF.multiply block column in
      let q = fst (LL.qr ~full_matrices:false (NL.transpose masked)) in
      let qt = UF.multiply (NL.transpose q) column in
      let big = AC.zeros ~dtype:(dtype ev) [| 2 * k; n |] in
      let updated = L.dynamic_update_slice big qt [ start; int_const [||] 0 ] in
      let update_full = L.slice_in_dim updated 0 k in
      let in_cluster =
        UF.logical_and
          (UF.greater_equal arange_k (L.broadcast_in_dim start [| k |] [||]))
          (UF.less arange_k (L.broadcast_in_dim stop [| k |] [||]))
      in
      NL.where_ (NL.reshape in_cluster [| k; 1 |]) update_full ev
    in
    match
      L.fori_loop (int_const [||] 0) num_clusters
        (fun i state -> [ cluster i (List.hd state) ])
        [ ev ]
    with
    | [ out ] -> out
    | _ -> failwith "eigh_tridiagonal: arity"
  in
  let batched v = L.broadcast_in_dim v [| k; n |] [| 1 |] in
  let alpha_shifted =
    UF.subtract
      (L.broadcast_in_dim alpha [| 1; n |] [| 1 |])
      (L.broadcast_in_dim eigvals [| k; 1 |] [| 0 |])
  in
  let dl_batched = batched dl and du_batched = batched du in
  let step state =
    match state with
    | [ i; v; nrm; _ ] ->
        let v_new =
          LL.tridiagonal_solve ~perturb_singular:true (solve_dt_of dl_batched)
            (solve_dt_of alpha_shifted)
            (solve_dt_of du_batched)
            (L.broadcast_in_dim (solve_dt_of v) [| k; n; 1 |] [| 0; 1 |])
        in
        let v_new = NL.squeeze ~axis:[| -1 |] v_new in
        let nrm_new =
          LN.vector_norm ~axis:(Some [| -1 |]) ~keepdims:true v_new
        in
        let v_new = UF.divide v_new (as_solve nrm_new) in
        let nrm_new = NL.reshape nrm_new [| k |] in
        [ L.add i (int_const [||] 1); orthogonalize v_new; nrm_new; nrm ]
    | _ -> failwith "eigh_tridiagonal: state"
  in
  let keep_going state =
    match state with
    | [ i; _; nrm; old ] ->
        UF.logical_and
          (UF.less i (int_const [||] 5))
          (RED.any
             (UF.greater_equal nrm
                (UF.multiply (real_const base_dtype 1.1) old)))
    | _ -> failwith "eigh_tridiagonal: state"
  in
  let result =
    L.while_loop keep_going step
      [
        int_const [||] 0;
        v0;
        AC.ones ~dtype:base_dtype [| k |];
        AC.zeros ~dtype:base_dtype [| k |];
      ]
  in
  List.nth result 1

let eigh_tridiagonal ?(eigvals_only = false) ?(select = "a") ?select_range ?tol
    ?key d e =
  let module L = Lax_api in
  let alpha = d and beta = e in
  if dtype alpha <> dtype beta then
    raise
      (Errors.jax_type_error
         (Printf.sprintf
            "diagonal and off-diagonal values must have same dtype, got %s and \
             %s"
            (D.short_name (dtype alpha))
            (D.short_name (dtype beta))));
  (match dtype alpha with
  | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> ()
  | dt ->
      raise
        (Errors.jax_type_error
           ("Only float32 and float64 inputs are supported as inputs to \
             jax.scipy.linalg.eigh_tridiagonal, got " ^ D.short_name dt)));
  let n = (shape alpha).(0) in
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
      if is_complex then
        let sq = UF.real (UF.multiply beta (UF.conj beta)) in
        (sq, UF.sqrt sq)
      else (UF.square beta, UF.abs beta)
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
    let m = (shape beta_abs).(0) in
    let row_sum =
      NL.concatenate
        [
          L.slice_in_dim beta_abs 0 1;
          UF.add
            (L.slice_in_dim beta_abs 0 (m - 1))
            (L.slice_in_dim beta_abs 1 m);
          L.slice_in_dim beta_abs (m - 1) m;
        ]
    in
    let lambda_max = RED.amax (UF.add alpha_real row_sum) in
    let lambda_min = RED.amin (UF.subtract alpha_real row_sum) in
    let t_norm = UF.maximum (UF.abs lambda_min) (UF.abs lambda_max) in
    let safemin =
      Float.max (round (1.0 /. largest)) (round (round (1.0 +. eps) *. tiny))
    in
    let pivmin =
      UF.multiply (c safemin) (UF.maximum (c 1.0) (RED.amax beta_sq))
    in
    let beta0 = L.index_in_dim ~keepdims:false beta_abs 0 in
    let alpha0_perturbation = UF.square (UF.multiply (c eps) beta0) in
    let abs_tol = UF.multiply (c eps) t_norm in
    let abs_tol =
      match tol with Some t -> UF.maximum (c t) abs_tol | None -> abs_tol
    in
    let max_it = nmant + 1 in
    let target_counts =
      match (select, select_range) with
      | "a", _ -> NL.arange ~dtype:D.I32 (float_of_int n)
      | "i", None ->
          invalid_arg "for select='i', select_range must be specified."
      | "i", Some (l, h) ->
          if l > h then invalid_arg "Got empty index range in select_range.";
          NL.arange ~dtype:D.I32 ~start:l (h +. 1.0)
      | "v", _ ->
          failwith "eigh_tridiagonal(..., select='v') is not implemented"
      | _ -> invalid_arg "'select must have a value in {'a', 'i', 'v'}."
    in
    let k = (shape target_counts).(0) in
    let fudge = 2.1 in
    let norm_slack =
      UF.multiply
        (c (round (round (float_of_int n *. round fudge) *. eps)))
        t_norm
    in
    let lower =
      UF.subtract
        (UF.subtract lambda_min norm_slack)
        (UF.multiply (c (round (2.0 *. fudge))) pivmin)
    in
    let upper =
      UF.add
        (UF.add lambda_max norm_slack)
        (UF.multiply (c (round fudge)) pivmin)
    in
    let spread v = L.broadcast_in_dim v [| k |] [||] in
    let lower = spread lower and upper = spread upper in
    let mid = UF.multiply (c 0.5) (UF.add upper lower) in
    let pivmin = spread pivmin in
    let alpha0_perturbation = spread alpha0_perturbation in
    let at v i = L.index_in_dim ~keepdims:false v i in
    let int_full v = L.full [| k |] (float_of_int v) D.I32 in
    let sturm x =
      let a0 = spread (at alpha_real 0) in
      let q = UF.subtract a0 x in
      let count = NL.where_ (UF.less q (c 0.0)) (int_full 1) (int_full 0) in
      let q = NL.where_ (UF.equal a0 x) alpha0_perturbation q in
      let state = ref (q, count) in
      for i = 1 to n - 1 do
        let q, count = !state in
        let q =
          UF.subtract
            (UF.subtract
               (spread (at alpha_real i))
               (UF.divide (spread (at beta_sq (i - 1))) q))
            x
        in
        let low = UF.less_equal q pivmin in
        let count = NL.where_ low (L.add count (int_full 1)) count in
        let q = NL.where_ low (UF.minimum q (UF.negative pivmin)) q in
        state := (q, count)
      done;
      snd !state
    in
    let cond state =
      match state with
      | [ i; lo; _; hi ] ->
          UF.logical_and
            (L.lt i (L.full [||] (float_of_int max_it) D.I32))
            (L.lt abs_tol (RED.amax (UF.subtract hi lo)))
      | _ -> failwith "eigh_tridiagonal: state"
    in
    let body state =
      match state with
      | [ i; lo; mid; hi ] ->
          let counts = sturm mid in
          let lo = NL.where_ (UF.less_equal counts target_counts) mid lo in
          let hi = NL.where_ (UF.greater counts target_counts) mid hi in
          [
            L.add i (L.full [||] 1.0 D.I32);
            lo;
            UF.multiply (c 0.5) (UF.add lo hi);
            hi;
          ]
      | _ -> failwith "eigh_tridiagonal: state"
    in
    let final =
      L.while_loop cond body [ L.full [||] 0.0 D.I32; lower; mid; upper ]
    in
    let eigvals = List.nth final 2 in
    if eigvals_only then [ eigvals ]
    else
      let key =
        match key with Some k -> k | None -> Random.Core.key (i32_scalar 42)
      in
      let vectors =
        tridiagonal_vectors ~key ~base_dtype ~solve_dtype alpha_real beta
          eigvals
      in
      [ eigvals; NL.transpose vectors ]
  end

let lu_factor a0 =
  Eager_jit.pair "scipy.linalg.lu_factor" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> lu_factor a0 | _ -> assert false)

let lu ?permute_l a0 =
  Eager_jit.many "scipy.linalg.lu" permute_l [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> lu ?permute_l a0
      | _ -> assert false)

let expm ?upper_triangular ?max_squarings a0 =
  Eager_jit.one "scipy.linalg.expm" (upper_triangular, max_squarings) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> expm ?upper_triangular ?max_squarings a0
      | _ -> assert false)

let expm_frechet ?compute_expm a0 a1 =
  Eager_jit.many "scipy.linalg.expm_frechet" compute_expm [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> expm_frechet ?compute_expm a0 a1
      | _ -> assert false)

let eigh_tridiagonal ?eigvals_only ?select ?select_range ?tol ?key a0 a1 =
  Eager_jit.many "scipy.linalg.eigh_tridiagonal"
    (eigvals_only, select, select_range, tol) [ key ] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ key ], [ a0; a1 ] ->
          eigh_tridiagonal ?eigvals_only ?select ?select_range ?tol ?key a0 a1
      | _ -> assert false)

let polar ?side ?method_ ?eps a0 =
  Eager_jit.many "scipy.linalg.polar" (side, method_, eps) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> polar ?side ?method_ ?eps a0
      | _ -> assert false)

let rsf2csf a0 a1 =
  Eager_jit.pair "scipy.linalg.rsf2csf" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> rsf2csf a0 a1
      | _ -> assert false)

let hessenberg ?calc_q a0 =
  Eager_jit.many "scipy.linalg.hessenberg" calc_q [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> hessenberg ?calc_q a0
      | _ -> assert false)

let cholesky ?lower a0 =
  Eager_jit.one "scipy.linalg.cholesky" lower [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> cholesky ?lower a0
      | _ -> assert false)

let svd ?full_matrices ?compute_uv a0 =
  Eager_jit.many "scipy.linalg.svd" (full_matrices, compute_uv) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> svd ?full_matrices ?compute_uv a0
      | _ -> assert false)

let eigh ?lower ?eigvals_only a0 =
  Eager_jit.many "scipy.linalg.eigh" (lower, eigvals_only) [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> eigh ?lower ?eigvals_only a0
      | _ -> assert false)

let schur ?output a0 =
  Eager_jit.many "scipy.linalg.schur" output [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> schur ?output a0
      | _ -> assert false)

let qr ?mode ?pivoting a0 =
  Eager_jit.many "scipy.linalg.qr" (mode, pivoting) [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> qr ?mode ?pivoting a0
      | _ -> assert false)

let solve ?lower ?assume_a a0 a1 =
  Eager_jit.one "scipy.linalg.solve" (lower, assume_a) [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> solve ?lower ?assume_a a0 a1
      | _ -> assert false)

let solve_triangular ?trans ?lower ?unit_diagonal a0 a1 =
  Eager_jit.one "scipy.linalg.solve_triangular" (trans, lower, unit_diagonal)
    [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> solve_triangular ?trans ?lower ?unit_diagonal a0 a1
      | _ -> assert false)

let sqrtm ?blocksize a0 =
  Eager_jit.one "scipy.linalg.sqrtm" blocksize [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> sqrtm ?blocksize a0
      | _ -> assert false)
