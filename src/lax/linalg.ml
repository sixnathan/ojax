open Types
module Lp = Lapack_seam.Lapack

let dims2 nd =
  let s = Ndarray.shape nd in
  (s.(0), s.(1))

let pack_cm nd m n =
  let a = Lp.alloc_f64 (m * n) in
  for j = 0 to n - 1 do
    for i = 0 to m - 1 do
      a.{i + (j * m)} <- Ndarray.get_f nd [| i; j |]
    done
  done;
  a

let unpack_cm a p q =
  let out = Array.make (p * q) 0.0 in
  for i = 0 to p - 1 do
    for j = 0 to q - 1 do
      out.((i * q) + j) <- a.{i + (j * p)}
    done
  done;
  out

let cholesky_impl inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let n, _ = dims2 a in
      let ca = pack_cm a n n in
      let info = Lp.potrf ~uplo:'L' ~n ~a:ca ~lda:n in
      let out = Array.make (n * n) 0.0 in
      if info = 0 then
        for i = 0 to n - 1 do
          for j = 0 to i do
            out.((i * n) + j) <- ca.{i + (j * n)}
          done
        done
      else Array.fill out 0 (n * n) Float.nan;
      [ Ndarray.of_floats dt [| n; n |] out ]
  | _ -> failwith "linalg: cholesky expects one operand"

let lu_impl inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let m, n = dims2 a in
      let ca = pack_cm a m n in
      let k = min m n in
      let ipiv = Lp.alloc_i32 (max k 1) in
      let _ = Lp.getrf ~m ~n ~a:ca ~lda:m ~ipiv in
      let lu = unpack_cm ca m n in
      let pivots = Array.init k (fun i -> Int32.to_int ipiv.{i} - 1) in
      let perm = Array.init m (fun i -> i) in
      for i = 0 to k - 1 do
        let j = pivots.(i) in
        let t = perm.(i) in
        perm.(i) <- perm.(j);
        perm.(j) <- t
      done;
      [
        Ndarray.of_floats dt [| m; n |] lu;
        Ndarray.of_floats Dtype.I32 [| k |] (Array.map float_of_int pivots);
        Ndarray.of_floats Dtype.I32 [| m |] (Array.map float_of_int perm);
      ]
  | _ -> failwith "linalg: lu expects one operand"

let qr_impl full_matrices inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let m, n = dims2 a in
      let mn = min m n in
      let k = if full_matrices then m else mn in
      if k > n then
        failwith "linalg: qr full_matrices with m>n deferred (padding, M5 gap)";
      let ca = pack_cm a m n in
      let tau = Lp.alloc_f64 (max mn 1) in
      let _ = Lp.geqrf ~m ~n ~a:ca ~lda:m ~tau in
      let r = Array.make (k * n) 0.0 in
      for i = 0 to k - 1 do
        for j = i to n - 1 do
          r.((i * n) + j) <- ca.{i + (j * m)}
        done
      done;
      let _ = Lp.orgqr ~m ~n:k ~k:mn ~a:ca ~lda:m ~tau in
      let q = Array.make (m * k) 0.0 in
      for i = 0 to m - 1 do
        for j = 0 to k - 1 do
          q.((i * k) + j) <- ca.{i + (j * m)}
        done
      done;
      [ Ndarray.of_floats dt [| m; k |] q; Ndarray.of_floats dt [| k; n |] r ]
  | _ -> failwith "linalg: qr expects one operand"

let householder_product_impl inputs =
  match inputs with
  | [ a; taus ] ->
      let dt = Ndarray.dtype a in
      let m, n = dims2 a in
      let k = (Ndarray.shape taus).(0) in
      let ca = pack_cm a m n in
      let tau = Lp.alloc_f64 (max k 1) in
      for i = 0 to k - 1 do
        tau.{i} <- Ndarray.get_f taus [| i |]
      done;
      let _ = Lp.orgqr ~m ~n ~k ~a:ca ~lda:m ~tau in
      [ Ndarray.of_floats dt [| m; n |] (unpack_cm ca m n) ]
  | _ -> failwith "linalg: householder_product expects two operands"

let lu_pivots_to_permutation_impl permutation_size inputs =
  match inputs with
  | [ pivots ] ->
      let k = (Ndarray.shape pivots).(0) in
      let m = permutation_size in
      let perm = Array.init m (fun i -> i) in
      for i = 0 to k - 1 do
        let j = Int64.to_int (Ndarray.get_i64 pivots [| i |]) in
        let t = perm.(i) in
        perm.(i) <- perm.(j);
        perm.(j) <- t
      done;
      [ Ndarray.of_floats Dtype.I32 [| m |] (Array.map float_of_int perm) ]
  | _ -> failwith "linalg: lu_pivots_to_permutation expects one operand"

let triangular_solve_impl left_side lower transpose_a _conjugate_a unit_diagonal
    inputs =
  match inputs with
  | [ a; b ] ->
      let dt = Ndarray.dtype b in
      let m, _ = dims2 a in
      let br, bc = dims2 b in
      let diag = if unit_diagonal then 'U' else 'N' in
      let uplo = if lower then 'L' else 'U' in
      let ca = pack_cm a m m in
      if left_side then begin
        let nrhs = bc in
        let cb = pack_cm b m nrhs in
        let trans = if transpose_a then 'T' else 'N' in
        let _ =
          Lp.trtrs ~uplo ~trans ~diag ~n:m ~nrhs ~a:ca ~lda:m ~b:cb ~ldb:m
        in
        [ Ndarray.of_floats dt [| m; nrhs |] (unpack_cm cb m nrhs) ]
      end
      else begin
        let p = br in
        let cb = Lp.alloc_f64 (m * p) in
        for j = 0 to p - 1 do
          for i = 0 to m - 1 do
            cb.{i + (j * m)} <- Ndarray.get_f b [| j; i |]
          done
        done;
        let trans = if transpose_a then 'N' else 'T' in
        let _ =
          Lp.trtrs ~uplo ~trans ~diag ~n:m ~nrhs:p ~a:ca ~lda:m ~b:cb ~ldb:m
        in
        let out = Array.make (p * m) 0.0 in
        for i = 0 to m - 1 do
          for j = 0 to p - 1 do
            out.((j * m) + i) <- cb.{i + (j * m)}
          done
        done;
        [ Ndarray.of_floats dt [| p; m |] out ]
      end
  | _ -> failwith "linalg: triangular_solve expects two operands"

let tridiagonal_solve_impl inputs =
  match inputs with
  | [ dl; d; du; b ] ->
      let dt = Ndarray.dtype b in
      let m = (Ndarray.shape d).(0) in
      let bshape = Ndarray.shape b in
      let kcols = if Array.length bshape >= 2 then bshape.(1) else 1 in
      let dlv i = Ndarray.get_f dl [| i |] in
      let dv i = Ndarray.get_f d [| i |] in
      let duv i = Ndarray.get_f du [| i |] in
      let bv i j = Ndarray.get_f b [| i; j |] in
      let cp = Array.make m 0.0 in
      let dp = Array.make (m * kcols) 0.0 in
      cp.(0) <- duv 0 /. dv 0;
      for j = 0 to kcols - 1 do
        dp.(j) <- bv 0 j /. dv 0
      done;
      for i = 1 to m - 1 do
        let den = dv i -. (dlv i *. cp.(i - 1)) in
        cp.(i) <- (if i < m - 1 then duv i else 0.0) /. den;
        for j = 0 to kcols - 1 do
          dp.((i * kcols) + j) <-
            (bv i j -. (dlv i *. dp.(((i - 1) * kcols) + j))) /. den
        done
      done;
      let x = Array.make (m * kcols) 0.0 in
      for j = 0 to kcols - 1 do
        x.(((m - 1) * kcols) + j) <- dp.(((m - 1) * kcols) + j)
      done;
      for i = m - 2 downto 0 do
        for j = 0 to kcols - 1 do
          x.((i * kcols) + j) <-
            dp.((i * kcols) + j) -. (cp.(i) *. x.(((i + 1) * kcols) + j))
        done
      done;
      [ Ndarray.of_floats dt bshape x ]
  | _ -> failwith "linalg: tridiagonal_solve expects four operands"

let to_complex_dtype = function
  | Dtype.F32 -> Dtype.Complex64
  | Dtype.F64 -> Dtype.Complex128
  | d -> d

let real_basetype = function
  | Dtype.Complex64 -> Dtype.F32
  | Dtype.Complex128 -> Dtype.F64
  | d -> d

let to_farray b n = Array.init n (fun i -> b.{i})

let eigh_impl lower inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let n, _ = dims2 a in
      let ca = pack_cm a n n in
      let w = Lp.alloc_f64 (max n 1) in
      let uplo = if lower then 'L' else 'U' in
      let _ = Lp.syevd ~jobz:'V' ~uplo ~n ~a:ca ~lda:n ~w in
      [
        Ndarray.of_floats dt [| n; n |] (unpack_cm ca n n);
        Ndarray.of_floats (real_basetype dt) [| n |] (to_farray w n);
      ]
  | _ -> failwith "linalg: eigh expects one operand"

let unpack_eigvecs n pv wi =
  let cv = Array.make (n * n) Complex.zero in
  let j = ref 0 in
  while !j < n do
    if wi.(!j) = 0.0 then begin
      for i = 0 to n - 1 do
        cv.((i * n) + !j) <- { Complex.re = pv.{i + (!j * n)}; im = 0.0 }
      done;
      incr j
    end
    else begin
      let jj = !j in
      for i = 0 to n - 1 do
        let re = pv.{i + (jj * n)} and im = pv.{i + ((jj + 1) * n)} in
        cv.((i * n) + jj) <- { Complex.re; im };
        cv.((i * n) + jj + 1) <- { Complex.re; im = -.im }
      done;
      j := jj + 2
    end
  done;
  cv

let eig_impl compute_left compute_right inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let cdt = to_complex_dtype dt in
      let n, _ = dims2 a in
      let ca = pack_cm a n n in
      let wri = Lp.alloc_f64 (2 * n) in
      let ldvl = if compute_left then n else 1 in
      let ldvr = if compute_right then n else 1 in
      let vl = Lp.alloc_f64 (n * ldvl) in
      let vr = Lp.alloc_f64 (n * ldvr) in
      let jobvl = if compute_left then 'V' else 'N' in
      let jobvr = if compute_right then 'V' else 'N' in
      let _ = Lp.geev ~jobvl ~jobvr ~n ~a:ca ~lda:n ~wri ~vl ~ldvl ~vr ~ldvr in
      let wi = Array.init n (fun i -> wri.{i + n}) in
      let w =
        Array.init n (fun i -> { Complex.re = wri.{i}; im = wri.{i + n} })
      in
      let left = if compute_left then [ (vl, ldvl) ] else [] in
      let right = if compute_right then [ (vr, ldvr) ] else [] in
      let vecs =
        List.map
          (fun (pv, _) ->
            Ndarray.of_complex cdt [| n; n |] (unpack_eigvecs n pv wi))
          (left @ right)
      in
      Ndarray.of_complex cdt [| n |] w :: vecs
  | _ -> failwith "linalg: eig expects one operand"

let hessenberg_impl inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let n, _ = dims2 a in
      let ca = pack_cm a n n in
      let tau = Lp.alloc_f64 (max (n - 1) 1) in
      let _ = Lp.gehrd ~n ~ilo:1 ~ihi:n ~a:ca ~lda:n ~tau in
      [
        Ndarray.of_floats dt [| n; n |] (unpack_cm ca n n);
        Ndarray.of_floats dt [| n - 1 |] (to_farray tau (n - 1));
      ]
  | _ -> failwith "linalg: hessenberg expects one operand"

let schur_impl compute_vectors inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let n, _ = dims2 a in
      let ca = pack_cm a n n in
      let wri = Lp.alloc_f64 (2 * n) in
      let ldvs = if compute_vectors then n else 1 in
      let vs = Lp.alloc_f64 (n * ldvs) in
      let sdim = Lp.alloc_i32 1 in
      let jobvs = if compute_vectors then 'V' else 'N' in
      let _ = Lp.gees ~jobvs ~n ~a:ca ~lda:n ~wri ~vs ~ldvs ~sdim in
      let t = Ndarray.of_floats dt [| n; n |] (unpack_cm ca n n) in
      if compute_vectors then
        [ t; Ndarray.of_floats dt [| n; n |] (unpack_cm vs n n) ]
      else [ t ]
  | _ -> failwith "linalg: schur expects one operand"

let svd_impl full_matrices compute_uv inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let rdt = real_basetype dt in
      let m, n = dims2 a in
      let mn = min m n in
      let ca = pack_cm a m n in
      let s = Lp.alloc_f64 (max mn 1) in
      if compute_uv then begin
        let jobz = if full_matrices then 'A' else 'S' in
        let ucols = if full_matrices then m else mn in
        let vrows = if full_matrices then n else mn in
        let u = Lp.alloc_f64 (m * ucols) in
        let vt = Lp.alloc_f64 (vrows * n) in
        let _ =
          Lp.gesdd ~jobz ~m ~n ~a:ca ~lda:m ~s ~u ~ldu:m ~vt ~ldvt:vrows
        in
        [
          Ndarray.of_floats dt [| m; ucols |] (unpack_cm u m ucols);
          Ndarray.of_floats rdt [| mn |] (to_farray s mn);
          Ndarray.of_floats dt [| vrows; n |] (unpack_cm vt vrows n);
        ]
      end
      else begin
        let u = Lp.alloc_f64 1 and vt = Lp.alloc_f64 1 in
        let _ =
          Lp.gesdd ~jobz:'N' ~m ~n ~a:ca ~lda:m ~s ~u ~ldu:1 ~vt ~ldvt:1
        in
        [ Ndarray.of_floats rdt [| mn |] (to_farray s mn) ]
      end
  | _ -> failwith "linalg: svd expects one operand"

let tridiagonal_impl lower inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let rdt = real_basetype dt in
      let n, _ = dims2 a in
      let ca = pack_cm a n n in
      let d = Lp.alloc_f64 (max n 1) in
      let e = Lp.alloc_f64 (max (n - 1) 1) in
      let tau = Lp.alloc_f64 (max (n - 1) 1) in
      let uplo = if lower then 'L' else 'U' in
      let _ = Lp.sytrd ~uplo ~n ~a:ca ~lda:n ~d ~e ~tau in
      [
        Ndarray.of_floats dt [| n; n |] (unpack_cm ca n n);
        Ndarray.of_floats rdt [| n |] (to_farray d n);
        Ndarray.of_floats rdt [| n - 1 |] (to_farray e (n - 1));
        Ndarray.of_floats dt [| n - 1 |] (to_farray tau (n - 1));
      ]
  | _ -> failwith "linalg: tridiagonal expects one operand"

let int_aval shape = { shape; dtype = Dtype.I32; weak_type = false }

let cholesky_aval avals =
  match avals with
  | [ a ] -> [ a ]
  | _ -> failwith "linalg: cholesky expects one aval"

let lu_aval avals =
  match avals with
  | [ a ] ->
      let m = a.shape.(0) and n = a.shape.(1) in
      [ a; int_aval [| min m n |]; int_aval [| m |] ]
  | _ -> failwith "linalg: lu expects one aval"

let qr_aval full_matrices avals =
  match avals with
  | [ a ] ->
      let m = a.shape.(0) and n = a.shape.(1) in
      let k = if full_matrices then m else min m n in
      [ { a with shape = [| m; k |] }; { a with shape = [| k; n |] } ]
  | _ -> failwith "linalg: qr expects one aval"

let householder_product_aval avals =
  match avals with
  | [ a; _ ] -> [ a ]
  | _ -> failwith "linalg: householder_product expects two avals"

let lu_pivots_to_permutation_aval permutation_size avals =
  match avals with
  | [ _ ] -> [ int_aval [| permutation_size |] ]
  | _ -> failwith "linalg: lu_pivots_to_permutation expects one aval"

let triangular_solve_aval avals =
  match avals with
  | [ _; b ] -> [ b ]
  | _ -> failwith "linalg: triangular_solve expects two avals"

let tridiagonal_solve_aval avals =
  match avals with
  | [ _; _; _; b ] -> [ b ]
  | _ -> failwith "linalg: tridiagonal_solve expects four avals"

let eigh_aval = function
  | [ a ] ->
      let n = a.shape.(0) in
      [
        { a with shape = [| n; n |] };
        { shape = [| n |]; dtype = real_basetype a.dtype; weak_type = false };
      ]
  | _ -> failwith "linalg: eigh expects one aval"

let eig_aval compute_left compute_right = function
  | [ a ] ->
      let n = a.shape.(0) in
      let cdt = to_complex_dtype a.dtype in
      let vec = { shape = [| n; n |]; dtype = cdt; weak_type = false } in
      let left = if compute_left then [ vec ] else [] in
      let right = if compute_right then [ vec ] else [] in
      { shape = [| n |]; dtype = cdt; weak_type = false } :: (left @ right)
  | _ -> failwith "linalg: eig expects one aval"

let hessenberg_aval = function
  | [ a ] ->
      let n = a.shape.(0) in
      [ a; { a with shape = [| n - 1 |] } ]
  | _ -> failwith "linalg: hessenberg expects one aval"

let schur_aval compute_vectors = function
  | [ a ] -> if compute_vectors then [ a; a ] else [ a ]
  | _ -> failwith "linalg: schur expects one aval"

let svd_aval full_matrices compute_uv = function
  | [ a ] ->
      let m = a.shape.(0) and n = a.shape.(1) in
      let mn = min m n in
      let rdt = real_basetype a.dtype in
      let s = { shape = [| mn |]; dtype = rdt; weak_type = false } in
      if compute_uv then
        let ucols = if full_matrices then m else mn in
        let vrows = if full_matrices then n else mn in
        [
          { a with shape = [| m; ucols |] };
          s;
          { a with shape = [| vrows; n |] };
        ]
      else [ s ]
  | _ -> failwith "linalg: svd expects one aval"

let tridiagonal_aval = function
  | [ a ] ->
      let n = a.shape.(0) in
      let rdt = real_basetype a.dtype in
      [
        a;
        { shape = [| n |]; dtype = rdt; weak_type = false };
        { shape = [| n - 1 |]; dtype = rdt; weak_type = false };
        { a with shape = [| n - 1 |] };
      ]
  | _ -> failwith "linalg: tridiagonal expects one aval"

let cholesky x = Core.bind1 Cholesky [ x ]

let lu x =
  match Core.bind Lu [ x ] with
  | [ a; b; c ] -> (a, b, c)
  | _ -> failwith "linalg: lu arity"

let qr ?(full_matrices = true) x =
  match Core.bind (Qr full_matrices) [ x ] with
  | [ q; r ] -> (q, r)
  | _ -> failwith "linalg: qr arity"

let householder_product a taus = Core.bind1 Householder_product [ a; taus ]

let lu_pivots_to_permutation ~permutation_size p =
  Core.bind1 (Lu_pivots_to_permutation permutation_size) [ p ]

let triangular_solve ?(left_side = false) ?(lower = false)
    ?(transpose_a = false) ?(conjugate_a = false) ?(unit_diagonal = false) a b =
  Core.bind1
    (Triangular_solve
       { left_side; lower; transpose_a; conjugate_a; unit_diagonal })
    [ a; b ]

let tridiagonal_solve dl d du b = Core.bind1 Tridiagonal_solve [ dl; d; du; b ]

let eig ?(compute_left_eigenvectors = true) ?(compute_right_eigenvectors = true)
    x =
  Core.bind
    (Eig
       {
         compute_left = compute_left_eigenvectors;
         compute_right = compute_right_eigenvectors;
       })
    [ x ]

let eigh ?(lower = true) x =
  match Core.bind (Eigh { lower }) [ x ] with
  | [ v; w ] -> (v, w)
  | _ -> failwith "linalg: eigh arity"

let hessenberg x =
  match Core.bind Hessenberg [ x ] with
  | [ a; taus ] -> (a, taus)
  | _ -> failwith "linalg: hessenberg arity"

let schur ?(compute_schur_vectors = true) x =
  Core.bind (Schur { compute_vectors = compute_schur_vectors }) [ x ]

let svd ?(full_matrices = true) ?(compute_uv = true) x =
  Core.bind (Svd { full_matrices; compute_uv }) [ x ]

let tridiagonal ?(lower = true) x =
  match Core.bind (Tridiagonal { lower }) [ x ] with
  | [ a; d; e; taus ] -> (a, d, e, taus)
  | _ -> failwith "linalg: tridiagonal arity"
