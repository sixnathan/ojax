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

let is_complex_nd nd =
  match Ndarray.dtype nd with
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let zpack nd m n =
  let a = Lp.alloc_f64 (2 * m * n) in
  for j = 0 to n - 1 do
    for i = 0 to m - 1 do
      let z = Ndarray.get_c nd [| i; j |] in
      let k = 2 * (i + (j * m)) in
      a.{k} <- z.Complex.re;
      a.{k + 1} <- z.Complex.im
    done
  done;
  a

let zat a k = { Complex.re = a.{2 * k}; im = a.{(2 * k) + 1} }

let zunpack a p q =
  Array.init (p * q) (fun f ->
      let i = f / max q 1 and j = f mod max q 1 in
      zat a (i + (j * p)))

let z_cholesky a =
  let dt = Ndarray.dtype a in
  let n = (Ndarray.shape a).(0) in
  let ca = zpack a n n in
  let info = Lp.zpotrf ~uplo:'L' ~n ~a:ca ~lda:n in
  let nan = { Complex.re = Float.nan; im = Float.nan } in
  [
    Ndarray.of_complex dt [| n; n |]
      (Array.init (n * n) (fun f ->
           let i = f / n and j = f mod n in
           if info <> 0 then nan
           else if j <= i then zat ca (i + (j * n))
           else Complex.zero));
  ]

let z_lu a =
  let dt = Ndarray.dtype a in
  let sh = Ndarray.shape a in
  let m = sh.(0) and n = sh.(1) in
  let ca = zpack a m n in
  let k = min m n in
  let ipiv = Lp.alloc_i32 (max k 1) in
  let _ = Lp.zgetrf ~m ~n ~a:ca ~lda:m ~ipiv in
  let pivots = Array.init k (fun i -> Int32.to_int ipiv.{i} - 1) in
  let perm = Array.init m (fun i -> i) in
  for i = 0 to k - 1 do
    let j = pivots.(i) in
    let t = perm.(i) in
    perm.(i) <- perm.(j);
    perm.(j) <- t
  done;
  [
    Ndarray.of_complex dt [| m; n |] (zunpack ca m n);
    Ndarray.of_floats Dtype.I32 [| k |] (Array.map float_of_int pivots);
    Ndarray.of_floats Dtype.I32 [| m |] (Array.map float_of_int perm);
  ]

let z_qr full_matrices a =
  let dt = Ndarray.dtype a in
  let sh = Ndarray.shape a in
  let m = sh.(0) and n = sh.(1) in
  let mn = min m n in
  let k = if full_matrices then m else mn in
  let ca = zpack a m n in
  let tau = Lp.alloc_f64 (2 * max mn 1) in
  let _ = Lp.zgeqrf ~m ~n ~a:ca ~lda:m ~tau in
  let r =
    Array.init (k * n) (fun f ->
        let i = f / n and j = f mod n in
        if j >= i then zat ca (i + (j * m)) else Complex.zero)
  in
  let work = Lp.alloc_f64 (2 * max (m * max k n) 1) in
  for x = 0 to (2 * m * min n k) - 1 do
    work.{x} <- ca.{x}
  done;
  let _ = Lp.zungqr ~m ~n:k ~k:mn ~a:work ~lda:m ~tau in
  [
    Ndarray.of_complex dt [| m; k |] (zunpack work m k);
    Ndarray.of_complex dt [| k; n |] r;
  ]

let z_householder_product a taus =
  let dt = Ndarray.dtype a in
  let sh = Ndarray.shape a in
  let m = sh.(0) and n = sh.(1) in
  let k = (Ndarray.shape taus).(0) in
  let ca = zpack a m n in
  let tau =
    zpack
      (Ndarray.of_complex dt [| k; 1 |]
         (Array.init k (fun i -> Ndarray.get_c taus [| i |])))
      k 1
  in
  let _ = Lp.zungqr ~m ~n ~k ~a:ca ~lda:m ~tau in
  [ Ndarray.of_complex dt [| m; n |] (zunpack ca m n) ]

let z_real_basetype = function Dtype.Complex128 -> Dtype.F64 | _ -> Dtype.F32

let z_eigh lower a =
  let dt = Ndarray.dtype a in
  let n = (Ndarray.shape a).(0) in
  let ca = zpack a n n in
  let w = Lp.alloc_f64 (max n 1) in
  let uplo = if lower then 'L' else 'U' in
  let _ = Lp.zheevd ~jobz:'V' ~uplo ~n ~a:ca ~lda:n ~w in
  [
    Ndarray.of_complex dt [| n; n |] (zunpack ca n n);
    Ndarray.of_floats (z_real_basetype dt) [| n |]
      (Array.init n (fun i -> w.{i}));
  ]

let z_svd gesvd full_matrices compute_uv a =
  let driver = if gesvd then Lp.zgesvd else Lp.zgesdd in
  let dt = Ndarray.dtype a in
  let sh = Ndarray.shape a in
  let m = sh.(0) and n = sh.(1) in
  let mn = min m n in
  let ca = zpack a m n in
  let s = Lp.alloc_f64 (max mn 1) in
  let real = Ndarray.of_floats (z_real_basetype dt) [| mn |] in
  if compute_uv then begin
    let jobz = if full_matrices then 'A' else 'S' in
    let ucols = if full_matrices then m else mn in
    let vrows = if full_matrices then n else mn in
    let u = Lp.alloc_f64 (2 * m * ucols) in
    let vt = Lp.alloc_f64 (2 * vrows * n) in
    let _ = driver ~jobz ~m ~n ~a:ca ~lda:m ~s ~u ~ldu:m ~vt ~ldvt:vrows in
    [
      Ndarray.of_complex dt [| m; ucols |] (zunpack u m ucols);
      real (Array.init mn (fun i -> s.{i}));
      Ndarray.of_complex dt [| vrows; n |] (zunpack vt vrows n);
    ]
  end
  else begin
    let u = Lp.alloc_f64 2 and vt = Lp.alloc_f64 2 in
    let _ = driver ~jobz:'N' ~m ~n ~a:ca ~lda:m ~s ~u ~ldu:1 ~vt ~ldvt:1 in
    [ real (Array.init mn (fun i -> s.{i})) ]
  end

let z_eig compute_left compute_right a =
  let dt = Ndarray.dtype a in
  let n = (Ndarray.shape a).(0) in
  let ca = zpack a n n in
  let w = Lp.alloc_f64 (2 * max n 1) in
  let ldvl = if compute_left then n else 1 in
  let ldvr = if compute_right then n else 1 in
  let vl = Lp.alloc_f64 (2 * n * ldvl) in
  let vr = Lp.alloc_f64 (2 * n * ldvr) in
  let jobvl = if compute_left then 'V' else 'N' in
  let jobvr = if compute_right then 'V' else 'N' in
  let _ = Lp.zgeev ~jobvl ~jobvr ~n ~a:ca ~lda:n ~w ~vl ~ldvl ~vr ~ldvr in
  let vectors flag buf =
    if flag then [ Ndarray.of_complex dt [| n; n |] (zunpack buf n n) ] else []
  in
  Ndarray.of_complex dt [| n |] (Array.init n (fun i -> zat w i))
  :: vectors compute_left vl
  @ vectors compute_right vr

let z_triangular_solve left_side lower transpose_a conjugate_a unit_diagonal a b
    =
  let dt = Ndarray.dtype b in
  let m = (Ndarray.shape a).(0) in
  let bs = Ndarray.shape b in
  let as_complex nd =
    if is_complex_nd nd then nd
    else
      let sh = Ndarray.shape nd in
      let total = Array.fold_left ( * ) 1 sh in
      Ndarray.of_complex dt sh
        (Array.init total (fun f ->
             let idx = Array.make (Array.length sh) 0 in
             let r = ref f in
             for d = Array.length sh - 1 downto 0 do
               idx.(d) <- !r mod sh.(d);
               r := !r / sh.(d)
             done;
             { Complex.re = Ndarray.get_f nd idx; im = 0.0 }))
  in
  let a = as_complex a in
  let ca = zpack a m m in
  let uplo = if lower then 'L' else 'U' in
  let diag = if unit_diagonal then 'U' else 'N' in
  let effective_transpose =
    if left_side then transpose_a else not transpose_a
  in
  let trans, conjugate_rhs =
    match (effective_transpose, conjugate_a) with
    | false, false -> ('N', false)
    | true, false -> ('T', false)
    | true, true -> ('C', false)
    | false, true -> ('N', true)
  in
  let nrhs = if left_side then bs.(1) else bs.(0) in
  let cb = Lp.alloc_f64 (2 * m * nrhs) in
  for c = 0 to nrhs - 1 do
    for r = 0 to m - 1 do
      let z =
        if left_side then Ndarray.get_c b [| r; c |]
        else Ndarray.get_c b [| c; r |]
      in
      let z = if conjugate_rhs then Complex.conj z else z in
      let k = 2 * (r + (c * m)) in
      cb.{k} <- z.Complex.re;
      cb.{k + 1} <- z.Complex.im
    done
  done;
  let _ = Lp.ztrtrs ~uplo ~trans ~diag ~n:m ~nrhs ~a:ca ~lda:m ~b:cb ~ldb:m in
  let read r c =
    let z = zat cb (r + (c * m)) in
    if conjugate_rhs then Complex.conj z else z
  in
  if left_side then
    [
      Ndarray.of_complex dt [| m; nrhs |]
        (Array.init (m * nrhs) (fun f -> read (f / nrhs) (f mod nrhs)));
    ]
  else
    [
      Ndarray.of_complex dt [| nrhs; m |]
        (Array.init (nrhs * m) (fun f -> read (f mod m) (f / m)));
    ]

let z_hessenberg a =
  let dt = Ndarray.dtype a in
  let n = (Ndarray.shape a).(0) in
  let ca = zpack a n n in
  let tau = Lp.alloc_f64 (2 * max (n - 1) 1) in
  let _ = Lp.zgehrd ~n ~ilo:1 ~ihi:n ~a:ca ~lda:n ~tau in
  [
    Ndarray.of_complex dt [| n; n |] (zunpack ca n n);
    Ndarray.of_complex dt [| n - 1 |] (Array.init (n - 1) (fun i -> zat tau i));
  ]

let z_schur compute_vectors a =
  let dt = Ndarray.dtype a in
  let n = (Ndarray.shape a).(0) in
  let ca = zpack a n n in
  let w = Lp.alloc_f64 (2 * max n 1) in
  let ldvs = if compute_vectors then n else 1 in
  let vs = Lp.alloc_f64 (2 * n * ldvs) in
  let jobvs = if compute_vectors then 'V' else 'N' in
  let _ = Lp.zgees ~jobvs ~n ~a:ca ~lda:n ~w ~vs ~ldvs in
  let t = Ndarray.of_complex dt [| n; n |] (zunpack ca n n) in
  if compute_vectors then
    [ t; Ndarray.of_complex dt [| n; n |] (zunpack vs n n) ]
  else [ t ]

let z_tridiagonal lower a =
  let dt = Ndarray.dtype a in
  let rdt = z_real_basetype dt in
  let n = (Ndarray.shape a).(0) in
  let ca = zpack a n n in
  let d = Lp.alloc_f64 (max n 1) in
  let e = Lp.alloc_f64 (max (n - 1) 1) in
  let tau = Lp.alloc_f64 (2 * max (n - 1) 1) in
  let uplo = if lower then 'L' else 'U' in
  let _ = Lp.zhetrd ~uplo ~n ~a:ca ~lda:n ~d ~e ~tau in
  [
    Ndarray.of_complex dt [| n; n |] (zunpack ca n n);
    Ndarray.of_floats rdt [| n |] (Array.init n (fun i -> d.{i}));
    Ndarray.of_floats rdt [| n - 1 |] (Array.init (n - 1) (fun i -> e.{i}));
    Ndarray.of_complex dt [| n - 1 |] (Array.init (n - 1) (fun i -> zat tau i));
  ]

let geqrf_impl a =
  let dt = Ndarray.dtype a in
  let sh = Ndarray.shape a in
  let m = sh.(0) and n = sh.(1) in
  let k = min m n in
  if is_complex_nd a then begin
    let ca = zpack a m n in
    let tau = Lp.alloc_f64 (2 * max k 1) in
    let _ = Lp.zgeqrf ~m ~n ~a:ca ~lda:m ~tau in
    [
      Ndarray.of_complex dt [| m; n |] (zunpack ca m n);
      Ndarray.of_complex dt [| k |] (Array.init k (fun i -> zat tau i));
    ]
  end
  else begin
    let ca = pack_cm a m n in
    let tau = Lp.alloc_f64 (max k 1) in
    let _ = Lp.geqrf ~m ~n ~a:ca ~lda:m ~tau in
    [
      Ndarray.of_floats dt [| m; n |] (unpack_cm ca m n);
      Ndarray.of_floats dt [| k |] (Array.init k (fun i -> tau.{i}));
    ]
  end

let geqp3_impl a =
  let dt = Ndarray.dtype a in
  let sh = Ndarray.shape a in
  let m = sh.(0) and n = sh.(1) in
  let k = min m n in
  let jpvt = Lp.alloc_i32 (max n 1) in
  for i = 0 to n - 1 do
    jpvt.{i} <- 0l
  done;
  let pivots () =
    Ndarray.of_floats Dtype.I32 [| n |]
      (Array.init n (fun i -> Int32.to_float jpvt.{i}))
  in
  if is_complex_nd a then begin
    let ca = zpack a m n in
    let tau = Lp.alloc_f64 (2 * max k 1) in
    let _ = Lp.zgeqp3 ~m ~n ~a:ca ~lda:m ~jpvt ~tau in
    [
      Ndarray.of_complex dt [| m; n |] (zunpack ca m n);
      pivots ();
      Ndarray.of_complex dt [| k |] (Array.init k (fun i -> zat tau i));
    ]
  end
  else begin
    let ca = pack_cm a m n in
    let tau = Lp.alloc_f64 (max k 1) in
    let _ = Lp.geqp3 ~m ~n ~a:ca ~lda:m ~jpvt ~tau in
    [
      Ndarray.of_floats dt [| m; n |] (unpack_cm ca m n);
      pivots ();
      Ndarray.of_floats dt [| k |] (Array.init k (fun i -> tau.{i}));
    ]
  end

let cholesky_impl inputs =
  match inputs with
  | [ a ] when is_complex_nd a -> z_cholesky a
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
  | [ a ] when is_complex_nd a -> z_lu a
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
  | [ a ] when is_complex_nd a -> z_qr full_matrices a
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let m, n = dims2 a in
      let mn = min m n in
      let k = if full_matrices then m else mn in
      let ca = pack_cm a m n in
      let tau = Lp.alloc_f64 (max mn 1) in
      let _ = Lp.geqrf ~m ~n ~a:ca ~lda:m ~tau in
      let r = Array.make (k * n) 0.0 in
      for i = 0 to k - 1 do
        for j = i to n - 1 do
          r.((i * n) + j) <- ca.{i + (j * m)}
        done
      done;
      let work = Lp.alloc_f64 (max (m * max k n) 1) in
      for j = 0 to min n k - 1 do
        for i = 0 to m - 1 do
          work.{i + (j * m)} <- ca.{i + (j * m)}
        done
      done;
      let _ = Lp.orgqr ~m ~n:k ~k:mn ~a:work ~lda:m ~tau in
      let q = Array.make (m * k) 0.0 in
      for i = 0 to m - 1 do
        for j = 0 to k - 1 do
          q.((i * k) + j) <- work.{i + (j * m)}
        done
      done;
      [ Ndarray.of_floats dt [| m; k |] q; Ndarray.of_floats dt [| k; n |] r ]
  | _ -> failwith "linalg: qr expects one operand"

let householder_product_impl inputs =
  match inputs with
  | [ a; taus ] when is_complex_nd a -> z_householder_product a taus
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

let triangular_solve_impl left_side lower transpose_a conjugate_a unit_diagonal
    inputs =
  match inputs with
  | [ a; b ] when is_complex_nd a || is_complex_nd b ->
      z_triangular_solve left_side lower transpose_a conjugate_a unit_diagonal a
        b
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

let tridiagonal_cell nd i j =
  if Array.length (Ndarray.shape nd) >= 2 then Ndarray.get_c nd [| i; j |]
  else Ndarray.get_c nd [| i |]

let tridiagonal_solve_impl ~perturb_singular inputs =
  match inputs with
  | [ dl; d; du; b ] ->
      let dt = Ndarray.dtype b in
      let m = (Ndarray.shape d).(0) in
      let bshape = Ndarray.shape b in
      let kcols = if Array.length bshape >= 2 then bshape.(1) else 1 in
      let single = dt = Dtype.F32 || dt = Dtype.Complex64 in
      let round_part v =
        if single then Int32.float_of_bits (Int32.bits_of_float v) else v
      in
      let round (z : Complex.t) =
        { Complex.re = round_part z.re; im = round_part z.im }
      in
      let is_complex = dt = Dtype.Complex64 || dt = Dtype.Complex128 in
      let ( *! ) x y = round (Complex.mul x y) in
      let ( /! ) x y =
        if is_complex then round (Complex.div x y)
        else
          { Complex.re = round_part (x.Complex.re /. y.Complex.re); im = 0.0 }
      in
      let ( -! ) x y = round (Complex.sub x y) in
      let fused x f y =
        if is_complex then x -! (f *! y)
        else
          let re =
            if single then
              round_part (x.Complex.re -. (f.Complex.re *. y.Complex.re))
            else Float.fma (-.f.Complex.re) y.Complex.re x.Complex.re
          in
          { Complex.re; im = 0.0 }
      in
      let cabs (z : Complex.t) =
        if is_complex then round_part (Complex.norm z) else Float.abs z.re
      in
      let zero = Complex.zero in
      let get nd i = Ndarray.get_c nd [| i |] in
      let a = Array.init m (get d) in
      let upper =
        Array.init m (fun i -> if i < m - 1 then get du i else zero)
      in
      let second = Array.make m zero in
      let lower =
        Array.init m (fun i -> if i < m - 1 then get dl (i + 1) else zero)
      in
      let x =
        Array.init kcols (fun j ->
            Array.init m (fun i -> tridiagonal_cell b i j))
      in
      let singular = ref false in
      let swap_rows k =
        Array.iter
          (fun col ->
            let tmp = col.(k) in
            col.(k) <- col.(k + 1);
            col.(k + 1) <- tmp)
          x
      in
      let scale1 = ref (round_part (cabs a.(0) +. cabs upper.(0))) in
      for k = 0 to m - 2 do
        let sub = lower.(k) in
        let swap =
          if perturb_singular then begin
            let scale2 =
              round_part
                (round_part (cabs sub +. cabs a.(k + 1))
                +. if k < m - 2 then cabs upper.(k + 1) else 0.0)
            in
            let piv1 =
              if a.(k) = zero then 0.0 else round_part (cabs a.(k) /. !scale1)
            in
            if sub = zero then begin
              scale1 := scale2;
              false
            end
            else begin
              let piv2 = round_part (cabs sub /. scale2) in
              let swapping = not (piv2 <= piv1) in
              if not swapping then scale1 := scale2;
              swapping
            end
          end
          else cabs a.(k) < cabs sub
        in
        if not swap then
          begin if a.(k) = zero then
            begin if sub <> zero then singular := true
            end
          else begin
            let factor = sub /! a.(k) in
            a.(k + 1) <- fused a.(k + 1) factor upper.(k);
            Array.iter
              (fun col -> col.(k + 1) <- fused col.(k + 1) factor col.(k))
              x
          end
          end
        else begin
          let factor = a.(k) /! sub in
          let next_a = a.(k + 1) in
          a.(k) <- sub;
          a.(k + 1) <- fused upper.(k) factor next_a;
          upper.(k) <- next_a;
          if k < m - 2 then begin
            second.(k) <- upper.(k + 1);
            upper.(k + 1) <- fused zero factor second.(k)
          end;
          swap_rows k;
          Array.iter
            (fun col -> col.(k + 1) <- fused col.(k + 1) factor col.(k))
            x
        end
      done;
      let tol =
        if not perturb_singular then 0.0
        else begin
          let largest = ref 0.0 in
          for k = 0 to m - 1 do
            largest :=
              Float.max !largest
                (Float.max
                   (cabs a.(k))
                   (Float.max (cabs upper.(k)) (cabs second.(k))))
          done;
          let eps, tiny =
            if single then (1.1920928955078125e-07, 1.1754943508222875e-38)
            else (2.220446049250313e-16, 2.2250738585072014e-308)
          in
          let t = round_part (eps *. !largest) in
          if t = 0.0 then tiny else t
        end
      in
      if not perturb_singular then
        Array.iter (fun p -> if p = zero then singular := true) a;
      let pivot k =
        if perturb_singular && cabs a.(k) < tol then
          if is_complex then begin
            let m = cabs a.(k) in
            if m = 0.0 then { Complex.re = tol; im = 0.0 }
            else Complex.mul a.(k) { Complex.re = tol /. m; im = 0.0 }
          end
          else { Complex.re = Float.copy_sign tol a.(k).re; im = 0.0 }
        else a.(k)
      in
      Array.iter
        (fun col ->
          for k = m - 1 downto 0 do
            let v = ref col.(k) in
            if k + 1 < m then v := fused !v upper.(k) col.(k + 1);
            if k + 2 < m then v := fused !v second.(k) col.(k + 2);
            col.(k) <- !v /! pivot k
          done)
        x;
      let nan_cell =
        { Complex.re = Float.nan; im = (if is_complex then Float.nan else 0.0) }
      in
      let at i j = if !singular then nan_cell else x.(j).(i) in
      let n_cells = m * kcols in
      if is_complex then
        [
          Ndarray.of_complex dt bshape
            (Array.init n_cells (fun f -> at (f / kcols) (f mod kcols)));
        ]
      else
        [
          Ndarray.of_floats dt bshape
            (Array.init n_cells (fun f ->
                 (at (f / kcols) (f mod kcols)).Complex.re));
        ]
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
  | [ a ] when is_complex_nd a -> z_eigh lower a
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
  | [ a ] when is_complex_nd a -> z_eig compute_left compute_right a
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
  | [ a ] when is_complex_nd a -> z_hessenberg a
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
  | [ a ] when is_complex_nd a -> z_schur compute_vectors a
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

let svd_impl ?(gesvd = false) full_matrices compute_uv inputs =
  let driver = if gesvd then Lp.gesvd else Lp.gesdd in
  match inputs with
  | [ a ] when is_complex_nd a -> z_svd gesvd full_matrices compute_uv a
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
        let _ = driver ~jobz ~m ~n ~a:ca ~lda:m ~s ~u ~ldu:m ~vt ~ldvt:vrows in
        [
          Ndarray.of_floats dt [| m; ucols |] (unpack_cm u m ucols);
          Ndarray.of_floats rdt [| mn |] (to_farray s mn);
          Ndarray.of_floats dt [| vrows; n |] (unpack_cm vt vrows n);
        ]
      end
      else begin
        let u = Lp.alloc_f64 1 and vt = Lp.alloc_f64 1 in
        let _ = driver ~jobz:'N' ~m ~n ~a:ca ~lda:m ~s ~u ~ldu:1 ~vt ~ldvt:1 in
        [ Ndarray.of_floats rdt [| mn |] (to_farray s mn) ]
      end
  | _ -> failwith "linalg: svd expects one operand"

let tridiagonal_impl lower inputs =
  match inputs with
  | [ a ] when is_complex_nd a -> z_tridiagonal lower a
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

let symmetrize x =
  let aval = Core.get_aval x in
  let rank = Array.length aval.shape in
  let perm =
    Array.init rank (fun i ->
        if i = rank - 1 then rank - 2 else if i = rank - 2 then rank - 1 else i)
  in
  let flipped = Core.bind1 (Transpose perm) [ x ] in
  let flipped =
    match aval.dtype with
    | Dtype.Complex64 | Dtype.Complex128 -> Core.bind1 Conj [ flipped ]
    | _ -> flipped
  in
  let half = Core.splat aval.dtype aval.shape 2.0 in
  Core.bind1 Div [ Core.bind1 Add [ x; flipped ]; half ]

let cholesky ?(symmetrize_input = true) x =
  Core.bind1 Cholesky [ (if symmetrize_input then symmetrize x else x) ]

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

let tridiagonal_solve ?(perturb_singular = false) dl d du b =
  Core.bind1 (Tridiagonal_solve { perturb_singular }) [ dl; d; du; b ]

let eig ?implementation ?(compute_left_eigenvectors = true)
    ?(compute_right_eigenvectors = true) ?(enable_eigvec_derivs = false) x =
  (match implementation with
  | None | Some Lax_config.Lapack -> ()
  | Some (Lax_config.Cusolver | Lax_config.Magma) ->
      invalid_arg "Only the lapack implementation is supported on CPU.");
  Core.bind
    (Eig
       {
         compute_left = compute_left_eigenvectors;
         compute_right = compute_right_eigenvectors;
         enable_eigvec_derivs;
       })
    [ x ]

let eigh ?implementation ?(lower = true) ?(symmetrize_input = true) x =
  (match implementation with
  | None | Some Lax_config.Eigh_qr -> ()
  | Some Lax_config.Eigh_jacobi ->
      failwith "Jacobi implementation is not supported on CPU"
  | Some Lax_config.Eigh_qdwh ->
      failwith "QDWH implementation is only supported on TPU");
  let x = if symmetrize_input then symmetrize x else x in
  match Core.bind (Eigh { lower }) [ x ] with
  | [ v; w ] -> (v, w)
  | _ -> failwith "linalg: eigh arity"

let hessenberg x =
  match Core.bind Hessenberg [ x ] with
  | [ a; taus ] -> (a, taus)
  | _ -> failwith "linalg: hessenberg arity"

let schur ?(compute_schur_vectors = true) x =
  Core.bind (Schur { compute_vectors = compute_schur_vectors }) [ x ]

let svd ?algorithm ?(full_matrices = true) ?(compute_uv = true) x =
  let gesvd =
    match algorithm with
    | None
    | Some Lax_config.Svd_default
    | Some Lax_config.Svd_divide_and_conquer ->
        false
    | Some Lax_config.Svd_qr -> true
    | Some Lax_config.Svd_jacobi | Some Lax_config.Svd_polar ->
        failwith
          "The SVD Jacobi and Polar algorithms are not implemented on CPU."
  in
  Core.bind (Svd { full_matrices; compute_uv; gesvd }) [ x ]

let tridiagonal ?(lower = true) x =
  match Core.bind (Tridiagonal { lower }) [ x ] with
  | [ a; d; e; taus ] -> (a, d, e, taus)
  | _ -> failwith "linalg: tridiagonal arity"
