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
