open Types
module Lp = Lapack_seam.Lapack

type 'r by_kind = { run : 'k. (float, 'k) Bigarray.kind -> 'r }

let is_complex_dtype = function
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let is_complex_nd nd = is_complex_dtype (Ndarray.dtype nd)

let by_dtype dt f =
  match dt with
  | (Dtype.F32 | Dtype.Complex64) when Lp.single_routines ->
      f.run Bigarray.float32
  | _ -> f.run Bigarray.float64

let to_complex_dtype = function
  | Dtype.F32 -> Dtype.Complex64
  | Dtype.F64 -> Dtype.Complex128
  | d -> d

let real_basetype = function
  | Dtype.Complex64 -> Dtype.F32
  | Dtype.Complex128 -> Dtype.F64
  | d -> d

let dims2 nd =
  let s = Ndarray.shape nd in
  (s.(0), s.(1))

let elt c (b : 'k Lp.real) k =
  if c then { Complex.re = b.{2 * k}; im = b.{(2 * k) + 1} }
  else { Complex.re = b.{k}; im = 0.0 }

let put c (b : 'k Lp.real) k (z : Complex.t) =
  if c then begin
    b.{2 * k} <- z.re;
    b.{(2 * k) + 1} <- z.im
  end
  else b.{k} <- z.re

let cell nd idx =
  if is_complex_nd nd then Ndarray.get_c nd idx
  else { Complex.re = Ndarray.get_f nd idx; im = 0.0 }

let width c = if c then 2 else 1
let alloc kind c n = Lp.alloc kind (max 1 (width c * n))

let pack kind c nd m n =
  let a = alloc kind c (m * n) in
  for j = 0 to n - 1 do
    for i = 0 to m - 1 do
      put c a (i + (j * m)) (cell nd [| i; j |])
    done
  done;
  a

let make dt shape f =
  let total = Array.fold_left ( * ) 1 shape in
  if is_complex_dtype dt then Ndarray.of_complex dt shape (Array.init total f)
  else Ndarray.of_floats dt shape (Array.init total (fun k -> (f k).Complex.re))

let of_cm dt c b p q =
  make dt [| p; q |] (fun f -> elt c b ((f / max q 1) + (f mod max q 1 * p)))

let of_vec dt c b n = make dt [| n |] (fun i -> elt c b i)
let nan_elt = { Complex.re = Float.nan; im = Float.nan }
let nan_like dt shape = make dt shape (fun _ -> nan_elt)
let guarded ok dt shape value = if ok then value () else nan_like dt shape

let cholesky_impl inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let c = is_complex_dtype dt in
      let n, _ = dims2 a in
      by_dtype dt
        {
          run =
            (fun kind ->
              let ca = pack kind c a n n in
              let info =
                if c then Lp.zpotrf ~uplo:'L' ~n ~a:ca ~lda:n
                else Lp.potrf ~uplo:'L' ~n ~a:ca ~lda:n
              in
              [
                make dt [| n; n |] (fun f ->
                    let i = f / n and j = f mod n in
                    if j > i then Complex.zero
                    else if info <> 0 then nan_elt
                    else elt c ca (i + (j * n)));
              ]);
        }
  | _ -> failwith "linalg: cholesky expects one operand"

let permutation_of_pivots m pivots =
  let perm = Array.init m (fun i -> i) in
  Array.iteri
    (fun i j ->
      let t = perm.(i) in
      perm.(i) <- perm.(j);
      perm.(j) <- t)
    pivots;
  perm

let lu_impl inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let c = is_complex_dtype dt in
      let m, n = dims2 a in
      let k = min m n in
      by_dtype dt
        {
          run =
            (fun kind ->
              let ca = pack kind c a m n in
              let ipiv = Lp.alloc_i32 (max k 1) in
              let info =
                if c then Lp.zgetrf ~m ~n ~a:ca ~lda:m ~ipiv
                else Lp.getrf ~m ~n ~a:ca ~lda:m ~ipiv
              in
              let pivots = Array.init k (fun i -> Int32.to_int ipiv.{i} - 1) in
              [
                guarded (info >= 0) dt [| m; n |] (fun () -> of_cm dt c ca m n);
                Ndarray.of_floats Dtype.I32 [| k |]
                  (Array.map float_of_int pivots);
                Ndarray.of_floats Dtype.I32 [| m |]
                  (Array.map float_of_int (permutation_of_pivots m pivots));
              ]);
        }
  | _ -> failwith "linalg: lu expects one operand"

let qr_impl full_matrices inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let c = is_complex_dtype dt in
      let m, n = dims2 a in
      let mn = min m n in
      let k = if full_matrices then m else mn in
      by_dtype dt
        {
          run =
            (fun kind ->
              let ca = pack kind c a m n in
              let tau = alloc kind c mn in
              let _ =
                if c then Lp.zgeqrf ~m ~n ~a:ca ~lda:m ~tau
                else Lp.geqrf ~m ~n ~a:ca ~lda:m ~tau
              in
              let r =
                make dt [| k; n |] (fun f ->
                    let i = f / max n 1 and j = f mod max n 1 in
                    if j >= i then elt c ca (i + (j * m)) else Complex.zero)
              in
              let work = alloc kind c (m * max k n) in
              for j = 0 to min n k - 1 do
                for i = 0 to m - 1 do
                  put c work (i + (j * m)) (elt c ca (i + (j * m)))
                done
              done;
              let _ =
                if c then Lp.zungqr ~m ~n:k ~k:mn ~a:work ~lda:m ~tau
                else Lp.orgqr ~m ~n:k ~k:mn ~a:work ~lda:m ~tau
              in
              [ of_cm dt c work m k; r ]);
        }
  | _ -> failwith "linalg: qr expects one operand"

let geqrf_impl a =
  let dt = Ndarray.dtype a in
  let c = is_complex_dtype dt in
  let m, n = dims2 a in
  let k = min m n in
  by_dtype dt
    {
      run =
        (fun kind ->
          let ca = pack kind c a m n in
          let tau = alloc kind c k in
          let _ =
            if c then Lp.zgeqrf ~m ~n ~a:ca ~lda:m ~tau
            else Lp.geqrf ~m ~n ~a:ca ~lda:m ~tau
          in
          [ of_cm dt c ca m n; of_vec dt c tau k ]);
    }

let geqp3_impl a jpvt_in =
  let dt = Ndarray.dtype a in
  let c = is_complex_dtype dt in
  let m, n = dims2 a in
  let k = min m n in
  by_dtype dt
    {
      run =
        (fun kind ->
          let ca = pack kind c a m n in
          let tau = alloc kind c k in
          let jpvt = Lp.alloc_i32 (max n 1) in
          for i = 0 to n - 1 do
            jpvt.{i} <- Int64.to_int32 (Ndarray.get_i64 jpvt_in [| i |])
          done;
          let _ =
            if c then Lp.zgeqp3 ~m ~n ~a:ca ~lda:m ~jpvt ~tau
            else Lp.geqp3 ~m ~n ~a:ca ~lda:m ~jpvt ~tau
          in
          [
            of_cm dt c ca m n;
            Ndarray.of_floats Dtype.I32 [| n |]
              (Array.init n (fun i -> Int32.to_float jpvt.{i}));
            of_vec dt c tau k;
          ]);
    }

let householder_product_impl inputs =
  match inputs with
  | [ a; taus ] ->
      let dt = Ndarray.dtype a in
      let c = is_complex_dtype dt in
      let m, n = dims2 a in
      let k = (Ndarray.shape taus).(0) in
      by_dtype dt
        {
          run =
            (fun kind ->
              let ca = pack kind c a m n in
              let tau = alloc kind c k in
              for i = 0 to k - 1 do
                put c tau i (cell taus [| i |])
              done;
              let _ =
                if c then Lp.zungqr ~m ~n ~k ~a:ca ~lda:m ~tau
                else Lp.orgqr ~m ~n ~k ~a:ca ~lda:m ~tau
              in
              [ of_cm dt c ca m n ]);
        }
  | _ -> failwith "linalg: householder_product expects two operands"

let qr_pivoted_impl full_matrices inputs =
  match inputs with
  | [ a ] -> (
      let dt = Ndarray.dtype a in
      let c = is_complex_dtype dt in
      let m, n = dims2 a in
      let mn = min m n in
      let k = if full_matrices then m else mn in
      let zeros = Ndarray.of_floats Dtype.I32 [| n |] (Array.make n 0.0) in
      match geqp3_impl a zeros with
      | [ h; jpvt; taus ] ->
          let r =
            make dt [| k; n |] (fun f ->
                let i = f / max n 1 and j = f mod max n 1 in
                if j >= i && i < m then cell h [| i; j |] else Complex.zero)
          in
          let base =
            make dt [| m; k |] (fun f ->
                let i = f / max k 1 and j = f mod max k 1 in
                if j < n then cell h [| i; j |] else Complex.zero)
          in
          ignore c;
          let q = List.hd (householder_product_impl [ base; taus ]) in
          let p =
            Ndarray.of_floats Dtype.I32 [| n |]
              (Array.init n (fun i ->
                   Int64.to_float (Ndarray.get_i64 jpvt [| i |]) -. 1.0))
          in
          [ q; r; p ]
      | _ -> failwith "linalg: geqp3 arity")
  | _ -> failwith "linalg: qr expects one operand"

let ormqr_impl ~left ~transpose a taus c_in =
  let dt = Ndarray.dtype a in
  let c = is_complex_dtype dt in
  let am, an = dims2 a in
  let k = (Ndarray.shape taus).(0) in
  let m, n = dims2 c_in in
  by_dtype dt
    {
      run =
        (fun kind ->
          let ca = pack kind c a am an in
          let tau = alloc kind c k in
          for i = 0 to k - 1 do
            put c tau i (cell taus [| i |])
          done;
          let cc = pack kind c c_in m n in
          let side = if left then 'L' else 'R' in
          let trans = if not transpose then 'N' else if c then 'C' else 'T' in
          let _ =
            if c then
              Lp.zunmqr ~side ~trans ~m ~n ~k ~a:ca ~lda:am ~tau ~c:cc ~ldc:m
            else Lp.ormqr ~side ~trans ~m ~n ~k ~a:ca ~lda:am ~tau ~c:cc ~ldc:m
          in
          [ of_cm dt c cc m n ]);
    }

let lu_pivots_to_permutation_impl permutation_size inputs =
  match inputs with
  | [ pivots ] ->
      let k = (Ndarray.shape pivots).(0) in
      let m = permutation_size in
      let perm =
        permutation_of_pivots m
          (Array.init k (fun i -> Int64.to_int (Ndarray.get_i64 pivots [| i |])))
      in
      [ Ndarray.of_floats Dtype.I32 [| m |] (Array.map float_of_int perm) ]
  | _ -> failwith "linalg: lu_pivots_to_permutation expects one operand"

let triangular_solve_impl left_side lower transpose_a conjugate_a unit_diagonal
    inputs =
  match inputs with
  | [ a; b ] ->
      let dt = Ndarray.dtype b in
      let c = is_complex_dtype dt in
      let order, _ = dims2 a in
      let rows, cols = dims2 b in
      let uplo = if lower then 'L' else 'U' in
      let diag = if unit_diagonal then 'U' else 'N' in
      let side = if left_side then 'L' else 'R' in
      let trans =
        if not transpose_a then 'N' else if conjugate_a && c then 'C' else 'T'
      in
      let conj_first = conjugate_a && c && not transpose_a in
      by_dtype dt
        {
          run =
            (fun kind ->
              let ca = alloc kind c (order * order) in
              for j = 0 to order - 1 do
                for i = 0 to order - 1 do
                  let z = cell a [| i; j |] in
                  put c ca
                    (i + (j * order))
                    (if conj_first then Complex.conj z else z)
                done
              done;
              let cb = pack kind c b rows cols in
              if c then
                Lp.ztrsm ~side ~uplo ~trans ~diag ~m:rows ~n:cols ~a:ca
                  ~lda:order ~b:cb ~ldb:(max rows 1)
              else
                Lp.trsm ~side ~uplo ~trans ~diag ~m:rows ~n:cols ~a:ca
                  ~lda:order ~b:cb ~ldb:(max rows 1);
              [ of_cm dt c cb rows cols ]);
        }
  | _ -> failwith "linalg: triangular_solve expects two operands"

let tridiagonal_cell nd i j =
  if Array.length (Ndarray.shape nd) >= 2 then Ndarray.get_c nd [| i; j |]
  else Ndarray.get_c nd [| i |]

let tridiagonal_solve_perturbed inputs =
  match inputs with
  | [ dl; d; du; b ] ->
      let dt = Ndarray.dtype b in
      let m = (Ndarray.shape d).(0) in
      let bshape = Ndarray.shape b in
      let cols = if Array.length bshape >= 2 then bshape.(1) else 1 in
      let c = is_complex_dtype dt in
      let solve kind =
        let band nd =
          let out = alloc kind c m in
          for i = 0 to m - 1 do
            put c out i (cell nd [| i |])
          done;
          out
        in
        let rhs = alloc kind c (m * cols) in
        for j = 0 to cols - 1 do
          for i = 0 to m - 1 do
            put c rhs (i + (j * m)) (tridiagonal_cell b i j)
          done
        done;
        let x = alloc kind c (m * cols) in
        Lp.tridiagonal_perturbed ~n:m ~nrhs:cols ~complex:c ~dl:(band dl)
          ~d:(band d) ~du:(band du) ~b:rhs ~x;
        [ make dt bshape (fun f -> elt c x ((f / cols) + (f mod cols * m))) ]
      in
      if dt = Dtype.F32 || dt = Dtype.Complex64 then solve Bigarray.float32
      else solve Bigarray.float64
  | _ -> failwith "linalg: tridiagonal_solve expects four operands"

let tridiagonal_solve_gtsv dl d du b =
  let dt = Ndarray.dtype b in
  let c = is_complex_dtype dt in
  let m = (Ndarray.shape d).(0) in
  let bshape = Ndarray.shape b in
  let cols = if Array.length bshape >= 2 then bshape.(1) else 1 in
  by_dtype dt
    {
      run =
        (fun kind ->
          let band nd off len =
            let out = alloc kind c len in
            for i = 0 to len - 1 do
              put c out i (cell nd [| i + off |])
            done;
            out
          in
          let cl = band dl 1 (m - 1) in
          let cd = band d 0 m in
          let cu = band du 0 (m - 1) in
          let cb = alloc kind c (m * cols) in
          for j = 0 to cols - 1 do
            for i = 0 to m - 1 do
              put c cb (i + (j * m)) (tridiagonal_cell b i j)
            done
          done;
          let info =
            if c then
              Lp.zgtsv ~n:m ~nrhs:cols ~dl:cl ~d:cd ~du:cu ~b:cb ~ldb:(max m 1)
            else
              Lp.gtsv ~n:m ~nrhs:cols ~dl:cl ~d:cd ~du:cu ~b:cb ~ldb:(max m 1)
          in
          [
            guarded (info = 0) dt bshape (fun () ->
                make dt bshape (fun f ->
                    elt c cb ((f / cols) + (f mod cols * m))));
          ]);
    }

let tridiagonal_solve_impl ~perturb_singular inputs =
  match inputs with
  | [ dl; d; du; b ] when not perturb_singular ->
      tridiagonal_solve_gtsv dl d du b
  | _ -> tridiagonal_solve_perturbed inputs

let eigh_impl lower inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let c = is_complex_dtype dt in
      let rdt = real_basetype dt in
      let n, _ = dims2 a in
      let uplo = if lower then 'L' else 'U' in
      by_dtype dt
        {
          run =
            (fun kind ->
              let ca = pack kind c a n n in
              let w = Lp.alloc kind (max n 1) in
              let info =
                if c then Lp.zheevd ~jobz:'V' ~uplo ~n ~a:ca ~lda:n ~w
                else Lp.syevd ~jobz:'V' ~uplo ~n ~a:ca ~lda:n ~w
              in
              [
                guarded (info = 0) dt [| n; n |] (fun () -> of_cm dt c ca n n);
                guarded (info = 0) rdt [| n |] (fun () -> of_vec rdt false w n);
              ]);
        }
  | _ -> failwith "linalg: eigh expects one operand"

let unpack_eigvecs n (pv : 'k Lp.real) wi =
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
      let c = is_complex_dtype dt in
      let cdt = to_complex_dtype dt in
      let n, _ = dims2 a in
      let jobvl = if compute_left then 'V' else 'N' in
      let jobvr = if compute_right then 'V' else 'N' in
      let ldvl = if compute_left then max n 1 else 1 in
      let ldvr = if compute_right then max n 1 else 1 in
      by_dtype dt
        {
          run =
            (fun kind ->
              let ca = pack kind c a n n in
              let w = Lp.alloc kind (max 1 (2 * n)) in
              let vl = alloc kind c (n * ldvl) in
              let vr = alloc kind c (n * ldvr) in
              let info =
                if c then
                  Lp.zgeev ~jobvl ~jobvr ~n ~a:ca ~lda:(max n 1) ~w ~vl ~ldvl
                    ~vr ~ldvr
                else
                  Lp.geev ~jobvl ~jobvr ~n ~a:ca ~lda:(max n 1) ~wri:w ~vl ~ldvl
                    ~vr ~ldvr
              in
              let ok = info = 0 in
              let values =
                guarded ok cdt [| n |] (fun () ->
                    if c then of_vec cdt true w n
                    else
                      make cdt [| n |] (fun i ->
                          { Complex.re = w.{i}; im = w.{i + n} }))
              in
              let vectors flag buf =
                if not flag then []
                else
                  [
                    guarded ok cdt [| n; n |] (fun () ->
                        if c then of_cm cdt true buf n n
                        else
                          let wi = Array.init n (fun i -> w.{i + n}) in
                          Ndarray.of_complex cdt [| n; n |]
                            (unpack_eigvecs n buf wi));
                  ]
              in
              (values :: vectors compute_left vl) @ vectors compute_right vr);
        }
  | _ -> failwith "linalg: eig expects one operand"

let hessenberg_impl inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let c = is_complex_dtype dt in
      let n, _ = dims2 a in
      by_dtype dt
        {
          run =
            (fun kind ->
              let ca = pack kind c a n n in
              let tau = alloc kind c (n - 1) in
              let info =
                if c then Lp.zgehrd ~n ~ilo:1 ~ihi:n ~a:ca ~lda:(max n 1) ~tau
                else Lp.gehrd ~n ~ilo:1 ~ihi:n ~a:ca ~lda:(max n 1) ~tau
              in
              [
                guarded (info = 0) dt [| n; n |] (fun () -> of_cm dt c ca n n);
                guarded (info = 0) dt
                  [| n - 1 |]
                  (fun () -> of_vec dt c tau (n - 1));
              ]);
        }
  | _ -> failwith "linalg: hessenberg expects one operand"

let schur_impl compute_vectors inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let c = is_complex_dtype dt in
      let n, _ = dims2 a in
      let jobvs = if compute_vectors then 'V' else 'N' in
      let ldvs = if compute_vectors then max n 1 else 1 in
      by_dtype dt
        {
          run =
            (fun kind ->
              let ca = pack kind c a n n in
              let w = Lp.alloc kind (max 1 (2 * n)) in
              let vs = alloc kind c (n * ldvs) in
              let info =
                if c then Lp.zgees ~jobvs ~n ~a:ca ~lda:(max n 1) ~w ~vs ~ldvs
                else
                  Lp.gees ~jobvs ~n ~a:ca ~lda:(max n 1) ~wri:w ~vs ~ldvs
                    ~sdim:(Lp.alloc_i32 1)
              in
              let ok = info = 0 in
              let t = guarded ok dt [| n; n |] (fun () -> of_cm dt c ca n n) in
              if compute_vectors then
                [ t; guarded ok dt [| n; n |] (fun () -> of_cm dt c vs n n) ]
              else [ t ]);
        }
  | _ -> failwith "linalg: schur expects one operand"

let eye_like dt p q =
  make dt [| p; q |] (fun f ->
      if f / max q 1 = f mod max q 1 then Complex.one else Complex.zero)

let svd_impl ?(gesvd = false) full_matrices compute_uv inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let c = is_complex_dtype dt in
      let rdt = real_basetype dt in
      let m, n = dims2 a in
      let mn = min m n in
      let ucols = if full_matrices then m else mn in
      let vrows = if full_matrices then n else mn in
      if mn = 0 then
        let s = Ndarray.of_floats rdt [| 0 |] [||] in
        if compute_uv then [ eye_like dt m ucols; s; eye_like dt vrows n ]
        else [ s ]
      else
        by_dtype dt
          {
            run =
              (fun kind ->
                let ca = pack kind c a m n in
                let s = Lp.alloc kind mn in
                let jobz =
                  if not compute_uv then 'N'
                  else if full_matrices then 'A'
                  else 'S'
                in
                let lu = if compute_uv then m else 1 in
                let lv = if compute_uv then vrows else 1 in
                let u = alloc kind c (lu * if compute_uv then ucols else 1) in
                let vt = alloc kind c (lv * if compute_uv then n else 1) in
                let info =
                  match (c, gesvd) with
                  | true, true ->
                      Lp.zgesvd ~jobz ~m ~n ~a:ca ~lda:m ~s ~u ~ldu:lu ~vt
                        ~ldvt:lv
                  | true, false ->
                      Lp.zgesdd ~jobz ~m ~n ~a:ca ~lda:m ~s ~u ~ldu:lu ~vt
                        ~ldvt:lv
                  | false, true ->
                      Lp.gesvd ~jobz ~m ~n ~a:ca ~lda:m ~s ~u ~ldu:lu ~vt
                        ~ldvt:lv
                  | false, false ->
                      Lp.gesdd ~jobz ~m ~n ~a:ca ~lda:m ~s ~u ~ldu:lu ~vt
                        ~ldvt:lv
                in
                let ok = info = 0 in
                let values =
                  guarded ok rdt [| mn |] (fun () -> of_vec rdt false s mn)
                in
                if compute_uv then
                  [
                    guarded ok dt [| m; ucols |] (fun () ->
                        of_cm dt c u m ucols);
                    values;
                    guarded ok dt [| vrows; n |] (fun () ->
                        of_cm dt c vt vrows n);
                  ]
                else [ values ]);
          }
  | _ -> failwith "linalg: svd expects one operand"

let tridiagonal_impl lower inputs =
  match inputs with
  | [ a ] ->
      let dt = Ndarray.dtype a in
      let c = is_complex_dtype dt in
      let rdt = real_basetype dt in
      let n, _ = dims2 a in
      let uplo = if lower then 'L' else 'U' in
      by_dtype dt
        {
          run =
            (fun kind ->
              let ca = pack kind c a n n in
              let d = Lp.alloc kind (max n 1) in
              let e = Lp.alloc kind (max (n - 1) 1) in
              let tau = alloc kind c (n - 1) in
              let info =
                if c then Lp.zhetrd ~uplo ~n ~a:ca ~lda:(max n 1) ~d ~e ~tau
                else Lp.sytrd ~uplo ~n ~a:ca ~lda:(max n 1) ~d ~e ~tau
              in
              let ok = info = 0 in
              [
                guarded ok dt [| n; n |] (fun () -> of_cm dt c ca n n);
                guarded ok rdt [| n |] (fun () -> of_vec rdt false d n);
                guarded ok rdt
                  [| n - 1 |]
                  (fun () -> of_vec rdt false e (n - 1));
                guarded ok dt [| n - 1 |] (fun () -> of_vec dt c tau (n - 1));
              ]);
        }
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

let qr_pivoted_aval full_matrices avals =
  match avals with
  | [ a ] ->
      let m = a.shape.(0) and n = a.shape.(1) in
      let k = if full_matrices then m else min m n in
      [
        { a with shape = [| m; k |] };
        { a with shape = [| k; n |] };
        { a with shape = [| n |]; dtype = Dtype.I32 };
      ]
  | _ -> failwith "linalg: qr expects one aval"

let geqrf_aval avals =
  match avals with
  | [ a ] -> [ a; { a with shape = [| min a.shape.(0) a.shape.(1) |] } ]
  | _ -> failwith "linalg: geqrf expects one aval"

let geqp3_aval avals =
  match avals with
  | [ a; jpvt ] ->
      [ a; jpvt; { a with shape = [| min a.shape.(0) a.shape.(1) |] } ]
  | _ -> failwith "linalg: geqp3 expects two avals"

let ormqr_aval ~left avals =
  match avals with
  | [ a; _; c ] ->
      let m = a.shape.(0) in
      let shape_text (s : int array) =
        match s with
        | [| x |] -> Printf.sprintf "(%d,)" x
        | _ ->
            "("
            ^ String.concat ", " (Array.to_list (Array.map string_of_int s))
            ^ ")"
      in
      if left && c.shape.(0) <> m then
        invalid_arg
          (Printf.sprintf
             "ormqr with left=True expects c to have the same number of rows \
              as the Householder matrix a. Got a shape %s and c shape %s."
             (shape_text a.shape) (shape_text c.shape));
      if (not left) && c.shape.(1) <> m then
        invalid_arg
          (Printf.sprintf
             "ormqr with left=False expects c to have the same number of \
              columns as the Householder matrix a has rows. Got a shape %s and \
              c shape %s."
             (shape_text a.shape) (shape_text c.shape));
      [ c ]
  | _ -> failwith "linalg: ormqr expects three avals"

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
  let half = Core.splat aval.dtype [||] 2.0 in
  let sum = Core.bind1 Add [ x; flipped ] in
  Core.bind1 Div [ sum; half ]

let tril ?(k = 0) m =
  let aval = Core.get_aval m in
  let rank = Array.length aval.shape in
  let n_rows = aval.shape.(rank - 2) and n_cols = aval.shape.(rank - 1) in
  let i =
    Core.bind1
      (Iota { dtype = Dtype.I32; shape = [| n_rows; n_cols |]; dimension = 0 })
      []
  in
  let plus = Core.bind1 Add [ i; Core.splat Dtype.I32 [||] (float_of_int k) ] in
  let j =
    Core.bind1
      (Iota { dtype = Dtype.I32; shape = [| n_rows; n_cols |]; dimension = 1 })
      []
  in
  let mask = Core.bind1 Ge [ plus; j ] in
  let b = rank - 2 in
  let mask =
    if b = 0 then mask
    else
      Core.bind1
        (Broadcast_in_dim { shape = aval.shape; dims = [| b; b + 1 |] })
        [ mask ]
  in
  let zeros = Core.splat aval.dtype aval.shape 0.0 in
  Core.bind1 Select_n [ mask; zeros; m ]

let cholesky ?(symmetrize_input = true) x =
  let x = if symmetrize_input then symmetrize x else x in
  let c = Core.bind1 Cholesky [ x ] in
  tril c

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

let lu_solve_core lu permutation b =
  let m = (Core.get_aval lu).shape.(0) in
  let bs = (Core.get_aval b).shape in
  let rest = Array.fold_left ( * ) 1 (Array.sub bs 1 (Array.length bs - 1)) in
  let x = Core.bind1 (Reshape [| m; rest |]) [ b ] in
  let lt = Core.bind1 Lt [ permutation; Core.splat Dtype.I32 [||] 0.0 ] in
  let plus =
    Core.bind1 Add [ permutation; Core.splat Dtype.I32 [||] (float_of_int m) ]
  in
  let idx0 = Core.bind1 Select_n [ lt; permutation; plus ] in
  let idx =
    Core.bind1
      (Broadcast_in_dim { shape = [| m; 1 |]; dims = [| 0 |] })
      [ idx0 ]
  in
  let x =
    Core.bind1
      (Gather
         {
           dimension_numbers =
             {
               offset_dims = [| 1 |];
               collapsed_slice_dims = [| 0 |];
               start_index_map = [| 0 |];
               g_operand_batching_dims = [||];
               g_start_indices_batching_dims = [||];
             };
           slice_sizes = [| 1; rest |];
         })
      [ x; idx ]
  in
  let x =
    triangular_solve ~left_side:true ~lower:true ~unit_diagonal:true lu x
  in
  let x = triangular_solve ~left_side:true ~lower:false lu x in
  Core.bind1 (Reshape bs) [ x ]

let solve_for_jvp a b =
  let r = Array.length (Core.get_aval a).shape in
  let lu, _, permutation = lu (Custom_derivatives.stop_gradient a) in
  let solve_vec v =
    match
      Jaxpr.named_jit ~name:"_lu_solve"
        (function
          | [ l; p; y ] -> [ lu_solve_core l p y ]
          | _ -> invalid_arg "lu_solve: arity")
        [ lu; permutation; v ]
    with
    | [ out ] -> out
    | _ -> invalid_arg "lu_solve: arity"
  in
  match
    Interpreters.Batching.vmap_out
      (function [ y ] -> [ solve_vec y ] | _ -> invalid_arg "solve: arity")
      [ Some (Array.length (Core.get_aval b).shape - 1) ]
      [ b ] (r - 1)
  with
  | [ out ] -> out
  | _ -> invalid_arg "solve: arity"

let () = Interpreters.Ad.solve_ref := solve_for_jvp
