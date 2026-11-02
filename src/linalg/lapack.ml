type 'k real = (float, 'k, Bigarray.c_layout) Bigarray.Array1.t
type f64 = Bigarray.float64_elt real
type i32 = (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t

exception Unavailable of string

external potrf_ : int array -> bool -> 'k real -> int = "ojax_lapack_potrf"

external getrf_ : int array -> bool -> 'k real -> i32 -> int
  = "ojax_lapack_getrf"

external geqrf_ : int array -> bool -> 'k real -> 'k real -> int
  = "ojax_lapack_geqrf"

external geqp3_ : int array -> bool -> 'k real -> i32 -> 'k real -> int
  = "ojax_lapack_geqp3"

external orgqr_ : int array -> bool -> 'k real -> 'k real -> int
  = "ojax_lapack_orgqr"

external ormqr_ : int array -> bool -> 'k real -> 'k real -> 'k real -> int
  = "ojax_lapack_ormqr"

external gesdd_ :
  int array -> bool -> 'k real -> 'k real -> 'k real -> 'k real -> int
  = "ojax_lapack_gesdd_bc" "ojax_lapack_gesdd"

external gesvd_ :
  int array -> bool -> 'k real -> 'k real -> 'k real -> 'k real -> int
  = "ojax_lapack_gesvd_bc" "ojax_lapack_gesvd"

external syevd_ : int array -> bool -> 'k real -> 'k real -> int
  = "ojax_lapack_syevd"

external geev_ :
  int array -> bool -> 'k real -> 'k real -> 'k real -> 'k real -> int
  = "ojax_lapack_geev_bc" "ojax_lapack_geev"

external gees_ :
  int array -> bool -> 'k real -> 'k real -> 'k real -> i32 -> int
  = "ojax_lapack_gees_bc" "ojax_lapack_gees"

external gehrd_ : int array -> bool -> 'k real -> 'k real -> int
  = "ojax_lapack_gehrd"

external sytrd_ :
  int array -> bool -> 'k real -> 'k real -> 'k real -> 'k real -> int
  = "ojax_lapack_sytrd_bc" "ojax_lapack_sytrd"

external trsm_ : int array -> bool -> 'k real -> 'k real -> unit
  = "ojax_lapack_trsm"

external gtsv_ :
  int array -> bool -> 'k real -> 'k real -> 'k real -> 'k real -> int
  = "ojax_lapack_gtsv_bc" "ojax_lapack_gtsv"

external trtrs_ : int array -> bool -> 'k real -> 'k real -> int
  = "ojax_lapack_trtrs"

external gecon_ : int array -> 'k real -> float -> 'k real -> int
  = "ojax_lapack_gecon"

external single_routines_ : unit -> bool = "ojax_lapack_single_routines"

let single_routines = single_routines_ ()
let alloc kind n = Bigarray.Array1.create kind Bigarray.c_layout n
let alloc_f64 n = alloc Bigarray.float64 n
let alloc_i32 n = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n

let guard name =
  if not Discover.available then
    raise
      (Unavailable
         (Printf.sprintf "linalg unavailable on this platform: %s (%s)" name
            Discover.backend))

let code = Char.code

let potrf ~uplo ~n ~a ~lda =
  guard "potrf";
  potrf_ [| code uplo; n; lda |] false a

let zpotrf ~uplo ~n ~a ~lda =
  guard "zpotrf";
  potrf_ [| code uplo; n; lda |] true a

let getrf ~m ~n ~a ~lda ~ipiv =
  guard "getrf";
  getrf_ [| m; n; lda |] false a ipiv

let zgetrf ~m ~n ~a ~lda ~ipiv =
  guard "zgetrf";
  getrf_ [| m; n; lda |] true a ipiv

let geqrf ~m ~n ~a ~lda ~tau =
  guard "geqrf";
  geqrf_ [| m; n; lda |] false a tau

let zgeqrf ~m ~n ~a ~lda ~tau =
  guard "zgeqrf";
  geqrf_ [| m; n; lda |] true a tau

let geqp3 ~m ~n ~a ~lda ~jpvt ~tau =
  guard "geqp3";
  geqp3_ [| m; n; lda |] false a jpvt tau

let zgeqp3 ~m ~n ~a ~lda ~jpvt ~tau =
  guard "zgeqp3";
  geqp3_ [| m; n; lda |] true a jpvt tau

let orgqr ~m ~n ~k ~a ~lda ~tau =
  guard "orgqr";
  orgqr_ [| m; n; k; lda |] false a tau

let zungqr ~m ~n ~k ~a ~lda ~tau =
  guard "zungqr";
  orgqr_ [| m; n; k; lda |] true a tau

let ormqr ~side ~trans ~m ~n ~k ~a ~lda ~tau ~c ~ldc =
  guard "ormqr";
  ormqr_ [| code side; code trans; m; n; k; lda; ldc |] false a tau c

let zunmqr ~side ~trans ~m ~n ~k ~a ~lda ~tau ~c ~ldc =
  guard "zunmqr";
  ormqr_ [| code side; code trans; m; n; k; lda; ldc |] true a tau c

let gesdd ~jobz ~m ~n ~a ~lda ~s ~u ~ldu ~vt ~ldvt =
  guard "gesdd";
  gesdd_ [| code jobz; m; n; lda; ldu; ldvt |] false a s u vt

let zgesdd ~jobz ~m ~n ~a ~lda ~s ~u ~ldu ~vt ~ldvt =
  guard "zgesdd";
  gesdd_ [| code jobz; m; n; lda; ldu; ldvt |] true a s u vt

let gesvd ~jobz ~m ~n ~a ~lda ~s ~u ~ldu ~vt ~ldvt =
  guard "gesvd";
  gesvd_ [| code jobz; m; n; lda; ldu; ldvt |] false a s u vt

let zgesvd ~jobz ~m ~n ~a ~lda ~s ~u ~ldu ~vt ~ldvt =
  guard "zgesvd";
  gesvd_ [| code jobz; m; n; lda; ldu; ldvt |] true a s u vt

let syevd ~jobz ~uplo ~n ~a ~lda ~w =
  guard "syevd";
  syevd_ [| code jobz; code uplo; n; lda |] false a w

let zheevd ~jobz ~uplo ~n ~a ~lda ~w =
  guard "zheevd";
  syevd_ [| code jobz; code uplo; n; lda |] true a w

let gehrd ~n ~ilo ~ihi ~a ~lda ~tau =
  guard "gehrd";
  gehrd_ [| n; ilo; ihi; lda |] false a tau

let zgehrd ~n ~ilo ~ihi ~a ~lda ~tau =
  guard "zgehrd";
  gehrd_ [| n; ilo; ihi; lda |] true a tau

let sytrd ~uplo ~n ~a ~lda ~d ~e ~tau =
  guard "sytrd";
  sytrd_ [| code uplo; n; lda |] false a d e tau

let zhetrd ~uplo ~n ~a ~lda ~d ~e ~tau =
  guard "zhetrd";
  sytrd_ [| code uplo; n; lda |] true a d e tau

let trsm ~side ~uplo ~trans ~diag ~m ~n ~a ~lda ~b ~ldb =
  guard "trsm";
  trsm_
    [| code side; code uplo; code trans; code diag; m; n; lda; ldb |]
    false a b

let ztrsm ~side ~uplo ~trans ~diag ~m ~n ~a ~lda ~b ~ldb =
  guard "ztrsm";
  trsm_
    [| code side; code uplo; code trans; code diag; m; n; lda; ldb |]
    true a b

let gtsv ~n ~nrhs ~dl ~d ~du ~b ~ldb =
  guard "gtsv";
  gtsv_ [| n; nrhs; ldb |] false dl d du b

let zgtsv ~n ~nrhs ~dl ~d ~du ~b ~ldb =
  guard "zgtsv";
  gtsv_ [| n; nrhs; ldb |] true dl d du b

external tridiagonal_perturbed_ :
  int array ->
  bool ->
  'k real ->
  'k real ->
  'k real ->
  'k real ->
  'k real ->
  unit = "ojax_tridiagonal_perturbed_host_bc" "ojax_tridiagonal_perturbed_host"

let tridiagonal_perturbed ~n ~nrhs ~complex ~dl ~d ~du ~b ~x =
  tridiagonal_perturbed_ [| n; nrhs |] complex dl d du b x

let trtrs ~uplo ~trans ~diag ~n ~nrhs ~a ~lda ~b ~ldb =
  guard "trtrs";
  trtrs_ [| code uplo; code trans; code diag; n; nrhs; lda; ldb |] false a b

let ztrtrs ~uplo ~trans ~diag ~n ~nrhs ~a ~lda ~b ~ldb =
  guard "ztrtrs";
  trtrs_ [| code uplo; code trans; code diag; n; nrhs; lda; ldb |] true a b

let geev ~jobvl ~jobvr ~n ~a ~lda ~wri ~vl ~ldvl ~vr ~ldvr =
  guard "geev";
  geev_ [| code jobvl; code jobvr; n; lda; ldvl; ldvr |] false a wri vl vr

let zgeev ~jobvl ~jobvr ~n ~a ~lda ~w ~vl ~ldvl ~vr ~ldvr =
  guard "zgeev";
  geev_ [| code jobvl; code jobvr; n; lda; ldvl; ldvr |] true a w vl vr

let gees ~jobvs ~n ~a ~lda ~wri ~vs ~ldvs ~sdim =
  guard "gees";
  gees_ [| code jobvs; n; lda; ldvs |] false a wri vs sdim

let zgees ~jobvs ~n ~a ~lda ~w ~vs ~ldvs =
  guard "zgees";
  gees_ [| code jobvs; n; lda; ldvs |] true a w vs (alloc_i32 1)

let gecon ~norm ~n ~a ~lda ~anorm ~rcond =
  guard "gecon";
  gecon_ [| code norm; n; lda |] a anorm rcond

external ffi_handler_ : string -> nativeint = "ojax_lapack_ffi_handler"

let ffi_targets =
  let target prec base = Printf.sprintf "lapack_%c%s_ffi" prec base in
  let both real_base complex_base handler =
    [
      (target 's' real_base, handler);
      (target 'd' real_base, handler);
      (target 'c' complex_base, handler);
      (target 'z' complex_base, handler);
    ]
  in
  List.concat_map
    (fun (real_base, complex_base, handler) ->
      both real_base complex_base (ffi_handler_ handler))
    [
      ("potrf", "potrf", "potrf");
      ("getrf", "getrf", "getrf");
      ("geqrf", "geqrf", "geqrf");
      ("geqp3", "geqp3", "geqp3");
      ("orgqr", "ungqr", "orgqr");
      ("ormqr", "unmqr", "ormqr");
      ("gesdd", "gesdd", "gesdd");
      ("gesvd", "gesvd", "gesvd");
      ("syevd", "heevd", "syevd");
      ("geev", "geev", "geev");
      ("gees", "gees", "gees");
      ("gehrd", "gehrd", "gehrd");
      ("sytrd", "hetrd", "sytrd");
      ("trsm", "trsm", "trsm");
      ("gtsv", "gtsv", "gtsv");
    ]
  @ [
      ("tridiagonal_solve_perturbed_ffi", ffi_handler_ "tridiagonal_perturbed");
      ("xla_ffi_python_cpu_callback", ffi_handler_ "host_callback");
      ("dce_sink", ffi_handler_ "dce_sink");
    ]
