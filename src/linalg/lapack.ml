type f64 = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
type i32 = (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t

exception Unavailable of string

external potrf_ : int array -> f64 -> int = "ojax_lapack_potrf"
external getrf_ : int array -> f64 -> i32 -> int = "ojax_lapack_getrf"
external geqrf_ : int array -> f64 -> f64 -> int = "ojax_lapack_geqrf"
external orgqr_ : int array -> f64 -> f64 -> int = "ojax_lapack_orgqr"

external gesdd_ : int array -> f64 -> f64 -> f64 -> f64 -> int
  = "ojax_lapack_gesdd"

external syevd_ : int array -> f64 -> f64 -> int = "ojax_lapack_syevd"

external geev_ : int array -> f64 -> f64 -> f64 -> f64 -> int
  = "ojax_lapack_geev"

external gees_ : int array -> f64 -> f64 -> f64 -> i32 -> int
  = "ojax_lapack_gees"

external trtrs_ : int array -> f64 -> f64 -> int = "ojax_lapack_trtrs"
external gecon_ : int array -> f64 -> float -> f64 -> int = "ojax_lapack_gecon"

let alloc_f64 n = Bigarray.Array1.create Bigarray.float64 Bigarray.c_layout n
let alloc_i32 n = Bigarray.Array1.create Bigarray.int32 Bigarray.c_layout n

let guard name =
  if not Discover.available then
    raise
      (Unavailable
         (Printf.sprintf "linalg unavailable on this platform: %s (%s)" name
            Discover.backend))

let potrf ~uplo ~n ~a ~lda =
  guard "potrf";
  potrf_ [| Char.code uplo; n; lda |] a

let getrf ~m ~n ~a ~lda ~ipiv =
  guard "getrf";
  getrf_ [| m; n; lda |] a ipiv

let geqrf ~m ~n ~a ~lda ~tau =
  guard "geqrf";
  geqrf_ [| m; n; lda |] a tau

let orgqr ~m ~n ~k ~a ~lda ~tau =
  guard "orgqr";
  orgqr_ [| m; n; k; lda |] a tau

let gesdd ~jobz ~m ~n ~a ~lda ~s ~u ~ldu ~vt ~ldvt =
  guard "gesdd";
  gesdd_ [| Char.code jobz; m; n; lda; ldu; ldvt |] a s u vt

let syevd ~jobz ~uplo ~n ~a ~lda ~w =
  guard "syevd";
  syevd_ [| Char.code jobz; Char.code uplo; n; lda |] a w

let geev ~jobvl ~jobvr ~n ~a ~lda ~wri ~vl ~ldvl ~vr ~ldvr =
  guard "geev";
  geev_ [| Char.code jobvl; Char.code jobvr; n; lda; ldvl; ldvr |] a wri vl vr

let gees ~jobvs ~n ~a ~lda ~wri ~vs ~ldvs ~sdim =
  guard "gees";
  gees_ [| Char.code jobvs; n; lda; ldvs |] a wri vs sdim

let trtrs ~uplo ~trans ~diag ~n ~nrhs ~a ~lda ~b ~ldb =
  guard "trtrs";
  trtrs_
    [| Char.code uplo; Char.code trans; Char.code diag; n; nrhs; lda; ldb |]
    a b

let gecon ~norm ~n ~a ~lda ~anorm ~rcond =
  guard "gecon";
  gecon_ [| Char.code norm; n; lda |] a anorm rcond
