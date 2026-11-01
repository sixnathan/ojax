type 'k real = (float, 'k, Bigarray.c_layout) Bigarray.Array1.t
type f64 = Bigarray.float64_elt real
type i32 = (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t

exception Unavailable of string

val single_routines : bool
val alloc : (float, 'k) Bigarray.kind -> int -> 'k real
val alloc_f64 : int -> f64
val alloc_i32 : int -> i32
val potrf : uplo:char -> n:int -> a:'k real -> lda:int -> int
val zpotrf : uplo:char -> n:int -> a:'k real -> lda:int -> int
val getrf : m:int -> n:int -> a:'k real -> lda:int -> ipiv:i32 -> int
val zgetrf : m:int -> n:int -> a:'k real -> lda:int -> ipiv:i32 -> int
val geqrf : m:int -> n:int -> a:'k real -> lda:int -> tau:'k real -> int
val zgeqrf : m:int -> n:int -> a:'k real -> lda:int -> tau:'k real -> int

val geqp3 :
  m:int -> n:int -> a:'k real -> lda:int -> jpvt:i32 -> tau:'k real -> int

val zgeqp3 :
  m:int -> n:int -> a:'k real -> lda:int -> jpvt:i32 -> tau:'k real -> int

val orgqr :
  m:int -> n:int -> k:int -> a:'k real -> lda:int -> tau:'k real -> int

val zungqr :
  m:int -> n:int -> k:int -> a:'k real -> lda:int -> tau:'k real -> int

val ormqr :
  side:char ->
  trans:char ->
  m:int ->
  n:int ->
  k:int ->
  a:'k real ->
  lda:int ->
  tau:'k real ->
  c:'k real ->
  ldc:int ->
  int

val zunmqr :
  side:char ->
  trans:char ->
  m:int ->
  n:int ->
  k:int ->
  a:'k real ->
  lda:int ->
  tau:'k real ->
  c:'k real ->
  ldc:int ->
  int

val gesdd :
  jobz:char ->
  m:int ->
  n:int ->
  a:'k real ->
  lda:int ->
  s:'k real ->
  u:'k real ->
  ldu:int ->
  vt:'k real ->
  ldvt:int ->
  int

val zgesdd :
  jobz:char ->
  m:int ->
  n:int ->
  a:'k real ->
  lda:int ->
  s:'k real ->
  u:'k real ->
  ldu:int ->
  vt:'k real ->
  ldvt:int ->
  int

val syevd :
  jobz:char -> uplo:char -> n:int -> a:'k real -> lda:int -> w:'k real -> int

val zheevd :
  jobz:char -> uplo:char -> n:int -> a:'k real -> lda:int -> w:'k real -> int

val gehrd :
  n:int -> ilo:int -> ihi:int -> a:'k real -> lda:int -> tau:'k real -> int

val zgehrd :
  n:int -> ilo:int -> ihi:int -> a:'k real -> lda:int -> tau:'k real -> int

val sytrd :
  uplo:char ->
  n:int ->
  a:'k real ->
  lda:int ->
  d:'k real ->
  e:'k real ->
  tau:'k real ->
  int

val zhetrd :
  uplo:char ->
  n:int ->
  a:'k real ->
  lda:int ->
  d:'k real ->
  e:'k real ->
  tau:'k real ->
  int

val gtsv :
  n:int ->
  nrhs:int ->
  dl:'k real ->
  d:'k real ->
  du:'k real ->
  b:'k real ->
  ldb:int ->
  int

val zgtsv :
  n:int ->
  nrhs:int ->
  dl:'k real ->
  d:'k real ->
  du:'k real ->
  b:'k real ->
  ldb:int ->
  int

val trtrs :
  uplo:char ->
  trans:char ->
  diag:char ->
  n:int ->
  nrhs:int ->
  a:'k real ->
  lda:int ->
  b:'k real ->
  ldb:int ->
  int

val ztrtrs :
  uplo:char ->
  trans:char ->
  diag:char ->
  n:int ->
  nrhs:int ->
  a:'k real ->
  lda:int ->
  b:'k real ->
  ldb:int ->
  int

val gesvd :
  jobz:char ->
  m:int ->
  n:int ->
  a:'k real ->
  lda:int ->
  s:'k real ->
  u:'k real ->
  ldu:int ->
  vt:'k real ->
  ldvt:int ->
  int

val zgesvd :
  jobz:char ->
  m:int ->
  n:int ->
  a:'k real ->
  lda:int ->
  s:'k real ->
  u:'k real ->
  ldu:int ->
  vt:'k real ->
  ldvt:int ->
  int

val trsm :
  side:char ->
  uplo:char ->
  trans:char ->
  diag:char ->
  m:int ->
  n:int ->
  a:'k real ->
  lda:int ->
  b:'k real ->
  ldb:int ->
  unit

val ztrsm :
  side:char ->
  uplo:char ->
  trans:char ->
  diag:char ->
  m:int ->
  n:int ->
  a:'k real ->
  lda:int ->
  b:'k real ->
  ldb:int ->
  unit

val geev :
  jobvl:char ->
  jobvr:char ->
  n:int ->
  a:'k real ->
  lda:int ->
  wri:'k real ->
  vl:'k real ->
  ldvl:int ->
  vr:'k real ->
  ldvr:int ->
  int

val zgeev :
  jobvl:char ->
  jobvr:char ->
  n:int ->
  a:'k real ->
  lda:int ->
  w:'k real ->
  vl:'k real ->
  ldvl:int ->
  vr:'k real ->
  ldvr:int ->
  int

val gees :
  jobvs:char ->
  n:int ->
  a:'k real ->
  lda:int ->
  wri:'k real ->
  vs:'k real ->
  ldvs:int ->
  sdim:i32 ->
  int

val zgees :
  jobvs:char ->
  n:int ->
  a:'k real ->
  lda:int ->
  w:'k real ->
  vs:'k real ->
  ldvs:int ->
  int

val gecon :
  norm:char ->
  n:int ->
  a:'k real ->
  lda:int ->
  anorm:float ->
  rcond:'k real ->
  int
