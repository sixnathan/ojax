type f64 = (float, Bigarray.float64_elt, Bigarray.c_layout) Bigarray.Array1.t
type i32 = (int32, Bigarray.int32_elt, Bigarray.c_layout) Bigarray.Array1.t

exception Unavailable of string

val alloc_f64 : int -> f64
val alloc_i32 : int -> i32
val potrf : uplo:char -> n:int -> a:f64 -> lda:int -> int
val getrf : m:int -> n:int -> a:f64 -> lda:int -> ipiv:i32 -> int
val geqrf : m:int -> n:int -> a:f64 -> lda:int -> tau:f64 -> int
val orgqr : m:int -> n:int -> k:int -> a:f64 -> lda:int -> tau:f64 -> int

val gesdd :
  jobz:char ->
  m:int ->
  n:int ->
  a:f64 ->
  lda:int ->
  s:f64 ->
  u:f64 ->
  ldu:int ->
  vt:f64 ->
  ldvt:int ->
  int

val syevd : jobz:char -> uplo:char -> n:int -> a:f64 -> lda:int -> w:f64 -> int

val geev :
  jobvl:char ->
  jobvr:char ->
  n:int ->
  a:f64 ->
  lda:int ->
  wri:f64 ->
  vl:f64 ->
  ldvl:int ->
  vr:f64 ->
  ldvr:int ->
  int

val gees :
  jobvs:char ->
  n:int ->
  a:f64 ->
  lda:int ->
  wri:f64 ->
  vs:f64 ->
  ldvs:int ->
  sdim:i32 ->
  int

val trtrs :
  uplo:char ->
  trans:char ->
  diag:char ->
  n:int ->
  nrhs:int ->
  a:f64 ->
  lda:int ->
  b:f64 ->
  ldb:int ->
  int

val gecon :
  norm:char -> n:int -> a:f64 -> lda:int -> anorm:float -> rcond:f64 -> int
