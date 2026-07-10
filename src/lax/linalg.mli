open Types

val cholesky_impl : Ndarray.t list -> Ndarray.t list
val lu_impl : Ndarray.t list -> Ndarray.t list
val qr_impl : bool -> Ndarray.t list -> Ndarray.t list
val householder_product_impl : Ndarray.t list -> Ndarray.t list
val lu_pivots_to_permutation_impl : int -> Ndarray.t list -> Ndarray.t list

val triangular_solve_impl :
  bool -> bool -> bool -> bool -> bool -> Ndarray.t list -> Ndarray.t list

val tridiagonal_solve_impl : Ndarray.t list -> Ndarray.t list
val cholesky_aval : aval list -> aval list
val lu_aval : aval list -> aval list
val qr_aval : bool -> aval list -> aval list
val householder_product_aval : aval list -> aval list
val lu_pivots_to_permutation_aval : int -> aval list -> aval list
val triangular_solve_aval : aval list -> aval list
val tridiagonal_solve_aval : aval list -> aval list
val cholesky : value -> value
val lu : value -> value * value * value
val qr : ?full_matrices:bool -> value -> value * value
val householder_product : value -> value -> value
val lu_pivots_to_permutation : permutation_size:int -> value -> value

val triangular_solve :
  ?left_side:bool ->
  ?lower:bool ->
  ?transpose_a:bool ->
  ?conjugate_a:bool ->
  ?unit_diagonal:bool ->
  value ->
  value ->
  value

val tridiagonal_solve : value -> value -> value -> value -> value
