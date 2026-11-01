open Types

val igamma : value list -> value list
val igammac : value list -> value list
val igamma_grad_a : value list -> value list
val betainc : value list -> value list
val bessel_i0e : value list -> value list
val complex_minmax : is_max:bool -> value list -> value list
val complex_reduce : is_max:bool -> axes:int array -> value list -> value list

val complex_sort :
  dimension:int -> is_stable:bool -> num_keys:int -> value list -> value list

val lu : value list -> value list
val lu_pivots_to_permutation : permutation_size:int -> value list -> value list
val tridiagonal_solve : value list -> value list
val qr_from_factors : full_matrices:bool -> value list -> value list
val svd : full_matrices:bool -> compute_uv:bool -> value list -> value list
val eigh_refined : lower:bool -> value list -> value list
val lapack_cholesky : value list -> value list
val lapack_lu : permutation_size:int -> value list -> value list
val lapack_guarded : value list -> value list
val lapack_svd : compute_uv:bool -> value list -> value list

val lapack_eig :
  real:bool ->
  compute_left:bool ->
  compute_right:bool ->
  value list ->
  value list

val lapack_schur : compute_vectors:bool -> value list -> value list
