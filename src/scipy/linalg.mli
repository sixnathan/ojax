val cholesky : ?lower:bool -> Types.value -> Types.value
val cho_factor : ?lower:bool -> Types.value -> Types.value * bool
val cho_solve : Types.value * bool -> Types.value -> Types.value
val det : Types.value -> Types.value
val inv : Types.value -> Types.value
val lu : ?permute_l:bool -> Types.value -> Types.value list
val lu_factor : Types.value -> Types.value * Types.value

val lu_solve :
  ?trans:int -> Types.value * Types.value -> Types.value -> Types.value

val qr : ?mode:string -> ?pivoting:bool -> Types.value -> Types.value list

val solve :
  ?lower:bool -> ?assume_a:string -> Types.value -> Types.value -> Types.value

val solve_triangular :
  ?trans:int ->
  ?lower:bool ->
  ?unit_diagonal:bool ->
  Types.value ->
  Types.value ->
  Types.value

val svd :
  ?full_matrices:bool -> ?compute_uv:bool -> Types.value -> Types.value list

val eigh : ?lower:bool -> ?eigvals_only:bool -> Types.value -> Types.value list
val schur : ?output:string -> Types.value -> Types.value list
val block_diag : Types.value list -> Types.value
val toeplitz : ?r:Types.value -> Types.value -> Types.value
val hessenberg : ?calc_q:bool -> Types.value -> Types.value list
val expm : ?upper_triangular:bool -> Types.value -> Types.value

val expm_frechet :
  ?compute_expm:bool -> Types.value -> Types.value -> Types.value

val polar : ?side:string -> ?method_:string -> Types.value -> Types.value list
val sqrtm : ?blocksize:int -> Types.value -> Types.value

val funm :
  ?disp:bool -> Types.value -> (Types.value -> Types.value) -> Types.value

val eigh_tridiagonal :
  ?eigvals_only:bool ->
  ?select:string ->
  ?select_range:Types.value ->
  Types.value ->
  Types.value ->
  Types.value
