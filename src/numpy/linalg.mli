type ord = Onone | Ofro | Onuc | Onum of float
type norm_axis = Anone | Aint of int | Apair of int * int

val cholesky :
  ?upper:bool -> ?symmetrize_input:bool -> Types.value -> Types.value

val svd :
  ?full_matrices:bool ->
  ?compute_uv:bool ->
  ?hermitian:bool ->
  Types.value ->
  Types.value list

val svdvals : Types.value -> Types.value
val solve : Types.value -> Types.value -> Types.value
val inv : Types.value -> Types.value
val slogdet : ?method_:string -> Types.value -> Types.value * Types.value
val det : Types.value -> Types.value
val eig : Types.value -> Types.value * Types.value
val eigvals : Types.value -> Types.value

val eigh :
  ?uplo:string ->
  ?symmetrize_input:bool ->
  Types.value ->
  Types.value * Types.value

val eigvalsh :
  ?uplo:string -> ?symmetrize_input:bool -> Types.value -> Types.value

val pinv : ?rtol:Types.value -> ?hermitian:bool -> Types.value -> Types.value
val matrix_power : Types.value -> int -> Types.value

val matrix_rank :
  ?rtol:Types.value -> ?hermitian:bool -> Types.value -> Types.value

val vector_norm :
  ?axis:int array option ->
  ?keepdims:bool ->
  ?ord:ord ->
  Types.value ->
  Types.value

val norm :
  ?ord:ord -> ?axis:norm_axis -> ?keepdims:bool -> Types.value -> Types.value

val matrix_norm : ?keepdims:bool -> ?ord:ord -> Types.value -> Types.value
val matrix_transpose : Types.value -> Types.value
val qr : ?mode:string -> Types.value -> Types.value list

val lstsq :
  ?rcond:float ->
  Types.value ->
  Types.value ->
  Types.value * Types.value * Types.value * Types.value

val cross : ?axis:int -> Types.value -> Types.value -> Types.value
val outer : Types.value -> Types.value -> Types.value
val matmul : ?preferred:Dtype.t -> Types.value -> Types.value -> Types.value

val vecdot :
  ?axis:int -> ?preferred:Dtype.t -> Types.value -> Types.value -> Types.value

val tensordot :
  ?preferred:Dtype.t ->
  ?axes:Tensor_contractions.td_axes ->
  Types.value ->
  Types.value ->
  Types.value

val diagonal : ?offset:int -> Types.value -> Types.value
val trace : ?offset:int -> ?dtype:Dtype.t -> Types.value -> Types.value
val tensorinv : ?ind:int -> Types.value -> Types.value
val tensorsolve : ?axes:int array -> Types.value -> Types.value -> Types.value
val multi_dot : Types.value list -> Types.value
val cond : ?p:ord -> Types.value -> Types.value
