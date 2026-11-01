val geqrf : Types.value -> Types.value * Types.value
val geqp3 : Types.value -> Types.value -> Types.value * Types.value * Types.value

val ormqr :
  ?left:bool ->
  ?transpose:bool ->
  Types.value ->
  Types.value ->
  Types.value ->
  Types.value

val cholesky_update : Types.value -> Types.value -> Types.value

val symmetric_product :
  ?alpha:float ->
  ?beta:float ->
  ?symmetrize_output:bool ->
  Types.value ->
  Types.value ->
  Types.value

val qdwh :
  ?is_hermitian:bool ->
  ?max_iterations:int ->
  ?eps:float ->
  Types.value ->
  Types.value * Types.value * Types.value * Types.value
