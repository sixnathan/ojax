type td_axes = Ax_int of int | Ax_pair of int array * int array

val dot :
  ?precision:Lax_config.precision ->
  ?preferred:Dtype.t ->
  Types.value ->
  Types.value ->
  Types.value

val matmul :
  ?precision:Lax_config.precision ->
  ?preferred:Dtype.t ->
  Types.value ->
  Types.value ->
  Types.value

val matvec : Types.value -> Types.value -> Types.value
val vecmat : Types.value -> Types.value -> Types.value

val vdot :
  ?precision:Lax_config.precision ->
  ?preferred:Dtype.t ->
  Types.value ->
  Types.value ->
  Types.value

val vecdot :
  ?axis:int ->
  ?precision:Lax_config.precision ->
  ?preferred:Dtype.t ->
  Types.value ->
  Types.value ->
  Types.value

val tensordot :
  ?precision:Lax_config.precision ->
  ?preferred:Dtype.t ->
  ?axes:td_axes ->
  Types.value ->
  Types.value ->
  Types.value

val inner :
  ?precision:Lax_config.precision ->
  ?preferred:Dtype.t ->
  Types.value ->
  Types.value ->
  Types.value

val outer : Types.value -> Types.value -> Types.value
