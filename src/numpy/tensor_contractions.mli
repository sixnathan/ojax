type td_axes = Ax_int of int | Ax_pair of int array * int array

val dot : ?preferred:Dtype.t -> Types.value -> Types.value -> Types.value
val matmul : ?preferred:Dtype.t -> Types.value -> Types.value -> Types.value
val matvec : Types.value -> Types.value -> Types.value
val vecmat : Types.value -> Types.value -> Types.value
val vdot : ?preferred:Dtype.t -> Types.value -> Types.value -> Types.value

val vecdot :
  ?axis:int -> ?preferred:Dtype.t -> Types.value -> Types.value -> Types.value

val tensordot :
  ?preferred:Dtype.t ->
  ?axes:td_axes ->
  Types.value ->
  Types.value ->
  Types.value

val inner : ?preferred:Dtype.t -> Types.value -> Types.value -> Types.value
val outer : Types.value -> Types.value -> Types.value
