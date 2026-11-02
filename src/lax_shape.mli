val aval_string : Types.aval -> string
val check : Types.primitive -> (unit -> Types.aval list) -> unit

val verify_conv :
  int array ->
  Types.conv_dims ->
  window_strides:int array ->
  lhs_dilation:int array ->
  rhs_dilation:int array ->
  unit

val verify : Types.primitive -> (unit -> Types.aval list) -> unit
