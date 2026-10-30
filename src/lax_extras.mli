type rng_algorithm = Rng_default | Three_fry | Philox

val approx_max_k :
  ?reduction_dimension:int -> Types.value -> int -> Types.value * Types.value

val approx_min_k :
  ?reduction_dimension:int -> Types.value -> int -> Types.value * Types.value

val broadcast_like : Types.value -> Types.value -> Types.value
val create_token : unit -> Types.value
val after_all : Types.value list -> Types.value
val ragged_dot : Types.value -> Types.value -> Types.value -> Types.value

val conv_general_dilated_local :
  ?window_strides:int array ->
  ?padding:Lax_api.padding ->
  ?lhs_dilation:int array ->
  ?rhs_dilation:int array ->
  ?dimension_numbers:Types.conv_dims ->
  Types.value ->
  Types.value ->
  filter_shape:int array ->
  Types.value

val rng_bit_generator :
  ?dtype:Dtype.t ->
  ?algorithm:rng_algorithm ->
  Types.value ->
  int array ->
  Types.value * Types.value
