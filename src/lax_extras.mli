type rng_algorithm = Lax_config.random_algorithm =
  | Rng_default
  | Three_fry
  | Philox

val approx_max_k :
  ?reduction_dimension:int -> Types.value -> int -> Types.value * Types.value

val approx_min_k :
  ?reduction_dimension:int -> Types.value -> int -> Types.value * Types.value

val broadcast_like : Types.value -> Types.value -> Types.value
val create_token : unit -> Types.value
val after_all : Types.value list -> Types.value

val ragged_dot :
  ?precision:Lax_config.precision ->
  ?preferred_element_type:Dtype.t ->
  ?group_offset:Types.value ->
  Types.value ->
  Types.value ->
  Types.value ->
  Types.value

val ragged_dot_general :
  ?precision:Lax_config.precision ->
  ?preferred_element_type:Dtype.t ->
  ?group_offset:Types.value ->
  Types.value ->
  Types.value ->
  Types.value ->
  Lax_config.ragged_dot_dimension_numbers ->
  Types.value

val scaled_dot :
  ?lhs_scale:Types.value ->
  ?rhs_scale:Types.value ->
  ?dimension_numbers:Lax_config.dot_dimension_numbers ->
  ?preferred_element_type:Dtype.t ->
  Types.value ->
  Types.value ->
  Types.value

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
