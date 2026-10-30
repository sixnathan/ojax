open Types

type padding = Same | Same_lower | Valid | Explicit of (int * int) array

val abs : value -> value
val acos : value -> value
val acosh : value -> value
val add : value -> value -> value
val asin : value -> value
val asinh : value -> value
val atan : value -> value
val atan2 : value -> value -> value
val atanh : value -> value
val bitwise_and : value -> value -> value
val bitwise_not : value -> value
val bitwise_or : value -> value -> value
val bitwise_xor : value -> value -> value
val cbrt : value -> value
val ceil : value -> value
val clz : value -> value
val complex : value -> value -> value
val conj : value -> value
val cos : value -> value
val cosh : value -> value
val div : value -> value -> value
val eq : value -> value -> value
val exp : value -> value
val exp2 : value -> value
val expm1 : value -> value
val floor : value -> value
val ge : value -> value -> value
val gt : value -> value -> value
val imag : value -> value
val integer_pow : value -> int -> value
val is_finite : value -> value
val le : value -> value -> value
val log : value -> value
val log1p : value -> value
val logistic : value -> value
val lt : value -> value -> value
val max : value -> value -> value
val min : value -> value -> value
val mul : value -> value -> value
val mulhi : value -> value -> value
val ne : value -> value -> value
val neg : value -> value
val nextafter : value -> value -> value
val population_count : value -> value
val pow : value -> value -> value
val real : value -> value
val reciprocal : value -> value
val rem : value -> value -> value
val round : value -> value
val rsqrt : value -> value
val shift_left : value -> value -> value
val shift_right_arithmetic : value -> value -> value
val shift_right_logical : value -> value -> value
val sign : value -> value
val sin : value -> value
val sinh : value -> value
val sqrt : value -> value
val square : value -> value
val sub : value -> value -> value
val tan : value -> value
val tanh : value -> value
val betainc : value -> value -> value -> value
val stop_gradient : value -> value
val optimization_barrier : value list -> value list
val convert_element_type : value -> Dtype.t -> value
val bitcast_convert_type : value -> Dtype.t -> value
val reduce_precision : value -> exponent_bits:int -> mantissa_bits:int -> value
val clamp : value -> value -> value -> value
val select : value -> value -> value -> value
val select_n : value -> value list -> value
val full : int array -> float -> Dtype.t -> value
val full_like : value -> float -> value
val empty : Dtype.t -> value
val iota : Dtype.t -> int -> value
val broadcasted_iota : Dtype.t -> int array -> int -> value
val reshape : ?dimensions:int array -> value -> int array -> value
val transpose : value -> int array -> value
val rev : value -> int array -> value
val tile : value -> int array -> value
val squeeze : value -> int array -> value
val expand_dims : value -> int array -> value
val broadcast_in_dim : value -> int array -> int array -> value
val broadcast : value -> int array -> value
val broadcast_to_rank : value -> int -> value
val broadcast_shapes : int array list -> int array
val collapse : value -> int -> int -> value
val concatenate : value list -> int -> value
val stack : value list -> int -> value
val unstack : value -> int -> value list
val split : value -> int array -> int -> value list
val pad : value -> value -> (int * int * int) array -> value
val slice : ?strides:int array -> value -> int array -> int array -> value
val slice_in_dim : ?stride:int -> ?axis:int -> value -> int -> int -> value
val index_in_dim : ?axis:int -> ?keepdims:bool -> value -> int -> value
val dynamic_slice : value -> value list -> int array -> value
val dynamic_update_slice : value -> value -> value list -> value
val dynamic_slice_in_dim : ?axis:int -> value -> value -> int -> value

val dynamic_index_in_dim :
  ?axis:int -> ?keepdims:bool -> value -> value -> value

val dynamic_update_slice_in_dim : ?axis:int -> value -> value -> value -> value
val dynamic_update_index_in_dim : ?axis:int -> value -> value -> value -> value
val gather : value -> value -> gather_dims -> int array -> value

val scatter :
  ?unique_indices:bool -> value -> value -> value -> scatter_dims -> value

val scatter_add : value -> value -> value -> scatter_dims -> value
val scatter_sub : value -> value -> value -> scatter_dims -> value

val scatter_mul :
  ?unique_indices:bool -> value -> value -> value -> scatter_dims -> value

val scatter_min : value -> value -> value -> scatter_dims -> value
val scatter_max : value -> value -> value -> scatter_dims -> value
val reduce_sum : value -> int array -> value
val reduce_prod : value -> int array -> value
val reduce_max : value -> int array -> value
val reduce_min : value -> int array -> value
val reduce_and : value -> int array -> value
val reduce_or : value -> int array -> value
val reduce_xor : value -> int array -> value
val reduce : value -> value -> (value -> value -> value) -> int array -> value

val reduce_window :
  value ->
  value ->
  (value -> value -> value) ->
  window_dimensions:int array ->
  window_strides:int array ->
  ?base_dilation:int array ->
  ?window_dilation:int array ->
  (int * int) array ->
  value

val argmax : value -> axis:int -> index_dtype:Dtype.t -> value
val argmin : value -> axis:int -> index_dtype:Dtype.t -> value

val sort :
  ?dimension:int -> ?is_stable:bool -> ?num_keys:int -> value list -> value list

val sort_key_val :
  ?dimension:int -> ?is_stable:bool -> value -> value -> value * value

val top_k : ?axis:int -> value -> int -> value * value
val dot : value -> value -> value

val dot_general :
  value ->
  value ->
  lhs_contract:int array ->
  rhs_contract:int array ->
  lhs_batch:int array ->
  rhs_batch:int array ->
  value

val batch_matmul : value -> value -> value

val padtype_to_pads :
  int array -> int array -> int array -> padding -> (int * int) array

val conv_general_permutations : String.t * String.t * String.t -> conv_dims
val conv_dimension_numbers : String.t * String.t * String.t -> conv_dims

val conv_general_dilated :
  ?lhs_dilation:int array ->
  ?rhs_dilation:int array ->
  ?dimension_numbers:conv_dims ->
  ?feature_group_count:int ->
  ?batch_group_count:int ->
  value ->
  value ->
  window_strides:int array ->
  padding ->
  value

val conv : value -> value -> window_strides:int array -> padding -> value

val conv_with_general_padding :
  ?lhs_dilation:int array ->
  ?rhs_dilation:int array ->
  value ->
  value ->
  window_strides:int array ->
  (int * int) array ->
  value

val conv_transpose :
  ?rhs_dilation:int array ->
  ?dimension_numbers:conv_dims ->
  ?transpose_kernel:bool ->
  value ->
  value ->
  strides:int array ->
  padding ->
  value

val cond :
  value ->
  (value list -> value list) ->
  (value list -> value list) ->
  value list ->
  value list

val scan :
  ?length:int ->
  ?reverse:bool ->
  (value list -> value list) ->
  value list ->
  value list ->
  value list

val while_loop :
  (value list -> value) ->
  (value list -> value list) ->
  value list ->
  value list

val cumsum : ?axis:int -> ?reverse:bool -> value -> value
val cumprod : ?axis:int -> ?reverse:bool -> value -> value
val cummax : ?axis:int -> ?reverse:bool -> value -> value
val cummin : ?axis:int -> ?reverse:bool -> value -> value
val cumlogsumexp : ?axis:int -> ?reverse:bool -> value -> value

val custom_linear_solve :
  ?symmetric:bool ->
  ?transpose_solve:((value list -> value list) -> value list -> value list) ->
  (value list -> value list) ->
  value list ->
  ((value list -> value list) -> value list -> value list) ->
  value list

val map : (value list -> value list) -> value list -> value list

val switch :
  value -> (value list -> value list) list -> value list -> value list

val fori_loop :
  value ->
  value ->
  (value -> value list -> value list) ->
  value list ->
  value list

val custom_root :
  (value list -> value list) ->
  value list ->
  ((value list -> value list) -> value list -> value list) ->
  ((value list -> value list) -> value list -> value list) ->
  value list

val platform_dependent : ?default:'a -> cpu:(unit -> 'b) -> unit -> 'b
val associative_scan : ?axis:int -> (value -> value -> value) -> value -> value
val shape_as_value : value -> value
val index_take : value -> value list -> int array -> value

val reduce_window_shape_tuple :
  int array ->
  int array ->
  int array ->
  (int * int) array ->
  ?base_dilation:int array ->
  ?window_dilation:int array ->
  unit ->
  int array

val conv_shape_tuple :
  int array -> int array -> int array -> (int * int) array -> int array

val conv_general_shape_tuple :
  int array -> int array -> int array -> padding -> conv_dims -> int array

val conv_transpose_shape_tuple :
  int array -> int array -> int array -> padding -> conv_dims -> int array

val conv_general_dilated_patches :
  ?window_strides:int array ->
  ?padding:padding ->
  ?lhs_dilation:int array ->
  ?rhs_dilation:int array ->
  ?dimension_numbers:conv_dims ->
  value ->
  filter_shape:int array ->
  value

val scatter_apply : value -> value -> (value -> value) -> scatter_dims -> value
val with_sharding_constraint : value -> 'a -> value
val pcast : value -> dtype:'a -> value

val composite :
  (value list -> value list) ->
  name:string ->
  attrs:(string * string) list ->
  version:int ->
  value list ->
  value list

val fft : value -> fft_type:fft_kind -> fft_lengths:int array -> value
val random_gamma_grad : value -> value -> value
val rng_uniform : value -> value -> int array -> value
