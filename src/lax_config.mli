type rounding_method = Away_from_zero | To_nearest_even
type gather_scatter_mode = Clip | Fill_or_drop | Promise_in_bounds | One_hot

val gather_scatter_mode_of_string : string -> gather_scatter_mode

type precision_level = Precision_default | Precision_high | Precision_highest

type dot_algorithm = {
  lhs_precision_type : Dtype.t;
  rhs_precision_type : Dtype.t;
  accumulation_type : Dtype.t;
  lhs_component_count : int;
  rhs_component_count : int;
  num_primitive_operations : int;
  allow_imprecise_accumulation : bool;
}

val dot_algorithm :
  ?lhs_component_count:int ->
  ?rhs_component_count:int ->
  ?num_primitive_operations:int ->
  ?allow_imprecise_accumulation:bool ->
  lhs_precision_type:Dtype.t ->
  rhs_precision_type:Dtype.t ->
  accumulation_type:Dtype.t ->
  unit ->
  dot_algorithm

type dot_algorithm_preset =
  | Preset_default
  | Any_f8_any_f8_f32
  | Any_f8_any_f8_f32_fast_accum
  | Any_f8_any_f8_any
  | Any_f8_any_f8_any_fast_accum
  | F16_f16_f16
  | F16_f16_f32
  | Bf16_bf16_bf16
  | Bf16_bf16_f32
  | Bf16_bf16_f32_x3
  | Bf16_bf16_f32_x6
  | Bf16_bf16_f32_x9
  | Tf32_tf32_f32
  | Tf32_tf32_f32_x3
  | F32_f32_f32
  | F64_f64_f64

val dot_algorithm_preset_name : dot_algorithm_preset -> string

type precision =
  | Precision of precision_level * precision_level
  | Algorithm of dot_algorithm
  | Preset of dot_algorithm_preset
  | Named of string
  | Named_pair of string * string

val precision_of_level : precision_level -> precision
val precision_argument_message : string -> string
val precision_of_string : string -> precision
val canonicalize_precision : precision -> precision

type fft_type = Types.fft_kind =
  | Fft_forward
  | Fft_inverse
  | Fft_real
  | Fft_inverse_real

val fft_type_of_string : string -> fft_type

type random_algorithm = Rng_default | Three_fry | Philox
type tolerance = private { atol : float; rtol : float; ulps : int }

val tolerance : ?atol:float -> ?rtol:float -> ?ulps:int -> unit -> tolerance

type accuracy_mode = Accuracy_highest | Accuracy_default
type accuracy = Tolerance of tolerance | Mode of accuracy_mode

type dot_dimension_numbers = Types.dot_dims = {
  lhs_contract : int array;
  rhs_contract : int array;
  lhs_batch : int array;
  rhs_batch : int array;
  untyped_dot : bool;
  dot_precision : (string * string) option;
}

val dot_dimension_numbers :
  ?lhs_batch:int array ->
  ?rhs_batch:int array ->
  lhs_contract:int array ->
  rhs_contract:int array ->
  unit ->
  dot_dimension_numbers

type gather_dimension_numbers = Types.gather_dims = {
  offset_dims : int array;
  collapsed_slice_dims : int array;
  start_index_map : int array;
  g_operand_batching_dims : int array;
  g_start_indices_batching_dims : int array;
  g_sorted_unique : bool;
}

val gather_dimension_numbers :
  ?operand_batching_dims:int array ->
  ?start_indices_batching_dims:int array ->
  offset_dims:int array ->
  collapsed_slice_dims:int array ->
  start_index_map:int array ->
  unit ->
  gather_dimension_numbers

type scatter_dimension_numbers = Types.scatter_dims = {
  update_window_dims : int array;
  inserted_window_dims : int array;
  scatter_dims_to_operand_dims : int array;
  s_operand_batching_dims : int array;
  s_scatter_indices_batching_dims : int array;
  s_sorted_unique : bool;
  s_clip : bool;
}

val scatter_dimension_numbers :
  ?operand_batching_dims:int array ->
  ?scatter_indices_batching_dims:int array ->
  update_window_dims:int array ->
  inserted_window_dims:int array ->
  scatter_dims_to_operand_dims:int array ->
  unit ->
  scatter_dimension_numbers

type conv_dimension_numbers = Types.conv_dims = {
  lhs_spec : int array;
  rhs_spec : int array;
  out_spec : int array;
}

type ragged_dot_dimension_numbers = {
  dot_dimension_numbers : dot_dimension_numbers;
  lhs_ragged_dimensions : int array;
  rhs_group_dimensions : int array;
}

type eig_implementation = Cusolver | Magma | Lapack
type eigh_implementation = Eigh_qr | Eigh_jacobi | Eigh_qdwh

type svd_algorithm =
  | Svd_default
  | Svd_qr
  | Svd_jacobi
  | Svd_polar
  | Svd_divide_and_conquer
