type rounding_method = Away_from_zero | To_nearest_even
type gather_scatter_mode = Clip | Fill_or_drop | Promise_in_bounds | One_hot

let gather_scatter_mode_of_string = function
  | "clip" -> Clip
  | "fill" | "drop" -> Fill_or_drop
  | "promise_in_bounds" -> Promise_in_bounds
  | "one_hot" -> One_hot
  | s -> invalid_arg (Printf.sprintf "Unknown gather mode \"%s\"" s)

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

let dot_algorithm ?(lhs_component_count = 1) ?(rhs_component_count = 1)
    ?(num_primitive_operations = 1) ?(allow_imprecise_accumulation = false)
    ~lhs_precision_type ~rhs_precision_type ~accumulation_type () =
  {
    lhs_precision_type;
    rhs_precision_type;
    accumulation_type;
    lhs_component_count;
    rhs_component_count;
    num_primitive_operations;
    allow_imprecise_accumulation;
  }

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

let preset_names =
  [
    (Preset_default, "DEFAULT");
    (Any_f8_any_f8_f32, "ANY_F8_ANY_F8_F32");
    (Any_f8_any_f8_f32_fast_accum, "ANY_F8_ANY_F8_F32_FAST_ACCUM");
    (Any_f8_any_f8_any, "ANY_F8_ANY_F8_ANY");
    (Any_f8_any_f8_any_fast_accum, "ANY_F8_ANY_F8_ANY_FAST_ACCUM");
    (F16_f16_f16, "F16_F16_F16");
    (F16_f16_f32, "F16_F16_F32");
    (Bf16_bf16_bf16, "BF16_BF16_BF16");
    (Bf16_bf16_f32, "BF16_BF16_F32");
    (Bf16_bf16_f32_x3, "BF16_BF16_F32_X3");
    (Bf16_bf16_f32_x6, "BF16_BF16_F32_X6");
    (Bf16_bf16_f32_x9, "BF16_BF16_F32_X9");
    (Tf32_tf32_f32, "TF32_TF32_F32");
    (Tf32_tf32_f32_x3, "TF32_TF32_F32_X3");
    (F32_f32_f32, "F32_F32_F32");
    (F64_f64_f64, "F64_F64_F64");
  ]

let dot_algorithm_preset_name p = List.assoc p preset_names

type precision =
  | Precision of precision_level * precision_level
  | Algorithm of dot_algorithm
  | Preset of dot_algorithm_preset

let precision_of_level level = Precision (level, level)

let precision_of_string s =
  match s with
  | "highest" | "float32" -> precision_of_level Precision_highest
  | "high" | "bfloat16_3x" | "tensorfloat32" ->
      precision_of_level Precision_high
  | "default" | "bfloat16" | "fastest" -> precision_of_level Precision_default
  | _ -> (
      match List.find_opt (fun (_, name) -> name = s) preset_names with
      | Some (preset, _) -> Preset preset
      | None ->
          invalid_arg
            (Printf.sprintf "Precision argument must be valid, but got %s." s))

type fft_type = Types.fft_kind =
  | Fft_forward
  | Fft_inverse
  | Fft_real
  | Fft_inverse_real

let fft_type_of_string = function
  | "fft" | "FFT" -> Fft_forward
  | "ifft" | "IFFT" -> Fft_inverse
  | "rfft" | "RFFT" -> Fft_real
  | "irfft" | "IRFFT" -> Fft_inverse_real
  | s -> invalid_arg (Printf.sprintf "Unknown FFT type '%s'" s)

type random_algorithm = Rng_default | Three_fry | Philox
type tolerance = { atol : float; rtol : float; ulps : int }

let tolerance ?(atol = 0.0) ?(rtol = 0.0) ?(ulps = 0) () =
  if atol < 0.0 || rtol < 0.0 || ulps < 0 then
    invalid_arg "Tolerances must be non-negative.";
  if atol = 0.0 && rtol = 0.0 && ulps = 0 then
    invalid_arg "At least one of atol, rtol, or ulps must be set.";
  { atol; rtol; ulps }

type accuracy_mode = Accuracy_highest | Accuracy_default
type accuracy = Tolerance of tolerance | Mode of accuracy_mode

type dot_dimension_numbers = Types.dot_dims = {
  lhs_contract : int array;
  rhs_contract : int array;
  lhs_batch : int array;
  rhs_batch : int array;
  dot_precision : (string * string) option;
  dot_preferred : bool;
}

let dot_dimension_numbers ?(lhs_batch = [||]) ?(rhs_batch = [||]) ~lhs_contract
    ~rhs_contract () =
  { lhs_contract; rhs_contract; lhs_batch; rhs_batch; dot_precision = None; dot_preferred = true }

type gather_dimension_numbers = Types.gather_dims = {
  offset_dims : int array;
  collapsed_slice_dims : int array;
  start_index_map : int array;
  g_operand_batching_dims : int array;
  g_start_indices_batching_dims : int array;
  g_sorted_unique : bool;
}

let gather_dimension_numbers ?(operand_batching_dims = [||])
    ?(start_indices_batching_dims = [||]) ~offset_dims ~collapsed_slice_dims
    ~start_index_map () =
  {
    offset_dims;
    collapsed_slice_dims;
    start_index_map;
    g_operand_batching_dims = operand_batching_dims;
    g_start_indices_batching_dims = start_indices_batching_dims;
    g_sorted_unique = false;
  }

type scatter_dimension_numbers = Types.scatter_dims = {
  update_window_dims : int array;
  inserted_window_dims : int array;
  scatter_dims_to_operand_dims : int array;
  s_operand_batching_dims : int array;
  s_scatter_indices_batching_dims : int array;
  s_sorted_unique : bool;
}

let scatter_dimension_numbers ?(operand_batching_dims = [||])
    ?(scatter_indices_batching_dims = [||]) ~update_window_dims
    ~inserted_window_dims ~scatter_dims_to_operand_dims () =
  {
    update_window_dims;
    inserted_window_dims;
    scatter_dims_to_operand_dims;
    s_operand_batching_dims = operand_batching_dims;
    s_scatter_indices_batching_dims = scatter_indices_batching_dims;
    s_sorted_unique = false;
  }

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
