type aval = { shape : int array; dtype : Dtype.t; weak_type : bool }

type value =
  | Concrete of Ndarray.t
  | Tracer of tracer
  | Device of Pjrt.Buffer.t

and tracer = { id : int; trace : trace; aval : aval; payload : payload }

and payload =
  | Eval
  | JVP of { primal : value; tangent : value }
  | Batch of { v : value; bdim : int option }
  | Jaxpr of unit
  | PE of { pval : partial_val; mutable recipe : recipe option }

and trace = { level : int; kind : trace_kind; global_data : global_data }
and trace_kind = KEval | KJVP | KBatch | KJaxpr | KPE
and global_data = GNone | GAxisSize of int | GBuilder of jaxpr_builder

and trace_debug = {
  func_src_info : string;
  traced_for : string;
  arg_names : string list option;
}

and jaxpr_builder = {
  mutable jb_eqns : eqn list;
  mutable jb_tracer_to_var : (int * var) list;
  mutable jb_const_tracers : (int * tracer) list;
  mutable jb_constvals : (var * value) list;
  mutable jb_tracers : tracer list;
  mutable jb_invars : var list;
  jb_debug : trace_debug;
}

and partial_val = { pv_aval : aval; pv_const : value option }

and recipe =
  | LambdaBinding
  | ConstRecipe of value
  | EqnRecipe of {
      er_prim : primitive;
      er_inputs : tracer list;
      er_avals_out : aval list;
      er_out : tracer option ref list;
    }

and primitive =
  | Add
  | Add_any
  | Stop_gradient
  | Sub
  | Mul
  | Div
  | Neg
  | Sin
  | Cos
  | Exp
  | Log
  | Tanh
  | Max
  | Min
  | Pow
  | Abs
  | Sign
  | Eq
  | Lt
  | Gt
  | Select_n
  | Convert_element_type of Dtype.t
  | Broadcast_in_dim of { shape : int array; dims : int array }
  | Reshape of int array
  | Reshape_perm of { new_sizes : int array; dimensions : int array }
  | Decomposable of {
      dname : string;
      dparams : (string * string) list;
      dimpl : int;
      davals : aval list;
    }
  | Reduce_sum of int array
  | Dot_general of dot_dims
  | Acos
  | Acosh
  | Asin
  | Asinh
  | Atan
  | Atanh
  | Cbrt
  | Ceil
  | Clz
  | Conj
  | Copy
  | Cosh
  | Exp2
  | Expm1
  | Floor
  | Imag
  | Integer_pow of int
  | Is_finite
  | Log1p
  | Logistic
  | Not
  | Population_count
  | Real
  | Round
  | Round_nearest_even
  | Rsqrt
  | Sinh
  | Sqrt
  | Square
  | Tan
  | And
  | Atan2
  | Complex
  | Eq_to
  | Ge
  | Le
  | Le_to
  | Lt_to
  | Mulhi
  | Ne
  | Nextafter
  | Or
  | Rem
  | Shift_left
  | Shift_right_arithmetic
  | Shift_right_logical
  | Xor
  | Concatenate of int
  | Pad of (int * int * int) array
  | Rev of int array
  | Split of { sizes : int array; axis : int }
  | Squeeze of int array
  | Stack of int
  | Tile of int array
  | Transpose of int array
  | Unstack of int
  | Argmax of { axis : int; index_dtype : Dtype.t }
  | Argmin of { axis : int; index_dtype : Dtype.t }
  | Reduce of { jaxpr : closed_jaxpr; dimensions : int array }
  | Reduce_and of int array
  | Reduce_max of int array
  | Reduce_min of int array
  | Reduce_or of int array
  | Reduce_prod of int array
  | Reduce_xor of int array
  | After_all
  | Bitcast_convert_type of Dtype.t
  | Clamp
  | Composite of closed_jaxpr
  | Create_token
  | Dce_sink
  | Empty of { shape : int array; dtype : Dtype.t }
  | Empty2 of Dtype.t
  | From_edtype of Dtype.t
  | Iota of { dtype : Dtype.t; shape : int array; dimension : int }
  | Optimization_barrier
  | Ragged_dot_general
  | Reduce_precision of { exponent_bits : int; mantissa_bits : int }
  | Rng_bit_generator of {
      rbg_shape : int array;
      rbg_dtype : Dtype.t;
      rbg_algorithm : string;
    }
  | Rng_uniform
  | Sort of { dimension : int; is_stable : bool; num_keys : int }
  | Tie
  | To_edtype of Dtype.t
  | Top_k of { k : int; axis : int }
  | Slice of {
      start_indices : int array;
      limit_indices : int array;
      strides : int array option;
    }
  | Dynamic_slice of { slice_sizes : int array }
  | Dynamic_update_slice
  | Gather of { dimension_numbers : gather_dims; slice_sizes : int array }
  | Scatter of { dimension_numbers : scatter_dims; unique_indices : bool }
  | Scatter_add of { dimension_numbers : scatter_dims }
  | Scatter_sub of { dimension_numbers : scatter_dims }
  | Scatter_mul of { dimension_numbers : scatter_dims; unique_indices : bool }
  | Scatter_min of { dimension_numbers : scatter_dims }
  | Scatter_max of { dimension_numbers : scatter_dims }
  | Conv_general_dilated of {
      window_strides : int array;
      padding : (int * int) array;
      lhs_dilation : int array;
      rhs_dilation : int array;
      dimension_numbers : conv_dims;
      feature_group_count : int;
      batch_group_count : int;
      conv_precision : (string * string) option;
    }
  | Reduce_window of { reducer : closed_jaxpr; window : window_dims }
  | Reduce_window_max of window_dims
  | Reduce_window_min of window_dims
  | Reduce_window_sum of window_dims
  | Select_and_gather_add of { select : window_select; window : window_dims }
  | Select_and_scatter of {
      select_jaxpr : closed_jaxpr;
      scatter_jaxpr : closed_jaxpr;
      window : window_dims;
    }
  | Select_and_scatter_add of { select : window_select; window : window_dims }
  | Bessel_i0e
  | Bessel_i1e
  | Digamma
  | Erf
  | Erf_inv
  | Erfc
  | Igamma
  | Igamma_grad_a
  | Igammac
  | Lgamma
  | Polygamma
  | Regularized_incomplete_beta
  | Zeta
  | Platform_index of string array option array
  | Xla_call of closed_jaxpr
  | Collective of { kind : collective_kind; axis_name : string }
  | Fft of { fft_type : fft_kind; fft_lengths : int array }
  | Custom_jvp_call of int
  | Custom_vjp_call of int
  | Custom_lin of { rule : int; n_res : int; out_avals : aval list }
  | Checkpoint_name of string
  | Cond of {
      branches : closed_jaxpr list;
      platforms : string array option array option;
    }
  | Scan of {
      length : int;
      reverse : bool;
      num_consts : int;
      num_carry : int;
      jaxpr : closed_jaxpr;
      unroll : int;
    }
  | While of {
      cond_nconsts : int;
      body_nconsts : int;
      cond : closed_jaxpr;
      body : closed_jaxpr;
    }
  | Cumsum of { axis : int; reverse : bool }
  | Cumprod of { axis : int; reverse : bool }
  | Cummax of { axis : int; reverse : bool }
  | Cummin of { axis : int; reverse : bool }
  | Cumlogsumexp of { axis : int; reverse : bool }
  | Custom_linear_solve of {
      matvec : closed_jaxpr;
      vecmat : closed_jaxpr option;
      solve : closed_jaxpr;
      transpose_solve : closed_jaxpr option;
      const_lengths : int array;
    }
  | Threefry2x32
  | Iota_2x32_shape of int array
  | Random_seed
  | Random_split of int array
  | Random_fold_in
  | Random_bits of { bit_width : int; shape : int array }
  | Random_wrap
  | Random_unwrap
  | Random_gamma of { log_space : bool }
  | Cholesky
  | Cholesky_update
  | Symmetric_product of { alpha : float; beta : float }
  | Householder_product
  | Lu
  | Lu_pivots_to_permutation of int
  | Qr of bool
  | Qr_pivoted of bool
  | Geqrf
  | Geqp3
  | Ormqr of { ormqr_left : bool; ormqr_transpose : bool }
  | Triangular_solve of {
      left_side : bool;
      lower : bool;
      transpose_a : bool;
      conjugate_a : bool;
      unit_diagonal : bool;
    }
  | Tridiagonal_solve of { perturb_singular : bool }
  | Eig of {
      compute_left : bool;
      compute_right : bool;
      enable_eigvec_derivs : bool;
    }
  | Eigh of { lower : bool }
  | Hessenberg
  | Schur of { compute_vectors : bool }
  | Svd of { full_matrices : bool; compute_uv : bool; gesvd : bool }
  | Tridiagonal of { lower : bool }
  | Ffi_call of ffi_call

and ffi_call = {
  ffi_target : string;
  ffi_result_avals : aval list;
  ffi_vmap_method : string option;
  ffi_has_side_effect : bool;
  ffi_input_layouts : int array list;
  ffi_output_layouts : int array list;
  ffi_aliases : (int * int) list;
  ffi_api_version : int;
  ffi_legacy_backend_config : string option;
  ffi_attributes : (string * ffi_attr) list;
}

and ffi_attr =
  | Ffi_bool of bool
  | Ffi_int of int64
  | Ffi_float of float
  | Ffi_string of string
  | Ffi_array of Ndarray.t
  | Ffi_dict of (string * ffi_attr) list
  | Ffi_seq of ffi_attr list
  | Ffi_dtype of Dtype.t

and dot_dims = {
  lhs_contract : int array;
  rhs_contract : int array;
  lhs_batch : int array;
  rhs_batch : int array;
  untyped_dot : bool;
  dot_precision : (string * string) option;
}

and gather_dims = {
  offset_dims : int array;
  collapsed_slice_dims : int array;
  start_index_map : int array;
  g_operand_batching_dims : int array;
  g_start_indices_batching_dims : int array;
  g_sorted_unique : bool;
}

and scatter_dims = {
  update_window_dims : int array;
  inserted_window_dims : int array;
  scatter_dims_to_operand_dims : int array;
  s_operand_batching_dims : int array;
  s_scatter_indices_batching_dims : int array;
  s_sorted_unique : bool;
  s_clip : bool;
}

and conv_dims = {
  lhs_spec : int array;
  rhs_spec : int array;
  out_spec : int array;
}

and window_dims = {
  window_dimensions : int array;
  window_strides : int array;
  w_padding : (int * int) array;
  base_dilation : int array;
  window_dilation : int array;
}

and window_select = Wge | Wle

and collective_kind =
  | Psum
  | Pmax
  | Pmin
  | All_gather of { axis : int; size : int }
  | Axis_index
  | Pbroadcast of int
  | Ppermute of (int * int) array
  | All_to_all of { split_axis : int; concat_axis : int; size : int }
  | Reduce_scatter of { scatter_dimension : int; size : int; tiled : bool }

and fft_kind = Fft_forward | Fft_inverse | Fft_real | Fft_inverse_real
and var = { vid : int; vaval : aval }
and atom = A_var of var | A_lit of Ndarray.t | DropVar of aval

and eqn = {
  prim : primitive;
  inputs : atom list;
  outs : var list;
  multiple_results : bool;
}

and jaxpr = { in_binders : var list; eqns : eqn list; outs : atom list }
and closed_jaxpr = { jid : int; jaxpr : jaxpr; consts : Ndarray.t list }
