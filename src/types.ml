type aval = { shape : int array; dtype : Dtype.t; weak_type : bool }

type value = Concrete of Ndarray.t | Tracer of tracer
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

and jaxpr_builder = {
  mutable jb_eqns : eqn list;
  mutable jb_tracer_to_var : (int * var) list;
  mutable jb_const_tracers : (int * tracer) list;
  mutable jb_constvals : (var * value) list;
  mutable jb_tracers : tracer list;
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
  | Rng_bit_generator
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
  | Cond of { t : closed_jaxpr; f : closed_jaxpr }
  | Scan of {
      length : int;
      reverse : bool;
      num_carry : int;
      jaxpr : closed_jaxpr;
    }
  | While of { cond : closed_jaxpr; body : closed_jaxpr }

and dot_dims = {
  lhs_contract : int array;
  rhs_contract : int array;
  lhs_batch : int array;
  rhs_batch : int array;
}

and gather_dims = {
  offset_dims : int array;
  collapsed_slice_dims : int array;
  start_index_map : int array;
  g_operand_batching_dims : int array;
  g_start_indices_batching_dims : int array;
}

and scatter_dims = {
  update_window_dims : int array;
  inserted_window_dims : int array;
  scatter_dims_to_operand_dims : int array;
  s_operand_batching_dims : int array;
  s_scatter_indices_batching_dims : int array;
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
