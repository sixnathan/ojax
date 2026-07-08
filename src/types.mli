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
  | Xla_call of closed_jaxpr
  | Cond of { t : closed_jaxpr; f : closed_jaxpr }

and dot_dims = {
  lhs_contract : int array;
  rhs_contract : int array;
  lhs_batch : int array;
  rhs_batch : int array;
}

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
