val identity : Types.value -> Types.value
val relu : Types.value -> Types.value
val relu6 : Types.value -> Types.value
val softplus : Types.value -> Types.value
val sparse_plus : Types.value -> Types.value
val soft_sign : Types.value -> Types.value
val sigmoid : Types.value -> Types.value
val sparse_sigmoid : Types.value -> Types.value
val silu : Types.value -> Types.value
val mish : Types.value -> Types.value
val log_sigmoid : Types.value -> Types.value
val hard_tanh : Types.value -> Types.value
val hard_sigmoid : Types.value -> Types.value
val hard_silu : Types.value -> Types.value
val selu : Types.value -> Types.value
val log1mexp : Types.value -> Types.value
val elu : ?alpha:float -> Types.value -> Types.value
val celu : ?alpha:float -> Types.value -> Types.value
val leaky_relu : ?negative_slope:float -> Types.value -> Types.value
val squareplus : ?b:float -> Types.value -> Types.value
val gelu : ?approximate:bool -> Types.value -> Types.value
val glu : ?axis:int -> Types.value -> Types.value
val softmax : ?axis:int -> Types.value -> Types.value
val log_softmax : ?axis:int -> Types.value -> Types.value
val standardize : ?axis:int -> ?epsilon:float -> Types.value -> Types.value
val logmeanexp : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val one_hot : ?axis:int -> num_classes:int -> Types.value -> Types.value

val scaled_dot_general :
  lhs_contract:int array ->
  rhs_contract:int array ->
  lhs_batch:int array ->
  rhs_batch:int array ->
  Types.value ->
  Types.value ->
  Types.value
