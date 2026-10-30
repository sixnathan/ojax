open Types

type layer = {
  init : value -> int array -> int array * value Tree_util.t;
  apply : ?rng:value -> value Tree_util.t -> value -> value;
}

val glorot : value -> int array -> value
val normal_init : value -> int array -> value

val dense :
  ?w_init:(value -> int array -> value) ->
  ?b_init:(value -> int array -> value) ->
  int ->
  layer

val elementwise : (value -> value) -> layer
val tanh : layer
val relu : layer
val exp : layer
val log_softmax : layer
val softmax : layer
val softplus : layer
val sigmoid : layer
val elu : layer
val leaky_relu : layer
val selu : layer
val gelu : layer
val identity : layer
val flatten : layer

val general_conv :
  ?strides:int array ->
  ?padding:Lax_api.padding ->
  ?w_init:(value -> int array -> value) ->
  ?b_init:(value -> int array -> value) ->
  int ->
  int array ->
  layer

val conv :
  ?strides:int array ->
  ?padding:Lax_api.padding ->
  ?w_init:(value -> int array -> value) ->
  ?b_init:(value -> int array -> value) ->
  int ->
  int array ->
  layer

val max_pool :
  ?strides:int array -> ?padding:Lax_api.padding -> int array -> layer

val sum_pool :
  ?strides:int array -> ?padding:Lax_api.padding -> int array -> layer

val avg_pool :
  ?strides:int array -> ?padding:Lax_api.padding -> int array -> layer

val dropout : ?mode:[ `Train | `Test ] -> float -> layer
val serial : layer list -> layer
val batch_norm : ?axis:int array -> ?epsilon:float -> unit -> layer
