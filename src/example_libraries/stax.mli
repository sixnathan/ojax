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

val general_conv_transpose :
  ?strides:int array ->
  ?padding:Lax_api.padding ->
  ?w_init:(value -> int array -> value) ->
  ?b_init:(value -> int array -> value) ->
  string * string * string ->
  int ->
  int array ->
  layer

val conv1d_transpose :
  ?strides:int array ->
  ?padding:Lax_api.padding ->
  ?w_init:(value -> int array -> value) ->
  ?b_init:(value -> int array -> value) ->
  int ->
  int array ->
  layer

val conv_transpose :
  ?strides:int array ->
  ?padding:Lax_api.padding ->
  ?w_init:(value -> int array -> value) ->
  ?b_init:(value -> int array -> value) ->
  int ->
  int array ->
  layer

type io_shape = Shape of int array | Shapes of io_shape list
type io_value = Value of value | Values of io_value list

type net = {
  net_init : value -> io_shape -> io_shape * value Tree_util.t;
  net_apply : ?rng:value -> value Tree_util.t -> io_value -> io_value;
}

val of_layer : layer -> net
val fan_out : int -> net
val fan_in_sum : net
val fan_in_concat : ?axis:int -> unit -> net
val serial_net : net list -> net
val parallel : net list -> net
val dropout : ?mode:[ `Train | `Test ] -> float -> layer
val serial : layer list -> layer
val batch_norm : ?axis:int array -> ?epsilon:float -> unit -> layer
