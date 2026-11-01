val dce_closed :
  Types.closed_jaxpr ->
  bool list ->
  instantiate:bool list ->
  Types.closed_jaxpr * bool list

val prune_unused_inputs : Types.closed_jaxpr -> Types.closed_jaxpr * int list
