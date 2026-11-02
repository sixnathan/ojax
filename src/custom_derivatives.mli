open Types

val custom_jvp :
  ?nondiff_argnums:int list ->
  ?name:string ->
  ?jvp_name:string ->
  (value list -> value list) ->
  jvp:(value list -> value list -> value list * value list) ->
  value list ->
  value list

val custom_vjp :
  ?nondiff_argnums:int list ->
  ?name:string ->
  (value list -> value list) ->
  fwd:(value list -> value list * value list) ->
  bwd:(value list -> value list -> value list) ->
  value list ->
  value list

val defjvps :
  ?name:string ->
  (value list -> value list) ->
  (value -> value list -> value list -> value list) option list ->
  value list ->
  value list

type symbolic_zero = { aval : aval }
type tangent = Symbolic_zero of symbolic_zero | Tangent of value
type custom_vjp_primal = { value : value; perturbed : bool }

val symbolic_zero_from_primal_value : value -> symbolic_zero
val instantiate_tangent : tangent -> value

val custom_vjp_primal_tree_values :
  custom_vjp_primal Tree_util.t -> value Tree_util.t

val custom_jvp_symbolic :
  (value list -> value list) ->
  jvp:(value list -> tangent list -> value list * tangent list) ->
  value list ->
  value list

val custom_vjp_symbolic :
  (value list -> value list) ->
  fwd:(custom_vjp_primal list -> value list * value list) ->
  bwd:(value list -> tangent list -> tangent option list) ->
  value list ->
  value list

val custom_gradient :
  (value list -> value list * (value list -> value list)) ->
  value list ->
  value list

val linear_call :
  forward:(value list -> value list -> value list) ->
  transpose:(value list -> value list -> value list) ->
  value list ->
  value list ->
  value list

val stop_gradient : value -> value

val closure_convert :
  (value list -> value list) ->
  aval list ->
  (value list -> value list) * value list

val pure_callback :
  (value list -> value list) ->
  result_avals:aval list ->
  value list ->
  value list

val io_callback :
  ?ordered:bool ->
  (value list -> value list) ->
  result_avals:aval list ->
  value list ->
  value list

val zero_from_primal : value -> value
