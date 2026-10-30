open Types

val custom_jvp :
  (value list -> value list) ->
  jvp:(value list -> value list -> value list * value list) ->
  value list ->
  value list

val custom_vjp :
  (value list -> value list) ->
  fwd:(value list -> value list * value list) ->
  bwd:(value list -> value list -> value list) ->
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
