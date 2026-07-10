open Types

val custom_jvp :
  primal:(value list -> value list) ->
  jvp:(value list -> value list) ->
  value list ->
  value list

val custom_vjp :
  primal:(value list -> value list) ->
  fwd:(value list -> value list) ->
  bwd:(value list -> value list) ->
  value list ->
  value list

val custom_gradient :
  (value list -> value list * (value list -> value list)) ->
  value list ->
  value list

val closure_convert :
  (value list -> value list) ->
  value list ->
  (value list -> value list) * value list
