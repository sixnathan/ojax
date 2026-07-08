open Types

val jit_flat : (value list -> value list) -> value list -> value list
val call : (value list -> value list) -> value list -> value list

val make_jaxpr :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  closed_jaxpr

val jit :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val jvp :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t list ->
  value Tree_util.t * value Tree_util.t

val linearize :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t * (value Tree_util.t list -> value Tree_util.t)

val vjp :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t * (value Tree_util.t -> value Tree_util.t list)

val grad :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val value_and_grad :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t * value Tree_util.t

val vmap :
  (value Tree_util.t list -> value Tree_util.t) ->
  int option list ->
  value Tree_util.t list ->
  value Tree_util.t
