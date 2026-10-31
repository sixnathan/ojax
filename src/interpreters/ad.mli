open Types

val install : unit -> unit

val jvp :
  (value list -> value list) ->
  value list ->
  value list ->
  value list * value list

val linearize :
  (value list -> value list) ->
  value list ->
  value list * (value list -> value list)

val vjp :
  (value list -> value list) ->
  value list ->
  value list * (value list -> value list)

val vjp_staged :
  (value list -> value list) ->
  value list ->
  value list * value list * (value list -> value list -> value list)

val aux_primal : value -> value
val grad : (value list -> value) -> value list -> value list

type tval = Prim of value | Undef of aval

val eval_jaxpr_transposed : jaxpr -> tval list -> value list -> value list
