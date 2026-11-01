open Types

val install : unit -> unit

val jvp :
  (value list -> value list) ->
  value list ->
  value list ->
  value list * value list

val jvp_instantiated :
  (value list -> value list) ->
  value list ->
  value list ->
  value list * value list

val solve_ref : (value -> value -> value) ref

val associative_scan_flat :
  (value list -> value list -> value list) ->
  axis:int ->
  reverse:bool ->
  value list ->
  value list

val lax_eye : Dtype.t -> int -> int -> value

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
