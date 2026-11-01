open Types

val typecheck_jaxpr : jaxpr -> unit
val eval_jaxpr : jaxpr -> value list -> value list
val jaxpr_as_fun : jaxpr -> value list -> value list
val eval_closed_jaxpr : closed_jaxpr -> value list -> value list
val make_jaxpr : aval list -> (value list -> value list) -> closed_jaxpr

val make_jaxpr_stateful :
  aval list ->
  (value list -> value list) ->
  closed_jaxpr * value list * Ref_state.cell list

val make_jaxpr_forbidding_refs :
  message:string ->
  aval list ->
  (value list -> value list) ->
  closed_jaxpr * value list

val make_jaxpr_hoisted :
  aval list -> (value list -> value list) -> closed_jaxpr * value list

val install : unit -> unit

val eager_jit :
  (string -> string -> (value list -> value list) -> value list -> value list)
  ref

val named_jit :
  name:string ->
  ?key:string ->
  (value list -> value list) ->
  value list ->
  value list
