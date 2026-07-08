open Types

val typecheck_jaxpr : jaxpr -> unit
val eval_jaxpr : jaxpr -> value list -> value list
val jaxpr_as_fun : jaxpr -> value list -> value list
val eval_closed_jaxpr : closed_jaxpr -> value list -> value list
val make_jaxpr : aval list -> (value list -> value list) -> closed_jaxpr
val install : unit -> unit
