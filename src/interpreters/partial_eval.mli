open Types

val partial_val_known : value -> partial_val
val partial_val_unknown : aval -> partial_val
val is_known : partial_val -> bool
val is_unknown : partial_val -> bool
val instantiate_const : trace -> tracer -> tracer
val toposort : tracer list -> (tracer -> tracer list) -> tracer list
val tracers_to_jaxpr : tracer list -> tracer list -> jaxpr * value list

val partial_eval_flat :
  (value list -> value list) ->
  partial_val list ->
  jaxpr * value list * partial_val list

val install : unit -> unit
