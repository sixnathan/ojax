open Types

val partial_val_known : value -> partial_val
val partial_val_unknown : aval -> partial_val
val is_known : partial_val -> bool
val is_unknown : partial_val -> bool
val new_unknown_arg : trace -> aval -> tracer
val instantiate_const : trace -> tracer -> tracer
val toposort : tracer list -> (tracer -> tracer list) -> tracer list

val tracers_to_jaxpr :
  ?literals:bool ->
  ?effects:(int * Types.primitive * Types.tracer list) list ->
  tracer list ->
  tracer list ->
  jaxpr * value list

val partial_eval_flat :
  (value list -> value list) ->
  partial_val list ->
  jaxpr * value list * partial_val list

val partial_eval_flat_instantiate :
  instantiate:(int -> bool) ->
  (value list -> value list) ->
  partial_val list ->
  jaxpr * value list * partial_val list

val known_tracer : trace -> value -> tracer
val is_unknown_value : value -> bool
val record_unknown : trace -> primitive -> tracer list -> value list
val custom_call_rule : (trace -> int -> tracer list -> value list option) ref
val install : unit -> unit
