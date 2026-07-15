open Types

type policy

val everything_saveable : policy
val nothing_saveable : policy
val checkpoint_dots : policy
val dots_with_no_batch_dims_saveable : policy
val save_only_these_names : string list -> policy
val policy_allows : policy -> ?name:string -> primitive -> bool
val checkpoint_name : value -> string -> value

val checkpoint :
  ?prevent_cse:bool ->
  ?policy:policy ->
  (value list -> value list) ->
  value list ->
  value list

val remat :
  ?prevent_cse:bool ->
  ?policy:policy ->
  (value list -> value list) ->
  value list ->
  value list

val saved_residuals :
  (value list -> value list) -> value list -> (aval * string) list
