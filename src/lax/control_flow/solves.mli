open Types

val solve_out_avals : closed_jaxpr -> aval list
val solve_impl : closed_jaxpr -> value list -> value list

val custom_linear_solve :
  ?symmetric:bool ->
  ?transpose_solve:((value list -> value list) -> value list -> value list) ->
  (value list -> value list) ->
  value list ->
  ((value list -> value list) -> value list -> value list) ->
  value list
