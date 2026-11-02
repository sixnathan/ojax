open Types

type rule = int -> bool list -> value list -> value list * bool list

val custom_vmap :
  ?rule_name:string ->
  (value list -> value list) ->
  rule:rule ->
  value list ->
  value list

val sequential_vmap : (value list -> value list) -> value list -> value list
