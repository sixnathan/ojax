open Types

type vmap_rule =
  axis_size:int -> in_batched:bool list -> value list -> value list * bool list

val custom_vmap :
  primal:(value list -> value list) ->
  vmap_rule:vmap_rule ->
  value list ->
  value list

val sequential_vmap : (value list -> value list) -> value list -> value list
