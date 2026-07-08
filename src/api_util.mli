val ensure_inbounds : int -> int list -> int list

val flatten_fun :
  ('a Tree_util.t list -> 'a Tree_util.t) ->
  Tree_util.treedef ->
  ('a list, 'a list) Linear_util.t * (unit -> Tree_util.treedef)

val argnums_partial :
  ('a Tree_util.t list -> 'a Tree_util.t) ->
  int list ->
  'a Tree_util.t list ->
  ('a Tree_util.t list -> 'a Tree_util.t) * 'a Tree_util.t list
