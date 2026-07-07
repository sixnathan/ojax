type 'a t =
  | Leaf of 'a
  | List of 'a t list
  | Tuple of 'a t list
  | Dict of (string * 'a t) list
  | Null

type treedef =
  | Leaf_def
  | List_def of treedef list
  | Tuple_def of treedef list
  | Dict_def of (string * treedef) list
  | Null_def

val tree_flatten : 'a t -> 'a list * treedef
val tree_leaves : 'a t -> 'a list
val tree_structure : 'a t -> treedef
val tree_unflatten : treedef -> 'b list -> 'b t
val tree_map : ('a -> 'b) -> 'a t -> 'b t
