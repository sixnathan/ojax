type 'a t =
  | Leaf of 'a
  | List of 'a t list
  | Tuple of 'a t list
  | Dict of (string * 'a t) list
  | Custom of string * string * 'a t list
  | Null

type treedef =
  | Leaf_def
  | List_def of treedef list
  | Tuple_def of treedef list
  | Dict_def of (string * treedef) list
  | Custom_def of string * string * treedef list
  | Null_def

val tree_flatten : 'a t -> 'a list * treedef
val tree_leaves : 'a t -> 'a list
val tree_structure : 'a t -> treedef
val tree_unflatten : treedef -> 'b list -> 'b t
val tree_map : ('a -> 'b) -> 'a t -> 'b t

type key_entry =
  | Dict_key of string
  | Sequence_key of int
  | Flattened_index_key of int
  | Get_attr_key of string

type key_path = key_entry list

val register_attribute_keys : string -> string list -> unit
val tree_leaves_with_path : 'a t -> (key_path * 'a) list
val tree_flatten_with_path : 'a t -> (key_path * 'a) list * treedef
val tree_map_with_path : (key_path -> 'a -> 'b) -> 'a t -> 'b t
val keystr : key_path -> string
val tree_reduce : ('acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
val tree_reduce_associative : ('a -> 'a -> 'a) -> ?identity:'a -> 'a t -> 'a
val tree_all : bool t -> bool
val tree_map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
val treedef_children : treedef -> treedef list
val treedef_is_leaf : treedef -> bool
val treedef_tuple : treedef list -> treedef
val flatten_one_level : 'a t -> 'a t list
val flatten_one_level_with_keys : 'a t -> (key_entry * 'a t) list
val is_tree_node : 'a t -> bool
val all_leaves : 'a t list -> bool
val tree_broadcast : 'a t -> 'b t -> 'a t
val count_leaves : treedef -> int
val tree_transpose : treedef -> treedef -> 'a t -> 'a t
