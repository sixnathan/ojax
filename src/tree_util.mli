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
val keystr : ?simple:bool -> ?separator:string -> key_path -> string
val tree_map_is_leaf : is_leaf:('a t -> bool) -> ('a t -> 'b t) -> 'a t -> 'b t
val tree_leaves_is_leaf : is_leaf:('a t -> bool) -> 'a t -> 'a t list
val tree_structure_is_leaf : is_leaf:('a t -> bool) -> 'a t -> treedef
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
val tree_broadcast_is_leaf : is_leaf:('a t -> bool) -> 'a t -> 'b t -> 'a t

val tree_flatten_is_leaf :
  is_leaf:('a t -> bool) -> 'a t -> 'a t list * treedef

val tree_leaves_with_path_is_leaf :
  ?is_leaf:('a t -> bool) ->
  ?is_leaf_takes_path:(key_path -> 'a t -> bool) ->
  'a t ->
  (key_path * 'a t) list

val tree_flatten_with_path_is_leaf :
  ?is_leaf:('a t -> bool) ->
  ?is_leaf_takes_path:(key_path -> 'a t -> bool) ->
  'a t ->
  (key_path * 'a t) list * treedef

val tree_map_with_path_is_leaf :
  ?is_leaf:('a t -> bool) ->
  ?is_leaf_takes_path:(key_path -> 'a t -> bool) ->
  (key_path -> 'a t -> 'b t) ->
  'a t ->
  'b t

val tree_reduce_is_leaf :
  is_leaf:('a t -> bool) -> ('acc -> 'a t -> 'acc) -> 'acc -> 'a t -> 'acc

val tree_all_is_leaf :
  is_leaf:('a t -> bool) -> ('a t -> bool) -> 'a t -> bool

val tree_reduce_no_init : ('a -> 'a -> 'a) -> 'a t -> 'a

val tree_reduce_associative_is_leaf :
  is_leaf:('a t -> bool) ->
  ('a t -> 'a t -> 'a t) ->
  ?identity:'a t ->
  'a t ->
  'a t

val all_leaves_is_leaf : is_leaf:('a t -> bool) -> 'a t list -> bool

type py_tree_def = treedef

val num_nodes : treedef -> int
val treedef_to_string : treedef -> string

type registry

val default_registry : registry
val registry_mem : registry -> string -> bool

type ('n, 'a) node_class

val register_pytree_node :
  ?flatten_with_keys:('n -> (key_entry * 'a t) list * string) ->
  string ->
  flatten:('n -> 'a t list * string) ->
  unflatten:(string -> 'a t list -> 'n) ->
  ('n, 'a) node_class

val register_pytree_node_class :
  string ->
  tree_flatten:('n -> 'a t list * string) ->
  tree_unflatten:(string -> 'a t list -> 'n) ->
  ('n, 'a) node_class

val register_pytree_with_keys :
  ?flatten_func:('n -> 'a t list * string) ->
  string ->
  flatten_with_keys:('n -> (key_entry * 'a t) list * string) ->
  unflatten:(string -> 'a t list -> 'n) ->
  ('n, 'a) node_class

val register_pytree_with_keys_class :
  string ->
  tree_flatten_with_keys:('n -> (key_entry * 'a t) list * string) ->
  tree_unflatten:(string -> 'a t list -> 'n) ->
  ('n, 'a) node_class

val register_static :
  string ->
  to_aux:('n -> string) ->
  of_aux:(string -> 'n) ->
  ('n, 'a) node_class

type field_kind = Data_field | Static_field

val static : unit -> field_kind

val register_dataclass :
  ?meta_fields:string list ->
  ?drop_fields:string list ->
  string ->
  data_fields:string list ->
  get:('n -> string -> 'a t) ->
  meta:('n -> string) ->
  make:(string -> 'a t list -> 'n) ->
  ('n, 'a) node_class

val node_to_tree : ('n, 'a) node_class -> 'n -> 'a t
val node_of_tree : ('n, 'a) node_class -> 'a t -> 'n

type ('a, 'b) partial

val partial : ?name:string -> ('a t list -> 'b) -> 'a t list -> ('a, 'b) partial
val partial_call : ('a, 'b) partial -> 'a t list -> 'b
val partial_to_tree : ('a, 'b) partial -> 'a t
val partial_of_tree : ('a, 'b) partial -> 'a t -> ('a, 'b) partial
