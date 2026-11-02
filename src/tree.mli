open Tree_util

val all : bool t -> bool
val broadcast : 'a t -> 'b t -> 'a t
val flatten : 'a t -> 'a list * treedef
val flatten_with_path : 'a t -> (key_path * 'a) list * treedef
val leaves : 'a t -> 'a list
val leaves_with_path : 'a t -> (key_path * 'a) list
val map : ('a -> 'b) -> 'a t -> 'b t
val map2 : ('a -> 'b -> 'c) -> 'a t -> 'b t -> 'c t
val map_with_path : (key_path -> 'a -> 'b) -> 'a t -> 'b t
val reduce : ('acc -> 'a -> 'acc) -> 'acc -> 'a t -> 'acc
val reduce_associative : ('a -> 'a -> 'a) -> ?identity:'a -> 'a t -> 'a
val structure : 'a t -> treedef
val transpose : treedef -> treedef -> 'a t -> 'a t
val unflatten : treedef -> 'b list -> 'b t
val all_is_leaf : is_leaf:('a t -> bool) -> ('a t -> bool) -> 'a t -> bool
val broadcast_is_leaf : is_leaf:('a t -> bool) -> 'a t -> 'b t -> 'a t

val flatten_is_leaf :
  is_leaf:('a t -> bool) -> 'a t -> 'a t list * treedef

val flatten_with_path_is_leaf :
  ?is_leaf:('a t -> bool) ->
  ?is_leaf_takes_path:(key_path -> 'a t -> bool) ->
  'a t ->
  (key_path * 'a t) list * treedef

val leaves_is_leaf : is_leaf:('a t -> bool) -> 'a t -> 'a t list

val leaves_with_path_is_leaf :
  ?is_leaf:('a t -> bool) ->
  ?is_leaf_takes_path:(key_path -> 'a t -> bool) ->
  'a t ->
  (key_path * 'a t) list

val map_is_leaf : is_leaf:('a t -> bool) -> ('a t -> 'b t) -> 'a t -> 'b t

val map_with_path_is_leaf :
  ?is_leaf:('a t -> bool) ->
  ?is_leaf_takes_path:(key_path -> 'a t -> bool) ->
  (key_path -> 'a t -> 'b t) ->
  'a t ->
  'b t

val reduce_is_leaf :
  is_leaf:('a t -> bool) -> ('acc -> 'a t -> 'acc) -> 'acc -> 'a t -> 'acc

val reduce_no_init : ('a -> 'a -> 'a) -> 'a t -> 'a

val reduce_associative_is_leaf :
  is_leaf:('a t -> bool) ->
  ('a t -> 'a t -> 'a t) ->
  ?identity:'a t ->
  'a t ->
  'a t

val structure_is_leaf : is_leaf:('a t -> bool) -> 'a t -> treedef
val static : unit -> field_kind
