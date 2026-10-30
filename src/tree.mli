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
