val mark : int array -> int array
val mark_where : (int -> bool) -> int array -> int array
val is_marked : int array -> int -> bool
val mark_prim : 'a -> 'a
val prim_marked : 'a -> bool

val mark_triples :
  (int -> int -> bool) -> (int * int * int) array -> (int * int * int) array

val triple_marked : (int * int * int) array -> int -> int -> bool
