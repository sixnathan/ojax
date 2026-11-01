val mark : int array -> int array
val mark_where : (int -> bool) -> int array -> int array
val is_marked : int array -> int -> bool
val mark_prim : 'a -> 'a
val prim_marked : 'a -> bool

val mark_triples_where :
  (int -> int -> bool) -> (int * int * int) array -> (int * int * int) array

val triple_marked : (int * int * int) array -> int -> int -> bool
val mark_sorted_unique : 'a -> 'a
val is_sorted_unique : 'a -> bool
val mark_clip : 'a -> 'a
val is_clip : 'a -> bool
val mark_exact_float : float -> 'a -> 'a
val exact_float : 'a -> float option
