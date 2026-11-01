val mark : int array -> int array
val mark_where : (int -> bool) -> int array -> int array
val is_marked : int array -> int -> bool
val mark_prim : 'a -> 'a
val prim_marked : 'a -> bool

val mark_triples_where :
  (int -> int -> bool) -> (int * int * int) array -> (int * int * int) array

val triple_marked : (int * int * int) array -> int -> int -> bool
val mark_scatter : 'a -> 'a
val scatter_marked : 'a -> bool
val mark_gather_fill : 'a -> string -> float option -> 'a
val gather_fill_of : 'a -> (string * float option) option
val mark_gather_clip : sorted:bool -> 'a -> 'a
val gather_clip_of : 'a -> bool option
