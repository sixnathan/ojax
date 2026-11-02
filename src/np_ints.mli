val mark : int array -> int array
val mark_where : (int -> bool) -> int array -> int array
val is_marked : int array -> int -> bool
val mark_prim : 'a -> 'a
val prim_marked : 'a -> bool

val mark_triples :
  (int -> int -> bool) -> (int * int * int) array -> (int * int * int) array

val triple_marked : (int * int * int) array -> int -> int -> bool
val mark_scatter : 'a -> 'a
val scatter_marked : 'a -> bool
val mark_gather_fill : 'a -> string -> float option -> 'a
val gather_fill_of : 'a -> (string * float option) option
val mark_gather_clip : sorted:bool -> 'a -> 'a
val gather_clip_of : 'a -> bool option
val mark_gather_flags : sorted:bool -> unique:bool -> 'a -> 'a
val gather_flags_of : 'a -> (bool * bool) option
val mark_scatter_flags : sorted:bool -> unique:bool -> 'a -> 'a
val scatter_flags_of : 'a -> (bool * bool) option
