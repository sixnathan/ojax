val mark : int array -> int array
val mark_where : (int -> bool) -> int array -> int array
val is_marked : int array -> int -> bool
val mark_prim : 'a -> 'a
val prim_marked : 'a -> bool
val mark_scatter : 'a -> 'a
val scatter_marked : 'a -> bool
val mark_gather_fill : 'a -> string -> float option -> 'a
val gather_fill_of : 'a -> (string * float option) option
