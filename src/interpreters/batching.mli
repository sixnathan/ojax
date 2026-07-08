open Types

val install : unit -> unit
val mapped_aval : int -> aval -> aval
val unmapped_aval : int -> int -> aval -> aval
val move_batch_axis : int -> int option -> int -> value -> value

val vmap_flat :
  (value list -> value list) -> int option list -> value list -> value list

val vmap :
  (value list -> value list) -> int option list -> value list -> value list
