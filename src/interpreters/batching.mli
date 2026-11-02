open Types

val install : unit -> unit
val mapped_aval : int -> aval -> aval
val unmapped_aval : int -> int -> aval -> aval
val move_batch_axis : int -> int option -> int -> value -> value

val vmap_flat :
  ?axis_name:string ->
  ?partial_permute:bool ->
  ?out_axis:int ->
  (value list -> value list) ->
  int option list ->
  value list ->
  value list

val vmap :
  ?axis_name:string ->
  (value list -> value list) ->
  int option list ->
  value list ->
  value list

val vmap_out :
  (value list -> value list) ->
  int option list ->
  value list ->
  int ->
  value list

val find_axis : string -> string * trace * int
val take_on_axis : (value -> int array -> int -> value) ref
