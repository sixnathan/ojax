open Types

val jit :
  string ->
  ?key:string ->
  (value list -> value list) ->
  value list ->
  value list

val jit1 : string -> ?key:string -> (value list -> value) -> value list -> value
val broadcast_to : value -> int array -> value
val broadcast_arrays : value list -> value list
val where : value -> value -> value -> value
val py_float : float -> Types.value
val py_int : int -> Types.value
