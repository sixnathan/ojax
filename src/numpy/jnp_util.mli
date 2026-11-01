open Types

val jit :
  string ->
  ?key:string ->
  (value list -> value list) ->
  value list ->
  value list

val jit1 : string -> ?key:string -> (value list -> value) -> value list -> value
val broadcast_to : value -> int array -> value
val promote_ranks : Types.value -> Types.value -> Types.value * Types.value
val broadcast_arrays : value list -> value list
val where : value -> value -> value -> value
val py_float : float -> value
val py_float_arg : float -> value
val py_int : int -> value
val py_complex : Complex.t -> value

val where_weak :
  ?x_weak:bool -> ?y_weak:bool -> value -> value -> value -> value
