open Types

val seed : unsafe:bool -> value -> value
val is_unsafe : value -> bool
val key_data : value -> value
val wrap : unsafe:bool -> value -> value
