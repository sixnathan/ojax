open Types

val format_message : string -> value list -> string
val callback : (value list -> unit) -> value list -> unit
val print : string -> value list -> unit
val inspect_array_sharding : value -> callback:(string -> unit) -> unit
val visualize_array_sharding : value -> unit
