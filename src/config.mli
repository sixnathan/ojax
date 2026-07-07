type 'a flag

val value : 'a flag -> 'a
val name : 'a flag -> string
val set : 'a flag -> 'a -> unit
val with_value : 'a flag -> 'a -> (unit -> 'b) -> 'b
val enable_x64 : bool flag
val x64_enabled : unit -> bool
