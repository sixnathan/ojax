type 'a flag

val value : 'a flag -> 'a
val name : 'a flag -> string
val set : 'a flag -> 'a -> unit
val with_value : 'a flag -> 'a -> (unit -> 'b) -> 'b
val enable_x64 : bool flag
val x64_enabled : unit -> bool
val disable_jit : bool flag
val debug_nans : bool flag
val debug_infs : bool flag
val log_compiles : bool flag
val no_tracing : bool flag
val no_execution : bool flag
val numpy_rank_promotion : string flag
