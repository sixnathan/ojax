val trace_annotation : string -> (unit -> 'a) -> 'a
val step_trace_annotation : string -> step_num:int -> (unit -> 'a) -> 'a
val annotate_function : 'a -> 'a
val start_trace : string -> unit
val stop_trace : unit -> unit
val trace : string -> (unit -> 'a) -> 'a
val device_memory_profile : unit -> string
val save_device_memory_profile : string -> unit
