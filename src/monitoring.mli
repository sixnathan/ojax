type metadata = Text of string | Number of int
type kwargs = (string * metadata) list
type event_listener = string -> kwargs -> unit
type duration_listener = string -> float -> kwargs -> unit
type time_span_listener = string -> float -> float -> kwargs -> unit
type scalar_listener = string -> float -> kwargs -> unit

val register_event_listener : event_listener -> unit
val register_event_duration_secs_listener : duration_listener -> unit
val register_event_time_span_listener : time_span_listener -> unit
val register_scalar_listener : scalar_listener -> unit
val unregister_event_listener : event_listener -> unit
val unregister_event_duration_listener : duration_listener -> unit
val unregister_event_time_span_listener : time_span_listener -> unit
val unregister_scalar_listener : scalar_listener -> unit
val get_event_listeners : unit -> event_listener list
val get_event_duration_listeners : unit -> duration_listener list
val get_event_time_span_listeners : unit -> time_span_listener list
val get_scalar_listeners : unit -> scalar_listener list
val clear_event_listeners : unit -> unit
val record_event : ?kwargs:kwargs -> string -> unit
val record_event_duration_secs : ?kwargs:kwargs -> string -> float -> unit
val record_event_time_span : ?kwargs:kwargs -> string -> float -> float -> unit
val record_scalar : ?kwargs:kwargs -> string -> float -> unit
