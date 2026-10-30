val register_event_listener : (string -> unit) -> unit
val register_event_duration_secs_listener : (string -> float -> unit) -> unit
val register_scalar_listener : (string -> float -> unit) -> unit
val clear_event_listeners : unit -> unit
val record_event : string -> unit
val record_event_duration_secs : string -> float -> unit
val record_scalar : string -> float -> unit
val record_event_time_span : string -> (unit -> 'a) -> 'a
