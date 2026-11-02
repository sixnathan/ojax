open Types

type field = { fname : string; conv : char option; spec : string }

val format_message : string -> value list -> string
val callback : (value list -> unit) -> value list -> unit

val print :
  ?ordered:bool -> ?kwargs:(string * value) list -> string -> value list -> unit

val print_sink : (string -> unit) ref
val parse_format : string -> (string * field option) list
val inspect_array_sharding : value -> callback:(string -> unit) -> unit

val visualize_sharding :
  ?scale:float ->
  ?min_width:int ->
  ?max_width:int ->
  int array ->
  Spmd.t ->
  unit

val visualize_array_sharding :
  ?scale:float -> ?min_width:int -> ?max_width:int -> value -> unit

val log : ?level:string -> string -> value list -> unit
val set_breakpoint_handler : (value list -> unit) -> unit
val breakpoint : value list -> unit
