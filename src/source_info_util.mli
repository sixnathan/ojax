type name_stack

val empty_name_stack : name_stack
val new_name_stack : string -> name_stack
val extend : name_stack -> string -> name_stack
val transform : name_stack -> string -> name_stack
val name_stack_str : name_stack -> string

type t

val new_source_info : unit -> t
val current : unit -> t
val name_stack : t -> name_stack
val current_name_stack : unit -> name_stack
val summarize : t -> string
val register_exclusion : string -> unit
val api_boundary : ('a -> 'b) -> 'a -> 'b
