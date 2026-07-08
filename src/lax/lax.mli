open Types

val impl : primitive -> Ndarray.t list -> Ndarray.t list
val abstract_eval : primitive -> aval list -> aval list
val install : unit -> unit
