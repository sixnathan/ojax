type t

val compile : Client.t -> string -> t
val execute : t -> Buffer.t array -> Buffer.t array
val num_outputs : t -> int
val destroy : t -> unit
