type t
type device = { id : int; process_index : int; kind : string }

val create : ?cpu_devices:int -> Abi.plugin -> t
val destroy : t -> unit
val platform : t -> string
val devices : t -> device array
