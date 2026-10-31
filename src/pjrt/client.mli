type t
type device = { id : int; process_index : int; kind : string }

type option_value =
  | String of string
  | Int64 of int64
  | Float of float
  | Bool of bool

val create :
  ?options:(string * option_value) list -> ?cpu_devices:int -> Abi.plugin -> t

val destroy : t -> unit
val platform_name : t -> string
val platform : t -> string
val platform_version : t -> string
val device_count : t -> int
val device_kind : t -> string
val devices : t -> device array
