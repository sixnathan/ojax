type t

type compiler_option =
  | Option_string of string
  | Option_bool of bool
  | Option_int of int
  | Option_float of float

val compile :
  ?options:(string * compiler_option) list -> Client.t -> string -> t

val compile_placed :
  Client.t -> string -> devices:int array -> propagate_outputs:bool -> t

val execute : t -> Buffer.t array -> Buffer.t array
val execute_devices : t -> Buffer.raw array array -> Buffer.raw array array

type output_sharding =
  | Replicated
  | Tiled of { dims : int array; grid : int array }

val output_shardings : t -> output_sharding list
val num_outputs : t -> int
val destroy : t -> unit
