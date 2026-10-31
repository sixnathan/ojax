type t

val compile : Client.t -> string -> t

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
