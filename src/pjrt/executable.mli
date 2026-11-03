type t

type compiler_option =
  | Option_string of string
  | Option_bool of bool
  | Option_int of int
  | Option_float of float

val compile :
  ?options:(string * compiler_option) list -> Client.t -> string -> t

val optimized_text :
  ?options:(string * compiler_option) list -> Client.t -> string -> string

val unoptimized_text :
  ?options:(string * compiler_option) list -> Client.t -> string -> string

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

type memory_stats = {
  generated_code_size_in_bytes : int;
  argument_size_in_bytes : int;
  output_size_in_bytes : int;
  alias_size_in_bytes : int;
  temp_size_in_bytes : int;
  host_generated_code_size_in_bytes : int;
  host_argument_size_in_bytes : int;
  host_output_size_in_bytes : int;
  host_alias_size_in_bytes : int;
  host_temp_size_in_bytes : int;
  peak_memory_in_bytes : int;
}

val cost_analysis : t -> (string * float) list
val memory_stats : t -> memory_stats
