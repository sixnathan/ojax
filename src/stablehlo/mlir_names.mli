val symbol_name : string -> string

val render :
  ?num_partitions:int ->
  ?module_attributes:string list ->
  ?result_paths:string list ->
  module_name:string ->
  string ->
  string

val hoist_constants : string -> string
val renumber_callbacks : string -> string
val add_platform_index_arg : arg:int -> string -> string
val drop_empty_backend_config : string -> string
val mentions_host_callback : string -> bool
