val symbol_name : string -> string

val render :
  ?num_partitions:int ->
  ?module_attributes:string list ->
  ?result_names:string list ->
  ?arg_attribute:(int -> string option) ->
  module_name:string ->
  string ->
  string

val hoist_constants : string -> string
val renumber_callbacks : string -> string
val add_platform_index_arg : arg:int -> string -> string
val drop_empty_backend_config : string -> string
val mentions_host_callback : string -> bool
val print_empty_dynamic_slices : string -> string
