val symbol_name : string -> string

val render :
  ?num_partitions:int ->
  ?module_attributes:string list ->
  ?result_names:string list ->
  module_name:string ->
  string ->
  string

val hoist_constants : string -> string
val renumber_callbacks : string -> string
