val render :
  ?num_partitions:int ->
  ?module_attributes:string list ->
  module_name:string ->
  string ->
  string

val hoist_constants : string -> string
