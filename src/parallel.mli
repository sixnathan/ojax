open Types

val psum : value -> axis_name:string -> value
val pmax : value -> axis_name:string -> value
val pmin : value -> axis_name:string -> value
val pmean : value -> axis_name:string -> value
val axis_size : axis_name:string -> int
val axis_index : axis_name:string -> value
val all_gather : ?axis:int -> value -> axis_name:string -> value
val pbroadcast : ?source:int -> value -> axis_name:string -> value
val ppermute : value -> axis_name:string -> perm:(int * int) list -> value

val all_to_all :
  ?tiled:bool ->
  value ->
  axis_name:string ->
  split_axis:int ->
  concat_axis:int ->
  value

val pswapaxes : value -> axis_name:string -> axis:int -> value

val psum_scatter :
  ?scatter_dimension:int -> ?tiled:bool -> value -> axis_name:string -> value

val pshuffle : value -> axis_name:string -> perm:int list -> value
