open Types

val psum :
  ?axis_index_groups:int list list -> value -> axis_name:string -> value

val pmax :
  ?axis_index_groups:int list list -> value -> axis_name:string -> value

val pmin :
  ?axis_index_groups:int list list -> value -> axis_name:string -> value

val pmean :
  ?axis_index_groups:int list list -> value -> axis_name:string -> value

val axis_size : axis_name:string -> int
val psend : value -> axis_name:string -> perm:(int * int) list -> value

val precv :
  value ->
  out_shape:Shape_dtype_struct.t ->
  axis_name:string ->
  perm:(int * int) list ->
  value

val axis_index : axis_name:string -> value

val all_gather :
  ?axis_index_groups:int list list ->
  ?axis:int ->
  ?tiled:bool ->
  ?to_:string ->
  value ->
  axis_name:string ->
  value

val pbroadcast : ?source:int -> value -> axis_name:string -> value
val ppermute : value -> axis_name:string -> perm:(int * int) list -> value

val all_to_all :
  ?axis_index_groups:int list list ->
  ?tiled:bool ->
  value ->
  axis_name:string ->
  split_axis:int ->
  concat_axis:int ->
  value

val pswapaxes :
  ?axis_index_groups:int list list ->
  value ->
  axis_name:string ->
  axis:int ->
  value

val psum_scatter :
  ?scatter_dimension:int ->
  ?axis_index_groups:int list list ->
  ?tiled:bool ->
  value ->
  axis_name:string ->
  value

val pshuffle : value -> axis_name:string -> perm:int list -> value

val ragged_all_to_all :
  ?axis_index_groups:int list list ->
  value ->
  value ->
  value ->
  value ->
  value ->
  value ->
  axis_name:string ->
  value
