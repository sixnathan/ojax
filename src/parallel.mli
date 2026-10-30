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
