open Types

val prod : int array -> int
val strides : int array -> int array
val decode : int -> int array -> int array
val free_axes : int -> int array -> int array -> int array
val reduce_shape : int array -> int array -> int array
val dot_general_shape : dot_dims -> int array -> int array -> int array
val all_weak : aval list -> bool
val dilate_dim : int -> int -> int
val pad_shape : (int * int * int) array -> int array -> int array
val concatenate_shape : int -> int array list -> int array
val insert_int : int array -> int -> int -> int array
val remove_int : int array -> int -> int array
val stack_shape : int -> int -> int array -> int array
val tile_shape : int array -> int array -> int array
val transpose_shape : int array -> int array -> int array
val squeeze_shape : int array -> int array -> int array
val split_shapes : int array -> int -> int array -> int array list
val unstack_shapes : int -> int array -> int array list
val argsort : int array -> int array
