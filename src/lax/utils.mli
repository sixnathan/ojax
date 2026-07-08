open Types

val prod : int array -> int
val strides : int array -> int array
val decode : int -> int array -> int array
val free_axes : int -> int array -> int array -> int array
val reduce_shape : int array -> int array -> int array
val dot_general_shape : dot_dims -> int array -> int array -> int array
val all_weak : aval list -> bool
