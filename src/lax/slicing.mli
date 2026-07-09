val slice_shape :
  int array -> int array -> int array option -> int array -> int array

val slice_impl :
  int array -> int array -> int array option -> Ndarray.t -> Ndarray.t

val dynamic_slice_impl : int array -> Ndarray.t list -> Ndarray.t
val dynamic_update_slice_impl : Ndarray.t list -> Ndarray.t
