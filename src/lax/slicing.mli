val slice_shape :
  int array -> int array -> int array option -> int array -> int array

val slice_impl :
  int array -> int array -> int array option -> Ndarray.t -> Ndarray.t

val dynamic_slice_impl : int array -> Ndarray.t list -> Ndarray.t
val dynamic_update_slice_impl : Ndarray.t list -> Ndarray.t

val gather_shape :
  Types.gather_dims -> int array -> int array -> int array -> int array

val gather_impl :
  Types.gather_dims -> int array -> Ndarray.t -> Ndarray.t -> Ndarray.t

val scatter_impl :
  (float -> float -> float) ->
  Types.scatter_dims ->
  Ndarray.t ->
  Ndarray.t ->
  Ndarray.t ->
  Ndarray.t

val scatter_combiner : Types.primitive -> float -> float -> float
