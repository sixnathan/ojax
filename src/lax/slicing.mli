val slice_shape :
  int array -> int array -> int array option -> int array -> int array

val check_dynamic_slice : int array -> int array -> int array list -> unit

val check_dynamic_update_slice :
  int array -> int array -> int array list -> unit

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

val gather_fill :
  ?flag:(Types.primitive -> Types.primitive) ->
  ?fill_value:float ->
  Types.value ->
  Types.value ->
  Types.gather_dims ->
  int array ->
  Types.value

val clamp_scatter_indices :
  Types.value -> Types.value -> Types.value -> Types.scatter_dims -> Types.value

val gather_fill_lowered :
  Types.primitive -> Types.value -> Types.value -> Types.value
