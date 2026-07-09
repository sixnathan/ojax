open Types

val out_shape : int array -> window_dims -> int array
val reduce_window_sum : window_dims -> Ndarray.t -> Ndarray.t
val reduce_window_max : window_dims -> Ndarray.t -> Ndarray.t
val reduce_window_min : window_dims -> Ndarray.t -> Ndarray.t

val reduce_window_general :
  reducer:(float -> float -> float) ->
  init:float ->
  window_dims ->
  Ndarray.t ->
  Ndarray.t

val select_and_gather_add :
  window_select -> window_dims -> Ndarray.t -> Ndarray.t -> Ndarray.t

val select_and_scatter_add :
  window_select -> window_dims -> Ndarray.t -> Ndarray.t -> Ndarray.t
