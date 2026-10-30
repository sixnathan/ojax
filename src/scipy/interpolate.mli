open Types

type t

val regular_grid_interpolator :
  ?method_:string ->
  ?bounds_error:bool ->
  ?fill_value:float option ->
  value list ->
  value ->
  t

val evaluate : t -> value -> value
