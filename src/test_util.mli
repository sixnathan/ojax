open Types

val check_jvp :
  ?eps:float -> ?tol:float -> (value list -> value list) -> value list -> unit

val check_vjp :
  ?eps:float -> ?tol:float -> (value list -> value list) -> value list -> unit

val check_grads :
  ?modes:[ `Fwd | `Rev ] list ->
  ?order:int ->
  ?eps:float ->
  ?tol:float ->
  (value list -> value list) ->
  value list ->
  unit
