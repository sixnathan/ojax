type mxstep = Mx_int of int | Mx_float of float

val odeint :
  ?rtol:float ->
  ?atol:float ->
  ?mxstep:mxstep ->
  ?hmax:float ->
  (Types.value Tree_util.t ->
  Types.value ->
  Types.value list ->
  Types.value Tree_util.t) ->
  Types.value Tree_util.t ->
  Types.value ->
  Types.value list ->
  Types.value Tree_util.t
