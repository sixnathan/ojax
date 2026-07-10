val invert_permutation : Types.value -> Types.value
val sem : ?axis:int -> ?ddof:int -> Types.value -> Types.value

val mode :
  ?axis:int ->
  ?nan_policy:string ->
  ?keepdims:bool ->
  Types.value ->
  Types.value

val rankdata :
  ?method_:string ->
  ?axis:int ->
  ?nan_policy:string ->
  Types.value ->
  Types.value
