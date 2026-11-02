type t

val frompyfunc :
  ?identity:float ->
  ?name:string ->
  (Types.value list -> Types.value list) ->
  nin:int ->
  nout:int ->
  t

val call : t -> Types.value list -> Types.value list
val reduce : ?axis:int -> t -> Types.value -> Types.value
val accumulate : ?axis:int -> t -> Types.value -> Types.value
val outer : t -> Types.value -> Types.value -> Types.value
