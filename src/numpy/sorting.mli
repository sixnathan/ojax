val sort :
  ?axis:int option ->
  ?stable:bool ->
  ?descending:bool ->
  Types.value ->
  Types.value

val argsort :
  ?axis:int option ->
  ?stable:bool ->
  ?descending:bool ->
  ?dtype:Dtype.t ->
  Types.value ->
  Types.value

val lexsort : ?axis:int -> Types.value list -> Types.value
val partition : ?axis:int -> Types.value -> kth:int -> Types.value
