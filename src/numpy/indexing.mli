val take :
  ?axis:int ->
  ?mode:string ->
  ?fill_value:float ->
  Types.value ->
  Types.value ->
  Types.value

val take_along_axis :
  ?axis:int ->
  ?mode:string ->
  ?fill_value:float ->
  Types.value ->
  Types.value ->
  Types.value

val put :
  ?mode:string -> Types.value -> Types.value -> Types.value -> Types.value

val put_along_axis :
  ?axis:int ->
  ?mode:string ->
  Types.value ->
  Types.value ->
  Types.value ->
  Types.value
