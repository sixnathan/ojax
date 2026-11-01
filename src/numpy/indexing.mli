val take :
  ?axis:int ->
  ?mode:string ->
  ?fill_value:float ->
  Types.value ->
  Types.value ->
  Types.value

val take_index :
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

type dslice =
  | Full
  | Dslice of { start : Types.value; size : int; stride : int }

val ds : ?size:int -> ?stride:int -> Types.value option -> dslice
val dslice_index : Types.value -> dslice list -> Types.value

val dynamic_at : Types.value -> Types.value -> Types.value
val dynamic_at2 : Types.value -> Types.value -> Types.value -> Types.value

val set_at2 :
  Types.value -> Types.value -> Types.value -> Types.value -> Types.value
