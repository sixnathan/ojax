type index =
  | Int of int
  | Range of { start : int option; stop : int option }
  | Arr of Types.value

val take_hook :
  (string option -> Types.value -> index list -> Types.value) option ref

val set_hook :
  (Types.value -> index list -> Types.value -> Types.value) option ref

val set_stepped_hook :
  (Types.value -> start:int -> step:int -> Types.value -> Types.value) option
  ref

val take : Types.value -> index list -> Types.value
val take_clip : Types.value -> index list -> Types.value
val set : Types.value -> index list -> Types.value -> Types.value

val set_stepped :
  Types.value -> start:int -> step:int -> Types.value -> Types.value

val index_dtype : int array -> Dtype.t
