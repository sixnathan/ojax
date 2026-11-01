type index =
  | Int of int
  | Range of { start : int option; stop : int option }
  | Arr of Types.value

val take : Types.value -> index list -> Types.value
val take_clip : Types.value -> index list -> Types.value
val set : Types.value -> index list -> Types.value -> Types.value
