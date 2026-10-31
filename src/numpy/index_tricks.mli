type item =
  | Int of int
  | Slice of { start : int option; stop : int option; step : int option }
  | Newaxis
  | Ellipsis

type expr = Item of item | Tuple of item list

val slice : ?start:int -> ?stop:int -> ?step:int -> unit -> item
val s_ : item list -> expr
val index_exp : item list -> item list
val items : expr -> item list
val apply : Types.value -> item list -> Types.value
val get : Types.value -> expr -> Types.value
