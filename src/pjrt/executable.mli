type t

type compiler_option =
  | Option_string of string
  | Option_bool of bool
  | Option_int of int
  | Option_float of float

val compile :
  ?options:(string * compiler_option) list -> Client.t -> string -> t

val execute : t -> Buffer.t array -> Buffer.t array
val num_outputs : t -> int
val destroy : t -> unit
