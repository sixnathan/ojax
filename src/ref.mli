open Types

type t
type kind = Ref_state.kind = Default | No_grad_no_remat

type index_item =
  | At of int
  | Range of int option * int option
  | All
  | Dynamic of value

val new_ref : ?kind:kind -> value -> t
val empty_ref : aval -> t
val free_ref : t -> unit
val freeze : t -> value
val get : ?index:index_item list -> t -> value
val set : ?index:index_item list -> t -> value -> unit
val swap : ?index:index_item list -> t -> value -> value
val addupdate : ?index:index_item list -> t -> value -> unit
val aval : t -> aval
val shape : t -> int array
val dtype : t -> Dtype.t
val ndim : t -> int
val size : t -> int
val to_string : t -> string
