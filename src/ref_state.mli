open Types

type kind = Default | No_grad_no_remat

type cell = {
  cell_id : int;
  mutable content : value;
  cell_aval : aval;
  level : int;
  depth : int;
  kind : kind;
  mutable consumed : bool;
}

type mode = Bare | Lift of (aval -> value) | Forbid of string
type scope

val make_cell : kind:kind -> value -> cell
val in_trace : unit -> bool
val with_scope : mode -> (scope -> 'a) -> 'a
val lifted : scope -> cell list
val contents : cell list -> value list
val pin : cell list -> value list -> unit
val read : ?operation:string -> cell -> value
val write : ?operation:string -> cell -> value -> unit
val consume : operation:string -> cell -> unit
val dtype_name : Dtype.t -> string
val aval_string : aval -> string
val cell_string : cell -> string
