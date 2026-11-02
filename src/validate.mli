val canonicalize_axis : int -> int -> int

val canonicalize_axis_tuple :
  ?allow_duplicate:bool -> int array -> int -> int array

val tuple_index : int array -> int -> int
val list_assign_index : int -> int -> int
val python_tuple : int array -> string
val slice_indices : ?start:int -> ?stop:int -> int -> int * int
val require_ndim_at_least : int -> min:int -> string -> unit
val require_ndim : int -> ndim:int -> string -> unit
val broadcast_shapes : int array list -> int array
val lax_broadcast_shapes : name:string -> int array list -> int array
val require_same_shape : int array -> int array -> string -> unit
val py_mod : int -> int -> int
val py_floordiv : int -> int -> int
val python_list : int list -> string
val check_shapelike : name:string -> arg:string -> int array -> unit
val aval_short : dtype_name:string -> weak:bool -> int array -> string
val assertion : ?message:string -> bool -> unit
