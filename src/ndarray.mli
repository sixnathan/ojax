type t

val half_of_bits : int -> float
val bits_of_half : float -> int
val bfloat_of_bits : int -> float
val bits_of_bfloat : float -> int
val of_floats : Dtype.t -> int array -> float array -> t
val of_int64s : Dtype.t -> int array -> int64 array -> t
val of_complex : Dtype.t -> int array -> Complex.t array -> t
val dtype : t -> Dtype.t
val shape : t -> int array
val get_f : t -> int array -> float
val set_f : t -> int array -> float -> unit
val get_i64 : t -> int array -> int64
val to_int64s : t -> int64 array
val get_c : t -> int array -> Complex.t
val set_c : t -> int array -> Complex.t -> unit
val map : Dtype.t -> (float -> float) -> t -> t
val map2 : Dtype.t -> (float -> float -> float) -> t -> t -> t
val fold : ('acc -> float -> 'acc) -> 'acc -> t -> 'acc
val canonicalize : Dtype.t -> t -> t
val weak : t -> bool
val as_weak : t -> t
val python_float : t -> float option
val with_python_float : t -> float -> t
