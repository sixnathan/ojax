type t

val of_floats : Dtype.t -> int array -> float array -> t
val dtype : t -> Dtype.t
val shape : t -> int array
val get_f : t -> int array -> float
val set_f : t -> int array -> float -> unit
val get_i64 : t -> int array -> int64
val map : Dtype.t -> (float -> float) -> t -> t
val map2 : Dtype.t -> (float -> float -> float) -> t -> t -> t
val fold : ('acc -> float -> 'acc) -> 'acc -> t -> 'acc
val canonicalize : Dtype.t -> t -> t
