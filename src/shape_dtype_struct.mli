open Types

type t = aval

val make : ?weak_type:bool -> int array -> Dtype.t -> t
val of_value : value -> t
val shape : t -> int array
val dtype : t -> Dtype.t
val weak_type : t -> bool
val ndim : t -> int
val size : t -> int
val update : ?shape:int array -> ?dtype:Dtype.t -> ?weak_type:bool -> t -> t
val equal : t -> t -> bool
val hash : t -> int
val to_string : t -> string
val traced_aval : t -> aval
