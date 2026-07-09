open Types

val default_prng_impl : unit -> Prng.prng_impl
val resolve_prng_impl : string option -> Prng.prng_impl
val key_impl : value -> string
val key_dtype : string option -> Dtype.t
val key : value -> value
val key_data : value -> value
val wrap_key_data : value -> value
val clone : value -> value
val fold_in : value -> value -> value
val split : value -> int -> value
val bits : value -> shape:int array -> value
val randint : value -> shape:int array -> minval:int -> maxval:int -> value
val uniform : value -> shape:int array -> minval:float -> maxval:float -> value
val normal : value -> shape:int array -> value

val truncated_normal :
  value -> lower:float -> upper:float -> shape:int array -> value

val permutation : value -> int -> value
val choice : value -> n:int -> shape:int array -> replace:bool -> value
