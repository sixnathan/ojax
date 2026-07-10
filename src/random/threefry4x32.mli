open Types

val threefry4x32_prng_impl : Prng.prng_impl
val threefry4x32_seed : value -> value
val threefry4x32_split : value -> int array -> value
val threefry4x32_fold_in : value -> value -> value

val threefry4x32_random_bits :
  value -> bit_width:int -> shape:int array -> value

val install : unit -> unit
