open Types

val philox2x32_prng_impl : Prng.prng_impl
val philox4x32_prng_impl : Prng.prng_impl
val philox2x32_seed : value -> value
val philox2x32_split : value -> int array -> value
val philox2x32_fold_in : value -> value -> value
val philox2x32_random_bits : value -> bit_width:int -> shape:int array -> value
val philox4x32_seed : value -> value
val philox4x32_split : value -> int array -> value
val philox4x32_fold_in : value -> value -> value
val philox4x32_random_bits : value -> bit_width:int -> shape:int array -> value
val install : unit -> unit
