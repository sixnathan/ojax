open Types

val threefry_prng_impl : Prng.prng_impl
val threefry_seed : value -> value
val threefry_2x32 : value -> value -> value
val threefry_split : value -> int array -> value
val threefry_fold_in : value -> value -> value
val threefry_random_bits : value -> int -> int array -> value
val install : unit -> unit
