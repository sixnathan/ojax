open Types

val rbg_prng_impl : Prng.prng_impl
val unsafe_rbg_prng_impl : Prng.prng_impl
val rbg_seed : value -> value
val rbg_split : value -> int array -> value
val rbg_fold_in : value -> value -> value
