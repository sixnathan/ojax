open Types

type prng_impl = {
  key_shape : int array;
  seed : Ndarray.t -> Ndarray.t;
  split : Ndarray.t -> int array -> Ndarray.t;
  random_bits : Ndarray.t -> int -> int array -> Ndarray.t;
  fold_in : Ndarray.t -> Ndarray.t -> Ndarray.t;
  name : string;
  tag : string;
}

val register_prng : prng_impl -> unit
val seed_with_impl : prng_impl -> Ndarray.t -> Ndarray.t
val iota_2x32_nd : int array -> Ndarray.t * Ndarray.t
val iota_2x32_shape : int array -> value * value
val random_seed : value -> value
val random_split : value -> int array -> value
val random_fold_in : value -> value -> value
val random_bits : value -> bit_width:int -> shape:int array -> value
val random_wrap : value -> value
val random_unwrap : value -> value
val install : unit -> unit
