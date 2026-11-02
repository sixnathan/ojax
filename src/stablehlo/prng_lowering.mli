open Types

val threefry2x32 : value -> value -> value -> value -> value list
val iota_counters : int array -> value * value
val random_bits : shape:int array -> value -> value
val random_split : shape:int array -> value -> value
val random_seed : value -> value
val fold_in : value -> value -> value
val random_bits64 : shape:int array -> value -> value
val rbg_random_seed : value -> value
val rbg_random_split : unsafe:bool -> shape:int array -> value -> value
val rbg_fold_in : unsafe:bool -> value -> value -> value
val rbg_random_bits : bit_width:int -> shape:int array -> value -> value
