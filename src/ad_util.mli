open Types

type zero = { z_aval : aval }

val zeros_like_aval : aval -> value
val zeros_like_value : value -> value
val symbolic_zero_of_aval : aval -> value
val symbolic_zero_like_value : value -> value
val is_symbolic_zero : value -> bool
val clear_symbolic_zeros : unit -> unit
val instantiate : zero -> value
val add_jaxvals : value -> value -> value
