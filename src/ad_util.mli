open Types

type zero = { z_aval : aval }

val zeros_like_aval : aval -> value
val zeros_like_value : value -> value
val instantiate : zero -> value
val add_jaxvals : value -> value -> value
