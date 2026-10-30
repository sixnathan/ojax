open Types

val convolve : ?mode:string -> ?method_:string -> value -> value -> value
val convolve2d : ?mode:string -> value -> value -> value
val correlate : ?mode:string -> ?method_:string -> value -> value -> value
val correlate2d : ?mode:string -> value -> value -> value
val fftconvolve : ?mode:string -> value -> value -> value
val detrend : ?axis:int -> ?type_:string -> value -> value
