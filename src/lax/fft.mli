open Types

val fft_impl : fft_kind -> int array -> Ndarray.t -> Ndarray.t
val fft_aval : fft_kind -> int array -> aval -> aval
