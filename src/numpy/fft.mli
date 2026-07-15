val fft : ?n:int -> ?axis:int -> ?norm:string -> Types.value -> Types.value
val ifft : ?n:int -> ?axis:int -> ?norm:string -> Types.value -> Types.value
val rfft : ?n:int -> ?axis:int -> ?norm:string -> Types.value -> Types.value
val irfft : ?n:int -> ?axis:int -> ?norm:string -> Types.value -> Types.value

val fftn :
  ?s:int array -> ?axes:int array -> ?norm:string -> Types.value -> Types.value

val ifftn :
  ?s:int array -> ?axes:int array -> ?norm:string -> Types.value -> Types.value

val rfftn :
  ?s:int array -> ?axes:int array -> ?norm:string -> Types.value -> Types.value

val irfftn :
  ?s:int array -> ?axes:int array -> ?norm:string -> Types.value -> Types.value

val fft2 :
  ?s:int array -> ?axes:int array -> ?norm:string -> Types.value -> Types.value

val fftfreq : ?d:float -> ?dtype:Dtype.t -> int -> Types.value
val rfftfreq : ?d:float -> ?dtype:Dtype.t -> int -> Types.value
val fftshift : ?axes:int array -> Types.value -> Types.value
val ifftshift : ?axes:int array -> Types.value -> Types.value
