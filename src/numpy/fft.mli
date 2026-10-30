open Types

val fft : ?n:int -> ?axis:int -> ?norm:string -> value -> value
val ifft : ?n:int -> ?axis:int -> ?norm:string -> value -> value
val rfft : ?n:int -> ?axis:int -> ?norm:string -> value -> value
val irfft : ?n:int -> ?axis:int -> ?norm:string -> value -> value
val hfft : ?n:int -> ?axis:int -> ?norm:string -> value -> value
val ihfft : ?n:int -> ?axis:int -> ?norm:string -> value -> value
val fftn : ?s:int list -> ?axes:int list -> ?norm:string -> value -> value
val ifftn : ?s:int list -> ?axes:int list -> ?norm:string -> value -> value
val rfftn : ?s:int list -> ?axes:int list -> ?norm:string -> value -> value
val irfftn : ?s:int list -> ?axes:int list -> ?norm:string -> value -> value
val fft2 : ?s:int list -> ?axes:int list -> ?norm:string -> value -> value
val ifft2 : ?s:int list -> ?axes:int list -> ?norm:string -> value -> value
val rfft2 : ?s:int list -> ?axes:int list -> ?norm:string -> value -> value
val irfft2 : ?s:int list -> ?axes:int list -> ?norm:string -> value -> value
val fftfreq : ?d:float -> ?dtype:Dtype.t -> int -> value
val rfftfreq : ?d:float -> ?dtype:Dtype.t -> int -> value
val fftshift : ?axes:int list -> value -> value
val ifftshift : ?axes:int list -> value -> value
