open Types

val convolve : ?mode:string -> ?method_:string -> value -> value -> value

val convolve2d :
  ?mode:string ->
  ?boundary:string ->
  ?fillvalue:float ->
  value ->
  value ->
  value

val correlate : ?mode:string -> ?method_:string -> value -> value -> value

val correlate2d :
  ?mode:string ->
  ?boundary:string ->
  ?fillvalue:float ->
  value ->
  value ->
  value

val fftconvolve : ?mode:string -> ?axes:int list -> value -> value -> value
val detrend : ?axis:int -> ?type_:string -> ?bp:int list -> value -> value

val stft :
  ?fs:float ->
  ?window:value ->
  ?nperseg:int ->
  ?noverlap:int ->
  ?nfft:int ->
  ?detrend:string ->
  ?return_onesided:bool ->
  ?boundary:string option ->
  ?padded:bool ->
  ?axis:int ->
  value ->
  value * value * value

val csd :
  ?fs:float ->
  ?window:value ->
  ?nperseg:int ->
  ?noverlap:int ->
  ?nfft:int ->
  ?detrend:string option ->
  ?return_onesided:bool ->
  ?scaling:string ->
  ?axis:int ->
  ?average:string ->
  value ->
  value option ->
  value * value

val welch :
  ?fs:float ->
  ?window:value ->
  ?nperseg:int ->
  ?noverlap:int ->
  ?nfft:int ->
  ?detrend:string option ->
  ?return_onesided:bool ->
  ?scaling:string ->
  ?axis:int ->
  ?average:string ->
  value ->
  value * value

val istft :
  ?fs:float ->
  ?window:value ->
  ?nperseg:int ->
  ?noverlap:int ->
  ?nfft:int ->
  ?input_onesided:bool ->
  ?boundary:bool ->
  ?time_axis:int ->
  ?freq_axis:int ->
  value ->
  value * value
