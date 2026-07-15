val fftconvolve :
  ?mode:string -> ?axes:int array -> Types.value -> Types.value -> Types.value

val stft :
  ?fs:float ->
  ?window:string ->
  ?nperseg:int ->
  ?noverlap:int ->
  ?nfft:int ->
  ?detrend:string ->
  ?return_onesided:bool ->
  ?boundary:string option ->
  ?padded:bool ->
  ?axis:int ->
  Types.value ->
  Types.value * Types.value * Types.value

val csd :
  ?fs:float ->
  ?window:string ->
  ?nperseg:int ->
  ?noverlap:int ->
  ?nfft:int ->
  ?detrend:string ->
  ?return_onesided:bool ->
  ?scaling:string ->
  ?axis:int ->
  ?average:string ->
  Types.value ->
  Types.value option ->
  Types.value * Types.value

val welch :
  ?fs:float ->
  ?window:string ->
  ?nperseg:int ->
  ?noverlap:int ->
  ?nfft:int ->
  ?detrend:string ->
  ?return_onesided:bool ->
  ?scaling:string ->
  ?axis:int ->
  ?average:string ->
  Types.value ->
  Types.value * Types.value

val istft :
  ?fs:float ->
  ?window:string ->
  ?nperseg:int ->
  ?noverlap:int ->
  ?nfft:int ->
  ?input_onesided:bool ->
  ?boundary:bool ->
  ?time_axis:int ->
  ?freq_axis:int ->
  Types.value ->
  Types.value * Types.value
