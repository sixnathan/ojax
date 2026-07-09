val check_where : string -> Types.value option -> Types.value option
val sum : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val prod : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val max : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val min : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val amax : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val amin : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val all : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val any : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val mean : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value

val var :
  ?axis:int array -> ?keepdims:bool -> ?ddof:int -> Types.value -> Types.value

val std :
  ?axis:int array -> ?keepdims:bool -> ?ddof:int -> Types.value -> Types.value

val ptp : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value

val count_nonzero :
  ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value

val average :
  ?axis:int array ->
  ?keepdims:bool ->
  ?weights:Types.value ->
  Types.value ->
  Types.value

val nansum : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val nanprod : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val nanmax : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val nanmin : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val nanmean : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value

val nanvar :
  ?axis:int array -> ?keepdims:bool -> ?ddof:int -> Types.value -> Types.value

val nanstd :
  ?axis:int array -> ?keepdims:bool -> ?ddof:int -> Types.value -> Types.value

val cumsum : ?axis:int -> Types.value -> Types.value
val cumprod : ?axis:int -> Types.value -> Types.value
val nancumsum : ?axis:int -> Types.value -> Types.value
val nancumprod : ?axis:int -> Types.value -> Types.value

val cumulative_sum :
  ?axis:int -> ?include_initial:bool -> Types.value -> Types.value

val cumulative_prod :
  ?axis:int -> ?include_initial:bool -> Types.value -> Types.value

val quantile :
  ?axis:int ->
  ?keepdims:bool ->
  ?method_:string ->
  Types.value ->
  Types.value ->
  Types.value

val nanquantile :
  ?axis:int ->
  ?keepdims:bool ->
  ?method_:string ->
  Types.value ->
  Types.value ->
  Types.value

val percentile :
  ?axis:int ->
  ?keepdims:bool ->
  ?method_:string ->
  Types.value ->
  Types.value ->
  Types.value

val nanpercentile :
  ?axis:int ->
  ?keepdims:bool ->
  ?method_:string ->
  Types.value ->
  Types.value ->
  Types.value

val median : ?axis:int -> ?keepdims:bool -> Types.value -> Types.value
val nanmedian : ?axis:int -> ?keepdims:bool -> Types.value -> Types.value
