val check_where : string -> Types.value option -> Types.value option

val sum :
  ?dtype:Dtype.t ->
  ?axis:int array ->
  ?keepdims:bool ->
  ?initial:Types.value ->
  ?where:Types.value ->
  ?promote_integers:bool ->
  Types.value ->
  Types.value

val prod :
  ?dtype:Dtype.t ->
  ?axis:int array ->
  ?keepdims:bool ->
  ?initial:Types.value ->
  ?where:Types.value ->
  ?promote_integers:bool ->
  Types.value ->
  Types.value

val max :
  ?axis:int array ->
  ?keepdims:bool ->
  ?initial:Types.value ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val min :
  ?axis:int array ->
  ?keepdims:bool ->
  ?initial:Types.value ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val amax :
  ?axis:int array ->
  ?keepdims:bool ->
  ?initial:Types.value ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val amin :
  ?axis:int array ->
  ?keepdims:bool ->
  ?initial:Types.value ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val all :
  ?axis:int array ->
  ?keepdims:bool ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val any :
  ?axis:int array ->
  ?keepdims:bool ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val mean :
  ?dtype:Dtype.t ->
  ?axis:int array ->
  ?keepdims:bool ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val var :
  ?dtype:Dtype.t ->
  ?axis:int array ->
  ?keepdims:bool ->
  ?ddof:int ->
  ?where:Types.value ->
  ?mean:Types.value ->
  ?correction:float ->
  Types.value ->
  Types.value

val std :
  ?dtype:Dtype.t ->
  ?axis:int array ->
  ?keepdims:bool ->
  ?ddof:int ->
  ?ddof_value:Types.value ->
  ?where:Types.value ->
  ?mean:Types.value ->
  ?correction:float ->
  Types.value ->
  Types.value

val ptp : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value

val count_nonzero :
  ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value

val average :
  ?axis:int array ->
  ?keepdims:bool ->
  ?weights:Types.value ->
  Types.value ->
  Types.value

val average_returned :
  ?axis:int array ->
  ?keepdims:bool ->
  ?weights:Types.value ->
  Types.value ->
  Types.value * Types.value

val nansum :
  ?dtype:Dtype.t ->
  ?axis:int array ->
  ?keepdims:bool ->
  ?initial:Types.value ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val nanprod :
  ?dtype:Dtype.t ->
  ?axis:int array ->
  ?keepdims:bool ->
  ?initial:Types.value ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val nanmax :
  ?axis:int array ->
  ?keepdims:bool ->
  ?initial:Types.value ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val nanmin :
  ?axis:int array ->
  ?keepdims:bool ->
  ?initial:Types.value ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val nanmean :
  ?dtype:Dtype.t ->
  ?axis:int array ->
  ?keepdims:bool ->
  ?where:Types.value ->
  Types.value ->
  Types.value

val nanvar :
  ?dtype:Dtype.t ->
  ?axis:int array ->
  ?keepdims:bool ->
  ?ddof:int ->
  ?ddof_value:Types.value ->
  ?where:Types.value ->
  ?mean:Types.value ->
  Types.value ->
  Types.value

val nanstd :
  ?dtype:Dtype.t ->
  ?axis:int array ->
  ?keepdims:bool ->
  ?ddof:int ->
  ?ddof_value:Types.value ->
  ?where:Types.value ->
  ?mean:Types.value ->
  Types.value ->
  Types.value

val cumsum : ?dtype:Dtype.t -> ?axis:int -> Types.value -> Types.value
val cumsum_by_keyword : ?axis:int -> Types.value -> Types.value
val cumprod : ?dtype:Dtype.t -> ?axis:int -> Types.value -> Types.value
val nancumsum : ?dtype:Dtype.t -> ?axis:int -> Types.value -> Types.value
val nancumprod : ?dtype:Dtype.t -> ?axis:int -> Types.value -> Types.value

val cumulative_sum :
  ?dtype:Dtype.t ->
  ?axis:int ->
  ?include_initial:bool ->
  Types.value ->
  Types.value

val cumulative_prod :
  ?dtype:Dtype.t ->
  ?axis:int ->
  ?include_initial:bool ->
  Types.value ->
  Types.value

val quantile :
  ?axis:int ->
  ?keepdims:bool ->
  ?method_:string ->
  ?weights:Types.value ->
  Types.value ->
  Types.value ->
  Types.value

val nanquantile :
  ?axis:int ->
  ?keepdims:bool ->
  ?method_:string ->
  ?weights:Types.value ->
  Types.value ->
  Types.value ->
  Types.value

val percentile :
  ?axis:int ->
  ?keepdims:bool ->
  ?method_:string ->
  ?weights:Types.value ->
  Types.value ->
  Types.value ->
  Types.value

val nanpercentile :
  ?axis:int ->
  ?keepdims:bool ->
  ?method_:string ->
  ?weights:Types.value ->
  Types.value ->
  Types.value ->
  Types.value

val median : ?axis:int -> ?keepdims:bool -> Types.value -> Types.value
val nanmedian : ?axis:int -> ?keepdims:bool -> Types.value -> Types.value

val take_along_axis_hook :
  (axis:int -> Types.value -> Types.value -> Types.value) ref
