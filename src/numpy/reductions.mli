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
