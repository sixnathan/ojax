val full : ?dtype:Dtype.t -> int array -> float -> Types.value
val zeros : ?dtype:Dtype.t -> int array -> Types.value
val ones : ?dtype:Dtype.t -> int array -> Types.value
val empty : ?dtype:Dtype.t -> int array -> Types.value

val full_like :
  ?dtype:Dtype.t -> ?shape:int array -> Types.value -> float -> Types.value

val zeros_like :
  ?dtype:Dtype.t -> ?shape:int array -> Types.value -> Types.value

val ones_like : ?dtype:Dtype.t -> ?shape:int array -> Types.value -> Types.value

val empty_like :
  ?dtype:Dtype.t -> ?shape:int array -> Types.value -> Types.value

val linspace :
  ?num:int -> ?endpoint:bool -> ?dtype:Dtype.t -> float -> float -> Types.value

val logspace :
  ?num:int ->
  ?endpoint:bool ->
  ?base:float ->
  ?dtype:Dtype.t ->
  float ->
  float ->
  Types.value

val geomspace :
  ?num:int -> ?endpoint:bool -> ?dtype:Dtype.t -> float -> float -> Types.value
