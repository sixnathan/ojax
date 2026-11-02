open Types

val maxwell : ?dtype:Dtype.t -> value -> shape:int array -> value

val double_sided_maxwell :
  ?dtype:Dtype.t ->
  value ->
  loc:float ->
  scale:float ->
  shape:int array ->
  value

val ball :
  ?dtype:Dtype.t -> ?p:float -> value -> d:int -> shape:int array -> value

val orthogonal :
  ?dtype:Dtype.t -> ?m:int -> value -> n:int -> shape:int array -> value

val multivariate_normal :
  ?shape:int array ->
  ?method_:string ->
  value ->
  mean:value ->
  cov:value ->
  value
