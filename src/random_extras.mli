open Types

val maxwell : value -> shape:int array -> value

val double_sided_maxwell :
  value -> loc:float -> scale:float -> shape:int array -> value

val ball : ?p:float -> value -> d:int -> shape:int array -> value
val orthogonal : ?m:int -> value -> n:int -> shape:int array -> value

val multivariate_normal :
  ?shape:int array ->
  ?method_:string ->
  value ->
  mean:value ->
  cov:value ->
  value
