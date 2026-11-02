open Types

val jit_region : string -> 'a -> value list -> (value list -> value) -> value
val float_dtype : unit -> Dtype.t
val rademacher_as : value -> shape:int array -> dt:Dtype.t -> value

val with_dtypes :
  ?float_dtype:Dtype.t -> ?int_dtype:Dtype.t -> (unit -> 'a) -> 'a

val default_prng_impl : unit -> Prng.prng_impl
val resolve_prng_impl : string option -> Prng.prng_impl
val key_impl : value -> string
val is_key : value -> bool
val key_dtype : string option -> Dtype.t
val key : ?impl:string -> ?dtype:string -> value -> value
val key_data : value -> value
val wrap_key_data : ?impl:string -> ?dtype:string -> value -> value
val clone : value -> value
val fold_in : value -> value -> value
val split : value -> int -> value
val bits : ?dtype:Dtype.t -> value -> shape:int array -> value
val check_float_dtype : string -> Dtype.t option -> unit
val check_inexact_dtype : string -> Dtype.t option -> unit

val randint :
  ?dtype:Dtype.t ->
  value ->
  shape:int array ->
  minval:int ->
  maxval:int ->
  value

val uniform :
  ?dtype:Dtype.t ->
  value ->
  shape:int array ->
  minval:float ->
  maxval:float ->
  value

val normal : ?dtype:Dtype.t -> value -> shape:int array -> value

val truncated_normal :
  ?dtype:Dtype.t ->
  value ->
  lower:float ->
  upper:float ->
  shape:int array ->
  value

val permutation : value -> int -> value
val choice : value -> a:int -> shape:int array -> replace:bool -> value
val exponential : ?dtype:Dtype.t -> value -> shape:int array -> value
val cauchy : ?dtype:Dtype.t -> value -> shape:int array -> value
val laplace : ?dtype:Dtype.t -> value -> shape:int array -> value
val logistic : ?dtype:Dtype.t -> value -> shape:int array -> value
val gumbel : ?dtype:Dtype.t -> ?mode:string -> value -> shape:int array -> value
val pareto : ?dtype:Dtype.t -> value -> shape:int array -> b:float -> value

val rayleigh :
  ?dtype:Dtype.t -> value -> shape:int array -> scale:float -> value

val weibull_min :
  ?dtype:Dtype.t ->
  value ->
  shape:int array ->
  scale:float ->
  concentration:float ->
  value

val lognormal :
  ?dtype:Dtype.t -> value -> shape:int array -> sigma:float -> value

val triangular :
  ?dtype:Dtype.t ->
  value ->
  shape:int array ->
  left:float ->
  mode:float ->
  right:float ->
  value

val wald : ?dtype:Dtype.t -> value -> shape:int array -> mean:float -> value
val geometric : ?dtype:Dtype.t -> value -> shape:int array -> p:float -> value
val bernoulli : ?mode:string -> value -> shape:int array -> p:float -> value
val rademacher : ?dtype:Dtype.t -> value -> shape:int array -> value
val categorical : value -> logits:value -> axis:int -> value
val gamma : ?dtype:Dtype.t -> value -> shape:int array -> a:float -> value
val loggamma : ?dtype:Dtype.t -> value -> shape:int array -> a:float -> value

val beta :
  ?dtype:Dtype.t -> value -> shape:int array -> a:float -> b:float -> value

val chisquare : ?dtype:Dtype.t -> value -> shape:int array -> df:float -> value
val t : ?dtype:Dtype.t -> value -> shape:int array -> df:float -> value

val f :
  ?dtype:Dtype.t ->
  value ->
  shape:int array ->
  dfnum:float ->
  dfden:float ->
  value

val generalized_normal :
  ?dtype:Dtype.t -> value -> shape:int array -> p:float -> value

val dirichlet :
  ?dtype:Dtype.t -> value -> alpha:value -> shape:int array -> value

val poisson : ?dtype:Dtype.t -> value -> shape:int array -> lam:float -> value

val binomial :
  ?dtype:Dtype.t -> value -> shape:int array -> n:float -> p:float -> value

val multinomial : value -> p:value -> n_trials:float -> value
val bernoulli_prob : value -> shape:int array -> p:value -> value

val uniform_range :
  value -> shape:int array -> minval:value -> maxval:value -> value

val random_gamma_grad : value -> value -> value
val gamma_value : value -> shape:int array -> a:value -> value
val loggamma_value : value -> shape:int array -> a:value -> value
val beta_value : value -> shape:int array -> a:value -> b:value -> value
val poisson_value : value -> shape:int array -> lam:value -> value
val bits_as : value -> shape:int array -> dtype:Dtype.t -> value
val prng_key : value -> value
val generalized_normal_value : value -> shape:int array -> p:value -> value

val permutation_array :
  ?axis:int -> ?independent:bool -> value -> value -> value

val choice_array :
  ?shape:int array ->
  ?replace:bool ->
  ?p:value ->
  ?axis:int ->
  ?mode:string ->
  value ->
  value ->
  value

val categorical_full :
  ?axis:int ->
  ?shape:int array ->
  ?replace:bool ->
  ?mode:string ->
  value ->
  value ->
  value

val multinomial_full :
  ?shape:int array ->
  ?dtype:Dtype.t ->
  ?unroll:int ->
  value ->
  n:value ->
  p:value ->
  value

val truncated_normal_int :
  value -> lower:int -> upper:int -> shape:int array -> value

val uniform_int_low : value -> shape:int array -> minval:int -> value
