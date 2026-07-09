open Types

val default_prng_impl : unit -> Prng.prng_impl
val resolve_prng_impl : string option -> Prng.prng_impl
val key_impl : value -> string
val key_dtype : string option -> Dtype.t
val key : value -> value
val key_data : value -> value
val wrap_key_data : value -> value
val clone : value -> value
val fold_in : value -> value -> value
val split : value -> int -> value
val bits : value -> shape:int array -> value
val randint : value -> shape:int array -> minval:int -> maxval:int -> value
val uniform : value -> shape:int array -> minval:float -> maxval:float -> value
val normal : value -> shape:int array -> value

val truncated_normal :
  value -> lower:float -> upper:float -> shape:int array -> value

val permutation : value -> int -> value
val choice : value -> n:int -> shape:int array -> replace:bool -> value
val exponential : value -> shape:int array -> value
val cauchy : value -> shape:int array -> value
val laplace : value -> shape:int array -> value
val logistic : value -> shape:int array -> value
val gumbel : value -> shape:int array -> value
val pareto : value -> shape:int array -> b:float -> value
val rayleigh : value -> shape:int array -> scale:float -> value

val weibull_min :
  value -> shape:int array -> scale:float -> concentration:float -> value

val lognormal : value -> shape:int array -> sigma:float -> value

val triangular :
  value -> shape:int array -> left:float -> mode:float -> right:float -> value

val wald : value -> shape:int array -> mean:float -> value
val geometric : value -> shape:int array -> p:float -> value
val bernoulli : value -> shape:int array -> p:float -> value
val rademacher : value -> shape:int array -> value
val categorical : value -> logits:value -> axis:int -> value
val gamma : value -> shape:int array -> a:float -> value
val loggamma : value -> shape:int array -> a:float -> value
val beta : value -> shape:int array -> a:float -> b:float -> value
val chisquare : value -> shape:int array -> df:float -> value
val t : value -> shape:int array -> df:float -> value
val f : value -> shape:int array -> dfnum:float -> dfden:float -> value
val generalized_normal : value -> shape:int array -> p:float -> value
val dirichlet : value -> alpha:value -> shape:int array -> value
val poisson : value -> shape:int array -> lam:float -> value
val binomial : value -> shape:int array -> count:float -> prob:float -> value
val multinomial : value -> p:value -> n_trials:float -> value
