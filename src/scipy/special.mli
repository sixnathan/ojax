val gammaln : Types.value -> Types.value
val gammasgn : Types.value -> Types.value
val loggamma : Types.value -> Types.value
val gamma : Types.value -> Types.value
val betaln : Types.value -> Types.value -> Types.value
val factorial : ?exact:bool -> Types.value -> Types.value
val comb : ?repetition:bool -> Types.value -> Types.value -> Types.value
val beta : Types.value -> Types.value -> Types.value
val betainc : Types.value -> Types.value -> Types.value -> Types.value
val digamma : Types.value -> Types.value
val gammainc : Types.value -> Types.value -> Types.value
val gammaincc : Types.value -> Types.value -> Types.value
val erf : Types.value -> Types.value
val erfc : Types.value -> Types.value
val erfinv : Types.value -> Types.value
val erfcx : Types.value -> Types.value
val dawsn : Types.value -> Types.value
val expit : Types.value -> Types.value
val logit : Types.value -> Types.value
val xlogy : Types.value -> Types.value -> Types.value
val xlog1py : Types.value -> Types.value -> Types.value
val entr : Types.value -> Types.value
val boxcox : Types.value -> Types.value -> Types.value
val boxcox1p : Types.value -> Types.value -> Types.value
val multigammaln : Types.value -> int -> Types.value
val rel_entr : Types.value -> Types.value -> Types.value
val kl_div : Types.value -> Types.value -> Types.value
val i0 : Types.value -> Types.value
val i0e : Types.value -> Types.value
val i1 : Types.value -> Types.value
val i1e : Types.value -> Types.value
val ndtr : Types.value -> Types.value
val ndtri : Types.value -> Types.value
val log_ndtr : ?series_order:int -> Types.value -> Types.value
val polygamma : Types.value -> Types.value -> Types.value
val zeta : ?q:Types.value -> Types.value -> Types.value
val wofz : Types.value -> Types.value
val softmax : ?axis:int -> Types.value -> Types.value
val log_softmax : ?axis:int -> Types.value -> Types.value
val poch : Types.value -> Types.value -> Types.value
val spence : Types.value -> Types.value
val sici : Types.value -> Types.value list
val owens_t : Types.value -> Types.value -> Types.value
val bernoulli : int -> Types.value
val bessel_jn : ?n_iter:int -> v:int -> Types.value -> Types.value
val hyp1f1 : Types.value -> Types.value -> Types.value -> Types.value
val expn : Types.value -> Types.value -> Types.value
val exp1 : Types.value -> Types.value

val hyp2f1 :
  Types.value -> Types.value -> Types.value -> Types.value -> Types.value

val expi : Types.value -> Types.value
val lpmn : int -> int -> Types.value -> Types.value list
val lpmn_values : int -> int -> Types.value -> bool -> Types.value

val sph_harm_y :
  ?diff_n:int ->
  ?n_max:int ->
  Types.value ->
  Types.value ->
  Types.value ->
  Types.value ->
  Types.value
