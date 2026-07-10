type bw_method = Scott | Silverman | Scalar of float
type t

val make : ?bw_method:bw_method -> ?weights:Types.value -> Types.value -> t
val dataset : t -> Types.value
val weights : t -> Types.value
val covariance : t -> Types.value
val inv_cov : t -> Types.value
val neff : t -> Types.value
val d : t -> int
val n : t -> int
val evaluate : t -> Types.value -> Types.value
val pdf : t -> Types.value -> Types.value
val logpdf : t -> Types.value -> Types.value
val integrate_gaussian : t -> Types.value -> Types.value -> Types.value
val integrate_box_1d : t -> float -> float -> Types.value
val integrate_kde : t -> t -> Types.value
val resample : ?shape:int array -> t -> Types.value -> Types.value
