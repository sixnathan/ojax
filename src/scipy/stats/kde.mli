open Types

type t

val gaussian_kde : ?bw_method:string -> ?weights:value -> value -> t
val evaluate : t -> value -> value
val pdf : t -> value -> value
val logpdf : t -> value -> value
val integrate_box_1d : t -> float -> float -> value
