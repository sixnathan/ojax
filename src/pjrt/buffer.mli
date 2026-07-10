type t

val of_host : Client.t -> Ndarray.t -> t
val to_host : t -> Ndarray.t
val dimensions : t -> int array
val element_type : t -> Dtype.t
val destroy : t -> unit
