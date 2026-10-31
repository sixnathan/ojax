type t

val of_host : Client.t -> Ndarray.t -> t
val to_host : t -> Ndarray.t
val dimensions : t -> int array
val element_type : t -> Dtype.t
val destroy : t -> unit
val to_dlpack : t -> nativeint
val of_dlpack : Client.t -> nativeint -> int -> t
val dlpack_delete : nativeint -> unit
val unsafe_pointer : t -> nativeint
