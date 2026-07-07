val default_int_dtype : unit -> Dtype.t
val default_float_dtype : unit -> Dtype.t
val canonicalize_dtype : Dtype.t -> Dtype.t
val promote_types : Dtype.t -> Dtype.t -> Dtype.t
val result_type : (Dtype.t * bool) list -> Dtype.t * bool
