val target_version : string
val element_type : Dtype.t -> string
val tensor_type : Dtype.t -> int array -> string
val tensor_type_of_aval : Types.aval -> string
val float_literal : Dtype.t -> float -> string
val int_literal : int64 -> string
val bool_literal : bool -> string
val dense : string -> string
val int_array_attr : int array -> string
val enum_attr : string -> string -> string
