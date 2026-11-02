val dtype : Types.value -> Dtype.t
val shape : Types.value -> int array
val ndim : Types.value -> int
val sc : Types.value -> float -> Types.value
val ninf : Types.value -> Types.value
val nan : Types.value -> Types.value
val f32s : float -> Types.value
val default_float : unit -> Dtype.t
val promote_dtypes : Types.value list -> Types.value list
val promote : ?name:string -> Types.value list -> Types.value list
val promote_complex : ?name:string -> Types.value list -> Types.value list
val lgamma : Types.value -> Types.value
val py_float : float -> Types.value
val where_px : Types.value -> float -> Types.value -> Types.value
val where_py : Types.value -> Types.value -> float -> Types.value
val where_ninf : Types.value -> Types.value -> Types.value
val where_nan : Types.value -> Types.value -> Types.value
val where_or_ninf : Types.value -> Types.value -> Types.value
val where_or_nan : Types.value -> Types.value -> Types.value

val defaults :
  Types.value option -> Types.value option -> Types.value * Types.value

val qbad : Types.value -> Types.value
val promote_dtypes_any_like : Types.value -> Types.value -> Types.value option
