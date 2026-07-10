val dtype : Types.value -> Dtype.t
val shape : Types.value -> int array
val ndim : Types.value -> int
val sc : Types.value -> float -> Types.value
val ninf : Types.value -> Types.value
val nan : Types.value -> Types.value
val f32s : float -> Types.value
val default_float : unit -> Dtype.t
val promote_dtypes : Types.value list -> Types.value list
val promote : Types.value list -> Types.value list
val lgamma : Types.value -> Types.value
