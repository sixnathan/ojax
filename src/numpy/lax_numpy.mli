type dtype_class =
  | Cdtype of Dtype.t
  | Signedinteger
  | Integer
  | Floating
  | Inexact
  | Number
  | Generic
  | Cbool

val transpose : ?axes:int array -> Types.value -> Types.value
val permute_dims : Types.value -> int array -> Types.value
val matrix_transpose : Types.value -> Types.value
val flip : ?axis:int array -> Types.value -> Types.value
val fliplr : Types.value -> Types.value
val flipud : Types.value -> Types.value
val reshape : Types.value -> int array -> Types.value
val ravel : Types.value -> Types.value
val rot90 : ?k:int -> ?axes:int * int -> Types.value -> Types.value
val trunc : Types.value -> Types.value
val fmin : Types.value -> Types.value -> Types.value
val fmax : Types.value -> Types.value -> Types.value
val diff : ?n:int -> ?axis:int -> Types.value -> Types.value
val ediff1d : Types.value -> Types.value
val angle : ?deg:bool -> Types.value -> Types.value
val iscomplex : Types.value -> Types.value
val isreal : Types.value -> Types.value
val iscomplexobj : Types.value -> bool
val isrealobj : Types.value -> bool
val isscalar : Types.value -> bool
val issubdtype : dtype_class -> dtype_class -> bool
val result_type : Types.value list -> Dtype.t
val convolve : ?mode:string -> Types.value -> Types.value -> Types.value
val correlate : ?mode:string -> Types.value -> Types.value -> Types.value
