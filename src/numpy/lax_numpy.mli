type dtype_class =
  | Cdtype of Dtype.t
  | Signedinteger
  | Integer
  | Floating
  | Inexact
  | Number
  | Generic
  | Cbool

type sections = Count of int | Indices of int array

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
val where_ : Types.value -> Types.value -> Types.value -> Types.value

val nan_to_num :
  ?nan:float -> ?posinf:float -> ?neginf:float -> Types.value -> Types.value

val isclose :
  ?rtol:float ->
  ?atol:float ->
  ?equal_nan:bool ->
  Types.value ->
  Types.value ->
  Types.value

val allclose :
  ?rtol:float ->
  ?atol:float ->
  ?equal_nan:bool ->
  Types.value ->
  Types.value ->
  Types.value

val clip : ?min:float -> ?max:float -> Types.value -> Types.value
val round : ?decimals:int -> Types.value -> Types.value
val around : ?decimals:int -> Types.value -> Types.value
val expand_dims : Types.value -> int array -> Types.value
val squeeze : ?axis:int array -> Types.value -> Types.value
val swapaxes : int -> int -> Types.value -> Types.value
val moveaxis : int array -> int array -> Types.value -> Types.value
val broadcast_to : Types.value -> int array -> Types.value
val broadcast_shapes_n : int array list -> int array
val broadcast_arrays : Types.value list -> Types.value list
val resize : Types.value -> int array -> Types.value
val unravel_index : Types.value -> int array -> Types.value list

val unwrap :
  ?discont:float -> ?axis:int -> ?period:float -> Types.value -> Types.value

val split : ?axis:int -> Types.value -> sections -> Types.value list
val array_split : ?axis:int -> Types.value -> sections -> Types.value list
val vsplit : Types.value -> sections -> Types.value list
val hsplit : Types.value -> sections -> Types.value list
val dsplit : Types.value -> sections -> Types.value list
val select : Types.value list -> Types.value list -> Types.value
