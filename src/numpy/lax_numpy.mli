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
val astype : Types.value -> Dtype.t -> Types.value
val copy : Types.value -> Types.value
val atleast_1d : Types.value -> Types.value
val atleast_2d : Types.value -> Types.value
val atleast_3d : Types.value -> Types.value
val concatenate : ?axis:int -> Types.value list -> Types.value
val concat : ?axis:int -> Types.value list -> Types.value
val stack : ?axis:int -> Types.value list -> Types.value
val unstack : ?axis:int -> Types.value -> Types.value list
val vstack : Types.value list -> Types.value
val hstack : Types.value list -> Types.value
val dstack : Types.value list -> Types.value
val column_stack : Types.value list -> Types.value
val tile : Types.value -> int array -> Types.value
val pad : Types.value -> (int * int) array -> float -> Types.value
val i0 : Types.value -> Types.value
val array_equal : ?equal_nan:bool -> Types.value -> Types.value -> Types.value
val array_equiv : Types.value -> Types.value -> Types.value

val arange :
  ?start:float -> ?step:float -> dtype:Dtype.t -> float -> Types.value

val eye : ?m:int -> ?k:int -> dtype:Dtype.t -> int -> Types.value
val identity : dtype:Dtype.t -> int -> Types.value
val indices : dtype:Dtype.t -> int array -> Types.value

val meshgrid :
  ?indexing:string -> ?sparse:bool -> Types.value list -> Types.value list

val ix_ : Types.value list -> Types.value list
val append : ?axis:int -> Types.value -> Types.value -> Types.value
val argmax : ?axis:int -> ?keepdims:bool -> Types.value -> Types.value

val cross :
  ?axisa:int ->
  ?axisb:int ->
  ?axisc:int ->
  ?axis:int ->
  Types.value ->
  Types.value ->
  Types.value

val diag : ?k:int -> Types.value -> Types.value
val diagflat : ?k:int -> Types.value -> Types.value

val diagonal :
  ?offset:int -> ?axis1:int -> ?axis2:int -> Types.value -> Types.value

val diag_indices : ?ndim:int -> int -> Types.value list
val diag_indices_from : Types.value -> Types.value list
val kron : Types.value -> Types.value -> Types.value
val repeat : ?axis:int -> Types.value -> int -> Types.value

val trace :
  ?offset:int ->
  ?axis1:int ->
  ?axis2:int ->
  ?dtype:Dtype.t ->
  Types.value ->
  Types.value

val trapezoid :
  ?x:Types.value -> ?dx:float -> ?axis:int -> Types.value -> Types.value

val tri : ?m:int -> ?k:int -> dtype:Dtype.t -> int -> Types.value
val tril : ?k:int -> Types.value -> Types.value
val triu : ?k:int -> Types.value -> Types.value
val vander : ?n:int -> ?increasing:bool -> Types.value -> Types.value
val argmin : ?axis:int -> ?keepdims:bool -> Types.value -> Types.value
val nanargmax : ?axis:int -> ?keepdims:bool -> Types.value -> Types.value
val nanargmin : ?axis:int -> ?keepdims:bool -> Types.value -> Types.value
val roll : ?axis:int array -> Types.value -> int array -> Types.value
val rollaxis : ?start:int -> int -> Types.value -> Types.value
val gcd : Types.value -> Types.value -> Types.value
val lcm : Types.value -> Types.value -> Types.value
val searchsorted : ?side:string -> Types.value -> Types.value -> Types.value
val digitize : ?right:bool -> Types.value -> Types.value -> Types.value

val cov :
  ?y:Types.value ->
  ?rowvar:bool ->
  ?bias:bool ->
  ?ddof:int ->
  ?dtype:Dtype.t ->
  Types.value ->
  Types.value

val corrcoef :
  ?y:Types.value -> ?rowvar:bool -> ?dtype:Dtype.t -> Types.value -> Types.value
