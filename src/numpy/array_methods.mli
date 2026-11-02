val all : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val any : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val sum : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val prod : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val max : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val min : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val mean : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value

val var :
  ?axis:int array -> ?keepdims:bool -> ?ddof:int -> Types.value -> Types.value

val std :
  ?axis:int array -> ?keepdims:bool -> ?ddof:int -> Types.value -> Types.value

val ptp : ?axis:int array -> ?keepdims:bool -> Types.value -> Types.value
val cumsum : ?axis:int -> Types.value -> Types.value
val cumprod : ?axis:int -> Types.value -> Types.value
val argmax : ?axis:int -> ?keepdims:bool -> Types.value -> Types.value
val argmin : ?axis:int -> ?keepdims:bool -> Types.value -> Types.value
val astype : Types.value -> Dtype.t -> Types.value
val clip : ?min:float -> ?max:float -> Types.value -> Types.value
val round : ?decimals:int -> Types.value -> Types.value
val copy : Types.value -> Types.value
val conj : Types.value -> Types.value
val conjugate : Types.value -> Types.value
val reshape : Types.value -> int array -> Types.value
val ravel : Types.value -> Types.value
val flatten : Types.value -> Types.value
val flat : Types.value -> Types.value
val transpose : ?axes:int array -> Types.value -> Types.value
val squeeze : ?axis:int array -> Types.value -> Types.value
val swapaxes : int -> int -> Types.value -> Types.value
val repeat : ?axis:int -> Types.value -> int -> Types.value

val diagonal :
  ?offset:int -> ?axis1:int -> ?axis2:int -> Types.value -> Types.value

val trace :
  ?offset:int ->
  ?axis1:int ->
  ?axis2:int ->
  ?dtype:Dtype.t ->
  Types.value ->
  Types.value

val searchsorted : ?side:string -> Types.value -> Types.value -> Types.value

val take :
  ?axis:int -> ?mode:string -> Types.value -> Types.value -> Types.value

val t : Types.value -> Types.value
val mt : Types.value -> Types.value
val real : Types.value -> Types.value
val imag : Types.value -> Types.value
val neg : Types.value -> Types.value
val pos : Types.value -> Types.value
val abs : Types.value -> Types.value
val invert : Types.value -> Types.value
val eq : Types.value -> Types.value -> Types.value
val ne : Types.value -> Types.value -> Types.value
val lt : Types.value -> Types.value -> Types.value
val le : Types.value -> Types.value -> Types.value
val gt : Types.value -> Types.value -> Types.value
val ge : Types.value -> Types.value -> Types.value
val add : Types.value -> Types.value -> Types.value
val sub : Types.value -> Types.value -> Types.value
val mul : Types.value -> Types.value -> Types.value
val truediv : Types.value -> Types.value -> Types.value
val floordiv : Types.value -> Types.value -> Types.value
val mod_ : Types.value -> Types.value -> Types.value
val pow : Types.value -> Types.value -> Types.value
val and_ : Types.value -> Types.value -> Types.value
val or_ : Types.value -> Types.value -> Types.value
val xor : Types.value -> Types.value -> Types.value
val lshift : Types.value -> Types.value -> Types.value
val rshift : Types.value -> Types.value -> Types.value
val view : Types.value -> Dtype.t -> Types.value
val byteswap : Types.value -> Types.value

type py_scalar =
  | Py_bool of bool
  | Py_int of string
  | Py_float of float
  | Py_complex of Complex.t

type py_list = Py_item of py_scalar | Py_list of py_list list

val item : ?args:int list -> Types.value -> py_scalar
val tolist : Types.value -> py_list
val tobytes : Types.value -> string
val itemsize : Types.value -> int
val nbytes : Types.value -> int
val flat : Types.value -> 'a
val float_repr : ?dot0:bool -> float -> string
val py_scalar_repr : py_scalar -> string
val py_list_repr : py_list -> string
