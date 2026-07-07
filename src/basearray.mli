type dim_size = int
type shape = int array
type dtype_like = Dtype of Dtype.t | Name of string
type static_scalar = Bool of bool | Int of int | Float of float
type array_like = Array of Ndarray.t | Scalar of static_scalar

val ndim : shape -> int
val size : shape -> int
val to_dtype : dtype_like -> Dtype.t
