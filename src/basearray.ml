type dim_size = int
type shape = int array
type dtype_like = Dtype of Dtype.t | Name of string
type static_scalar = Bool of bool | Int of int | Float of float
type array_like = Array of Ndarray.t | Scalar of static_scalar

let ndim (s : shape) : int = Array.length s
let size (s : shape) : int = Array.fold_left ( * ) 1 s

let to_dtype : dtype_like -> Dtype.t = function
  | Dtype d -> d
  | Name "float32" | Name "f32" -> Dtype.F32
  | Name "float64" | Name "f64" -> Dtype.F64
  | Name "int32" | Name "i32" -> Dtype.I32
  | Name "int64" | Name "i64" -> Dtype.I64
  | Name "bool" -> Dtype.Bool
  | Name s -> invalid_arg (Printf.sprintf "data type '%s' not understood" s)
