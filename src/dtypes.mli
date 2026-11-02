exception Type_promotion_error of string

val default_int_dtype : unit -> Dtype.t
val default_float_dtype : unit -> Dtype.t
val canonicalize_dtype : Dtype.t -> Dtype.t
val check_and_canonicalize_user_dtype : ?fun_name:string -> Dtype.t -> Dtype.t
val promote_types : Dtype.t -> Dtype.t -> Dtype.t
val result_type : (Dtype.t * bool) list -> Dtype.t * bool

type finfo = {
  bits : int;
  eps : float;
  epsneg : float;
  fmax : float;
  fmin : float;
  tiny : float;
  smallest_subnormal : float;
  precision : int;
  resolution : float;
  nmant : int;
  nexp : int;
}

val finfo : Dtype.t -> finfo

type iinfo = { imin : float; imax : float; ibits : int }

val iinfo : Dtype.t -> iinfo

val issubdtype :
  Dtype.t ->
  [ `Bool
  | `Complex
  | `Floating
  | `Generic
  | `Inexact
  | `Integer
  | `Number
  | `Signed
  | `Unsigned
  | `Extended
  | `Prng_key ] ->
  bool

val supports_inf : Dtype.t -> bool
val max_identity : Dtype.t -> float
val min_identity : Dtype.t -> float
val float0 : Dtype.t
val bfloat16 : Dtype.t
val itemsize_bits : Dtype.t -> int

type scalar_type = Scalar_bool | Scalar_int | Scalar_float | Scalar_complex

val scalar_type_of : Dtype.t -> scalar_type
val isdtype : Dtype.t -> string -> bool
val to_inexact_dtype : Dtype.t -> Dtype.t
