exception Type_promotion_error of string

val default_int_dtype : unit -> Dtype.t
val default_float_dtype : unit -> Dtype.t
val canonicalize_dtype : Dtype.t -> Dtype.t
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
  | `Unsigned ] ->
  bool

val isdtype : Dtype.t -> string -> bool
val to_inexact_dtype : Dtype.t -> Dtype.t
