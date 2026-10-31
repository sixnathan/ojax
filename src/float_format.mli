type special = Ieee | Fn | Fnuz | Unsigned_fn | No_special

type t = {
  bits : int;
  nmant : int;
  nexp : int;
  min_exp : int;
  fmax : float;
  special : special;
}

val of_dtype : Dtype.t -> t option
val round : t -> float -> float
val encode : t -> float -> int
val decode : t -> int -> float
val next_after : t -> float -> float -> float
