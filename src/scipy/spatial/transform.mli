module Rotation : sig
  type t

  val of_quat : Types.value -> t
  val quat : t -> Types.value
  val single : t -> bool
  val len : t -> int
  val from_quat : Types.value -> t
  val from_matrix : Types.value -> t
  val from_mrp : Types.value -> t
  val from_rotvec : ?degrees:bool -> Types.value -> t
  val identity : ?dtype:Dtype.t -> unit -> t
  val concatenate : t list -> t
  val from_euler : string -> Types.value -> degrees:bool -> t
  val as_matrix : t -> Types.value
  val as_mrp : t -> Types.value
  val as_rotvec : ?degrees:bool -> t -> Types.value
  val magnitude : t -> Types.value
  val inv : t -> t
  val as_quat : ?canonical:bool -> ?scalar_first:bool -> t -> Types.value
  val as_euler : ?degrees:bool -> string -> t -> Types.value
  val apply : ?inverse:bool -> t -> Types.value -> Types.value
  val compose : t -> t -> t
  val getitem : t -> Types.value -> t
  val getrow : t -> int -> t
  val mean : ?weights:Types.value -> t -> t
end

module Slerp : sig
  type t

  val init : Types.value -> Rotation.t -> t
  val apply : t -> Types.value -> Rotation.t
end
