open Types

type t

val stateful_rng : ?impl:string -> int -> t
val key : ?shape:int array -> t -> value
val random : ?size:int array -> t -> value
val uniform : ?low:float -> ?high:float -> ?size:int array -> t -> value
val normal : ?loc:float -> ?scale:float -> ?size:int array -> t -> value
val integers : ?high:int -> ?size:int array -> low:int -> t -> value
val split : int -> t -> t
val base_key : t -> value
val spawn : int -> t -> t list
