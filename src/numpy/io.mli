val save : string -> Types.value -> unit
val load : string -> Types.value
val savez : string -> (string * Types.value) list -> unit
val load_npz : string -> (string * Types.value) list

val frombuffer :
  ?dtype:Dtype.t -> ?count:int -> ?offset:int -> string -> Types.value

val fromstring : ?dtype:Dtype.t -> ?sep:string -> string -> Types.value

val fromfile :
  ?dtype:Dtype.t ->
  ?count:int ->
  ?sep:string ->
  ?offset:int ->
  string ->
  Types.value

val fromiter : ?dtype:Dtype.t -> ?count:int -> float Seq.t -> Types.value
