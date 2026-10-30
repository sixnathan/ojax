open Types

val dct : ?type_:int -> ?n:int -> ?axis:int -> ?norm:string -> value -> value
val idct : ?type_:int -> ?n:int -> ?axis:int -> ?norm:string -> value -> value

val dctn :
  ?type_:int -> ?s:int list -> ?axes:int list -> ?norm:string -> value -> value

val idctn :
  ?type_:int -> ?s:int list -> ?axes:int list -> ?norm:string -> value -> value
