open Types

val hilbert : int -> value
val circulant : value -> value
val hankel : ?r:value -> value -> value
val fiedler : value -> value
val companion : value -> value
val fiedler_companion : value -> value
val leslie : value -> value -> value
val dft : ?scale:string -> int -> value
val pascal : ?kind:string -> int -> value
val invpascal : ?kind:string -> int -> value
val helmert : ?full:bool -> int -> value
val hadamard : int -> value
val convolution_matrix : ?mode:string -> value -> int -> value
val invhilbert : int -> value
val solve_sylvester : value -> value -> value -> value
