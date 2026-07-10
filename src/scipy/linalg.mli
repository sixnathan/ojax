val cholesky : ?lower:bool -> Types.value -> Types.value
val cho_factor : ?lower:bool -> Types.value -> Types.value * bool
val cho_solve : Types.value * bool -> Types.value -> Types.value
val det : Types.value -> Types.value
val inv : Types.value -> Types.value
val lu : ?permute_l:bool -> Types.value -> Types.value list
val lu_factor : Types.value -> Types.value * Types.value

val lu_solve :
  ?trans:int -> Types.value * Types.value -> Types.value -> Types.value

val qr : ?mode:string -> ?pivoting:bool -> Types.value -> Types.value list

val solve :
  ?lower:bool -> ?assume_a:string -> Types.value -> Types.value -> Types.value

val solve_triangular :
  ?trans:int ->
  ?lower:bool ->
  ?unit_diagonal:bool ->
  Types.value ->
  Types.value ->
  Types.value
