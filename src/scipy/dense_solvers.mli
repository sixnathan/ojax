val solve_triangular :
  ?trans:int ->
  ?lower:bool ->
  ?unit_diagonal:bool ->
  Types.value ->
  Types.value ->
  Types.value

val solve :
  ?lower:bool -> ?assume_a:string -> Types.value -> Types.value -> Types.value
