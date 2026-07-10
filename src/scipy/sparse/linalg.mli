val cg :
  ?tol:float ->
  ?atol:float ->
  ?maxiter:int ->
  ?m:Types.value ->
  ?x0:Types.value ->
  Types.value ->
  Types.value ->
  Types.value * Types.value option

val gmres :
  ?tol:float ->
  ?atol:float ->
  ?restart:int ->
  ?maxiter:int ->
  ?m:Types.value ->
  ?solve_method:string ->
  ?x0:Types.value ->
  Types.value ->
  Types.value ->
  Types.value * Types.value option

val bicgstab :
  ?tol:float ->
  ?atol:float ->
  ?maxiter:int ->
  ?m:Types.value ->
  ?x0:Types.value ->
  Types.value ->
  Types.value ->
  Types.value * Types.value option
