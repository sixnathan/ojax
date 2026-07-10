type optimize_results = {
  x : Types.value;
  success : bool;
  status : int;
  fun_ : Types.value;
  jac : Types.value;
  hess_inv : Types.value option;
  nfev : int;
  njev : int;
  nit : int;
}

val minimize :
  (Types.value -> Types.value) ->
  Types.value ->
  method_:string ->
  ?maxiter:int ->
  ?gtol:float ->
  ?norm:float ->
  unit ->
  optimize_results
