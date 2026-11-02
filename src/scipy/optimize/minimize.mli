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
  ?tol:float ->
  ?options:(string * float) list ->
  unit ->
  optimize_results

val minimize_args :
  (Types.value -> Types.value list -> Types.value) ->
  Types.value ->
  ?args:Types.value list ->
  method_:string ->
  ?maxiter:int ->
  ?gtol:float ->
  ?norm:float ->
  ?tol:float ->
  ?options:(string * float) list ->
  unit ->
  optimize_results
