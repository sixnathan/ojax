type bfgs_results = {
  converged : bool;
  failed : bool;
  k : int;
  nfev : int;
  ngev : int;
  nhev : int;
  x_k : Types.value;
  f_k : Types.value;
  g_k : Types.value;
  h_k : Types.value;
  old_old_fval : Types.value;
  status : int;
  line_search_status : int;
}

val minimize_bfgs :
  (Types.value -> Types.value) ->
  Types.value ->
  ?maxiter:int ->
  ?norm:float ->
  ?gtol:float ->
  ?line_search_maxiter:int ->
  unit ->
  bfgs_results
