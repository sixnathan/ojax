type lbfgs_results = {
  converged : bool;
  failed : bool;
  k : int;
  nfev : int;
  ngev : int;
  x_k : Types.value;
  f_k : Types.value;
  g_k : Types.value;
  s_history : Types.value;
  y_history : Types.value;
  rho_history : Types.value;
  gamma : Types.value;
  status : int;
  ls_status : int;
}

val minimize_lbfgs :
  (Types.value -> Types.value) ->
  Types.value ->
  ?maxiter:int ->
  ?norm:float ->
  ?maxcor:int ->
  ?ftol:float ->
  ?gtol:float ->
  ?maxfun:int ->
  ?maxgrad:int ->
  ?maxls:int ->
  unit ->
  lbfgs_results
