val concrete : Types.value -> Ndarray.t

type line_search_results = {
  failed : bool;
  nit : int;
  nfev : int;
  ngev : int;
  k : int;
  a_k : Types.value;
  f_k : Types.value;
  g_k : Types.value;
  status : int;
}

val line_search :
  (Types.value -> Types.value) ->
  Types.value ->
  Types.value ->
  ?old_fval:Types.value ->
  ?old_old_fval:Types.value ->
  ?gfk:Types.value ->
  ?c1:float ->
  ?c2:float ->
  ?maxiter:int ->
  unit ->
  line_search_results
