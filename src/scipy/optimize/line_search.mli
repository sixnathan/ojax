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

type line_search_values = {
  v_failed : Types.value;
  v_nit : Types.value;
  v_nfev : Types.value;
  v_ngev : Types.value;
  v_k : Types.value;
  v_a_k : Types.value;
  v_f_k : Types.value;
  v_g_k : Types.value;
  v_status : Types.value;
}

val line_search_values :
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
  line_search_values

val value_and_grad :
  (Types.value -> Types.value) -> Types.value -> Types.value * Types.value

val to_int : Types.value -> int
val to_bool : Types.value -> bool
val highest : Lax_config.precision
