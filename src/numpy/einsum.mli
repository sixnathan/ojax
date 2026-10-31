val einsum :
  ?optimize:Einsum_paths.optimize ->
  ?precision:Lax_config.precision ->
  ?preferred_element_type:Dtype.t ->
  string ->
  Types.value list ->
  Types.value

val einsum_path :
  ?optimize:Einsum_paths.optimize ->
  string ->
  Types.value list ->
  int list list * Einsum_paths.path_info
