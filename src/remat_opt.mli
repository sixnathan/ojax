open Types

val optimize_remat_fwd :
  (value list -> value list) ->
  (value list -> value list * value list) ->
  value list ->
  value list * value list

val batch :
  batch_jaxpr:
    (int ->
    closed_jaxpr ->
    aval list ->
    int option list ->
    closed_jaxpr * bool list) ->
  move:(int -> int option -> int -> value -> value) ->
  int ->
  int ->
  value list ->
  int option list ->
  (value list * int option list) option

val dce :
  dce_closed:
    (closed_jaxpr -> bool list -> bool list -> closed_jaxpr * bool list) ->
  int ->
  bool list ->
  (int * bool list) option option

val is_remat_opt : int -> bool
