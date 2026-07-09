open Types

val scan_out_avals : length:int -> num_carry:int -> closed_jaxpr -> aval list

val scan_impl :
  length:int ->
  reverse:bool ->
  num_carry:int ->
  closed_jaxpr ->
  value list ->
  value list

val scan :
  ?reverse:bool ->
  (value list -> value list) ->
  value list ->
  value list ->
  value list

val while_out_avals : closed_jaxpr -> aval list
val while_impl : closed_jaxpr -> closed_jaxpr -> value list -> value list

val while_loop :
  (value list -> value) ->
  (value list -> value list) ->
  value list ->
  value list
