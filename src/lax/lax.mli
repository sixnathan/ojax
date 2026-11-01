open Types

val impl : primitive -> Ndarray.t list -> Ndarray.t list
val abstract_eval : primitive -> aval list -> aval list
val install : unit -> unit

val cond :
  value ->
  (value list -> value list) ->
  (value list -> value list) ->
  value list ->
  value list

val switch :
  value -> (value list -> value list) list -> value list -> value list

val platform_index : platforms:string array option array -> value

val lower_scan :
  length:int ->
  reverse:bool ->
  num_consts:int ->
  num_carry:int ->
  unroll:int ->
  Types.closed_jaxpr ->
  Types.value list ->
  Types.value list

val dynamic_index_in_dim :
  ?axis:int ->
  ?keepdims:bool ->
  ?allow_negative_indices:bool ->
  Types.value ->
  Types.value ->
  Types.value

val dynamic_update_index_in_dim :
  ?allow_negative_indices:bool ->
  Types.value ->
  Types.value ->
  Types.value ->
  int ->
  Types.value

val scan :
  ?length:int ->
  ?reverse:bool ->
  ?unroll:int ->
  (value list -> value list) ->
  value list ->
  value list ->
  value list

val while_loop :
  (value list -> value) ->
  (value list -> value list) ->
  value list ->
  value list

val cumsum : ?axis:int -> ?reverse:bool -> value -> value
val cumprod : ?axis:int -> ?reverse:bool -> value -> value
val cummax : ?axis:int -> ?reverse:bool -> value -> value
val cummin : ?axis:int -> ?reverse:bool -> value -> value
val cumlogsumexp : ?axis:int -> ?reverse:bool -> value -> value

val custom_linear_solve :
  ?symmetric:bool ->
  ?transpose_solve:((value list -> value list) -> value list -> value list) ->
  ?linear_matvec:(value list -> value list) ->
  (value list -> value list) ->
  value list ->
  ((value list -> value list) -> value list -> value list) ->
  value list

val solve_with_lu : value -> value -> value

module Linalg = Linalg
module Special = Special
module Slicing = Slicing
module Fft = Fft

val complex_bilinear :
  (Types.value -> Types.value -> Types.value) ->
  Types.value ->
  Types.value ->
  Types.value
