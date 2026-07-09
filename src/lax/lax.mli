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

val platform_index : platforms:string array option array -> value

val scan :
  ?reverse:bool ->
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
