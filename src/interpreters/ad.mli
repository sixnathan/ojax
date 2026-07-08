open Types

val install : unit -> unit

val jvp :
  (value list -> value list) ->
  value list ->
  value list ->
  value list * value list

val linearize :
  (value list -> value list) ->
  value list ->
  value list * (value list -> value list)

val vjp :
  (value list -> value list) ->
  value list ->
  value list * (value list -> value list)

val grad : (value list -> value) -> value list -> value list
