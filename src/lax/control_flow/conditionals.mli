open Types

val cond :
  value ->
  (value list -> value list) ->
  (value list -> value list) ->
  value list ->
  value list

val platform_index : platforms:string array option array -> value
