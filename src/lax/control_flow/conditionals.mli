open Types

val cond :
  value ->
  (value list -> value list) ->
  (value list -> value list) ->
  value list ->
  value list

val platform_index : platforms:string array option array -> value

val switch :
  value -> (value list -> value list) list -> value list -> value list

val branch_call :
  ?platforms:string array option array ->
  value ->
  (value list -> value list) list ->
  value list ->
  value list
