open Types

val one :
  ?jit:string ->
  ?vals_first:bool ->
  string ->
  'statics ->
  value option list ->
  value list ->
  (value option list -> value list -> value) ->
  value

val many :
  ?jit:string ->
  ?vals_first:bool ->
  string ->
  'statics ->
  value option list ->
  value list ->
  (value option list -> value list -> value list) ->
  value list

val pair :
  ?jit:string ->
  ?vals_first:bool ->
  string ->
  'statics ->
  value option list ->
  value list ->
  (value option list -> value list -> value * value) ->
  value * value
