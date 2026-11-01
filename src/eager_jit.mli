open Types

val one :
  string ->
  'statics ->
  value option list ->
  value list ->
  (value option list -> value list -> value) ->
  value

val many :
  string ->
  'statics ->
  value option list ->
  value list ->
  (value option list -> value list -> value list) ->
  value list

val pair :
  string ->
  'statics ->
  value option list ->
  value list ->
  (value option list -> value list -> value * value) ->
  value * value
