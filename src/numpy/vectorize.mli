val vectorize :
  ?excluded:int list ->
  ?signature:string ->
  (Types.value list -> Types.value) ->
  Types.value list ->
  Types.value
