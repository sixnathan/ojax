val check_rank_promotion :
  signature:string option -> int array list -> int list -> unit

val vectorize :
  ?excluded:int list ->
  ?signature:string ->
  (Types.value list -> Types.value) ->
  Types.value list ->
  Types.value

val vectorize_many :
  ?excluded:int list ->
  ?signature:string ->
  (Types.value list -> Types.value list) ->
  Types.value list ->
  Types.value list
