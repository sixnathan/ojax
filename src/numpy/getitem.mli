type index =
  | Int of int
  | Slice of { start : int option; stop : int option; step : int option }
  | Newaxis
  | Ellipsis
  | Arr of Types.value
  | Bool of bool

val take :
  ?indices_are_sorted:bool ->
  ?unique_indices:bool ->
  ?mode:string ->
  ?fill_value:Types.value ->
  ?normalize_indices:bool ->
  Types.value ->
  index list ->
  Types.value

type update =
  ?indices_are_sorted:bool ->
  ?unique_indices:bool ->
  ?mode:string ->
  ?normalize_indices:bool ->
  Types.value ->
  index list ->
  Types.value ->
  Types.value

val at_set : update
val at_add : update
val at_subtract : update
val at_multiply : update
val at_divide : update
val at_power : update
val at_min : update
val at_max : update

val at_apply :
  ?indices_are_sorted:bool ->
  ?unique_indices:bool ->
  ?mode:string ->
  ?normalize_indices:bool ->
  Types.value ->
  index list ->
  (Types.value -> Types.value) ->
  Types.value
