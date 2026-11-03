type error_category = Failed_check | Nan | Oob | Division_by_zero

val user_checks : error_category list
val nan_checks : error_category list
val index_checks : error_category list
val div_checks : error_category list
val float_checks : error_category list
val automatic_checks : error_category list
val all_checks : error_category list

type error

val init_error : error
val reset_codes : unit -> unit
val error_leaves : error -> Types.value list
val with_leaves : error -> Types.value list -> error
val get : error -> string option
val throw : error -> unit

val check :
  ?kwargs:(string * Types.value) list ->
  Types.value ->
  string ->
  Types.value list ->
  unit

val checkify :
  ?errors:error_category list ->
  (Types.value list -> Types.value list) ->
  Types.value list ->
  error * Types.value list
