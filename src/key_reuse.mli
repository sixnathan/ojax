open Types

type signature
type arg_kind

val idle : unit -> bool
val suspended : (unit -> 'a) -> 'a
val apply : primitive -> value list -> (unit -> value list) -> value list
val mark_typed : value -> value
val mark_raw : value -> value
val is_typed : value -> bool
val known_raw : value -> bool
val consume : string -> value -> unit
val arg_kinds : value list -> arg_kind list
val kinds_key : arg_kind list -> string
val new_signature : unit -> signature

val instrument :
  signature ->
  arg_kind list ->
  (value list -> value list) ->
  value list ->
  value list

val call_with_signature :
  signature -> value list -> (unit -> value list) -> value list

val note_moved : bool list -> unit
val note_while : primitive -> aval list -> aval list -> unit

val cond_regions :
  (value list -> value list) list ->
  value list ->
  ((value list -> value list) list -> value list) ->
  value list

val scan_region :
  num_carry:int ->
  (value list -> value list) ->
  value list ->
  ((value list -> value list) -> value list) ->
  value list

val while_regions :
  (value list -> value list) ->
  (value list -> value list) ->
  value list ->
  ((value list -> value list) -> (value list -> value list) -> value list) ->
  value list
