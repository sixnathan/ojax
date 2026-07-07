exception Store_exception of string

type 'a store

val new_store : unit -> 'a store
val store : 'a store -> 'a -> unit
val store_val : 'a store -> 'a
val reset : 'a store -> unit

type ('a, 'b) t

val wrap_init : ('a -> 'b) -> ('a, 'b) t
val call_wrapped : ('a, 'b) t -> 'a -> 'b
val transformation2 : (('a -> 'b) -> 'c -> 'd) -> ('a, 'b) t -> ('c, 'd) t

val transformation_with_aux2 :
  (('a -> 'b) -> 'aux store -> 'c -> 'd) ->
  ('a, 'b) t ->
  ('c, 'd) t * (unit -> 'aux)

val merge_linear_aux : (unit -> 'a) -> (unit -> 'a) -> bool * 'a
