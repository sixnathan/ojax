val safe_map : ('a -> 'b) -> 'a list -> 'b list
val safe_map2 : ('a -> 'b -> 'c) -> 'a list -> 'b list -> 'c list
val foreach : ('a -> unit) -> 'a list -> unit
val safe_zip : 'a list -> 'b list -> ('a * 'b) list
val unzip2 : ('a * 'b) list -> 'a list * 'b list
val unzip3 : ('a * 'b * 'c) list -> 'a list * 'b list * 'c list
val subvals : 'a list -> (int * 'a) list -> 'a list
val split_list : 'a list -> int list -> 'a list list
val split_half : 'a list -> 'a list * 'a list
val partition_list : bool list -> 'a list -> 'a list * 'a list
val merge_lists : bool list -> 'a list -> 'a list -> 'a list
val subs_list : int option list -> 'a list -> 'a list -> 'a list

val subs_list2 :
  int option list -> int option list -> 'a list -> 'a list -> 'a list -> 'a list

val concatenate : 'a list list -> 'a list
val flatten : 'a list list -> 'a list
val unflatten : 'a list -> int list -> 'a list list
