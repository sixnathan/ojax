type t = {
  aliases : int option array;
  buffer_donor : bool array;
  unusable : bool array;
}

val set_up_aliases : Types.aval list -> Types.aval list -> bool list -> t
val arg_attribute : t -> int -> string option
val warn_unusable : Types.aval list -> t -> unit
