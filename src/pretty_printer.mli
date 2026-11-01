open Types

val primitive_name : primitive -> string
val encode_var : int -> string
val python_repr_float : float -> string
val aval_short : aval -> string
val jaxpr_to_string : jaxpr -> string
val closed_jaxpr_to_string : closed_jaxpr -> string
val primitive_structure : closed_jaxpr -> string
val closed_jaxpr_to_jax_string : closed_jaxpr -> string
val register_call_name : int -> string -> unit

val custom_call :
  (int -> int option -> aval list -> (string * string * closed_jaxpr) option)
  ref

val has_call_name : int -> bool
val call_name : int -> string
val registered_call_name : int -> string option
val atom_type : atom -> aval
