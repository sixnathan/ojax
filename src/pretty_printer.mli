open Types

val primitive_name : primitive -> string
val encode_var : int -> string
val python_repr_float : float -> string
val aval_short : aval -> string
val jaxpr_to_string : jaxpr -> string
val closed_jaxpr_to_string : closed_jaxpr -> string
val primitive_structure : closed_jaxpr -> string
val closed_jaxpr_to_jax_string : closed_jaxpr -> string
val eqn_to_jax_string : eqn -> string
val register_call_name : int -> string -> unit
val custom_num_consts : (eqn -> int option) ref

val custom_call :
  (int -> int option -> aval list -> (string * string * closed_jaxpr) option)
  ref

type jax_param =
  | Plain of string
  | Sub of closed_jaxpr
  | Subs of closed_jaxpr list

val register_custom_view : int -> string -> (string * jax_param) list -> unit
val note_varying : int -> string list -> unit
val has_call_name : int -> bool
val call_name : int -> string
val registered_call_name : int -> string option
val atom_type : atom -> aval
val mark_dropped : var -> unit
val register_callback_view : int -> string -> (string * string) list -> unit
val custom_lin_vjp : (int -> bool) ref
