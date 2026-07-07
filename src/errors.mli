module Effects : sig
  type eff
  type t = eff list

  val no_effects : t
  val is_empty : t -> bool
  val equal : t -> t -> bool
end

exception Jax_type_error of string
exception Jax_index_error of string
exception Concretization_type_error of string
exception Non_concrete_boolean_index_error of string
exception Tracer_array_conversion_error of string
exception Tracer_integer_conversion_error of string
exception Tracer_bool_conversion_error of string
exception Unexpected_tracer_error of string
exception Key_reuse_error of string

val jax_type_error : string -> exn
val jax_index_error : string -> exn

val concretization_type_error :
  ?context:string -> error_repr:string -> origin_msg:string -> unit -> exn

val non_concrete_boolean_index_error : tracer:string -> unit -> exn

val tracer_array_conversion_error :
  error_repr:string -> origin_msg:string -> unit -> exn

val tracer_integer_conversion_error :
  error_repr:string -> origin_msg:string -> unit -> exn

val tracer_bool_conversion_error :
  error_repr:string -> origin_msg:string -> unit -> exn

val unexpected_tracer_error : string -> exn
val key_reuse_error : string -> exn
