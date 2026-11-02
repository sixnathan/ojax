exception Value_error of string
exception Type_error of string
exception Index_error of string
exception Key_error of string
exception Name_error of string
exception Mlir_error of string
exception Overflow_error of string
exception Not_implemented_error of string
exception Assertion_error of string
exception Runtime_error of string
exception Zero_division_error of string
exception Jax_type_error of string
exception Jax_index_error of string
exception Concretization_type_error of string
exception Non_concrete_boolean_index_error of string
exception Tracer_array_conversion_error of string
exception Tracer_integer_conversion_error of string
exception Tracer_bool_conversion_error of string
exception Unexpected_tracer_error of string
exception Key_reuse_error of string
exception Inconclusive_dimension_operation of string
exception Jax_runtime_error of string
exception Jaxpr_type_error of string
exception Type_promotion_error of string

val value_error : string -> 'a
val type_error : string -> 'a
val index_error : string -> 'a
val class_name : exn -> string option
val message_of : exn -> string option
val parent : string -> string option
val mro : string -> string list
val is_subclass : string -> of_:string -> bool
val is_instance : exn -> string -> bool
val to_string : exn -> string option
