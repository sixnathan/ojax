module Effects = struct
  type eff = |
  type t = eff list

  let no_effects : t = []
  let is_empty (e : t) : bool = e = []
  let equal (a : t) (b : t) : bool = a = b
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

let error_page = "https://docs.jax.dev/en/latest/errors.html"
let module_name = "jax.errors"

let with_page (class_name : string) (message : string) : string =
  Printf.sprintf "%s\nSee %s#%s.%s" message error_page module_name class_name

let jax_type_error (message : string) : exn =
  Jax_type_error (with_page "JAXTypeError" message)

let jax_index_error (message : string) : exn =
  Jax_index_error (with_page "JAXIndexError" message)

let concretization_type_error ?(context = "") ~error_repr ~origin_msg () : exn =
  Concretization_type_error
    (with_page "ConcretizationTypeError"
       (Printf.sprintf
          "Abstract tracer value encountered where concrete value is expected: \
           %s\n\
           %s%s\n"
          error_repr context origin_msg))

let non_concrete_boolean_index_error ~tracer () : exn =
  Non_concrete_boolean_index_error
    (with_page "NonConcreteBooleanIndexError"
       (Printf.sprintf "Array boolean indices must be concrete; got %s\n" tracer))

let tracer_array_conversion_error ~error_repr ~origin_msg () : exn =
  Tracer_array_conversion_error
    (with_page "TracerArrayConversionError"
       (Printf.sprintf
          "The numpy.ndarray conversion method __array__() was called on %s%s"
          error_repr origin_msg))

let tracer_integer_conversion_error ~error_repr ~origin_msg () : exn =
  Tracer_integer_conversion_error
    (with_page "TracerIntegerConversionError"
       (Printf.sprintf "The __index__() method was called on %s%s" error_repr
          origin_msg))

let tracer_bool_conversion_error ~error_repr ~origin_msg () : exn =
  Tracer_bool_conversion_error
    (with_page "TracerBoolConversionError"
       (Printf.sprintf "Attempted boolean conversion of %s.%s" error_repr
          origin_msg))

let unexpected_tracer_error (message : string) : exn =
  Unexpected_tracer_error (with_page "UnexpectedTracerError" message)

let key_reuse_error (message : string) : exn =
  Key_reuse_error (with_page "KeyReuseError" message)
