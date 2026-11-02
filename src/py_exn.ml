exception Value_error of string
exception Mlir_error of string
exception Type_error of string
exception Index_error of string
exception Key_error of string
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

let value_error message = raise (Value_error message)
let type_error message = raise (Type_error message)
let index_error message = raise (Index_error message)

let classify = function
  | Value_error m -> Some ("ValueError", m)
  | Type_error m -> Some ("TypeError", m)
  | Index_error m -> Some ("IndexError", m)
  | Key_error m -> Some ("KeyError", m)
  | Overflow_error m -> Some ("OverflowError", m)
  | Not_implemented_error m -> Some ("NotImplementedError", m)
  | Assertion_error m -> Some ("AssertionError", m)
  | Runtime_error m -> Some ("RuntimeError", m)
  | Zero_division_error m -> Some ("ZeroDivisionError", m)
  | Mlir_error m -> Some ("MLIRError", m)
  | Jax_type_error m -> Some ("JAXTypeError", m)
  | Jax_index_error m -> Some ("JAXIndexError", m)
  | Concretization_type_error m -> Some ("ConcretizationTypeError", m)
  | Non_concrete_boolean_index_error m ->
      Some ("NonConcreteBooleanIndexError", m)
  | Tracer_array_conversion_error m -> Some ("TracerArrayConversionError", m)
  | Tracer_integer_conversion_error m -> Some ("TracerIntegerConversionError", m)
  | Tracer_bool_conversion_error m -> Some ("TracerBoolConversionError", m)
  | Unexpected_tracer_error m -> Some ("UnexpectedTracerError", m)
  | Key_reuse_error m -> Some ("KeyReuseError", m)
  | Inconclusive_dimension_operation m ->
      Some ("InconclusiveDimensionOperation", m)
  | Jax_runtime_error m -> Some ("JaxRuntimeError", m)
  | Jaxpr_type_error m -> Some ("JaxprTypeError", m)
  | Type_promotion_error m -> Some ("TypePromotionError", m)
  | Division_by_zero -> Some ("ZeroDivisionError", "")
  | Assert_failure _ -> Some ("AssertionError", "")
  | _ -> None

let class_name e = Option.map fst (classify e)
let message_of e = Option.map snd (classify e)

let parent = function
  | "ValueError" | "TypeError" | "LookupError" | "ArithmeticError"
  | "AssertionError" | "RuntimeError" | "InconclusiveDimensionOperation"
  | "MLIRError" ->
      Some "Exception"
  | "IndexError" | "KeyError" -> Some "LookupError"
  | "OverflowError" | "ZeroDivisionError" -> Some "ArithmeticError"
  | "NotImplementedError" | "JaxRuntimeError" -> Some "RuntimeError"
  | "JAXTypeError" | "JaxprTypeError" -> Some "TypeError"
  | "JAXIndexError" -> Some "IndexError"
  | "ConcretizationTypeError" | "TracerArrayConversionError"
  | "TracerIntegerConversionError" | "UnexpectedTracerError" | "KeyReuseError"
    ->
      Some "JAXTypeError"
  | "TracerBoolConversionError" -> Some "ConcretizationTypeError"
  | "NonConcreteBooleanIndexError" -> Some "JAXIndexError"
  | "TypePromotionError" -> Some "ValueError"
  | _ -> None

let rec mro cls =
  match parent cls with None -> [ cls ] | Some up -> cls :: mro up

let is_subclass cls ~of_ = List.mem of_ (mro cls)

let is_instance e cls =
  match class_name e with
  | Some name -> is_subclass name ~of_:cls
  | None -> false

let python_repr message =
  let quote = if String.contains message '\'' then '"' else '\'' in
  Printf.sprintf "%c%s%c" quote message quote

let to_string e =
  match (e, classify e) with
  | _, None -> None
  | Key_error m, _ -> Some ("KeyError: " ^ python_repr m)
  | _, Some (cls, "") -> Some cls
  | _, Some (cls, m) -> Some (cls ^ ": " ^ m)

let () = Printexc.register_printer to_string
