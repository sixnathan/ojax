module E = Ojax.Errors

let see cls =
  "\nSee https://docs.jax.dev/en/latest/errors.html#jax.errors." ^ cls

let er = "traced array with shape int32[4]"
let om = "\nORIGIN"

let messages () =
  Alcotest.check_raises "JAXTypeError"
    (E.Jax_type_error ("hello" ^ see "JAXTypeError"))
    (fun () -> raise (E.jax_type_error "hello"));
  Alcotest.check_raises "JAXIndexError"
    (E.Jax_index_error ("idx" ^ see "JAXIndexError"))
    (fun () -> raise (E.jax_index_error "idx"));
  Alcotest.check_raises "UnexpectedTracerError"
    (E.Unexpected_tracer_error
       ("Encountered an unexpected tracer." ^ see "UnexpectedTracerError"))
    (fun () ->
      raise (E.unexpected_tracer_error "Encountered an unexpected tracer."));
  Alcotest.check_raises "KeyReuseError"
    (E.Key_reuse_error ("dup" ^ see "KeyReuseError"))
    (fun () -> raise (E.key_reuse_error "dup"));
  Alcotest.check_raises "NonConcreteBooleanIndexError"
    (E.Non_concrete_boolean_index_error
       ("Array boolean indices must be concrete; got ShapedArray(bool[10])\n"
       ^ see "NonConcreteBooleanIndexError"))
    (fun () ->
      raise
        (E.non_concrete_boolean_index_error ~tracer:"ShapedArray(bool[10])" ()));
  Alcotest.check_raises "TracerArrayConversionError"
    (E.Tracer_array_conversion_error
       ("The numpy.ndarray conversion method __array__() was called on " ^ er
      ^ om
       ^ see "TracerArrayConversionError"))
    (fun () ->
      raise (E.tracer_array_conversion_error ~error_repr:er ~origin_msg:om ()));
  Alcotest.check_raises "TracerIntegerConversionError"
    (E.Tracer_integer_conversion_error
       ("The __index__() method was called on " ^ er ^ om
       ^ see "TracerIntegerConversionError"))
    (fun () ->
      raise (E.tracer_integer_conversion_error ~error_repr:er ~origin_msg:om ()));
  Alcotest.check_raises "TracerBoolConversionError"
    (E.Tracer_bool_conversion_error
       ("Attempted boolean conversion of " ^ er ^ "." ^ om
       ^ see "TracerBoolConversionError"))
    (fun () ->
      raise (E.tracer_bool_conversion_error ~error_repr:er ~origin_msg:om ()));
  Alcotest.check_raises "ConcretizationTypeError"
    (E.Concretization_type_error
       ("Abstract tracer value encountered where concrete value is expected: "
      ^ er ^ "\n" ^ "CTX\n" ^ om ^ "\n"
       ^ see "ConcretizationTypeError"))
    (fun () ->
      raise
        (E.concretization_type_error ~context:"CTX\n" ~error_repr:er
           ~origin_msg:om ()))

let effects_stub () =
  Alcotest.(check bool)
    "no_effects is empty" true
    (E.Effects.is_empty E.Effects.no_effects);
  Alcotest.(check bool)
    "no_effects equals itself" true
    (E.Effects.equal E.Effects.no_effects E.Effects.no_effects)

let () =
  Alcotest.run "errors"
    [
      ("messages", [ Alcotest.test_case "exact jax text" `Quick messages ]);
      ("effects", [ Alcotest.test_case "no_effects stub" `Quick effects_stub ]);
    ]
