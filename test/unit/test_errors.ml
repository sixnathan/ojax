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

let bincount_index_overflow () =
  let x dtype value =
    Ojax.Types.Concrete (Ojax.Ndarray.of_floats dtype [| 1 |] [| value |])
  in
  Alcotest.check_raises "int8 maximum"
    (Invalid_argument "Python integer 128 out of bounds for int8") (fun () ->
      ignore (Ojax.Numpy.Extras.bincount (x Ojax.Dtype.I8 127.0)));
  Alcotest.check_raises "int8 length"
    (Invalid_argument "Python integer 128 out of bounds for int8") (fun () ->
      ignore (Ojax.Numpy.Extras.bincount ~length:128 (x Ojax.Dtype.I8 1.0)));
  Alcotest.check_raises "int16 maximum"
    (Invalid_argument "Python integer 32768 out of bounds for int16") (fun () ->
      ignore (Ojax.Numpy.Extras.bincount (x Ojax.Dtype.I16 32767.0)));
  Alcotest.check_raises "int32 maximum"
    (Invalid_argument "Python integer 2147483648 out of bounds for int32")
    (fun () ->
      ignore (Ojax.Numpy.Extras.bincount (x Ojax.Dtype.I32 2147483647.0)))

let () =
  Alcotest.run "errors"
    [
      ("messages", [ Alcotest.test_case "exact jax text" `Quick messages ]);
      ( "bincount",
        [ Alcotest.test_case "index overflow" `Quick bincount_index_overflow ]
      );
      ("effects", [ Alcotest.test_case "no_effects stub" `Quick effects_stub ]);
    ]
