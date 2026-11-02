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

let jax_validation () =
  let z dt sh =
    Ojax.Types.Concrete
      (Ojax.Ndarray.of_floats dt sh
         (Array.make (Array.fold_left ( * ) 1 sh) 0.0))
  in
  let f = Ojax.Dtype.F32 in
  let value_error name msg g =
    Alcotest.check_raises name (Invalid_argument msg) (fun () -> ignore (g ()))
  in
  let type_error name msg g =
    Alcotest.check_raises name (E.Jax_type_error msg) (fun () -> ignore (g ()))
  in
  let index_error name msg g =
    Alcotest.check_raises name (E.Jax_index_error msg) (fun () -> ignore (g ()))
  in
  let module L = Ojax.Lax_api in
  type_error "slice limit bound"
    "slice limit_indices must be less than or equal to operand shape, got \
     limit_indices (5, 2) for operand shape (0, 2)." (fun () ->
      L.slice_in_dim (z f [| 0; 2 |]) 1 5);
  type_error "slice negative limit"
    "slice limit_indices must have every element be nonnegative, got (2, -1)."
    (fun () -> L.slice_in_dim ~axis:1 (z f [| 2; 0 |]) 1 (-1));
  type_error "slice ordering"
    "slice limit_indices must be greater than or equal to start_indices, got \
     start_indices (1,) and limit_indices (0,)." (fun () ->
      L.slice_in_dim (z f [| 0 |]) 1 0);
  value_error "lax.dot scalars"
    "jax.lax.dot: dimension_numbers must be specified when not performing \
     simple un-batched matrix-matrix, matrix-vector, or vector-vector \
     products; got lhs_shape=() rhs_shape=()" (fun () ->
      L.dot (z f [||]) (z f [||]));
  value_error "cumsum on a scalar"
    "axis 0 is out of bounds for array of dimension 0" (fun () ->
      Ojax.Numpy.Reductions.cumsum ~axis:0 (z f [||]));
  index_error "take from an empty axis"
    "Cannot do a non-empty jnp.take() from an empty axis." (fun () ->
      Ojax.Numpy.Indexing.take ~axis:0
        (z f [| 0; 2 |])
        (z Ojax.Dtype.I32 [| 2 |]));
  index_error "integer index into an empty axis"
    "index is out of bounds for axis 1 with size 0" (fun () ->
      Ojax.Numpy.Nd_index.take
        (z f [| 3; 0 |])
        [
          Ojax.Numpy.Nd_index.Range { start = None; stop = None };
          Ojax.Numpy.Nd_index.Int 0;
        ]);
  value_error "fiedler_companion empty"
    "fiedler_companion requires the last axis of 'a' to have nonzero length, \
     but got an array of shape (0,)." (fun () ->
      Ojax.Scipy.Linalg_special.fiedler_companion (z f [| 0 |]));
  value_error "convolve empty"
    "zero-size arrays not supported in convolutions, got shapes (0,) and (6,)."
    (fun () -> Ojax.Scipy.Signal.convolve (z f [| 0 |]) (z f [| 6 |]))

let () =
  Alcotest.run "errors"
    [
      ( "validation",
        [ Alcotest.test_case "jax shape checks" `Quick jax_validation ] );
      ("messages", [ Alcotest.test_case "exact jax text" `Quick messages ]);
      ( "bincount",
        [ Alcotest.test_case "index overflow" `Quick bincount_index_overflow ]
      );
      ("effects", [ Alcotest.test_case "no_effects stub" `Quick effects_stub ]);
    ]
