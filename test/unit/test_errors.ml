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
    (Ojax.Py_exn.Overflow_error "Python integer 128 out of bounds for int8")
    (fun () -> ignore (Ojax.Numpy.Extras.bincount (x Ojax.Dtype.I8 127.0)));
  Alcotest.check_raises "int8 length"
    (Ojax.Py_exn.Overflow_error "Python integer 128 out of bounds for int8")
    (fun () ->
      ignore (Ojax.Numpy.Extras.bincount ~length:128 (x Ojax.Dtype.I8 1.0)));
  Alcotest.check_raises "int16 maximum"
    (Ojax.Py_exn.Overflow_error "Python integer 32768 out of bounds for int16")
    (fun () -> ignore (Ojax.Numpy.Extras.bincount (x Ojax.Dtype.I16 32767.0)));
  Alcotest.check_raises "int32 maximum"
    (Ojax.Py_exn.Overflow_error
       "Python integer 2147483648 out of bounds for int32") (fun () ->
      ignore (Ojax.Numpy.Extras.bincount (x Ojax.Dtype.I32 2147483647.0)))

let jax_validation () =
  let z dt sh =
    Ojax.Types.Concrete
      (Ojax.Ndarray.of_floats dt sh
         (Array.make (Array.fold_left ( * ) 1 sh) 0.0))
  in
  let f = Ojax.Dtype.F32 in
  let value_error name msg g =
    Alcotest.check_raises name (Ojax.Py_exn.Value_error msg) (fun () ->
        ignore (g ()))
  in
  let type_error name msg g =
    Alcotest.check_raises name (Ojax.Py_exn.Type_error msg) (fun () ->
        ignore (g ()))
  in
  let index_error name msg g =
    Alcotest.check_raises name (Ojax.Py_exn.Index_error msg) (fun () ->
        ignore (g ()))
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

let python_classes () =
  let module P = Ojax.Py_exn in
  let check name wanted got = Alcotest.(check string) name wanted got in
  let printed e = Option.value (P.to_string e) ~default:"none" in
  check "value error" "ValueError: bad" (printed (P.Value_error "bad"));
  check "printexc" "TypeError: bad" (Printexc.to_string (P.Type_error "bad"));
  check "key error reprs" "KeyError: 'k'" (printed (P.Key_error "k"));
  check "no message" "AssertionError" (printed (P.Assertion_error ""));
  check "division" "ZeroDivisionError" (printed Division_by_zero);
  check "foreign" "none" (printed Not_found);
  Alcotest.(check (list string))
    "tracer bool mro"
    [
      "TracerBoolConversionError";
      "ConcretizationTypeError";
      "JAXTypeError";
      "TypeError";
      "Exception";
    ]
    (P.mro "TracerBoolConversionError");
  Alcotest.(check bool)
    "index lookup" true
    (P.is_subclass "IndexError" ~of_:"LookupError");
  Alcotest.(check bool)
    "not implemented is runtime" true
    (P.is_subclass "NotImplementedError" ~of_:"RuntimeError");
  Alcotest.(check bool)
    "promotion is value" true
    (P.is_instance (Ojax.Dtypes.Type_promotion_error "x") "ValueError");
  Alcotest.(check bool)
    "type is not jax type" false
    (P.is_instance (P.Type_error "x") "JAXTypeError");
  Alcotest.(check (option string))
    "errors share the class" (Some "JAXTypeError")
    (P.class_name (E.Jax_type_error "x"))

let validators () =
  let module V = Ojax.Validate in
  let module P = Ojax.Py_exn in
  let value name msg f =
    Alcotest.check_raises name (P.Value_error msg) (fun () -> ignore (f ()))
  in
  let index name msg f =
    Alcotest.check_raises name (P.Index_error msg) (fun () -> ignore (f ()))
  in
  let same = Alcotest.(check int) in
  same "negative axis" 2 (V.canonicalize_axis (-1) 3);
  value "axis" "axis 3 is out of bounds for array of dimension 3" (fun () ->
      V.canonicalize_axis 3 3);
  value "repeated" "repeated axis: (1, 1)" (fun () ->
      V.canonicalize_axis_tuple [| 1; -1 |] 2);
  same "tuple" 4 (V.tuple_index [| 3; 4 |] (-1));
  index "tuple" "tuple index out of range" (fun () -> V.tuple_index [||] 0);
  index "list" "list assignment index out of range" (fun () ->
      V.list_assign_index 0 0);
  Alcotest.(check (pair int int))
    "slice" (2, 3)
    (V.slice_indices ~start:2 ~stop:9 3);
  Alcotest.(check (pair int int))
    "slice negative" (0, 1)
    (V.slice_indices ~start:(-9) ~stop:(-2) 3);
  Alcotest.(check (array int))
    "broadcast" [| 3; 4 |]
    (V.broadcast_shapes [ [| 3; 1 |]; [| 4 |] ]);
  value "broadcast"
    "Incompatible shapes for broadcasting: shapes=[(3, 1), (4, 1)]" (fun () ->
      V.broadcast_shapes [ [| 3; 1 |]; [| 4; 1 |] ]);
  value "ndim" "needs two" (fun () ->
      V.require_ndim_at_least 1 ~min:2 "needs two";
      0)

let collapse_like_jax () =
  let module L = Ojax.Lax_api in
  let x sh =
    Ojax.Types.Concrete
      (Ojax.Ndarray.of_floats Ojax.Dtype.F32 sh
         (Array.make (Array.fold_left ( * ) 1 sh) 0.0))
  in
  let shape v = (Ojax.Core.get_aval v).Ojax.Types.shape in
  let check name wanted got = Alcotest.(check (array int)) name wanted got in
  check "scalar" [| 1 |] (shape (L.collapse (x [||]) 1 3));
  check "past the end" [| 2; 3 |] (shape (L.collapse (x [| 2; 3 |]) 1 3));
  check "negative stop" [| 6; 4 |] (shape (L.collapse (x [| 2; 3; 4 |]) 0 (-1)));
  Alcotest.check_raises "reversed"
    (Ojax.Py_exn.Value_error
       "Invalid dimension range passed to collapse: (2, 3, 4)[2:1]") (fun () ->
      ignore (L.collapse (x [| 2; 3; 4 |]) 2 1))

let () =
  Alcotest.run "errors"
    [
      ( "python classes",
        [
          Alcotest.test_case "classes and printer" `Quick python_classes;
          Alcotest.test_case "validators" `Quick validators;
          Alcotest.test_case "collapse" `Quick collapse_like_jax;
        ] );
      ( "validation",
        [ Alcotest.test_case "jax shape checks" `Quick jax_validation ] );
      ("messages", [ Alcotest.test_case "exact jax text" `Quick messages ]);
      ( "bincount",
        [ Alcotest.test_case "index overflow" `Quick bincount_index_overflow ]
      );
      ("effects", [ Alcotest.test_case "no_effects stub" `Quick effects_stub ]);
    ]
