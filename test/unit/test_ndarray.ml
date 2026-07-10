let big_i64 = 9007199254740994.0
let big_i64_int = 9007199254740994L

let i64_roundtrip () =
  let a = Ojax.Ndarray.of_floats Ojax.Dtype.I64 [| 1 |] [| big_i64 |] in
  Alcotest.(check int64)
    "i64 > 2^53 round-trips byte-identical through exact int64 storage"
    big_i64_int
    (Ojax.Ndarray.get_i64 a [| 0 |])

let i64_canonicalize () =
  let a = Ojax.Ndarray.of_floats Ojax.Dtype.I64 [| 1 |] [| big_i64 |] in
  let c = Ojax.Ndarray.canonicalize Ojax.Dtype.I64 a in
  Alcotest.(check int64)
    "canonicalize I64 preserves > 2^53 value byte-identical" big_i64_int
    (Ojax.Ndarray.get_i64 c [| 0 |])

let i32_storage () =
  let a =
    Ojax.Ndarray.of_floats Ojax.Dtype.I32 [| 2 |] [| -1.0; 2147483647.0 |]
  in
  Alcotest.(check int64)
    "i32 stores signed min-representative exactly" (-1L)
    (Ojax.Ndarray.get_i64 a [| 0 |]);
  Alcotest.(check int64)
    "i32 stores max exactly" 2147483647L
    (Ojax.Ndarray.get_i64 a [| 1 |])

let bool_storage () =
  let a = Ojax.Ndarray.of_floats Ojax.Dtype.Bool [| 3 |] [| 0.0; 1.0; 5.0 |] in
  Alcotest.(check (list (float 0.0)))
    "bool stores 0/1 exactly (nonzero -> 1)" [ 0.0; 1.0; 1.0 ]
    [
      Ojax.Ndarray.get_f a [| 0 |];
      Ojax.Ndarray.get_f a [| 1 |];
      Ojax.Ndarray.get_f a [| 2 |];
    ]

let f32_canonicalize () =
  let a = Ojax.Ndarray.of_floats Ojax.Dtype.F32 [| 1 |] [| 0.1 |] in
  let c = Ojax.Ndarray.canonicalize Ojax.Dtype.F32 a in
  let expected = Int32.float_of_bits (Int32.bits_of_float 0.1) in
  Alcotest.(check (float 0.0))
    "canonicalize F32 rounds to single precision" expected
    (Ojax.Ndarray.get_f c [| 0 |])

let c re im = { Complex.re; im }

let check_complex name (e : Complex.t) (a : Complex.t) =
  Alcotest.(check (float 0.0)) (name ^ ".re") e.re a.re;
  Alcotest.(check (float 0.0)) (name ^ ".im") e.im a.im

let complex128_roundtrip () =
  let vals = [| c 1.5 (-2.5); c 0.0 3.25; c (-7.0) 0.0 |] in
  let a = Ojax.Ndarray.of_complex Ojax.Dtype.Complex128 [| 3 |] vals in
  Alcotest.check
    (Alcotest.testable
       (fun ppf d -> Format.pp_print_string ppf (Ojax.Dtype.short_name d))
       ( = ))
    "dtype c128" Ojax.Dtype.Complex128 (Ojax.Ndarray.dtype a);
  check_complex "e0" vals.(0) (Ojax.Ndarray.get_c a [| 0 |]);
  check_complex "e1" vals.(1) (Ojax.Ndarray.get_c a [| 1 |]);
  check_complex "e2" vals.(2) (Ojax.Ndarray.get_c a [| 2 |])

let f32r v = Int32.float_of_bits (Int32.bits_of_float v)

let complex64_storage_rounds () =
  let a =
    Ojax.Ndarray.of_complex Ojax.Dtype.Complex64 [| 1 |] [| c 0.1 0.2 |]
  in
  check_complex "c64 store rounds to f32"
    (c (f32r 0.1) (f32r 0.2))
    (Ojax.Ndarray.get_c a [| 0 |])

let complex64_canonicalize () =
  let a =
    Ojax.Ndarray.of_complex Ojax.Dtype.Complex128 [| 1 |] [| c 0.1 0.2 |]
  in
  let cc = Ojax.Ndarray.canonicalize Ojax.Dtype.Complex64 a in
  check_complex "canonicalize c128->c64 rounds components"
    (c (f32r 0.1) (f32r 0.2))
    (Ojax.Ndarray.get_c cc [| 0 |])

let complex_set () =
  let a =
    Ojax.Ndarray.of_complex Ojax.Dtype.Complex128 [| 2 |]
      [| c 0.0 0.0; c 0.0 0.0 |]
  in
  Ojax.Ndarray.set_c a [| 1 |] (c 4.0 (-9.0));
  check_complex "set/get" (c 4.0 (-9.0)) (Ojax.Ndarray.get_c a [| 1 |])

let () =
  Alcotest.run "ndarray"
    [
      ( "i64-exact-storage",
        [
          Alcotest.test_case "roundtrip >2^53 byte-identical" `Quick
            i64_roundtrip;
          Alcotest.test_case "canonicalize >2^53 byte-identical" `Quick
            i64_canonicalize;
        ] );
      ( "per-kind-backing",
        [
          Alcotest.test_case "i32 int32 backing" `Quick i32_storage;
          Alcotest.test_case "bool int8 backing" `Quick bool_storage;
        ] );
      ( "canonicalize",
        [ Alcotest.test_case "f32 rounding" `Quick f32_canonicalize ] );
      ( "complex-backing",
        [
          Alcotest.test_case "c128 round-trip exact" `Quick complex128_roundtrip;
          Alcotest.test_case "c64 storage rounds to f32" `Quick
            complex64_storage_rounds;
          Alcotest.test_case "canonicalize c128->c64" `Quick
            complex64_canonicalize;
          Alcotest.test_case "set_c/get_c" `Quick complex_set;
        ] );
    ]
