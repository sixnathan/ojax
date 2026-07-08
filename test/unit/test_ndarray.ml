let big_i64 = 9007199254740994.0
let big_i64_int = 9007199254740994L

let i64_roundtrip () =
  let a = Ojax.Ndarray.of_floats Ojax.Dtype.I64 [| 1 |] [| big_i64 |] in
  Alcotest.(check int64)
    "f64-representable i64 > 2^53 round-trips through abstract accessor"
    big_i64_int
    (Ojax.Ndarray.get_i64 a [| 0 |])

let i64_canonicalize () =
  let a = Ojax.Ndarray.of_floats Ojax.Dtype.I64 [| 1 |] [| big_i64 |] in
  let c = Ojax.Ndarray.canonicalize Ojax.Dtype.I64 a in
  Alcotest.(check int64)
    "canonicalize I64 preserves f64-representable > 2^53 value" big_i64_int
    (Ojax.Ndarray.get_i64 c [| 0 |])

let f32_canonicalize () =
  let a = Ojax.Ndarray.of_floats Ojax.Dtype.F32 [| 1 |] [| 0.1 |] in
  let c = Ojax.Ndarray.canonicalize Ojax.Dtype.F32 a in
  let expected = Int32.float_of_bits (Int32.bits_of_float 0.1) in
  Alcotest.(check (float 0.0))
    "canonicalize F32 rounds to single precision" expected
    (Ojax.Ndarray.get_f c [| 0 |])

let () =
  Alcotest.run "ndarray"
    [
      ( "i64-f64-representable-lock",
        [
          Alcotest.test_case "roundtrip f64-representable >2^53" `Quick
            i64_roundtrip;
          Alcotest.test_case "canonicalize f64-representable >2^53" `Quick
            i64_canonicalize;
        ] );
      ( "canonicalize",
        [ Alcotest.test_case "f32 rounding" `Quick f32_canonicalize ] );
    ]
