module D = Ojax.Dtype
module Dt = Ojax.Dtypes

let dt =
  Alcotest.testable
    (fun ppf d -> Format.pp_print_string ppf (D.short_name d))
    ( = )

let pair = Alcotest.pair dt Alcotest.bool

let check_result name x64 args expected =
  Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
      Alcotest.check pair name expected (Dt.result_type args))

let x64_off () =
  check_result "bool,i32" false
    [ (D.Bool, false); (D.I32, false) ]
    (D.I32, false);
  check_result "f32,i32" false [ (D.F32, false); (D.I32, false) ] (D.F32, false);
  check_result "f32w,i32" false [ (D.F32, true); (D.I32, false) ] (D.F32, true);
  check_result "f32w,i32w" false [ (D.F32, true); (D.I32, true) ] (D.F32, true);
  check_result "i32w,f32" false [ (D.I32, true); (D.F32, false) ] (D.F32, false);
  check_result "i32w,i32w" false [ (D.I32, true); (D.I32, true) ] (D.I32, true)

let x64_on () =
  check_result "bool,i32" true [ (D.Bool, false); (D.I32, false) ] (D.I32, false);
  check_result "f32,f64" true [ (D.F32, false); (D.F64, false) ] (D.F64, false);
  check_result "f32,i32" true [ (D.F32, false); (D.I32, false) ] (D.F32, false);
  check_result "f64w,i32" true [ (D.F64, true); (D.I32, false) ] (D.F64, true);
  check_result "f64w,i64w" true [ (D.F64, true); (D.I64, true) ] (D.F64, true);
  check_result "i64w,f32" true [ (D.I64, true); (D.F32, false) ] (D.F32, false);
  check_result "i64w,i64w" true [ (D.I64, true); (D.I64, true) ] (D.I64, true);
  check_result "i64,f32" true [ (D.I64, false); (D.F32, false) ] (D.F32, false)

let promote () =
  Alcotest.check dt "i32,f32" D.F32 (Dt.promote_types D.I32 D.F32);
  Alcotest.check dt "i32,i64" D.I64 (Dt.promote_types D.I32 D.I64);
  Alcotest.check dt "bool,i32" D.I32 (Dt.promote_types D.Bool D.I32);
  Alcotest.check dt "f32,f64" D.F64 (Dt.promote_types D.F32 D.F64);
  Alcotest.check dt "i64,f32" D.F32 (Dt.promote_types D.I64 D.F32)

let promote_complex () =
  Alcotest.check dt "f32,c64" D.Complex64 (Dt.promote_types D.F32 D.Complex64);
  Alcotest.check dt "c64,f32" D.Complex64 (Dt.promote_types D.Complex64 D.F32);
  Alcotest.check dt "f64,c128" D.Complex128
    (Dt.promote_types D.F64 D.Complex128);
  Alcotest.check dt "f32,c128" D.Complex128
    (Dt.promote_types D.F32 D.Complex128);
  Alcotest.check dt "f64,c64" D.Complex128 (Dt.promote_types D.F64 D.Complex64);
  Alcotest.check dt "c64,f64" D.Complex128 (Dt.promote_types D.Complex64 D.F64);
  Alcotest.check dt "c64,c128" D.Complex128
    (Dt.promote_types D.Complex64 D.Complex128);
  Alcotest.check dt "i32,c64" D.Complex64 (Dt.promote_types D.I32 D.Complex64);
  Alcotest.check dt "i64,c64" D.Complex64 (Dt.promote_types D.I64 D.Complex64);
  Alcotest.check dt "u32,c64" D.Complex64
    (Dt.promote_types D.Uint32 D.Complex64);
  Alcotest.check dt "bool,c64" D.Complex64 (Dt.promote_types D.Bool D.Complex64);
  Alcotest.check dt "i32,c128" D.Complex128
    (Dt.promote_types D.I32 D.Complex128);
  Alcotest.check dt "c64,c64" D.Complex64
    (Dt.promote_types D.Complex64 D.Complex64)

let result_complex () =
  check_result "c64,f32" false
    [ (D.Complex64, false); (D.F32, false) ]
    (D.Complex64, false);
  check_result "c64w single" false [ (D.Complex64, true) ] (D.Complex64, true);
  check_result "wi,c64w" false
    [ (D.I32, true); (D.Complex64, true) ]
    (D.Complex64, true);
  check_result "bool,c64w" false
    [ (D.Bool, false); (D.Complex64, true) ]
    (D.Complex64, true);
  check_result "c64,f64" true
    [ (D.Complex64, false); (D.F64, false) ]
    (D.Complex128, false);
  check_result "f64,c128" true
    [ (D.F64, false); (D.Complex128, false) ]
    (D.Complex128, false)

let canonicalize () =
  Ojax.Config.with_value Ojax.Config.enable_x64 false (fun () ->
      Alcotest.check dt "off f64" D.F32 (Dt.canonicalize_dtype D.F64);
      Alcotest.check dt "off i64" D.I32 (Dt.canonicalize_dtype D.I64);
      Alcotest.check dt "off f32" D.F32 (Dt.canonicalize_dtype D.F32));
  Ojax.Config.with_value Ojax.Config.enable_x64 false (fun () ->
      Alcotest.check dt "off c128" D.Complex64
        (Dt.canonicalize_dtype D.Complex128);
      Alcotest.check dt "off c64" D.Complex64
        (Dt.canonicalize_dtype D.Complex64));
  Ojax.Config.with_value Ojax.Config.enable_x64 true (fun () ->
      Alcotest.check dt "on f64" D.F64 (Dt.canonicalize_dtype D.F64);
      Alcotest.check dt "on i64" D.I64 (Dt.canonicalize_dtype D.I64);
      Alcotest.check dt "on c128" D.Complex128
        (Dt.canonicalize_dtype D.Complex128))

let () =
  Alcotest.run "dtypes"
    [
      ( "result_type",
        [
          Alcotest.test_case "x64_off" `Quick x64_off;
          Alcotest.test_case "x64_on" `Quick x64_on;
        ] );
      ( "promote_types",
        [
          Alcotest.test_case "pairs" `Quick promote;
          Alcotest.test_case "complex pairs" `Quick promote_complex;
        ] );
      ( "result_type_complex",
        [ Alcotest.test_case "complex" `Quick result_complex ] );
      ("canonicalize", [ Alcotest.test_case "x64 map" `Quick canonicalize ]);
    ]
