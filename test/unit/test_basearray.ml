module B = Ojax.Basearray
module D = Ojax.Dtype

let dt =
  Alcotest.testable
    (fun ppf d -> Format.pp_print_string ppf (D.short_name d))
    ( = )

let shape_helpers () =
  Alcotest.(check int) "ndim scalar" 0 (B.ndim [||]);
  Alcotest.(check int) "ndim 2d" 2 (B.ndim [| 2; 3 |]);
  Alcotest.(check int) "size scalar" 1 (B.size [||]);
  Alcotest.(check int) "size 2d" 6 (B.size [| 2; 3 |])

let resolve () =
  Alcotest.check dt "passthrough" D.F32 (B.to_dtype (B.Dtype D.F32));
  Alcotest.check dt "float32" D.F32 (B.to_dtype (B.Name "float32"));
  Alcotest.check dt "f64" D.F64 (B.to_dtype (B.Name "f64"));
  Alcotest.check dt "int64" D.I64 (B.to_dtype (B.Name "int64"));
  Alcotest.check dt "bool" D.Bool (B.to_dtype (B.Name "bool"))

let resolve_unknown () =
  Alcotest.check_raises "unknown name"
    (Invalid_argument "data type 'complex64' not understood") (fun () ->
      ignore (B.to_dtype (B.Name "complex64")))

let () =
  Alcotest.run "basearray"
    [
      ("shape", [ Alcotest.test_case "ndim/size" `Quick shape_helpers ]);
      ( "dtype_like",
        [
          Alcotest.test_case "resolve" `Quick resolve;
          Alcotest.test_case "unknown raises" `Quick resolve_unknown;
        ] );
    ]
