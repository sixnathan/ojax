module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module NL = Ojax.Numpy.Lax_numpy

let () = Ojax.Lax.install ()
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)
let i32 shape xs = T.Concrete (Nd.of_floats D.I32 shape xs)

let read v =
  match v with
  | T.Concrete nd ->
      let n = Array.fold_left ( * ) 1 (Nd.shape nd) in
      let a = Array.make n 0.0 in
      let _ =
        Nd.fold
          (fun i x ->
            a.(i) <- x;
            i + 1)
          0 nd
      in
      a
  | T.Tracer _ -> failwith "expected concrete"

let shape_of v = (Ojax.Core.get_aval v).T.shape
let dtype_of v = (Ojax.Core.get_aval v).T.dtype
let farr = Alcotest.(array (float 1e-6))

let test_shape_ops () =
  let x = f32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  Alcotest.(check (array int))
    "transpose" [| 3; 2 |]
    (shape_of (NL.transpose x));
  Alcotest.(check (array int))
    "reshape" [| 6 |]
    (shape_of (NL.reshape x [| 6 |]));
  Alcotest.(check (array int))
    "reshape neg1" [| 3; 2 |]
    (shape_of (NL.reshape x [| -1; 2 |]));
  Alcotest.(check (array int)) "ravel" [| 6 |] (shape_of (NL.ravel x));
  Alcotest.check farr "fliplr" [| 3.; 2.; 1.; 6.; 5.; 4. |] (read (NL.fliplr x));
  Alcotest.check farr "flipud" [| 4.; 5.; 6.; 1.; 2.; 3. |] (read (NL.flipud x))

let test_trunc () =
  let x = f32 [| 4 |] [| -1.7; 2.3; -0.4; 3.9 |] in
  Alcotest.check farr "trunc" [| -1.; 2.; -0.; 3. |] (read (NL.trunc x));
  let xi = i32 [| 3 |] [| 1.; 2.; 3. |] in
  Alcotest.check farr "trunc int passthrough" [| 1.; 2.; 3. |]
    (read (NL.trunc xi))

let test_diff () =
  let x = f32 [| 5 |] [| 1.; 3.; 6.; 10.; 15. |] in
  Alcotest.check farr "diff" [| 2.; 3.; 4.; 5. |] (read (NL.diff x));
  Alcotest.check farr "ediff1d" [| 2.; 3.; 4.; 5. |] (read (NL.ediff1d x))

let test_predicates () =
  let x = f32 [| 3 |] [| 1.; 2.; 3. |] in
  let s = f32 [||] [| 1. |] in
  Alcotest.(check bool) "iscomplexobj" false (NL.iscomplexobj x);
  Alcotest.(check bool) "isrealobj" true (NL.isrealobj x);
  Alcotest.(check bool) "isscalar false" false (NL.isscalar x);
  Alcotest.(check bool) "isscalar true" true (NL.isscalar s);
  Alcotest.(check bool)
    "issubdtype f32 floating" true
    (NL.issubdtype (NL.Cdtype D.F32) NL.Floating);
  Alcotest.(check bool)
    "issubdtype f32 integer" false
    (NL.issubdtype (NL.Cdtype D.F32) NL.Integer);
  Alcotest.(check bool)
    "issubdtype i32 integer" true
    (NL.issubdtype (NL.Cdtype D.I32) NL.Integer);
  Alcotest.(check bool)
    "issubdtype i32 number" true
    (NL.issubdtype (NL.Cdtype D.I32) NL.Number);
  Alcotest.(check bool)
    "issubdtype bool generic" true
    (NL.issubdtype (NL.Cdtype D.Bool) NL.Generic)

let test_result_type () =
  let a = f32 [| 2 |] [| 1.; 2. |] in
  let b = i32 [| 2 |] [| 3.; 4. |] in
  Alcotest.(check bool)
    "result_type f32 i32 -> f32" true
    (NL.result_type [ a; b ] = D.F32)

let test_angle () =
  let x = f32 [| 2 |] [| 2.0; -3.0 |] in
  let r = read (NL.angle x) in
  Alcotest.(check bool) "angle pos = 0" true (Float.abs r.(0) < 1e-6);
  Alcotest.(check bool)
    "angle neg = pi" true
    (Float.abs (r.(1) -. Float.pi) < 1e-6);
  Alcotest.(check bool) "angle dtype" true (dtype_of (NL.angle x) = D.F32)

let () =
  Alcotest.run "numpy_lax_numpy"
    [
      ( "lax_numpy",
        [
          Alcotest.test_case "shape_ops" `Quick test_shape_ops;
          Alcotest.test_case "trunc" `Quick test_trunc;
          Alcotest.test_case "diff" `Quick test_diff;
          Alcotest.test_case "predicates" `Quick test_predicates;
          Alcotest.test_case "result_type" `Quick test_result_type;
          Alcotest.test_case "angle" `Quick test_angle;
        ] );
    ]
