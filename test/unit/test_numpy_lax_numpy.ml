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

let bool_ shape xs = T.Concrete (Nd.of_floats D.Bool shape xs)

let test_nan_to_num () =
  let x =
    f32 [| 5 |] [| 0.0; Float.nan; 1.0; Float.infinity; Float.neg_infinity |]
  in
  let r = read (NL.nan_to_num x) in
  Alcotest.(check bool) "nan->0" true (r.(1) = 0.0);
  Alcotest.(check bool) "posinf->max" true (r.(3) > 3.0e38);
  Alcotest.(check bool) "neginf->min" true (r.(4) < -3.0e38);
  Alcotest.(check bool) "finite kept" true (r.(0) = 0.0 && r.(2) = 1.0)

let test_close () =
  let a = f32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let b = f32 [| 3 |] [| 1.0; 2.0; 3.0000001 |] in
  Alcotest.check farr "isclose equal" [| 1.; 1.; 1. |] (read (NL.isclose a b));
  Alcotest.check farr "allclose true" [| 1. |] (read (NL.allclose a b));
  let c = f32 [| 3 |] [| 1.0; 2.0; 9.0 |] in
  Alcotest.check farr "allclose false" [| 0. |] (read (NL.allclose a c))

let test_clip () =
  let x = f32 [| 5 |] [| -1.0; 0.5; 1.5; 2.5; 3.5 |] in
  Alcotest.check farr "clip" [| 0.; 0.5; 1.5; 2.; 2. |]
    (read (NL.clip ~min:0.0 ~max:2.0 x))

let test_where_select () =
  let c = bool_ [| 4 |] [| 1.; 0.; 1.; 0. |] in
  let x = f32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let y = f32 [| 4 |] [| 10.; 20.; 30.; 40. |] in
  Alcotest.check farr "where" [| 1.; 20.; 3.; 40. |] (read (NL.where_ c x y));
  let c0 = bool_ [| 4 |] [| 0.; 1.; 0.; 0. |] in
  let c1 = bool_ [| 4 |] [| 1.; 0.; 0.; 0. |] in
  let ch0 = i32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let ch1 = i32 [| 4 |] [| 10.; 20.; 30.; 40. |] in
  Alcotest.check farr "select" [| 10.; 2.; 0.; 0. |]
    (read (NL.select [ c0; c1 ] [ ch0; ch1 ]))

let test_moveaxis_swapaxes () =
  let x = f32 [| 2; 3; 4; 5 |] (Array.make 120 0.0) in
  Alcotest.(check (array int))
    "swapaxes" [| 5; 3; 4; 2 |]
    (shape_of (NL.swapaxes 0 3 x));
  Alcotest.(check (array int))
    "moveaxis" [| 4; 5; 3; 2 |]
    (shape_of (NL.moveaxis [| 0; 1 |] [| -1; -2 |] x));
  Alcotest.(check (array int))
    "expand_dims" [| 1; 2; 3; 4; 5; 1 |]
    (shape_of (NL.expand_dims x [| 0; 5 |]))

let test_squeeze () =
  let x = f32 [| 1; 3; 1 |] [| 1.; 2.; 3. |] in
  Alcotest.(check (array int)) "squeeze all" [| 3 |] (shape_of (NL.squeeze x));
  Alcotest.(check (array int))
    "squeeze axis" [| 1; 3 |]
    (shape_of (NL.squeeze ~axis:[| 2 |] x))

let test_broadcast () =
  Alcotest.(check (array int))
    "broadcast_shapes_n" [| 5; 3; 4 |]
    (NL.broadcast_shapes_n [ [| 3; 1 |]; [| 1; 4 |]; [| 5; 1; 1 |] ]);
  let a = f32 [| 1; 3 |] [| 1.; 2.; 3. |] in
  let b = f32 [| 2; 1 |] [| 10.; 20. |] in
  match NL.broadcast_arrays [ a; b ] with
  | [ a2; b2 ] ->
      Alcotest.(check (array int)) "ba shape a" [| 2; 3 |] (shape_of a2);
      Alcotest.(check (array int)) "ba shape b" [| 2; 3 |] (shape_of b2);
      Alcotest.check farr "ba b vals"
        [| 10.; 10.; 10.; 20.; 20.; 20. |]
        (read b2)
  | _ -> Alcotest.fail "broadcast_arrays arity"

let test_split_resize () =
  let x = f32 [| 6 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  (match NL.split x (NL.Count 3) with
  | [ p; q; r ] ->
      Alcotest.check farr "split 0" [| 1.; 2. |] (read p);
      Alcotest.check farr "split 1" [| 3.; 4. |] (read q);
      Alcotest.check farr "split 2" [| 5.; 6. |] (read r)
  | _ -> Alcotest.fail "split arity");
  (match NL.array_split x (NL.Count 4) with
  | [ p; q; r; s ] ->
      Alcotest.(check int) "array_split 0 len" 2 (Array.length (read p));
      Alcotest.(check int) "array_split 1 len" 2 (Array.length (read q));
      Alcotest.(check int) "array_split 2 len" 1 (Array.length (read r));
      Alcotest.(check int) "array_split 3 len" 1 (Array.length (read s))
  | _ -> Alcotest.fail "array_split arity");
  Alcotest.check farr "resize"
    [| 1.; 2.; 3.; 4.; 5.; 6.; 1.; 2.; 3. |]
    (read (NL.resize x [| 3; 3 |]))

let test_unravel () =
  let idx = i32 [| 3 |] [| 1.; 3.; 5. |] in
  match NL.unravel_index idx [| 2; 3 |] with
  | [ r0; r1 ] ->
      Alcotest.check farr "unravel row" [| 0.; 1.; 1. |] (read r0);
      Alcotest.check farr "unravel col" [| 1.; 0.; 2. |] (read r1)
  | _ -> Alcotest.fail "unravel arity"

let test_round () =
  let x = f32 [| 4 |] [| 1.4; 2.6; -1.7; 3.2 |] in
  Alcotest.check farr "round" [| 1.; 3.; -2.; 3. |] (read (NL.round x))

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
          Alcotest.test_case "nan_to_num" `Quick test_nan_to_num;
          Alcotest.test_case "close" `Quick test_close;
          Alcotest.test_case "clip" `Quick test_clip;
          Alcotest.test_case "where_select" `Quick test_where_select;
          Alcotest.test_case "moveaxis_swapaxes" `Quick test_moveaxis_swapaxes;
          Alcotest.test_case "squeeze" `Quick test_squeeze;
          Alcotest.test_case "broadcast" `Quick test_broadcast;
          Alcotest.test_case "split_resize" `Quick test_split_resize;
          Alcotest.test_case "unravel" `Quick test_unravel;
          Alcotest.test_case "round" `Quick test_round;
        ] );
    ]
