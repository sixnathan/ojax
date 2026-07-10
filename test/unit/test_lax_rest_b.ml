module L = Ojax.Lax
module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching

let () = L.install ()
let cval dtype shape xs = T.Concrete (Nd.of_floats dtype shape xs)
let f32 shape xs = cval D.F32 shape xs

let out_floats v =
  match v with
  | T.Concrete a ->
      let n = Array.fold_left ( * ) 1 (Nd.shape a) in
      let arr = Array.make n 0.0 in
      let _ =
        Nd.fold
          (fun i x ->
            arr.(i) <- x;
            i + 1)
          0 a
      in
      arr
  | T.Tracer _ -> Alcotest.fail "not concrete"
  | T.Device _ -> Alcotest.fail "not concrete"

let approx = Alcotest.(float 1e-6)
let arr = Alcotest.(array approx)

let test_optimization_barrier_impl () =
  let r =
    C.bind T.Optimization_barrier
      [ f32 [| 2 |] [| 1.; 2. |]; f32 [| 2 |] [| 3.; 4. |] ]
  in
  Alcotest.(check int) "two outputs" 2 (List.length r);
  Alcotest.check arr "out0" [| 1.; 2. |] (out_floats (List.nth r 0));
  Alcotest.check arr "out1" [| 3.; 4. |] (out_floats (List.nth r 1))

let test_reduce_precision_identity () =
  let r =
    C.bind1
      (T.Reduce_precision { exponent_bits = 8; mantissa_bits = 23 })
      [ f32 [| 3 |] [| 1.5; -2.25; 3.0 |] ]
  in
  Alcotest.check arr "full-mantissa identity" [| 1.5; -2.25; 3.0 |]
    (out_floats r)

let test_reduce_precision_round () =
  let r =
    C.bind1
      (T.Reduce_precision { exponent_bits = 8; mantissa_bits = 1 })
      [ f32 [| 2 |] [| 1.25; 1.5 |] ]
  in
  Alcotest.check arr "round to 1 mantissa bit" [| 1.0; 1.5 |] (out_floats r)

let test_sort_impl () =
  let r =
    C.bind1
      (T.Sort { dimension = 0; is_stable = true; num_keys = 1 })
      [ f32 [| 4 |] [| 3.; 1.; 2.; 0. |] ]
  in
  Alcotest.check arr "sorted" [| 0.; 1.; 2.; 3. |] (out_floats r)

let test_sort_2d () =
  let r =
    C.bind1
      (T.Sort { dimension = 1; is_stable = true; num_keys = 1 })
      [ f32 [| 2; 3 |] [| 3.; 1.; 2.; 6.; 4.; 5. |] ]
  in
  Alcotest.check arr "sorted rows" [| 1.; 2.; 3.; 4.; 5.; 6. |] (out_floats r)

let test_tie_impl () =
  let r =
    C.bind1 T.Tie [ f32 [| 2 |] [| 1.; 2. |]; f32 [| 2 |] [| 7.; 8. |] ]
  in
  Alcotest.check arr "tie returns second" [| 7.; 8. |] (out_floats r)

let test_top_k_impl () =
  let r =
    C.bind (T.Top_k { k = 2; axis = 0 }) [ f32 [| 4 |] [| 3.; 1.; 4.; 2. |] ]
  in
  Alcotest.(check int) "two outputs" 2 (List.length r);
  Alcotest.check arr "values" [| 4.; 3. |] (out_floats (List.nth r 0));
  Alcotest.check arr "indices" [| 2.; 0. |] (out_floats (List.nth r 1))

let test_reduce_precision_jvp () =
  let x = f32 [| 3 |] [| 1.5; -2.25; 3.0 |] in
  let tx = f32 [| 3 |] [| 1.; 1.; 1. |] in
  let _, to_ =
    Ad.jvp
      (fun a ->
        [
          C.bind1
            (T.Reduce_precision { exponent_bits = 8; mantissa_bits = 23 })
            a;
        ])
      [ x ] [ tx ]
  in
  Alcotest.check arr "reduce_precision jvp linear" [| 1.; 1.; 1. |]
    (out_floats (List.hd to_))

let test_tie_jvp () =
  let x = f32 [| 2 |] [| 1.; 2. |] in
  let y = f32 [| 2 |] [| 3.; 4. |] in
  let tx = f32 [| 2 |] [| 5.; 6. |] in
  let ty = f32 [| 2 |] [| 7.; 8. |] in
  let _, to_ = Ad.jvp (fun a -> [ C.bind1 T.Tie a ]) [ x; y ] [ tx; ty ] in
  Alcotest.check arr "tie jvp = tangent of second" [| 7.; 8. |]
    (out_floats (List.hd to_))

let test_optimization_barrier_vmap () =
  let x = f32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  let r =
    Batching.vmap (fun a -> C.bind T.Optimization_barrier a) [ Some 0 ] [ x ]
  in
  Alcotest.check arr "opt_barrier vmap" [| 1.; 2.; 3.; 4. |]
    (out_floats (List.hd r))

let test_reduce_precision_vmap () =
  let x = f32 [| 2; 3 |] [| 1.5; 2.5; 3.5; 4.5; 5.5; 6.5 |] in
  let r =
    Batching.vmap
      (fun a ->
        [
          C.bind1
            (T.Reduce_precision { exponent_bits = 8; mantissa_bits = 23 })
            a;
        ])
      [ Some 0 ] [ x ]
  in
  Alcotest.check arr "reduce_precision vmap identity"
    [| 1.5; 2.5; 3.5; 4.5; 5.5; 6.5 |]
    (out_floats (List.hd r))

let () =
  Alcotest.run "lax_rest_b"
    [
      ( "impl",
        [
          Alcotest.test_case "optimization_barrier" `Quick
            test_optimization_barrier_impl;
          Alcotest.test_case "reduce_precision_identity" `Quick
            test_reduce_precision_identity;
          Alcotest.test_case "reduce_precision_round" `Quick
            test_reduce_precision_round;
          Alcotest.test_case "sort" `Quick test_sort_impl;
          Alcotest.test_case "sort_2d" `Quick test_sort_2d;
          Alcotest.test_case "tie" `Quick test_tie_impl;
          Alcotest.test_case "top_k" `Quick test_top_k_impl;
        ] );
      ( "transforms",
        [
          Alcotest.test_case "reduce_precision_jvp" `Quick
            test_reduce_precision_jvp;
          Alcotest.test_case "tie_jvp" `Quick test_tie_jvp;
          Alcotest.test_case "optimization_barrier_vmap" `Quick
            test_optimization_barrier_vmap;
          Alcotest.test_case "reduce_precision_vmap" `Quick
            test_reduce_precision_vmap;
        ] );
    ]
