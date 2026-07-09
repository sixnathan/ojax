module T = Ojax.Types
module C = Ojax.Core
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module Lax = Ojax.Lax
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching

let () = Ojax.Lax.install ()
let b1 = C.bind1
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)

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

let shape_of v = match v with T.Concrete nd -> Nd.shape nd | _ -> [||]

let approx name a b =
  Array.iter2
    (fun x y ->
      if Float.abs (x -. y) > 1e-5 then Alcotest.failf "%s: %f <> %f" name x y)
    a b

let sum1 v = b1 (T.Reduce_sum [| 0 |]) [ v ]

let expect_raise name f =
  match f () with
  | _ -> Alcotest.failf "%s: expected raise" name
  | exception Failure _ -> ()

let test_cumsum_eval () =
  let x = f32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  approx "cumsum" (read (Lax.cumsum x)) [| 1.; 3.; 6.; 10. |];
  approx "cumsum_rev" (read (Lax.cumsum ~reverse:true x)) [| 10.; 9.; 7.; 4. |]

let test_cumsum_axis1 () =
  let x = f32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let y = Lax.cumsum ~axis:1 x in
  if shape_of y <> [| 2; 3 |] then Alcotest.fail "cumsum axis1 shape";
  approx "cumsum_axis1" (read y) [| 1.; 3.; 6.; 4.; 9.; 15. |]

let test_cumprod_eval () =
  let x = f32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  approx "cumprod" (read (Lax.cumprod x)) [| 1.; 2.; 6.; 24. |]

let test_cummax_eval () =
  let x = f32 [| 4 |] [| 1.; 3.; 2.; 5. |] in
  approx "cummax" (read (Lax.cummax x)) [| 1.; 3.; 3.; 5. |]

let test_cummin_eval () =
  let x = f32 [| 4 |] [| 5.; 3.; 4.; 1. |] in
  approx "cummin" (read (Lax.cummin x)) [| 5.; 3.; 3.; 1. |]

let test_cumlogsumexp_eval () =
  let x = f32 [| 3 |] [| 0.; 1.; 2. |] in
  approx "cumlogsumexp"
    (read (Lax.cumlogsumexp x))
    [| 0.; 1.313261687; 2.407605964 |]

let test_cumsum_jvp () =
  let _, to_ =
    Ad.jvp
      (fun ins -> [ Lax.cumsum (List.hd ins) ])
      [ f32 [| 4 |] [| 1.; 2.; 3.; 4. |] ]
      [ f32 [| 4 |] [| 1.; 1.; 1.; 1. |] ]
  in
  match to_ with
  | [ t ] -> approx "cumsum_jvp" (read t) [| 1.; 2.; 3.; 4. |]
  | _ -> Alcotest.fail "cumsum jvp arity"

let test_cumsum_grad () =
  match
    Ad.grad
      (fun ins -> sum1 (Lax.cumsum (List.hd ins)))
      [ f32 [| 4 |] [| 1.; 2.; 3.; 4. |] ]
  with
  | [ g ] -> approx "cumsum_grad" (read g) [| 4.; 3.; 2.; 1. |]
  | _ -> Alcotest.fail "cumsum grad arity"

let test_cumsum_vmap () =
  let outs =
    Batching.vmap
      (fun ins -> [ Lax.cumsum (List.hd ins) ])
      [ Some 0 ]
      [ f32 [| 2; 3 |] [| 1.; 2.; 3.; 10.; 20.; 30. |] ]
  in
  match outs with
  | [ y ] ->
      if shape_of y <> [| 2; 3 |] then Alcotest.fail "cumsum vmap shape";
      approx "cumsum_vmap" (read y) [| 1.; 3.; 6.; 10.; 30.; 60. |]
  | _ -> Alcotest.fail "cumsum vmap arity"

let test_cummax_vmap () =
  let outs =
    Batching.vmap
      (fun ins -> [ Lax.cummax (List.hd ins) ])
      [ Some 0 ]
      [ f32 [| 2; 3 |] [| 1.; 3.; 2.; 30.; 20.; 25. |] ]
  in
  match outs with
  | [ y ] -> approx "cummax_vmap" (read y) [| 1.; 3.; 3.; 30.; 30.; 30. |]
  | _ -> Alcotest.fail "cummax vmap arity"

let jvp_of f =
  Ad.jvp
    (fun ins -> [ f (List.hd ins) ])
    [ f32 [| 4 |] [| 1.; 2.; 3.; 4. |] ]
    [ f32 [| 4 |] [| 1.; 1.; 1.; 1. |] ]

let test_nonlinear_jvp_raises () =
  expect_raise "cumprod jvp" (fun () -> jvp_of Lax.cumprod);
  expect_raise "cummax jvp" (fun () -> jvp_of Lax.cummax);
  expect_raise "cummin jvp" (fun () -> jvp_of Lax.cummin);
  expect_raise "cumlogsumexp jvp" (fun () -> jvp_of Lax.cumlogsumexp)

let grad_of f =
  Ad.grad
    (fun ins -> sum1 (f (List.hd ins)))
    [ f32 [| 4 |] [| 1.; 2.; 3.; 4. |] ]

let test_nonlinear_grad_raises () =
  expect_raise "cumprod grad" (fun () -> grad_of Lax.cumprod);
  expect_raise "cummax grad" (fun () -> grad_of Lax.cummax);
  expect_raise "cummin grad" (fun () -> grad_of Lax.cummin);
  expect_raise "cumlogsumexp grad" (fun () -> grad_of Lax.cumlogsumexp)

let () =
  Alcotest.run "lax_cumulatives"
    [
      ( "eval",
        [
          Alcotest.test_case "cumsum" `Quick test_cumsum_eval;
          Alcotest.test_case "cumsum_axis1" `Quick test_cumsum_axis1;
          Alcotest.test_case "cumprod" `Quick test_cumprod_eval;
          Alcotest.test_case "cummax" `Quick test_cummax_eval;
          Alcotest.test_case "cummin" `Quick test_cummin_eval;
          Alcotest.test_case "cumlogsumexp" `Quick test_cumlogsumexp_eval;
        ] );
      ( "autodiff",
        [
          Alcotest.test_case "cumsum_jvp" `Quick test_cumsum_jvp;
          Alcotest.test_case "cumsum_grad" `Quick test_cumsum_grad;
          Alcotest.test_case "nonlinear_jvp_raises" `Quick
            test_nonlinear_jvp_raises;
          Alcotest.test_case "nonlinear_grad_raises" `Quick
            test_nonlinear_grad_raises;
        ] );
      ( "vmap",
        [
          Alcotest.test_case "cumsum_vmap" `Quick test_cumsum_vmap;
          Alcotest.test_case "cummax_vmap" `Quick test_cummax_vmap;
        ] );
    ]
