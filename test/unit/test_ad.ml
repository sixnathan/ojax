module Ad = Ojax.Interpreters.Ad
module C = Ojax.Core
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray

let () = Ojax.Lax.install ()
let b1 = C.bind1
let ndf = Nd.of_floats
let scalar x = T.Concrete (ndf D.F32 [||] [| x |])
let vec xs = T.Concrete (ndf D.F32 [| Array.length xs |] xs)

let get0 v =
  match v with T.Concrete a -> Nd.get_f a [||] | _ -> Alcotest.fail "concrete"

let geti v i =
  match v with
  | T.Concrete a -> Nd.get_f a [| i |]
  | _ -> Alcotest.fail "concrete"

let sin1 args = [ b1 T.Sin args ]

let cubic args =
  match args with
  | [ x ] -> [ b1 T.Sub [ b1 T.Mul [ x; b1 T.Mul [ x; x ] ]; x ] ]
  | _ -> assert false

let sum_sin args = [ b1 (T.Reduce_sum [| 0 |]) [ b1 T.Sin args ] ]

let test_jvp_sin () =
  let po, to_ = Ad.jvp sin1 [ scalar 0.7 ] [ scalar 1.0 ] in
  Alcotest.(check (float 1e-6)) "primal" (sin 0.7) (get0 (List.hd po));
  Alcotest.(check (float 1e-6)) "tangent" (cos 0.7) (get0 (List.hd to_))

let test_jvp_vec_mul () =
  let f args =
    match args with
    | [ x; y ] -> [ b1 T.Mul [ b1 T.Sin [ x ]; y ] ]
    | _ -> assert false
  in
  let x = vec [| 0.1; 0.5; 1.0 |] and y = vec [| 2.0; 3.0; 4.0 |] in
  let tx = vec [| 1.0; 1.0; 1.0 |] and ty = vec [| 0.0; 0.0; 0.0 |] in
  let _, to_ = Ad.jvp f [ x; y ] [ tx; ty ] in
  List.iteri
    (fun i xi ->
      let yi = [| 2.0; 3.0; 4.0 |].(i) in
      Alcotest.(check (float 1e-6))
        "d(sin x * y)/dx"
        (cos xi *. yi)
        (geti (List.hd to_) i))
    [ 0.1; 0.5; 1.0 ]

let test_grad_cubic () =
  let g = Ad.grad (fun a -> List.hd (cubic a)) [ scalar 1.3 ] in
  Alcotest.(check (float 1e-6))
    "3x^2-1"
    ((3.0 *. 1.3 *. 1.3) -. 1.0)
    (get0 (List.hd g))

let test_grad_sum_sin () =
  let x = vec [| 0.2; 0.9; 1.5 |] in
  let g = Ad.grad (fun a -> List.hd (sum_sin a)) [ x ] in
  List.iteri
    (fun i xi ->
      Alcotest.(check (float 1e-6)) "cos" (cos xi) (geti (List.hd g) i))
    [ 0.2; 0.9; 1.5 ]

let test_grad2_sin () =
  let g1 f = fun xs -> List.hd (Ad.grad (fun a -> List.hd (f a)) xs) in
  let g2 = Ad.grad (fun xs -> g1 sin1 xs) [ scalar 0.6 ] in
  Alcotest.(check (float 1e-6)) "-sin" (-.sin 0.6) (get0 (List.hd g2))

let test_linearize () =
  let po, f_lin = Ad.linearize sin1 [ scalar 0.4 ] in
  Alcotest.(check (float 1e-6)) "primal" (sin 0.4) (get0 (List.hd po));
  let t = f_lin [ scalar 2.0 ] in
  Alcotest.(check (float 1e-6)) "jvp" (cos 0.4 *. 2.0) (get0 (List.hd t))

let () =
  Alcotest.run "ad"
    [
      ( "ad",
        [
          Alcotest.test_case "jvp_sin" `Quick test_jvp_sin;
          Alcotest.test_case "jvp_vec_mul" `Quick test_jvp_vec_mul;
          Alcotest.test_case "grad_cubic" `Quick test_grad_cubic;
          Alcotest.test_case "grad_sum_sin" `Quick test_grad_sum_sin;
          Alcotest.test_case "grad2_sin" `Quick test_grad2_sin;
          Alcotest.test_case "linearize" `Quick test_linearize;
        ] );
    ]
