module Cd = Ojax.Custom_derivatives
module Ad = Ojax.Interpreters.Ad
module C = Ojax.Core
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray

let () = Ojax.Lax.install ()
let b1 = C.bind1
let ndf = Nd.of_floats
let scalar x = T.Concrete (ndf D.F32 [||] [| x |])
let konst c = scalar c

let get0 v =
  match v with T.Concrete a -> Nd.get_f a [||] | _ -> Alcotest.fail "concrete"

let close x y = Float.abs (x -. y) <= 1e-5

let custom_jvp_sin args =
  let primal a = [ b1 T.Sin a ] in
  let jvp inp =
    match inp with
    | [ x; t ] ->
        let dbl = b1 T.Mul [ konst 2.0; b1 T.Cos [ x ] ] in
        [ b1 T.Sin [ x ]; b1 T.Mul [ dbl; t ] ]
    | _ -> Alcotest.fail "custom_jvp_sin jvp arity"
  in
  Cd.custom_jvp ~primal ~jvp args

let custom_vjp_exp args =
  let primal a = [ b1 T.Exp a ] in
  let fwd a =
    let y = b1 T.Exp a in
    [ y; y ]
  in
  let bwd inp =
    match inp with
    | [ res; ct ] -> [ b1 T.Mul [ b1 T.Mul [ konst 3.0; res ]; ct ] ]
    | _ -> Alcotest.fail "custom_vjp_exp bwd arity"
  in
  Cd.custom_vjp ~primal ~fwd ~bwd args

let custom_gradient_exp args =
  let fun_ a =
    let y = b1 T.Exp a in
    let rule cts =
      match cts with
      | [ g ] -> [ b1 T.Mul [ b1 T.Mul [ konst 4.0; y ]; g ] ]
      | _ -> Alcotest.fail "custom_gradient rule arity"
    in
    ([ y ], rule)
  in
  Cd.custom_gradient fun_ args

let test_custom_jvp_forward () =
  let out = custom_jvp_sin [ scalar 0.5 ] in
  Alcotest.(check bool) "sin(0.5)" true (close (get0 (List.hd out)) (sin 0.5))

let test_custom_jvp_grad () =
  let g = Ad.grad (fun a -> List.hd (custom_jvp_sin a)) [ scalar 0.5 ] in
  Alcotest.(check bool)
    "grad = 2 cos(0.5)" true
    (close (get0 (List.hd g)) (2.0 *. cos 0.5))

let test_custom_jvp_jvp () =
  let po, to_ = Ad.jvp custom_jvp_sin [ scalar 0.5 ] [ scalar 1.0 ] in
  Alcotest.(check bool) "primal" true (close (get0 (List.hd po)) (sin 0.5));
  Alcotest.(check bool)
    "tangent = 2 cos(0.5)" true
    (close (get0 (List.hd to_)) (2.0 *. cos 0.5))

let test_custom_vjp_forward () =
  let out = custom_vjp_exp [ scalar 0.3 ] in
  Alcotest.(check bool) "exp(0.3)" true (close (get0 (List.hd out)) (exp 0.3))

let test_custom_vjp_grad () =
  let g = Ad.grad (fun a -> List.hd (custom_vjp_exp a)) [ scalar 0.3 ] in
  Alcotest.(check bool)
    "grad = 3 exp(0.3)" true
    (close (get0 (List.hd g)) (3.0 *. exp 0.3))

let test_custom_gradient_grad () =
  let out = custom_gradient_exp [ scalar 0.3 ] in
  Alcotest.(check bool)
    "forward exp(0.3)" true
    (close (get0 (List.hd out)) (exp 0.3));
  let g = Ad.grad (fun a -> List.hd (custom_gradient_exp a)) [ scalar 0.3 ] in
  Alcotest.(check bool)
    "grad = 4 exp(0.3)" true
    (close (get0 (List.hd g)) (4.0 *. exp 0.3))

let vec xs = T.Concrete (ndf D.F32 [| Array.length xs |] xs)

let get_i v i =
  match v with
  | T.Concrete a -> Nd.get_f a [| i |]
  | _ -> Alcotest.fail "concrete"

let test_closure_convert () =
  let c = vec [| 2.0; 3.0 |] in
  let fun_ a = [ b1 T.Add [ List.hd a; c ] ] in
  let converted, consts = Cd.closure_convert fun_ [ vec [| 0.0; 0.0 |] ] in
  Alcotest.(check int) "one const" 1 (List.length consts);
  Alcotest.(check bool)
    "const value" true
    (close (get_i (List.hd consts) 1) 3.0);
  let out = converted (vec [| 5.0; 5.0 |] :: consts) in
  Alcotest.(check bool) "5 + 3" true (close (get_i (List.hd out) 1) 8.0)

let () =
  Alcotest.run "custom_derivatives"
    [
      ( "custom_jvp",
        [
          Alcotest.test_case "forward" `Quick test_custom_jvp_forward;
          Alcotest.test_case "jvp" `Quick test_custom_jvp_jvp;
          Alcotest.test_case "grad" `Quick test_custom_jvp_grad;
        ] );
      ( "custom_vjp",
        [
          Alcotest.test_case "forward" `Quick test_custom_vjp_forward;
          Alcotest.test_case "grad" `Quick test_custom_vjp_grad;
        ] );
      ( "custom_gradient",
        [ Alcotest.test_case "grad" `Quick test_custom_gradient_grad ] );
      ( "closure_convert",
        [ Alcotest.test_case "hoist" `Quick test_closure_convert ] );
    ]
