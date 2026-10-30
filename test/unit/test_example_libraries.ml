open Ojax
open Ojax.Types
module Tree = Ojax.Tree_util
module O = Ojax.Example_libraries.Optimizers
module S = Ojax.Example_libraries.Stax
module L = Ojax.Lax_api

let () =
  Lax.install ();
  Random.Prng.install ();
  Random.Threefry.install ()

let f32 shape xs = Concrete (Ndarray.of_floats Dtype.F32 shape xs)

let scalar_of = function
  | Concrete nd -> Ndarray.get_f nd [||]
  | Device b -> Ndarray.get_f (Pjrt.Buffer.to_host b) [||]
  | Tracer _ -> Alcotest.fail "expected a concrete scalar"

let key seed =
  Random.Core.key
    (Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int seed |]))

let inputs = f32 [| 4; 2 |] [| 0.; 0.; 0.; 1.; 1.; 0.; 1.; 1. |]
let targets = f32 [| 4; 1 |] [| 0.; 1.; 1.; 0. |]

let loss_of apply params =
  let out = apply params inputs in
  let diff = L.sub out targets in
  L.reduce_sum (L.mul diff diff) [| 0; 1 |]

let train_with make_opt steps =
  let net = S.serial [ S.dense 16; S.tanh; S.dense 16; S.tanh; S.dense 1 ] in
  let _, params = net.S.init (key 0) [| 4; 2 |] in
  let apply p x = net.S.apply p x in
  let opt = make_opt () in
  let loss p = Tree.Leaf (loss_of apply p) in
  let step =
    Api.jit (fun args ->
        match args with
        | [ p ] ->
            Api.grad
              (function [ q ] -> loss q | _ -> Alcotest.fail "arity")
              [ p ]
        | _ -> Alcotest.fail "arity")
  in
  let state = ref (opt.O.init params) in
  let first = scalar_of (loss_of apply params) in
  for i = 0 to steps - 1 do
    let g = step [ opt.O.get_params !state ] in
    state := opt.O.update i g !state
  done;
  (first, scalar_of (loss_of apply (opt.O.get_params !state)))

let trains_xor () =
  let first, final = train_with (fun () -> O.adam (O.constant 0.02)) 400 in
  Alcotest.(check bool)
    (Printf.sprintf "loss fell from %g to %g" first final)
    true
    (final < 0.02 && final < first)

let trains_with_sgd_momentum () =
  let first, final =
    train_with (fun () -> O.momentum (O.constant 0.02) ~mass:0.9) 200
  in
  Alcotest.(check bool) "loss fell" true (final < first)

let conv_pipeline () =
  let net =
    S.serial
      [
        S.conv 4 [| 3; 3 |] ~padding:L.Same;
        S.relu;
        S.max_pool [| 2; 2 |] ~strides:[| 2; 2 |];
        S.flatten;
        S.dense 3;
        S.log_softmax;
      ]
  in
  let out_shape, params = net.S.init (key 1) [| 2; 8; 8; 1 |] in
  Alcotest.(check (array int)) "output shape" [| 2; 3 |] out_shape;
  let x =
    f32 [| 2; 8; 8; 1 |] (Array.init 128 (fun i -> sin (float_of_int i)))
  in
  let out = net.S.apply params x in
  Alcotest.(check (array int))
    "applied shape" [| 2; 3 |] (Core.get_aval out).shape;
  let loss = function
    | [ p ] ->
        let o = net.S.apply p x in
        Tree.Leaf (L.reduce_sum o [| 0; 1 |])
    | _ -> Alcotest.fail "arity"
  in
  let grads = Api.grad loss [ params ] in
  let norm = scalar_of (O.l2_norm grads) in
  Alcotest.(check bool) "conv gradients flow" true (norm > 0.0)

let schedules () =
  let check name expected actual =
    Alcotest.(check (float 1e-9)) name expected actual
  in
  check "exponential" 0.5 (O.exponential_decay 1.0 10 0.5 10);
  check "inverse time" 0.5 (O.inverse_time_decay 1.0 10 1.0 10);
  check "polynomial" 0.25 (O.polynomial_decay 1.0 10 0.0 ~power:2.0 5);
  check "piecewise" 3.0 (O.piecewise_constant [ 2; 4 ] [ 1.0; 2.0; 3.0 ] 5)

let check_grads_detects () =
  let good = function [ x ] -> [ L.sin x ] | _ -> Alcotest.fail "arity" in
  let x = Concrete (Ndarray.of_floats Dtype.F64 [| 3 |] [| 0.1; 0.2; 0.3 |]) in
  Test_util.check_grads ~order:2 good [ x ];
  let wrong =
    Custom_derivatives.custom_jvp good ~jvp:(fun p t ->
        (good p, List.map (fun v -> L.mul v (L.full_like v 5.0)) t))
  in
  match Test_util.check_jvp wrong [ x ] with
  | () -> Alcotest.fail "check_jvp should reject a wrong derivative"
  | exception Failure _ -> ()

let () =
  Alcotest.run "example_libraries"
    [
      ( "training",
        [
          Alcotest.test_case "xor with adam" `Quick trains_xor;
          Alcotest.test_case "momentum" `Quick trains_with_sgd_momentum;
          Alcotest.test_case "conv pipeline" `Quick conv_pipeline;
        ] );
      ("schedules", [ Alcotest.test_case "values" `Quick schedules ]);
      ( "test_util",
        [ Alcotest.test_case "check_grads" `Quick check_grads_detects ] );
    ]
