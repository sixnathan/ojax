open Ojax
open Ojax.Types
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching

let () = Lax.install ()
let scalar x = Concrete (Ndarray.of_floats Dtype.F32 [||] [| x |])
let vec xs = Concrete (Ndarray.of_floats Dtype.F32 [| Array.length xs |] xs)

let to_nd = function
  | Concrete nd -> nd
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> Alcotest.fail "expected a concrete value"

let value v = Ndarray.get_f (to_nd v) [||]

let check name expected actual =
  Alcotest.(check (float 1e-5)) name expected actual

let mul a b = Core.bind1 Mul [ a; b ]
let sole = function [ x ] -> x | _ -> Alcotest.fail "expected one value"
let grad_of f x = sole (Ad.grad (fun a -> sole (f a)) [ x ])

let square_custom =
  Custom_derivatives.custom_jvp
    (fun a -> [ mul (sole a) (sole a) ])
    ~jvp:(fun p t -> ([ mul (sole p) (sole p) ], [ mul (scalar 3.0) (sole t) ]))

let jvp_uses_rule () =
  let po, to_ = Ad.jvp square_custom [ scalar 2.0 ] [ scalar 1.0 ] in
  check "primal" 4.0 (value (sole po));
  check "tangent" 3.0 (value (sole to_))

let grad_through_jvp_rule () =
  check "grad" 3.0 (value (grad_of square_custom (scalar 2.0)))

let jit_keeps_jvp_rule () =
  let jitted = Api.jit_flat square_custom in
  check "value" 4.0 (value (sole (jitted [ scalar 2.0 ])));
  check "grad" 3.0 (value (grad_of jitted (scalar 2.0)))

let vmap_jvp_rule () =
  let out =
    Batching.vmap_flat
      (fun a -> [ grad_of square_custom (sole a) ])
      [ Some 0 ]
      [ vec [| 1.0; 2.0 |] ]
  in
  let nd = to_nd (sole out) in
  check "first" 3.0 (Ndarray.get_f nd [| 0 |]);
  check "second" 3.0 (Ndarray.get_f nd [| 1 |])

let grad_of_vmap_jvp_rule () =
  let summed a =
    let out = Batching.vmap_flat square_custom [ Some 0 ] a in
    [ Core.bind1 (Reduce_sum [| 0 |]) out ]
  in
  let nd = to_nd (grad_of summed (vec [| 1.0; 2.0 |])) in
  check "first" 3.0 (Ndarray.get_f nd [| 0 |]);
  check "second" 3.0 (Ndarray.get_f nd [| 1 |])

let clipped =
  Custom_derivatives.custom_vjp
    (fun a -> a)
    ~fwd:(fun a -> (a, a))
    ~bwd:(fun res cts -> [ mul (mul (scalar 5.0) (sole res)) (sole cts) ])

let vjp_uses_rule () = check "grad" 10.0 (value (grad_of clipped (scalar 2.0)))

let vjp_chain_rule () =
  let f a = [ mul (scalar 3.0) (sole (clipped [ mul (sole a) (sole a) ])) ] in
  check "chained" 240.0 (value (grad_of f (scalar 2.0)))

let vjp_nested () =
  let g x = grad_of clipped (sole x) in
  check "second order" 5.0 (value (grad_of (fun x -> [ g x ]) (scalar 2.0)))

let vjp_jit () =
  let jitted = Api.jit_flat clipped in
  check "value" 2.0 (value (sole (jitted [ scalar 2.0 ])));
  check "grad" 10.0 (value (grad_of jitted (scalar 2.0)))

let vjp_vmap_of_grad () =
  let out =
    Batching.vmap_flat
      (fun a -> [ grad_of clipped (sole a) ])
      [ Some 0 ]
      [ vec [| 1.0; 2.0 |] ]
  in
  let nd = to_nd (sole out) in
  check "first" 5.0 (Ndarray.get_f nd [| 0 |]);
  check "second" 10.0 (Ndarray.get_f nd [| 1 |])

let vjp_grad_of_vmap () =
  let summed a =
    let out = Batching.vmap_flat clipped [ Some 0 ] a in
    [ Core.bind1 (Reduce_sum [| 0 |]) out ]
  in
  let nd = to_nd (grad_of summed (vec [| 1.0; 2.0 |])) in
  check "first" 5.0 (Ndarray.get_f nd [| 0 |]);
  check "second" 10.0 (Ndarray.get_f nd [| 1 |])

let vjp_forward_raises () =
  match Ad.jvp clipped [ scalar 2.0 ] [ scalar 1.0 ] with
  | _ -> Alcotest.fail "expected forward mode to raise"
  | exception Failure msg ->
      Alcotest.(check string)
        "message"
        "can't apply forward-mode autodiff (jvp) to a custom_vjp function." msg

let vjp_unused_input () =
  let g =
    Custom_derivatives.custom_vjp
      (fun a -> a)
      ~fwd:(fun a -> (a, []))
      ~bwd:(fun _ cts -> cts)
  in
  let f a = [ mul (sole a) (sole (g [ scalar 7.0 ])) ] in
  check "constant input" 7.0 (value (grad_of f (scalar 2.0)))

let custom_gradient_rule () =
  let f =
    Custom_derivatives.custom_gradient (fun a ->
        let x = sole a in
        ([ mul x x ], fun cts -> [ mul (mul (scalar 4.0) x) (sole cts) ]))
  in
  check "value" 4.0 (value (sole (f [ scalar 2.0 ])));
  check "grad" 8.0 (value (grad_of f (scalar 2.0)))

let linear_call_rule () =
  let f =
    Custom_derivatives.linear_call
      ~forward:(fun res x -> [ mul (sole res) (sole x) ])
      ~transpose:(fun res ct -> [ mul (sole res) (sole ct) ])
  in
  check "forward" 6.0 (value (sole (f [ scalar 3.0 ] [ scalar 2.0 ])));
  let g a = f [ scalar 3.0 ] a in
  check "grad" 3.0 (value (grad_of g (scalar 2.0)));
  let po, to_ = Ad.jvp g [ scalar 2.0 ] [ scalar 1.0 ] in
  check "jvp primal" 6.0 (value (sole po));
  check "jvp tangent" 3.0 (value (sole to_))

let checkpointed calls =
  Api.checkpoint (fun a ->
      incr calls;
      let x = sole a in
      [ mul (mul x x) x ])

let checkpoint_recomputes () =
  let calls = ref 0 in
  let f = checkpointed calls in
  let g = grad_of f (scalar 2.0) in
  check "grad" 12.0 (value g);
  Alcotest.(check bool) "forward ran again on the way back" true (!calls >= 2)

let checkpoint_orders () =
  let f = checkpointed (ref 0) in
  let _, t = Ad.jvp f [ scalar 2.0 ] [ scalar 1.0 ] in
  check "jvp" 12.0 (value (sole t));
  let first a = [ grad_of f (sole a) ] in
  check "second order" 12.0 (value (grad_of first (scalar 2.0)));
  let forward a =
    let _, t = Ad.jvp f a [ scalar 1.0 ] in
    t
  in
  let _, tt = Ad.jvp forward [ scalar 2.0 ] [ scalar 1.0 ] in
  check "forward over forward" 12.0 (value (sole tt));
  let reverse_of_forward = grad_of forward (scalar 2.0) in
  check "reverse over forward" 12.0 (value reverse_of_forward)

let checkpoint_jit_vmap () =
  let f = checkpointed (ref 0) in
  let jitted = Api.jit_flat (fun a -> [ grad_of f (sole a) ]) [ scalar 2.0 ] in
  check "jit of grad" 12.0 (value (sole jitted));
  let batched =
    Batching.vmap
      (fun a -> [ grad_of f (sole a) ])
      [ Some 0 ]
      [ vec [| 1.0; 2.0 |] ]
  in
  check "vmap of grad" 12.0 (Ndarray.get_f (to_nd (sole batched)) [| 1 |])

let () =
  Alcotest.run "custom_derivatives"
    [
      ( "custom_jvp",
        [
          Alcotest.test_case "jvp" `Quick jvp_uses_rule;
          Alcotest.test_case "grad" `Quick grad_through_jvp_rule;
          Alcotest.test_case "jit" `Quick jit_keeps_jvp_rule;
          Alcotest.test_case "vmap of grad" `Quick vmap_jvp_rule;
          Alcotest.test_case "grad of vmap" `Quick grad_of_vmap_jvp_rule;
        ] );
      ( "custom_vjp",
        [
          Alcotest.test_case "grad" `Quick vjp_uses_rule;
          Alcotest.test_case "chain" `Quick vjp_chain_rule;
          Alcotest.test_case "nested" `Quick vjp_nested;
          Alcotest.test_case "jit" `Quick vjp_jit;
          Alcotest.test_case "vmap of grad" `Quick vjp_vmap_of_grad;
          Alcotest.test_case "grad of vmap" `Quick vjp_grad_of_vmap;
          Alcotest.test_case "forward raises" `Quick vjp_forward_raises;
          Alcotest.test_case "constant input" `Quick vjp_unused_input;
        ] );
      ( "checkpoint",
        [
          Alcotest.test_case "recomputes" `Quick checkpoint_recomputes;
          Alcotest.test_case "derivative orders" `Quick checkpoint_orders;
          Alcotest.test_case "jit and vmap" `Quick checkpoint_jit_vmap;
        ] );
      ( "others",
        [
          Alcotest.test_case "custom_gradient" `Quick custom_gradient_rule;
          Alcotest.test_case "linear_call" `Quick linear_call_rule;
        ] );
    ]
