module Api = Ojax.Api
module Ad = Ojax.Interpreters.Ad
module C = Ojax.Core
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray
module Tree = Ojax.Tree_util

let () = Ojax.Lax.install ()
let b1 = C.bind1
let scalar x = T.Concrete (Nd.of_floats D.F32 [||] [| x |])
let vec xs = T.Concrete (Nd.of_floats D.F32 [| Array.length xs |] xs)

let get0 v =
  match v with T.Concrete a -> Nd.get_f a [||] | _ -> Alcotest.fail "concrete"

let geti v i =
  match v with
  | T.Concrete a -> Nd.get_f a [| i |]
  | _ -> Alcotest.fail "concrete"

let leaf v = Tree.Leaf v
let unleaf = function Tree.Leaf v -> v | _ -> Alcotest.fail "expected leaf"
let add a b = b1 T.Add [ a; b ]
let mul a b = b1 T.Mul [ a; b ]
let sub a b = b1 T.Sub [ a; b ]
let sin_ a = b1 T.Sin [ a ]

let poly args =
  match args with
  | [ x ] -> [ sub (mul (sin_ x) (scalar 2.0)) x ]
  | _ -> assert false

let poly_tree args = leaf (List.hd (poly (List.map unleaf args)))

let test_jit_eq_f () =
  let x = scalar 0.7 in
  let fx = List.hd (poly [ x ]) in
  let jx = List.hd (Api.jit_flat poly [ x ]) in
  Alcotest.(check (float 1e-6)) "jit==f" (get0 fx) (get0 jx)

let test_jit_memo () =
  let jf = Api.jit_flat poly in
  let a = get0 (List.hd (jf [ scalar 0.3 ])) in
  let b = get0 (List.hd (jf [ scalar 0.9 ])) in
  Alcotest.(check (float 1e-6))
    "reuse a"
    (get0 (List.hd (poly [ scalar 0.3 ])))
    a;
  Alcotest.(check (float 1e-6))
    "reuse b"
    (get0 (List.hd (poly [ scalar 0.9 ])))
    b

let test_jvp_of_jit () =
  let po, to_ = Ad.jvp (Api.jit_flat poly) [ scalar 0.5 ] [ scalar 1.0 ] in
  Alcotest.(check (float 1e-6))
    "primal"
    (get0 (List.hd (poly [ scalar 0.5 ])))
    (get0 (List.hd po));
  Alcotest.(check (float 1e-5))
    "tangent"
    ((cos 0.5 *. 2.0) -. 1.0)
    (get0 (List.hd to_))

let test_grad_of_jit () =
  let g = Ad.grad (fun a -> List.hd (Api.jit_flat poly a)) [ scalar 0.5 ] in
  Alcotest.(check (float 1e-5))
    "grad"
    ((cos 0.5 *. 2.0) -. 1.0)
    (get0 (List.hd g))

let test_vmap_of_jit () =
  let xs = vec [| 0.1; 0.2; 0.3 |] in
  let out =
    Ojax.Interpreters.Batching.vmap (Api.jit_flat poly) [ Some 0 ] [ xs ]
  in
  let o = List.hd out in
  for i = 0 to 2 do
    let x = geti xs i in
    Alcotest.(check (float 1e-6)) "vmap jit" ((sin x *. 2.0) -. x) (geti o i)
  done

let test_pytree_jit () =
  let jf = Api.jit poly_tree in
  let out = unleaf (jf [ leaf (scalar 0.7) ]) in
  Alcotest.(check (float 1e-6))
    "pytree jit"
    (get0 (List.hd (poly [ scalar 0.7 ])))
    (get0 out)

let test_pytree_grad () =
  let g = unleaf (Api.grad poly_tree [ leaf (scalar 0.5) ]) in
  Alcotest.(check (float 1e-5)) "pytree grad" ((cos 0.5 *. 2.0) -. 1.0) (get0 g)

let test_value_and_grad () =
  let v, g = Api.value_and_grad poly_tree [ leaf (scalar 0.5) ] in
  Alcotest.(check (float 1e-6))
    "value"
    (get0 (List.hd (poly [ scalar 0.5 ])))
    (get0 (unleaf v));
  Alcotest.(check (float 1e-5))
    "grad"
    ((cos 0.5 *. 2.0) -. 1.0)
    (get0 (unleaf g))

let test_pytree_vmap () =
  let xs = vec [| 0.1; 0.2; 0.3 |] in
  let out = unleaf (Api.vmap poly_tree [ Some 0 ] [ leaf xs ]) in
  for i = 0 to 2 do
    let x = geti xs i in
    Alcotest.(check (float 1e-6))
      "pytree vmap"
      ((sin x *. 2.0) -. x)
      (geti out i)
  done

let call1 f x = List.hd (Api.call (fun a -> [ f (List.hd a) ]) [ x ])
let call0 f = List.hd (Api.call (fun _ -> [ f () ]) [])

let fun_with_nested_calls_2 (x : T.value) : T.value =
  let one = scalar 1.0 in
  let bar (y : T.value) : T.value =
    let baz (w : T.value) : T.value =
      let q = List.hd (Api.call (fun _ -> [ y ]) [ x ]) in
      let q = add q (call0 (fun () -> y)) in
      let q =
        add q (List.hd (Api.call (fun a -> [ add w (List.hd a) ]) [ y ]))
      in
      let inner = call1 (fun _ -> mul (call1 (fun z -> sin_ z) x) y) one in
      add inner q
    in
    let p, t = Ad.jvp (fun a -> [ baz (List.hd a) ]) [ add x one ] [ y ] in
    add (List.hd t) (mul x (List.hd p))
  in
  call1 bar x

let test_nested_calls_2 () =
  let f a = [ fun_with_nested_calls_2 (List.hd a) ] in
  let direct = get0 (List.hd (f [ scalar 3.0 ])) in
  let jitted = get0 (List.hd (Api.jit_flat f [ scalar 3.0 ])) in
  Alcotest.(check (float 1e-5)) "nested jit==f" direct jitted;
  let xs = vec [| 0.0; 1.0; 2.0 |] in
  let out = Ojax.Interpreters.Batching.vmap f [ Some 0 ] [ xs ] in
  let o = List.hd out in
  for i = 0 to 2 do
    let d = get0 (List.hd (f [ scalar (geti xs i) ])) in
    Alcotest.(check (float 1e-5)) "nested vmap" d (geti o i)
  done

let () =
  Alcotest.run "api"
    [
      ( "jit",
        [
          Alcotest.test_case "jit_eq_f" `Quick test_jit_eq_f;
          Alcotest.test_case "jit_memo" `Quick test_jit_memo;
          Alcotest.test_case "jvp_of_jit" `Quick test_jvp_of_jit;
          Alcotest.test_case "grad_of_jit" `Quick test_grad_of_jit;
          Alcotest.test_case "vmap_of_jit" `Quick test_vmap_of_jit;
        ] );
      ( "pytree",
        [
          Alcotest.test_case "jit" `Quick test_pytree_jit;
          Alcotest.test_case "grad" `Quick test_pytree_grad;
          Alcotest.test_case "value_and_grad" `Quick test_value_and_grad;
          Alcotest.test_case "vmap" `Quick test_pytree_vmap;
        ] );
      ( "nested",
        [
          Alcotest.test_case "fun_with_nested_calls_2" `Quick
            test_nested_calls_2;
        ] );
    ]
