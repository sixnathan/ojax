module Batching = Ojax.Interpreters.Batching
module Ad = Ojax.Interpreters.Ad
module C = Ojax.Core
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray

let () = Ojax.Lax.install ()
let b1 = C.bind1
let ndf = Nd.of_floats
let mat r c xs = T.Concrete (ndf D.F32 [| r; c |] xs)
let vec xs = T.Concrete (ndf D.F32 [| Array.length xs |] xs)
let as_nd v = match v with T.Concrete a -> a | _ -> Alcotest.fail "concrete"

let getij v i j =
  match v with T.Concrete a -> Nd.get_f a [| i; j |] | _ -> Alcotest.fail "nd"

let test_mapped_aval () =
  let a : T.aval = { shape = [| 5; 3 |]; dtype = D.F32; weak_type = false } in
  let m = Batching.mapped_aval 0 a in
  Alcotest.(check (array int)) "mapped" [| 3 |] m.shape;
  let u = Batching.unmapped_aval 5 0 m in
  Alcotest.(check (array int)) "unmapped" [| 5; 3 |] u.shape

let test_vmap_unary () =
  let x = mat 2 3 [| 0.1; 0.2; 0.3; 0.4; 0.5; 0.6 |] in
  let out = Batching.vmap (fun a -> [ b1 T.Sin a ]) [ Some 0 ] [ x ] in
  let o = List.hd out in
  for i = 0 to 1 do
    for j = 0 to 2 do
      let expected = sin (Nd.get_f (as_nd x) [| i; j |]) in
      Alcotest.(check (float 1e-6)) "sin" expected (getij o i j)
    done
  done

let test_vmap_moveaxis () =
  let x = mat 3 2 [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let out = Batching.vmap (fun a -> [ b1 T.Neg a ]) [ Some 1 ] [ x ] in
  let o = List.hd out in
  Alcotest.(check (array int)) "shape" [| 2; 3 |] (Nd.shape (as_nd o));
  Alcotest.(check (float 1e-6)) "neg[0,1]" (-3.0) (getij o 0 1)

let test_vmap_binop_broadcast () =
  let x = mat 2 3 [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let y = vec [| 10.0; 20.0; 30.0 |] in
  let out = Batching.vmap (fun a -> [ b1 T.Add a ]) [ Some 0; None ] [ x; y ] in
  let o = List.hd out in
  Alcotest.(check (float 1e-6)) "add[1,2]" 36.0 (getij o 1 2)

let test_jacfwd_via_vmap () =
  let x = vec [| 0.3; 0.7; 1.1 |] in
  let basis = mat 3 3 [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |] in
  let pushfwd a =
    match a with
    | [ x; t ] ->
        let _, to_ = Ad.jvp (fun z -> [ b1 T.Sin z ]) [ x ] [ t ] in
        [ List.hd to_ ]
    | _ -> assert false
  in
  let out = Batching.vmap pushfwd [ None; Some 0 ] [ x; basis ] in
  let o = List.hd out in
  Alcotest.(check (array int)) "jac shape" [| 3; 3 |] (Nd.shape (as_nd o));
  for i = 0 to 2 do
    let expected = cos (Nd.get_f (as_nd x) [| i |]) in
    Alcotest.(check (float 1e-6)) "diag cos" expected (getij o i i);
    Alcotest.(check (float 1e-6)) "off diag" 0.0 (getij o i ((i + 1) mod 3))
  done

let () =
  Alcotest.run "batching"
    [
      ( "batching",
        [
          Alcotest.test_case "mapped_aval" `Quick test_mapped_aval;
          Alcotest.test_case "vmap_unary" `Quick test_vmap_unary;
          Alcotest.test_case "vmap_moveaxis" `Quick test_vmap_moveaxis;
          Alcotest.test_case "vmap_binop_broadcast" `Quick
            test_vmap_binop_broadcast;
          Alcotest.test_case "jacfwd_via_vmap" `Quick test_jacfwd_via_vmap;
        ] );
    ]
