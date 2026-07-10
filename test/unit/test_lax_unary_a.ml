module L = Ojax.Lax
module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching

let () = L.install ()
let cval dtype shape xs = T.Concrete (Nd.of_floats dtype shape xs)
let scalar x = cval D.F32 [| 1 |] [| x |]

let get0 v =
  match v with
  | T.Concrete a -> Nd.get_f a [| 0 |]
  | T.Tracer _ -> Alcotest.fail "not concrete"
  | T.Device _ -> Alcotest.fail "not concrete"

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

let close = Alcotest.(check (float 1e-5))
let flist = Alcotest.(check (list (float 1e-5)))

let impl_scalar prim x =
  match C.bind prim [ scalar x ] with
  | [ v ] -> get0 v
  | _ -> Alcotest.fail "arity"

let impls () =
  close "acos" (acos 0.25) (impl_scalar T.Acos 0.25);
  close "acosh" (Float.acosh 1.5) (impl_scalar T.Acosh 1.5);
  close "asin" (asin 0.25) (impl_scalar T.Asin 0.25);
  close "asinh" (Float.asinh 0.7) (impl_scalar T.Asinh 0.7);
  close "atan" (atan 0.7) (impl_scalar T.Atan 0.7);
  close "atanh" (Float.atanh 0.3) (impl_scalar T.Atanh 0.3);
  close "cbrt" (Float.cbrt (-8.0)) (impl_scalar T.Cbrt (-8.0));
  close "ceil" 3.0 (impl_scalar T.Ceil 2.2);
  close "floor" 2.0 (impl_scalar T.Floor 2.8);
  close "cosh" (Float.cosh 0.7) (impl_scalar T.Cosh 0.7);
  close "exp2" 8.0 (impl_scalar T.Exp2 3.0);
  close "expm1" (Float.expm1 0.5) (impl_scalar T.Expm1 0.5);
  close "copy" 4.5 (impl_scalar T.Copy 4.5)

let clz_case () =
  match C.bind T.Clz [ cval D.I32 [| 4 |] [| 0.; 1.; 255.; 1024. |] ] with
  | [ v ] -> flist "clz" [ 32.; 31.; 24.; 21. ] (Array.to_list (out_floats v))
  | _ -> Alcotest.fail "arity"

let jvp_deriv prim x =
  let _, to_ =
    Ad.jvp (fun a -> [ C.bind1 prim a ]) [ scalar x ] [ scalar 1.0 ]
  in
  get0 (List.hd to_)

let jvps () =
  close "asin'" (1.0 /. sqrt (1.0 -. (0.25 *. 0.25))) (jvp_deriv T.Asin 0.25);
  close "acos'" (-1.0 /. sqrt (1.0 -. (0.25 *. 0.25))) (jvp_deriv T.Acos 0.25);
  close "atan'" (1.0 /. (1.0 +. (0.7 *. 0.7))) (jvp_deriv T.Atan 0.7);
  close "asinh'" (1.0 /. sqrt ((0.7 *. 0.7) +. 1.0)) (jvp_deriv T.Asinh 0.7);
  close "acosh'" (1.0 /. sqrt ((1.5 *. 1.5) -. 1.0)) (jvp_deriv T.Acosh 1.5);
  close "atanh'" (1.0 /. (1.0 -. (0.3 *. 0.3))) (jvp_deriv T.Atanh 0.3);
  close "cbrt'"
    (1.0 /. 3.0 *. Float.pow (Float.cbrt 2.0) (-2.0))
    (jvp_deriv T.Cbrt 2.0);
  close "cosh'" (Float.sinh 0.7) (jvp_deriv T.Cosh 0.7);
  close "exp2'" (Float.log 2.0 *. 8.0) (jvp_deriv T.Exp2 3.0);
  close "expm1'" (Float.exp 0.5) (jvp_deriv T.Expm1 0.5);
  close "ceil'" 0.0 (jvp_deriv T.Ceil 2.2);
  close "floor'" 0.0 (jvp_deriv T.Floor 2.2);
  close "copy'" 1.0 (jvp_deriv T.Copy 4.0)

let vmaps () =
  let xs = cval D.F32 [| 3 |] [| 0.1; 0.5; 0.9 |] in
  let out = Batching.vmap (fun a -> [ C.bind1 T.Cosh a ]) [ Some 0 ] [ xs ] in
  flist "vmap cosh"
    [ Float.cosh 0.1; Float.cosh 0.5; Float.cosh 0.9 ]
    (Array.to_list (out_floats (List.hd out)))

let () =
  Alcotest.run "lax_unary_a"
    [
      ( "impl",
        [
          Alcotest.test_case "unary" `Quick impls;
          Alcotest.test_case "clz" `Quick clz_case;
        ] );
      ("jvp", [ Alcotest.test_case "derivatives" `Quick jvps ]);
      ("vmap", [ Alcotest.test_case "cosh" `Quick vmaps ]);
    ]
