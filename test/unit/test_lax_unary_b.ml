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

let bind1 prim v =
  match C.bind prim [ v ] with [ r ] -> r | _ -> Alcotest.fail "arity"

let impl_scalar prim x = get0 (bind1 prim (scalar x))

let impls () =
  close "integer_pow" 8.0 (impl_scalar (T.Integer_pow 3) 2.0);
  close "log1p" (Float.log1p 0.5) (impl_scalar T.Log1p 0.5);
  close "logistic"
    (1.0 /. (1.0 +. Float.exp (-0.7)))
    (impl_scalar T.Logistic 0.7);
  close "round_up" 3.0 (impl_scalar T.Round 2.6);
  close "round_half" 3.0 (impl_scalar T.Round 2.5);
  close "round_neg" (-3.0) (impl_scalar T.Round (-2.5));
  close "rsqrt" 0.5 (impl_scalar T.Rsqrt 4.0);
  close "sinh" (Float.sinh 0.7) (impl_scalar T.Sinh 0.7);
  close "sqrt" 3.0 (impl_scalar T.Sqrt 9.0);
  close "square" 9.0 (impl_scalar T.Square 3.0);
  close "tan" (Float.tan 0.3) (impl_scalar T.Tan 0.3)

let is_finite_case () =
  let xs = cval D.F32 [| 3 |] [| 0.0; 1.5; -2.0 |] in
  flist "is_finite" [ 1.0; 1.0; 1.0 ]
    (Array.to_list (out_floats (bind1 T.Is_finite xs)))

let not_case () =
  let b = cval D.Bool [| 2 |] [| 0.0; 1.0 |] in
  flist "not bool" [ 1.0; 0.0 ] (Array.to_list (out_floats (bind1 T.Not b)));
  let i = cval D.I32 [| 2 |] [| 0.0; 5.0 |] in
  flist "not int" [ -1.0; -6.0 ] (Array.to_list (out_floats (bind1 T.Not i)))

let popcount_case () =
  let i = cval D.I32 [| 4 |] [| 0.0; 1.0; 7.0; 255.0 |] in
  flist "population_count" [ 0.0; 1.0; 3.0; 8.0 ]
    (Array.to_list (out_floats (bind1 T.Population_count i)))

let jvp_deriv prim x =
  let _, to_ =
    Ad.jvp (fun a -> [ C.bind1 prim a ]) [ scalar x ] [ scalar 1.0 ]
  in
  get0 (List.hd to_)

let jvps () =
  close "integer_pow'" 12.0 (jvp_deriv (T.Integer_pow 3) 2.0);
  close "log1p'" (1.0 /. 1.5) (jvp_deriv T.Log1p 0.5);
  let l = 1.0 /. (1.0 +. Float.exp (-0.7)) in
  close "logistic'" (l *. (1.0 -. l)) (jvp_deriv T.Logistic 0.7);
  close "sinh'" (Float.cosh 0.7) (jvp_deriv T.Sinh 0.7);
  close "sqrt'" 0.25 (jvp_deriv T.Sqrt 4.0);
  close "rsqrt'" (-0.0625) (jvp_deriv T.Rsqrt 4.0);
  close "square'" 6.0 (jvp_deriv T.Square 3.0);
  close "tan'" (1.0 +. (Float.tan 0.3 *. Float.tan 0.3)) (jvp_deriv T.Tan 0.3);
  close "round'" 0.0 (jvp_deriv T.Round 2.6);
  close "is_finite'" 0.0 (jvp_deriv T.Is_finite 1.0)

let vmaps () =
  let xs = cval D.F32 [| 3 |] [| 1.0; 4.0; 9.0 |] in
  let out = Batching.vmap (fun a -> [ C.bind1 T.Sqrt a ]) [ Some 0 ] [ xs ] in
  flist "vmap sqrt" [ 1.0; 2.0; 3.0 ] (Array.to_list (out_floats (List.hd out)))

let () =
  Alcotest.run "lax_unary_b"
    [
      ( "impl",
        [
          Alcotest.test_case "unary" `Quick impls;
          Alcotest.test_case "is_finite" `Quick is_finite_case;
          Alcotest.test_case "not" `Quick not_case;
          Alcotest.test_case "population_count" `Quick popcount_case;
        ] );
      ("jvp", [ Alcotest.test_case "derivatives" `Quick jvps ]);
      ("vmap", [ Alcotest.test_case "sqrt" `Quick vmaps ]);
    ]
