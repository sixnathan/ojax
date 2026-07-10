module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module L = Ojax.Lax
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
let close4 = Alcotest.(check (float 1e-4))
let flist = Alcotest.(check (list (float 1e-4)))
let scalar64 x = cval D.F64 [| 1 |] [| x |]
let bind1 prim v = get0 (C.bind1 prim [ v ])
let bind2 prim a b = get0 (C.bind1 prim [ scalar a; scalar b ])
let bind2_64 prim a b = get0 (C.bind1 prim [ scalar64 a; scalar64 b ])
let bind3 prim a b c = get0 (C.bind1 prim [ scalar a; scalar b; scalar c ])

let unary_impls () =
  close "lgamma" 0.2846829 (bind1 T.Lgamma (scalar 2.5));
  close "digamma" 0.7031566 (bind1 T.Digamma (scalar 2.5));
  close "erf" 0.6778012 (bind1 T.Erf (scalar 0.7));
  close "erfc" 0.3221988 (bind1 T.Erfc (scalar 0.7));
  close4 "erf_inv" 0.2724627 (bind1 T.Erf_inv (scalar 0.3));
  close "bessel_i0e" 0.3085083 (bind1 T.Bessel_i0e (scalar 2.0));
  close "bessel_i1e" 0.2152693 (bind1 T.Bessel_i1e (scalar 2.0))

let nary_impls () =
  close "igamma" 0.4421746 (bind2 T.Igamma 2.0 1.5);
  close "igammac" 0.5578254 (bind2 T.Igammac 2.0 1.5);
  close "igamma_grad_a" (-0.3134886) (bind2_64 T.Igamma_grad_a 2.0 1.5);
  close "zeta" 0.2020569 (bind2 T.Zeta 3.0 2.0);
  close "polygamma" 0.4903578 (bind2 T.Polygamma 1.0 2.5);
  close4 "beta" 0.5248 (bind3 T.Regularized_incomplete_beta 2.0 3.0 0.4)

let jvp1 prim x =
  let _, to_ =
    Ad.jvp (fun a -> [ C.bind1 prim a ]) [ scalar x ] [ scalar 1.0 ]
  in
  get0 (List.hd to_)

let jvp_arg prim a b ~ta ~tb =
  let _, to_ =
    Ad.jvp
      (fun args -> [ C.bind1 prim args ])
      [ scalar a; scalar b ]
      [ scalar ta; scalar tb ]
  in
  get0 (List.hd to_)

let two_over_sqrt_pi = 2.0 /. Float.sqrt (4.0 *. Float.atan 1.0)

let jvps () =
  close4 "lgamma'" 0.7031566 (jvp1 T.Lgamma 2.5);
  close4 "digamma'" 0.4903578 (jvp1 T.Digamma 2.5);
  close4 "erf'"
    (two_over_sqrt_pi *. Float.exp (-.(0.5 *. 0.5)))
    (jvp1 T.Erf 0.5);
  close4 "erfc'"
    (-.two_over_sqrt_pi *. Float.exp (-.(0.5 *. 0.5)))
    (jvp1 T.Erfc 0.5);
  close4 "bessel_i0e'" (0.2152693 -. 0.3085083) (jvp1 T.Bessel_i0e 2.0);
  let gradx = Float.exp (-1.5 +. ((2.0 -. 1.0) *. Float.log 1.5) -. 0.0) in
  close4 "igamma_x'" gradx (jvp_arg T.Igamma 2.0 1.5 ~ta:0.0 ~tb:1.0);
  close4 "igammac_x'" (-.gradx) (jvp_arg T.Igammac 2.0 1.5 ~ta:0.0 ~tb:1.0)

let vmaps () =
  let xs = cval D.F32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let out = Batching.vmap (fun a -> [ C.bind1 T.Lgamma a ]) [ Some 0 ] [ xs ] in
  flist "vmap lgamma"
    [ 0.0; 0.0; Float.log 2.0 ]
    (Array.to_list (out_floats (List.hd out)));
  let a = cval D.F32 [| 2 |] [| 2.0; 3.0 |] in
  let x = cval D.F32 [| 2 |] [| 1.5; 2.5 |] in
  let out2 =
    Batching.vmap
      (fun args -> [ C.bind1 T.Igamma args ])
      [ Some 0; Some 0 ] [ a; x ]
  in
  flist "vmap igamma"
    [ bind2 T.Igamma 2.0 1.5; bind2 T.Igamma 3.0 2.5 ]
    (Array.to_list (out_floats (List.hd out2)))

let () =
  Alcotest.run "lax_special"
    [
      ( "impl",
        [
          Alcotest.test_case "unary" `Quick unary_impls;
          Alcotest.test_case "nary" `Quick nary_impls;
        ] );
      ("jvp", [ Alcotest.test_case "derivatives" `Quick jvps ]);
      ("vmap", [ Alcotest.test_case "vectorized" `Quick vmaps ]);
    ]
