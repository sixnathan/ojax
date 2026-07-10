module L = Ojax.Lax
module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching

let () = L.install ()
let cval dtype shape xs = T.Concrete (Nd.of_floats dtype shape xs)

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

let get0 v =
  match v with
  | T.Concrete a -> Nd.get_f a [| 0 |]
  | T.Tracer _ -> Alcotest.fail "not concrete"
  | T.Device _ -> Alcotest.fail "not concrete"

let close = Alcotest.(check (float 1e-5))
let flist = Alcotest.(check (list (float 1e-5)))

let bind2 prim a b =
  match C.bind prim [ a; b ] with [ r ] -> r | _ -> Alcotest.fail "arity"

let impls () =
  let ai = cval D.I32 [| 3 |] [| 3.0; 0.0; 5.0 |] in
  let bi = cval D.I32 [| 3 |] [| 1.0; 2.0; 5.0 |] in
  flist "and int32" [ 1.0; 0.0; 5.0 ]
    (Array.to_list (out_floats (bind2 T.And ai bi)));
  flist "eq_to int32" [ 0.0; 0.0; 1.0 ]
    (Array.to_list (out_floats (bind2 T.Eq_to ai bi)));
  let ab = cval D.Bool [| 3 |] [| 1.0; 0.0; 1.0 |] in
  let bb = cval D.Bool [| 3 |] [| 1.0; 1.0; 0.0 |] in
  flist "and bool" [ 1.0; 0.0; 0.0 ]
    (Array.to_list (out_floats (bind2 T.And ab bb)));
  let af = cval D.F32 [| 2 |] [| 1.0; 2.0 |] in
  let bf = cval D.F32 [| 2 |] [| 2.0; 1.0 |] in
  flist "atan2"
    [ Float.atan2 1.0 2.0; Float.atan2 2.0 1.0 ]
    (Array.to_list (out_floats (bind2 T.Atan2 af bf)));
  flist "ge" [ 0.0; 1.0 ] (Array.to_list (out_floats (bind2 T.Ge af bf)));
  flist "le" [ 1.0; 0.0 ] (Array.to_list (out_floats (bind2 T.Le af bf)));
  flist "le_to" [ 1.0; 0.0 ] (Array.to_list (out_floats (bind2 T.Le_to af bf)));
  flist "lt_to" [ 1.0; 0.0 ] (Array.to_list (out_floats (bind2 T.Lt_to af bf)));
  let am = cval D.I32 [| 2 |] [| 100000.0; 2.0 |] in
  let bm = cval D.I32 [| 2 |] [| 100000.0; 3.0 |] in
  flist "mulhi int32" [ 2.0; 0.0 ]
    (Array.to_list (out_floats (bind2 T.Mulhi am bm)))

let jvp2 prim (x, y) (tx, ty) =
  let _, to_ = Ad.jvp (fun a -> [ C.bind1 prim a ]) [ x; y ] [ tx; ty ] in
  get0 (List.hd to_)

let s x = cval D.F32 [| 1 |] [| x |]

let jvps () =
  close "atan2 d/dx" 0.4 (jvp2 T.Atan2 (s 1.0, s 2.0) (s 1.0, s 0.0));
  close "atan2 d/dy" (-0.2) (jvp2 T.Atan2 (s 1.0, s 2.0) (s 0.0, s 1.0));
  close "ge zero-tangent" 0.0 (jvp2 T.Ge (s 1.0, s 2.0) (s 1.0, s 1.0))

let vmaps () =
  let xs = cval D.F32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let ys = cval D.F32 [| 3 |] [| 2.0; 1.0; 4.0 |] in
  let out =
    Batching.vmap (fun a -> [ C.bind1 T.Atan2 a ]) [ Some 0; Some 0 ] [ xs; ys ]
  in
  flist "vmap atan2"
    [ Float.atan2 1.0 2.0; Float.atan2 2.0 1.0; Float.atan2 3.0 4.0 ]
    (Array.to_list (out_floats (List.hd out)))

let () =
  Alcotest.run "lax_binary_a"
    [
      ("impl", [ Alcotest.test_case "binary" `Quick impls ]);
      ("jvp", [ Alcotest.test_case "atan2" `Quick jvps ]);
      ("vmap", [ Alcotest.test_case "atan2" `Quick vmaps ]);
    ]
