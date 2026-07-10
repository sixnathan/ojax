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

let close = Alcotest.(check (float 1e-6))
let flist = Alcotest.(check (list (float 1e-6)))

let bind2 prim a b =
  match C.bind prim [ a; b ] with [ r ] -> r | _ -> Alcotest.fail "arity"

let impls () =
  let ai = cval D.I32 [| 3 |] [| 3.0; 0.0; 5.0 |] in
  let bi = cval D.I32 [| 3 |] [| 1.0; 2.0; 5.0 |] in
  flist "ne int32" [ 1.0; 1.0; 0.0 ]
    (Array.to_list (out_floats (bind2 T.Ne ai bi)));
  flist "or int32" [ 3.0; 2.0; 5.0 ]
    (Array.to_list (out_floats (bind2 T.Or ai bi)));
  flist "xor int32" [ 2.0; 2.0; 0.0 ]
    (Array.to_list (out_floats (bind2 T.Xor ai bi)));
  let ab = cval D.Bool [| 3 |] [| 1.0; 0.0; 1.0 |] in
  let bb = cval D.Bool [| 3 |] [| 1.0; 1.0; 0.0 |] in
  flist "or bool" [ 1.0; 1.0; 1.0 ]
    (Array.to_list (out_floats (bind2 T.Or ab bb)));
  flist "xor bool" [ 0.0; 1.0; 1.0 ]
    (Array.to_list (out_floats (bind2 T.Xor ab bb)));
  let rf = cval D.F32 [| 2 |] [| 7.5; -7.5 |] in
  let rg = cval D.F32 [| 2 |] [| 2.0; 2.0 |] in
  flist "rem float" [ 1.5; -1.5 ]
    (Array.to_list (out_floats (bind2 T.Rem rf rg)));
  let ri = cval D.I32 [| 2 |] [| 7.0; -7.0 |] in
  let rj = cval D.I32 [| 2 |] [| 3.0; 3.0 |] in
  flist "rem int32" [ 1.0; -1.0 ]
    (Array.to_list (out_floats (bind2 T.Rem ri rj)));
  let sv = cval D.I32 [| 3 |] [| 1.0; 3.0; 5.0 |] in
  let sh = cval D.I32 [| 3 |] [| 2.0; 1.0; 3.0 |] in
  flist "shift_left int32" [ 4.0; 6.0; 40.0 ]
    (Array.to_list (out_floats (bind2 T.Shift_left sv sh)));
  let nv = cval D.I32 [| 3 |] [| -8.0; -3.0; 5.0 |] in
  let nh = cval D.I32 [| 3 |] [| 1.0; 1.0; 1.0 |] in
  flist "shift_right_arithmetic int32" [ -4.0; -2.0; 2.0 ]
    (Array.to_list (out_floats (bind2 T.Shift_right_arithmetic nv nh)));
  let lv = cval D.I32 [| 2 |] [| -8.0; 255.0 |] in
  let lh = cval D.I32 [| 2 |] [| 1.0; 4.0 |] in
  flist "shift_right_logical int32" [ 2147483644.0; 15.0 ]
    (Array.to_list (out_floats (bind2 T.Shift_right_logical lv lh)))

let nextafters () =
  let x = cval D.F32 [| 1 |] [| 1.0 |] in
  let y = cval D.F32 [| 1 |] [| 2.0 |] in
  let expected = Int32.float_of_bits (Int32.add (Int32.bits_of_float 1.0) 1l) in
  let r = get0 (bind2 T.Nextafter x y) in
  Alcotest.(check bool) "nextafter up moved" true (r > 1.0);
  Alcotest.(check bool) "nextafter up exact" true (r = expected);
  let y2 = cval D.F32 [| 1 |] [| 0.0 |] in
  let r2 = get0 (bind2 T.Nextafter x y2) in
  Alcotest.(check bool) "nextafter down moved" true (r2 < 1.0)

let jvp2 prim (x, y) (tx, ty) =
  let _, to_ = Ad.jvp (fun a -> [ C.bind1 prim a ]) [ x; y ] [ tx; ty ] in
  get0 (List.hd to_)

let s x = cval D.F32 [| 1 |] [| x |]

let jvps () =
  close "rem d/dx" 1.0 (jvp2 T.Rem (s 7.5, s 2.0) (s 1.0, s 0.0));
  close "rem d/dy" (-3.0) (jvp2 T.Rem (s 7.5, s 2.0) (s 0.0, s 1.0));
  close "ne zero-tangent" 0.0 (jvp2 T.Ne (s 1.0, s 2.0) (s 1.0, s 1.0))

let vmaps () =
  let xs = cval D.I32 [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let ys = cval D.I32 [| 3 |] [| 1.0; 1.0; 1.0 |] in
  let out =
    Batching.vmap
      (fun a -> [ C.bind1 T.Shift_left a ])
      [ Some 0; Some 0 ] [ xs; ys ]
  in
  flist "vmap shift_left" [ 2.0; 4.0; 6.0 ]
    (Array.to_list (out_floats (List.hd out)))

let () =
  Alcotest.run "lax_binary_b"
    [
      ("impl", [ Alcotest.test_case "binary" `Quick impls ]);
      ("nextafter", [ Alcotest.test_case "nextafter" `Quick nextafters ]);
      ("jvp", [ Alcotest.test_case "rem" `Quick jvps ]);
      ("vmap", [ Alcotest.test_case "shift_left" `Quick vmaps ]);
    ]
