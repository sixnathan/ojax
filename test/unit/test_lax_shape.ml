module L = Ojax.Lax
module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching

let () = L.install ()
let cval dtype shape xs = T.Concrete (Nd.of_floats dtype shape xs)
let f32 shape xs = cval D.F32 shape xs

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

let shape_of v =
  match v with
  | T.Concrete a -> Nd.shape a
  | T.Tracer _ -> Alcotest.fail "not concrete"

let farr = Alcotest.(array (float 1e-6))
let iarr = Alcotest.(array int)
let b1 = C.bind1
let bind = C.bind

let sum_all v =
  let a = C.get_aval v in
  let axes = Array.init (Array.length a.T.shape) (fun i -> i) in
  b1 (T.Reduce_sum axes) [ v ]

let two = f32 [||] [| 2.0 |]
let mat = f32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |]

let test_impl () =
  let x = f32 [| 3 |] [| 1.; 2.; 3. |] in
  let y = f32 [| 3 |] [| 4.; 5.; 6. |] in
  Alcotest.check farr "concatenate"
    [| 1.; 2.; 3.; 4.; 5.; 6. |]
    (out_floats (b1 (T.Concatenate 0) [ x; y ]));
  Alcotest.check farr "stack"
    [| 1.; 2.; 3.; 4.; 5.; 6. |]
    (out_floats (b1 (T.Stack 0) [ x; y ]));
  let pv = f32 [||] [| 0.0 |] in
  let padded = b1 (T.Pad [| (1, 0, 0); (0, 1, 1) |]) [ mat; pv ] in
  Alcotest.check iarr "pad shape" [| 3; 6 |] (shape_of padded);
  Alcotest.check farr "pad"
    [| 0.; 0.; 0.; 0.; 0.; 0.; 0.; 0.; 1.; 0.; 2.; 0.; 3.; 0.; 4.; 0.; 5.; 0. |]
    (out_floats padded);
  Alcotest.check farr "rev"
    [| 2.; 1.; 0.; 5.; 4.; 3. |]
    (out_floats (b1 (T.Rev [| 1 |]) [ mat ]));
  Alcotest.check farr "transpose"
    [| 0.; 3.; 1.; 4.; 2.; 5. |]
    (out_floats (b1 (T.Transpose [| 1; 0 |]) [ mat ]));
  Alcotest.check farr "tile" [| 1.; 2.; 1.; 2. |]
    (out_floats (b1 (T.Tile [| 2 |]) [ f32 [| 2 |] [| 1.; 2. |] ]));
  Alcotest.check farr "squeeze"
    [| 0.; 1.; 2.; 3.; 4.; 5. |]
    (out_floats (b1 (T.Squeeze [| 0 |]) [ f32 [| 1; 2; 3 |] (out_floats mat) ]));
  let sp =
    bind
      (T.Split { sizes = [| 1; 3 |]; axis = 0 })
      [ f32 [| 4 |] [| 0.; 1.; 2.; 3. |] ]
  in
  Alcotest.check farr "split0" [| 0. |] (out_floats (List.nth sp 0));
  Alcotest.check farr "split1" [| 1.; 2.; 3. |] (out_floats (List.nth sp 1));
  let us = bind (T.Unstack 0) [ mat ] in
  Alcotest.check farr "unstack0" [| 0.; 1.; 2. |] (out_floats (List.nth us 0));
  Alcotest.check farr "unstack1" [| 3.; 4.; 5. |] (out_floats (List.nth us 1))

let test_jvp () =
  let x = f32 [| 3 |] [| 1.; 2.; 3. |] in
  let y = f32 [| 3 |] [| 4.; 5.; 6. |] in
  let tx = f32 [| 3 |] [| 0.5; 0.5; 0.5 |] in
  let ty = f32 [| 3 |] [| 1.; 2.; 3. |] in
  let _, to_ =
    Ad.jvp (fun a -> [ b1 (T.Concatenate 0) a ]) [ x; y ] [ tx; ty ]
  in
  Alcotest.check farr "concat jvp"
    [| 0.5; 0.5; 0.5; 1.; 2.; 3. |]
    (out_floats (List.hd to_));
  let tm = f32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let _, tt =
    Ad.jvp (fun a -> [ b1 (T.Transpose [| 1; 0 |]) a ]) [ mat ] [ tm ]
  in
  Alcotest.check farr "transpose jvp"
    [| 1.; 4.; 2.; 5.; 3.; 6. |]
    (out_floats (List.hd tt));
  let pv = f32 [||] [| 0.0 |] in
  let tpv = f32 [||] [| 0.0 |] in
  let cfg = [| (1, 0, 0); (0, 1, 1) |] in
  let _, tp = Ad.jvp (fun a -> [ b1 (T.Pad cfg) a ]) [ mat; pv ] [ tm; tpv ] in
  Alcotest.check farr "pad jvp"
    (out_floats (b1 (T.Pad cfg) [ tm; tpv ]))
    (out_floats (List.hd tp))

let grad1 f x = List.hd (Ad.grad f [ x ])

let test_grad () =
  let x3 = f32 [| 3 |] [| 1.; 2.; 3. |] in
  let g_transpose =
    grad1 (fun a -> sum_all (b1 (T.Transpose [| 1; 0 |]) a)) mat
  in
  Alcotest.check farr "transpose grad"
    [| 1.; 1.; 1.; 1.; 1.; 1. |]
    (out_floats g_transpose);
  let g_tile =
    grad1 (fun a -> sum_all (b1 (T.Tile [| 2 |]) a)) (f32 [| 2 |] [| 1.; 2. |])
  in
  Alcotest.check farr "tile grad" [| 2.; 2. |] (out_floats g_tile);
  let g_rev = grad1 (fun a -> sum_all (b1 (T.Rev [| 1 |]) a)) mat in
  Alcotest.check farr "rev grad" [| 1.; 1.; 1.; 1.; 1.; 1. |] (out_floats g_rev);
  let g_squeeze =
    grad1
      (fun a -> sum_all (b1 (T.Squeeze [| 0 |]) a))
      (f32 [| 1; 3 |] [| 1.; 2.; 3. |])
  in
  Alcotest.check farr "squeeze grad" [| 1.; 1.; 1. |] (out_floats g_squeeze);
  let split_fn a =
    let outs = bind (T.Split { sizes = [| 1; 3 |]; axis = 0 }) a in
    let s0 = sum_all (List.nth outs 0) in
    let s1 = sum_all (List.nth outs 1) in
    b1 T.Add [ s0; b1 T.Mul [ two; s1 ] ]
  in
  let g_split = grad1 split_fn (f32 [| 4 |] [| 0.; 1.; 2.; 3. |]) in
  Alcotest.check farr "split grad" [| 1.; 2.; 2.; 2. |] (out_floats g_split);
  let unstack_fn a =
    let outs = bind (T.Unstack 0) a in
    let s0 = sum_all (List.nth outs 0) in
    let s1 = sum_all (List.nth outs 1) in
    b1 T.Add [ s0; b1 T.Mul [ two; s1 ] ]
  in
  let g_unstack = grad1 unstack_fn mat in
  Alcotest.check farr "unstack grad"
    [| 1.; 1.; 1.; 2.; 2.; 2. |]
    (out_floats g_unstack);
  let gs = Ad.grad (fun a -> sum_all (b1 (T.Concatenate 0) a)) [ x3; x3 ] in
  Alcotest.check farr "concat grad x" [| 1.; 1.; 1. |]
    (out_floats (List.nth gs 0));
  Alcotest.check farr "concat grad y" [| 1.; 1.; 1. |]
    (out_floats (List.nth gs 1));
  let gstk = Ad.grad (fun a -> sum_all (b1 (T.Stack 0) a)) [ x3; x3 ] in
  Alcotest.check farr "stack grad x" [| 1.; 1.; 1. |]
    (out_floats (List.nth gstk 0));
  Alcotest.check farr "stack grad y" [| 1.; 1.; 1. |]
    (out_floats (List.nth gstk 1))

let test_pad_grad_raises () =
  let pv = f32 [||] [| 0.0 |] in
  let cfg = [| (1, 0, 0); (0, 1, 1) |] in
  match
    Ad.grad (fun a -> sum_all (b1 (T.Pad cfg) [ List.hd a; pv ])) [ mat ]
  with
  | _ -> Alcotest.fail "expected pad grad to raise"
  | exception Failure _ -> ()

let test_vmap () =
  let x = f32 [| 3; 2 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let r_tile =
    Batching.vmap (fun a -> [ b1 (T.Tile [| 2 |]) a ]) [ Some 0 ] [ x ]
  in
  Alcotest.check iarr "vmap tile shape" [| 3; 4 |] (shape_of (List.hd r_tile));
  Alcotest.check farr "vmap tile"
    [| 1.; 2.; 1.; 2.; 3.; 4.; 3.; 4.; 5.; 6.; 5.; 6. |]
    (out_floats (List.hd r_tile));
  let xs =
    f32 [| 3; 4 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11.; 12. |]
  in
  let r_split =
    Batching.vmap
      (fun a -> bind (T.Split { sizes = [| 2; 2 |]; axis = 0 }) a)
      [ Some 0 ] [ xs ]
  in
  Alcotest.check iarr "vmap split0 shape" [| 3; 2 |]
    (shape_of (List.nth r_split 0));
  Alcotest.check farr "vmap split0"
    [| 1.; 2.; 5.; 6.; 9.; 10. |]
    (out_floats (List.nth r_split 0));
  Alcotest.check farr "vmap split1"
    [| 3.; 4.; 7.; 8.; 11.; 12. |]
    (out_floats (List.nth r_split 1));
  let a2 = f32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let b2 = f32 [| 2; 3 |] [| 7.; 8.; 9.; 10.; 11.; 12. |] in
  let r_cat =
    Batching.vmap
      (fun a -> [ b1 (T.Concatenate 0) a ])
      [ Some 0; Some 0 ] [ a2; b2 ]
  in
  Alcotest.check iarr "vmap concat shape" [| 2; 6 |] (shape_of (List.hd r_cat));
  Alcotest.check farr "vmap concat"
    [| 1.; 2.; 3.; 7.; 8.; 9.; 4.; 5.; 6.; 10.; 11.; 12. |]
    (out_floats (List.hd r_cat));
  let r_unstack =
    Batching.vmap (fun a -> bind (T.Unstack 0) a) [ Some 0 ] [ a2 ]
  in
  Alcotest.check iarr "vmap unstack0 shape" [| 2 |]
    (shape_of (List.nth r_unstack 0));
  Alcotest.check farr "vmap unstack0" [| 1.; 4. |]
    (out_floats (List.nth r_unstack 0))

let () =
  Alcotest.run "lax_shape"
    [
      ( "shape-layout",
        [
          Alcotest.test_case "impl" `Quick test_impl;
          Alcotest.test_case "jvp" `Quick test_jvp;
          Alcotest.test_case "grad" `Quick test_grad;
          Alcotest.test_case "pad-grad-raises" `Quick test_pad_grad_raises;
          Alcotest.test_case "vmap" `Quick test_vmap;
        ] );
    ]
