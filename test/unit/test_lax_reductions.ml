module L = Ojax.Lax
module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module J = Ojax.Jaxpr
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching

let () = L.install ()
let cval dtype shape xs = T.Concrete (Nd.of_floats dtype shape xs)
let f32 shape xs = cval D.F32 shape xs
let i32 shape xs = cval D.I32 shape xs

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

let shape_of v =
  match v with
  | T.Concrete a -> Nd.shape a
  | T.Tracer _ -> Alcotest.fail "not concrete"
  | T.Device _ -> Alcotest.fail "not concrete"

let dtype_of v =
  match v with
  | T.Concrete a -> Nd.dtype a
  | T.Tracer _ -> Alcotest.fail "not concrete"
  | T.Device _ -> Alcotest.fail "not concrete"

let farr = Alcotest.(array (float 1e-6))
let iarr = Alcotest.(array int)
let b1 = C.bind1
let mat = f32 [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |]

let add_reducer =
  let sc = { T.shape = [||]; dtype = D.F32; weak_type = false } in
  J.make_jaxpr [ sc; sc ] (fun a -> [ b1 T.Add a ])

let test_impl () =
  Alcotest.check farr "reduce_max ax0" [| 3.; 4.; 5. |]
    (out_floats (b1 (T.Reduce_max [| 0 |]) [ mat ]));
  Alcotest.check farr "reduce_max ax1" [| 2.; 5. |]
    (out_floats (b1 (T.Reduce_max [| 1 |]) [ mat ]));
  Alcotest.check farr "reduce_min ax1" [| 0.; 3. |]
    (out_floats (b1 (T.Reduce_min [| 1 |]) [ mat ]));
  Alcotest.check farr "reduce_prod ax0" [| 0.; 4.; 10. |]
    (out_floats (b1 (T.Reduce_prod [| 0 |]) [ mat ]));
  let ix = i32 [| 2; 3 |] [| 7.; 3.; 5.; 6.; 2.; 4. |] in
  Alcotest.check farr "reduce_and ax0" [| 6.; 2.; 4. |]
    (out_floats (b1 (T.Reduce_and [| 0 |]) [ ix ]));
  Alcotest.check farr "reduce_or ax0" [| 7.; 3.; 5. |]
    (out_floats (b1 (T.Reduce_or [| 0 |]) [ ix ]));
  Alcotest.check farr "reduce_xor ax0" [| 1.; 1.; 1. |]
    (out_floats (b1 (T.Reduce_xor [| 0 |]) [ ix ]));
  let bx = cval D.Bool [| 2; 3 |] [| 1.; 0.; 1.; 1.; 1.; 0. |] in
  Alcotest.check farr "reduce_and bool ax0" [| 1.; 0.; 0. |]
    (out_floats (b1 (T.Reduce_and [| 0 |]) [ bx ]))

let test_argminmax () =
  let a = b1 (T.Argmax { axis = 0; index_dtype = D.I32 }) [ mat ] in
  Alcotest.check iarr "argmax ax0 dtype" [| 3 |] (shape_of a);
  Alcotest.check farr "argmax ax0" [| 1.; 1.; 1. |] (out_floats a);
  (match dtype_of a with
  | D.I32 -> ()
  | _ -> Alcotest.fail "argmax dtype not i32");
  Alcotest.check farr "argmax ax1" [| 2.; 2. |]
    (out_floats (b1 (T.Argmax { axis = 1; index_dtype = D.I32 }) [ mat ]));
  Alcotest.check farr "argmin ax0" [| 0.; 0.; 0. |]
    (out_floats (b1 (T.Argmin { axis = 0; index_dtype = D.I32 }) [ mat ]))

let test_reduce_general () =
  let init = f32 [||] [| 10.0 |] in
  let r =
    b1 (T.Reduce { jaxpr = add_reducer; dimensions = [| 0 |] }) [ mat; init ]
  in
  Alcotest.check iarr "reduce shape" [| 3 |] (shape_of r);
  Alcotest.check farr "reduce add init" [| 13.; 15.; 17. |] (out_floats r)

let test_jvp () =
  let tx = f32 [| 2; 3 |] [| 10.; 20.; 30.; 40.; 50.; 60. |] in
  let _, to_ =
    Ad.jvp (fun a -> [ b1 (T.Reduce_max [| 0 |]) a ]) [ mat ] [ tx ]
  in
  Alcotest.check farr "reduce_max jvp" [| 40.; 50.; 60. |]
    (out_floats (List.hd to_));
  let _, to2 =
    Ad.jvp (fun a -> [ b1 (T.Reduce_min [| 1 |]) a ]) [ mat ] [ tx ]
  in
  Alcotest.check farr "reduce_min jvp" [| 10.; 40. |] (out_floats (List.hd to2))

let test_grad () =
  let f a =
    let m = b1 (T.Reduce_max [| 0 |]) a in
    b1 (T.Reduce_sum [| 0 |]) [ m ]
  in
  let g = Ad.grad f [ mat ] in
  Alcotest.check farr "grad reduce_max"
    [| 0.; 0.; 0.; 1.; 1.; 1. |]
    (out_floats (List.hd g))

let test_prod_jvp_raises () =
  match Ad.jvp (fun a -> [ b1 (T.Reduce_prod [| 0 |]) a ]) [ mat ] [ mat ] with
  | _ -> Alcotest.fail "reduce_prod jvp should raise"
  | exception Failure _ -> ()

let test_vmap () =
  let x =
    f32 [| 2; 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11. |]
  in
  let r =
    Batching.vmap (fun a -> [ b1 (T.Reduce_max [| 0 |]) a ]) [ Some 0 ] [ x ]
  in
  Alcotest.check iarr "vmap reduce_max shape" [| 2; 3 |] (shape_of (List.hd r));
  Alcotest.check farr "vmap reduce_max"
    [| 3.; 4.; 5.; 9.; 10.; 11. |]
    (out_floats (List.hd r));
  let ra =
    Batching.vmap
      (fun a -> [ b1 (T.Argmax { axis = 0; index_dtype = D.I32 }) a ])
      [ Some 0 ] [ x ]
  in
  Alcotest.check farr "vmap argmax"
    [| 1.; 1.; 1.; 1.; 1.; 1. |]
    (out_floats (List.hd ra));
  let init = f32 [||] [| 0.0 |] in
  let rg =
    Batching.vmap
      (fun a ->
        match a with
        | [ op; iv ] ->
            [
              b1
                (T.Reduce { jaxpr = add_reducer; dimensions = [| 0 |] })
                [ op; iv ];
            ]
        | _ -> assert false)
      [ Some 0; None ] [ x; init ]
  in
  Alcotest.check iarr "vmap reduce shape" [| 2; 3 |] (shape_of (List.hd rg));
  Alcotest.check farr "vmap reduce"
    [| 3.; 5.; 7.; 15.; 17.; 19. |]
    (out_floats (List.hd rg))

let () =
  Alcotest.run "lax_reductions"
    [
      ( "reductions",
        [
          Alcotest.test_case "impl" `Quick test_impl;
          Alcotest.test_case "argminmax" `Quick test_argminmax;
          Alcotest.test_case "reduce-general" `Quick test_reduce_general;
          Alcotest.test_case "jvp" `Quick test_jvp;
          Alcotest.test_case "grad" `Quick test_grad;
          Alcotest.test_case "prod-jvp-raises" `Quick test_prod_jvp_raises;
          Alcotest.test_case "vmap" `Quick test_vmap;
        ] );
    ]
