module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching
module L = Ojax.Lax

let () = L.install ()
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)

let dims1 =
  {
    T.lhs_spec = [| 0; 1; 2 |];
    rhs_spec = [| 0; 1; 2 |];
    out_spec = [| 0; 1; 2 |];
  }

let dims2 =
  {
    T.lhs_spec = [| 0; 1; 2; 3 |];
    rhs_spec = [| 0; 1; 2; 3 |];
    out_spec = [| 0; 1; 2; 3 |];
  }

let conv ~ws ~pad ~ld ~rd ~dims ?(fgc = 1) lhs rhs =
  C.bind1
    (T.Conv_general_dilated
       {
         window_strides = ws;
         padding = pad;
         lhs_dilation = ld;
         rhs_dilation = rd;
         dimension_numbers = dims;
         feature_group_count = fgc;
         batch_group_count = 1;
       })
    [ lhs; rhs ]

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
  | T.Tracer _ -> Alcotest.fail "tracer"
  | T.Device _ -> Alcotest.fail "tracer"

let farr = Alcotest.(check (array (float 1e-6)))
let iarr = Alcotest.(check (array int))

let conv1d_impl () =
  let lhs = f32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let rhs = f32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
  let y =
    conv lhs rhs ~ws:[| 1 |]
      ~pad:[| (0, 0) |]
      ~ld:[| 1 |] ~rd:[| 1 |] ~dims:dims1
  in
  iarr "conv1d shape" [| 1; 1; 3 |] (shape_of y);
  farr "conv1d vals" [| -2.; -2.; -2. |] (out_floats y)

let conv2d_impl () =
  let lhs =
    f32 [| 1; 1; 3; 3 |] (Array.init 9 (fun i -> float_of_int (i + 1)))
  in
  let rhs = f32 [| 1; 1; 2; 2 |] [| 1.; 1.; 1.; 1. |] in
  let y =
    conv lhs rhs ~ws:[| 1; 1 |]
      ~pad:[| (0, 0); (0, 0) |]
      ~ld:[| 1; 1 |] ~rd:[| 1; 1 |] ~dims:dims2
  in
  iarr "conv2d shape" [| 1; 1; 2; 2 |] (shape_of y);
  farr "conv2d vals" [| 12.; 16.; 24.; 28. |] (out_floats y)

let conv_depthwise_impl () =
  let lhs = f32 [| 1; 2; 1; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let rhs = f32 [| 2; 1; 1; 2 |] [| 1.; 1.; 1.; 1. |] in
  let y =
    conv lhs rhs ~ws:[| 1; 1 |]
      ~pad:[| (0, 0); (0, 0) |]
      ~ld:[| 1; 1 |] ~rd:[| 1; 1 |] ~dims:dims2 ~fgc:2
  in
  iarr "depthwise shape" [| 1; 2; 1; 2 |] (shape_of y);
  farr "depthwise vals" [| 3.; 5.; 9.; 11. |] (out_floats y)

let jvp_conv () =
  let lhs = f32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let rhs = f32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
  let t_lhs = f32 [| 1; 1; 5 |] (Array.make 5 0.0) in
  let t_rhs = f32 [| 1; 1; 3 |] [| 1.; 1.; 1. |] in
  let fn a =
    match a with
    | [ l; r ] ->
        [
          conv l r ~ws:[| 1 |]
            ~pad:[| (0, 0) |]
            ~ld:[| 1 |] ~rd:[| 1 |] ~dims:dims1;
        ]
    | _ -> assert false
  in
  let po, to_ = Ad.jvp fn [ lhs; rhs ] [ t_lhs; t_rhs ] in
  farr "jvp primal" [| -2.; -2.; -2. |] (out_floats (List.hd po));
  farr "jvp tangent" [| 6.; 9.; 12. |] (out_floats (List.hd to_))

let transpose_raises () =
  let lhs = f32 [| 1; 1; 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let rhs = f32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
  let fn a =
    C.bind1
      (T.Reduce_sum [| 0; 1; 2 |])
      [
        conv lhs (List.hd a) ~ws:[| 1 |]
          ~pad:[| (0, 0) |]
          ~ld:[| 1 |] ~rd:[| 1 |] ~dims:dims1;
      ]
  in
  match Ad.grad fn [ rhs ] with
  | _ -> Alcotest.fail "expected conv transpose failure"
  | exception Failure _ -> ()

let vmap_raises () =
  let lhs = f32 [| 2; 1; 1; 5 |] (Array.init 10 float_of_int) in
  let rhs = f32 [| 1; 1; 3 |] [| 1.; 0.; -1. |] in
  let fn a =
    match a with
    | [ l ] ->
        [
          conv l rhs ~ws:[| 1 |]
            ~pad:[| (0, 0) |]
            ~ld:[| 1 |] ~rd:[| 1 |] ~dims:dims1;
        ]
    | _ -> assert false
  in
  match Batching.vmap fn [ Some 0 ] [ lhs ] with
  | _ -> Alcotest.fail "expected conv vmap failure"
  | exception Failure _ -> ()

let () =
  Alcotest.run "lax_convolution"
    [
      ( "impl",
        [
          Alcotest.test_case "conv1d" `Quick conv1d_impl;
          Alcotest.test_case "conv2d" `Quick conv2d_impl;
          Alcotest.test_case "depthwise" `Quick conv_depthwise_impl;
        ] );
      ("jvp", [ Alcotest.test_case "conv" `Quick jvp_conv ]);
      ("transpose", [ Alcotest.test_case "raises" `Quick transpose_raises ]);
      ("vmap", [ Alcotest.test_case "raises" `Quick vmap_raises ]);
    ]
