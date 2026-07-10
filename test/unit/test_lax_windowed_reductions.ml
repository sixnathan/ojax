module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching
module J = Ojax.Jaxpr
module L = Ojax.Lax

let () = L.install ()
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)

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

let win1 ~w ~s ~p ~bd ~wd : T.window_dims =
  {
    window_dimensions = [| w |];
    window_strides = [| s |];
    w_padding = [| p |];
    base_dilation = [| bd |];
    window_dilation = [| wd |];
  }

let ramp6 = f32 [| 6 |] [| 0.; 1.; 2.; 3.; 4.; 5. |]

let sum_impl () =
  let w = win1 ~w:2 ~s:1 ~p:(0, 0) ~bd:1 ~wd:1 in
  let y = C.bind1 (T.Reduce_window_sum w) [ ramp6 ] in
  iarr "sum shape" [| 5 |] (shape_of y);
  farr "sum vals" [| 1.; 3.; 5.; 7.; 9. |] (out_floats y)

let max_impl () =
  let w = win1 ~w:2 ~s:1 ~p:(0, 0) ~bd:1 ~wd:1 in
  let y = C.bind1 (T.Reduce_window_max w) [ ramp6 ] in
  farr "max vals" [| 1.; 2.; 3.; 4.; 5. |] (out_floats y)

let min_impl () =
  let w = win1 ~w:2 ~s:1 ~p:(0, 0) ~bd:1 ~wd:1 in
  let y = C.bind1 (T.Reduce_window_min w) [ ramp6 ] in
  farr "min vals" [| 0.; 1.; 2.; 3.; 4. |] (out_floats y)

let general_impl () =
  let sc = { T.shape = [||]; dtype = D.F32; weak_type = false } in
  let reducer = J.make_jaxpr [ sc; sc ] (fun args -> [ C.bind1 T.Mul args ]) in
  let w = win1 ~w:2 ~s:1 ~p:(0, 0) ~bd:1 ~wd:1 in
  let init = f32 [||] [| 1.0 |] in
  let y = C.bind1 (T.Reduce_window { reducer; window = w }) [ ramp6; init ] in
  farr "mul vals" [| 0.; 2.; 6.; 12.; 20. |] (out_floats y)

let gather_add_impl () =
  let w = win1 ~w:2 ~s:1 ~p:(0, 0) ~bd:1 ~wd:1 in
  let t = f32 [| 6 |] [| 0.; 10.; 20.; 30.; 40.; 50. |] in
  let y =
    C.bind1
      (T.Select_and_gather_add { select = T.Wge; window = w })
      [ t; ramp6 ]
  in
  farr "sga ge" [| 10.; 20.; 30.; 40.; 50. |] (out_floats y)

let scatter_add_impl () =
  let w = win1 ~w:2 ~s:1 ~p:(0, 0) ~bd:1 ~wd:1 in
  let source = f32 [| 5 |] [| 1.; 1.; 1.; 1.; 1. |] in
  let y =
    C.bind1
      (T.Select_and_scatter_add { select = T.Wge; window = w })
      [ source; ramp6 ]
  in
  iarr "ssa shape" [| 6 |] (shape_of y);
  farr "ssa ge" [| 0.; 1.; 1.; 1.; 1.; 1. |] (out_floats y)

let sum_jvp () =
  let w = win1 ~w:2 ~s:1 ~p:(0, 0) ~bd:1 ~wd:1 in
  let t = f32 [| 6 |] [| 1.; 1.; 1.; 1.; 1.; 1. |] in
  let fn a = [ C.bind1 (T.Reduce_window_sum w) a ] in
  let _, to_ = Ad.jvp fn [ ramp6 ] [ t ] in
  farr "sum tangent" [| 2.; 2.; 2.; 2.; 2. |] (out_floats (List.hd to_))

let max_jvp () =
  let w = win1 ~w:2 ~s:1 ~p:(0, 0) ~bd:1 ~wd:1 in
  let t = f32 [| 6 |] [| 0.; 10.; 20.; 30.; 40.; 50. |] in
  let fn a = [ C.bind1 (T.Reduce_window_max w) a ] in
  let _, to_ = Ad.jvp fn [ ramp6 ] [ t ] in
  farr "max tangent" [| 10.; 20.; 30.; 40.; 50. |] (out_floats (List.hd to_))

let general_jvp_raises () =
  let sc = { T.shape = [||]; dtype = D.F32; weak_type = false } in
  let reducer = J.make_jaxpr [ sc; sc ] (fun args -> [ C.bind1 T.Mul args ]) in
  let w = win1 ~w:2 ~s:1 ~p:(0, 0) ~bd:1 ~wd:1 in
  let init = f32 [||] [| 1.0 |] in
  let fn a = [ C.bind1 (T.Reduce_window { reducer; window = w }) a ] in
  match Ad.jvp fn [ ramp6; init ] [ ramp6; init ] with
  | _ -> Alcotest.fail "expected reduce_window jvp failure"
  | exception Failure _ -> ()

let transpose_raises () =
  let w = win1 ~w:2 ~s:1 ~p:(0, 0) ~bd:1 ~wd:1 in
  let fn a = C.bind1 (T.Reduce_window_sum w) a in
  match
    Ad.grad (fun a -> C.bind1 (T.Reduce_sum [| 0 |]) [ fn a ]) [ ramp6 ]
  with
  | _ -> Alcotest.fail "expected reduce_window_sum transpose failure"
  | exception Failure _ -> ()

let vmap_raises () =
  let w = win1 ~w:2 ~s:1 ~p:(0, 0) ~bd:1 ~wd:1 in
  let batched = f32 [| 2; 6 |] (Array.init 12 float_of_int) in
  let fn a = [ C.bind1 (T.Reduce_window_sum w) a ] in
  match Batching.vmap fn [ Some 0 ] [ batched ] with
  | _ -> Alcotest.fail "expected reduce_window vmap failure"
  | exception Failure _ -> ()

let () =
  Alcotest.run "lax_windowed_reductions"
    [
      ( "impl",
        [
          Alcotest.test_case "sum" `Quick sum_impl;
          Alcotest.test_case "max" `Quick max_impl;
          Alcotest.test_case "min" `Quick min_impl;
          Alcotest.test_case "general" `Quick general_impl;
          Alcotest.test_case "gather_add" `Quick gather_add_impl;
          Alcotest.test_case "scatter_add" `Quick scatter_add_impl;
        ] );
      ( "jvp",
        [
          Alcotest.test_case "sum" `Quick sum_jvp;
          Alcotest.test_case "max" `Quick max_jvp;
          Alcotest.test_case "general_raises" `Quick general_jvp_raises;
        ] );
      ("transpose", [ Alcotest.test_case "raises" `Quick transpose_raises ]);
      ("vmap", [ Alcotest.test_case "raises" `Quick vmap_raises ]);
    ]
