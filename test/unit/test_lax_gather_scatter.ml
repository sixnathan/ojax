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
let i32 shape xs = cval D.I32 shape (Array.map float_of_int xs)

let take_dnums =
  {
    T.offset_dims = [||];
    collapsed_slice_dims = [| 0 |];
    start_index_map = [| 0 |];
    g_operand_batching_dims = [||];
    g_start_indices_batching_dims = [||];
  }

let scalar_sdnums =
  {
    T.update_window_dims = [||];
    inserted_window_dims = [| 0 |];
    scatter_dims_to_operand_dims = [| 0 |];
    s_operand_batching_dims = [||];
    s_scatter_indices_batching_dims = [||];
  }

let gather operand indices ~dimension_numbers ~slice_sizes =
  C.bind1 (T.Gather { dimension_numbers; slice_sizes }) [ operand; indices ]

let scatter_add operand indices updates =
  C.bind1
    (T.Scatter_add { dimension_numbers = scalar_sdnums })
    [ operand; indices; updates ]

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
let rsum v = C.bind1 (T.Reduce_sum [| 0 |]) [ v ]

let gather_impl () =
  let x = f32 [| 10 |] (Array.init 10 (fun i -> float_of_int (i * 10))) in
  let idx = i32 [| 4; 1 |] [| 0; 1; 1; 9 |] in
  let y = gather x idx ~dimension_numbers:take_dnums ~slice_sizes:[| 1 |] in
  iarr "take shape" [| 4 |] (shape_of y);
  farr "take vals" [| 0.; 10.; 10.; 90. |] (out_floats y);
  let m = f32 [| 10; 3 |] (Array.init 30 float_of_int) in
  let ridx = i32 [| 2; 1 |] [| 0; 2 |] in
  let rows =
    gather m ridx
      ~dimension_numbers:
        {
          T.offset_dims = [| 1 |];
          collapsed_slice_dims = [| 0 |];
          start_index_map = [| 0 |];
          g_operand_batching_dims = [||];
          g_start_indices_batching_dims = [||];
        }
      ~slice_sizes:[| 1; 3 |]
  in
  iarr "rows shape" [| 2; 3 |] (shape_of rows);
  farr "rows vals" [| 0.; 1.; 2.; 6.; 7.; 8. |] (out_floats rows)

let scatter_impls () =
  let op = f32 [| 5 |] [| 1.; 1.; 1.; 1.; 1. |] in
  let idx = i32 [| 3; 1 |] [| 1; 2; 4 |] in
  let up = f32 [| 3 |] [| 2.; 3.; 4. |] in
  farr "add" [| 1.; 3.; 4.; 1.; 5. |] (out_floats (scatter_add op idx up));
  let dup = i32 [| 3; 1 |] [| 1; 1; 4 |] in
  farr "add dup" [| 1.; 6.; 1.; 1.; 5. |] (out_floats (scatter_add op dup up));
  let sub =
    C.bind1
      (T.Scatter_sub { dimension_numbers = scalar_sdnums })
      [ op; idx; up ]
  in
  farr "sub" [| 1.; -1.; -2.; 1.; -3. |] (out_floats sub);
  let mul =
    C.bind1
      (T.Scatter_mul
         { dimension_numbers = scalar_sdnums; unique_indices = true })
      [ f32 [| 5 |] [| 2.; 2.; 2.; 2.; 2. |]; idx; up ]
  in
  farr "mul" [| 2.; 4.; 6.; 2.; 8. |] (out_floats mul);
  let set =
    C.bind1
      (T.Scatter { dimension_numbers = scalar_sdnums; unique_indices = true })
      [ op; i32 [| 2; 1 |] [| 0; 3 |]; f32 [| 2 |] [| 9.; 8. |] ]
  in
  farr "set" [| 9.; 1.; 1.; 8.; 1. |] (out_floats set);
  let mx =
    C.bind1
      (T.Scatter_max { dimension_numbers = scalar_sdnums })
      [
        f32 [| 4 |] [| 5.; 5.; 5.; 5. |];
        i32 [| 3; 1 |] [| 0; 1; 1 |];
        f32 [| 3 |] [| 9.; 2.; 7. |];
      ]
  in
  farr "max" [| 9.; 7.; 5.; 5. |] (out_floats mx);
  let mn =
    C.bind1
      (T.Scatter_min { dimension_numbers = scalar_sdnums })
      [
        f32 [| 4 |] [| 5.; 5.; 5.; 5. |];
        i32 [| 3; 1 |] [| 0; 1; 1 |];
        f32 [| 3 |] [| 9.; 2.; 7. |];
      ]
  in
  farr "min" [| 5.; 2.; 5.; 5. |] (out_floats mn)

let jvp_gather () =
  let x = f32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let t = f32 [| 5 |] [| 10.; 20.; 30.; 40.; 50. |] in
  let idx = i32 [| 3; 1 |] [| 0; 2; 4 |] in
  let fn a =
    [
      gather (List.hd a) idx ~dimension_numbers:take_dnums ~slice_sizes:[| 1 |];
    ]
  in
  let po, to_ = Ad.jvp fn [ x ] [ t ] in
  farr "jvp gather primal" [| 1.; 3.; 5. |] (out_floats (List.hd po));
  farr "jvp gather tangent" [| 10.; 30.; 50. |] (out_floats (List.hd to_))

let grad_gather () =
  let x = f32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let idx = i32 [| 3; 1 |] [| 0; 1; 1 |] in
  let g =
    List.hd
      (Ad.grad
         (fun xs ->
           rsum
             (gather (List.hd xs) idx ~dimension_numbers:take_dnums
                ~slice_sizes:[| 1 |]))
         [ x ])
  in
  farr "grad gather" [| 1.; 2.; 0.; 0.; 0. |] (out_floats g)

let grad_scatter_add () =
  let op = f32 [| 5 |] [| 1.; 1.; 1.; 1.; 1. |] in
  let up = f32 [| 3 |] [| 1.; 1.; 1. |] in
  let idx = i32 [| 3; 1 |] [| 1; 2; 4 |] in
  let gs =
    Ad.grad
      (fun a ->
        match a with
        | [ o; u ] -> rsum (scatter_add o idx u)
        | _ -> assert false)
      [ op; up ]
  in
  match gs with
  | [ go; gu ] ->
      farr "grad op" [| 1.; 1.; 1.; 1.; 1. |] (out_floats go);
      farr "grad up" [| 1.; 1.; 1. |] (out_floats gu)
  | _ -> Alcotest.fail "expected two grads"

let grad_scatter_set () =
  let op = f32 [| 5 |] [| 1.; 1.; 1.; 1.; 1. |] in
  let up = f32 [| 2 |] [| 9.; 8. |] in
  let idx = i32 [| 2; 1 |] [| 0; 3 |] in
  let gs =
    Ad.grad
      (fun a ->
        match a with
        | [ o; u ] ->
            rsum
              (C.bind1
                 (T.Scatter
                    { dimension_numbers = scalar_sdnums; unique_indices = true })
                 [ o; idx; u ])
        | _ -> assert false)
      [ op; up ]
  in
  match gs with
  | [ go; gu ] ->
      farr "grad set op" [| 0.; 1.; 1.; 0.; 1. |] (out_floats go);
      farr "grad set up" [| 1.; 1. |] (out_floats gu)
  | _ -> Alcotest.fail "expected two grads"

let grad_scatter_mul () =
  let op = f32 [| 5 |] [| 2.; 3.; 4.; 5.; 6. |] in
  let up = f32 [| 2 |] [| 10.; 20. |] in
  let idx = i32 [| 2; 1 |] [| 1; 3 |] in
  let gs =
    Ad.grad
      (fun a ->
        match a with
        | [ o; u ] ->
            rsum
              (C.bind1
                 (T.Scatter_mul
                    { dimension_numbers = scalar_sdnums; unique_indices = true })
                 [ o; idx; u ])
        | _ -> assert false)
      [ op; up ]
  in
  match gs with
  | [ go; gu ] ->
      farr "grad mul op" [| 1.; 10.; 1.; 20.; 1. |] (out_floats go);
      farr "grad mul up" [| 3.; 5. |] (out_floats gu)
  | _ -> Alcotest.fail "expected two grads"

let vmap_gather_operand () =
  let x = f32 [| 2; 10 |] (Array.init 20 float_of_int) in
  let idx = i32 [| 4; 1 |] [| 0; 1; 1; 9 |] in
  let fn a =
    [
      gather (List.hd a) idx ~dimension_numbers:take_dnums ~slice_sizes:[| 1 |];
    ]
  in
  let out = List.hd (Batching.vmap fn [ Some 0 ] [ x ]) in
  iarr "vmap g op shape" [| 2; 4 |] (shape_of out);
  farr "vmap g op vals"
    [| 0.; 1.; 1.; 9.; 10.; 11.; 11.; 19. |]
    (out_floats out)

let vmap_gather_indices () =
  let x = f32 [| 10 |] (Array.init 10 (fun i -> float_of_int (i * 10))) in
  let idx = i32 [| 2; 3; 1 |] [| 0; 1; 2; 3; 4; 5 |] in
  let fn a =
    match a with
    | [ o; i ] ->
        [ gather o i ~dimension_numbers:take_dnums ~slice_sizes:[| 1 |] ]
    | _ -> assert false
  in
  let out = List.hd (Batching.vmap fn [ None; Some 0 ] [ x; idx ]) in
  iarr "vmap g idx shape" [| 2; 3 |] (shape_of out);
  farr "vmap g idx vals" [| 0.; 10.; 20.; 30.; 40.; 50. |] (out_floats out)

let vmap_gather_both () =
  let x = f32 [| 2; 10 |] (Array.init 20 float_of_int) in
  let idx = i32 [| 2; 3; 1 |] [| 0; 1; 2; 7; 8; 9 |] in
  let fn a =
    match a with
    | [ o; i ] ->
        [ gather o i ~dimension_numbers:take_dnums ~slice_sizes:[| 1 |] ]
    | _ -> assert false
  in
  let out = List.hd (Batching.vmap fn [ Some 0; Some 0 ] [ x; idx ]) in
  iarr "vmap g both shape" [| 2; 3 |] (shape_of out);
  farr "vmap g both vals" [| 0.; 1.; 2.; 17.; 18.; 19. |] (out_floats out)

let vmap_scatter_add () =
  let op = f32 [| 2; 10 |] (Array.make 20 0.0) in
  let up = f32 [| 2; 3 |] [| 1.; 1.; 1.; 2.; 2.; 2. |] in
  let idx = i32 [| 3; 1 |] [| 1; 2; 4 |] in
  let fn a =
    match a with [ o; i; u ] -> [ scatter_add o i u ] | _ -> assert false
  in
  let out =
    List.hd (Batching.vmap fn [ Some 0; None; Some 0 ] [ op; idx; up ])
  in
  iarr "vmap sadd shape" [| 2; 10 |] (shape_of out);
  farr "vmap sadd vals"
    [|
      0.;
      1.;
      1.;
      0.;
      1.;
      0.;
      0.;
      0.;
      0.;
      0.;
      0.;
      2.;
      2.;
      0.;
      2.;
      0.;
      0.;
      0.;
      0.;
      0.;
    |]
    (out_floats out)

let scatter_min_max_jvp_raises () =
  let op = f32 [| 3 |] [| 1.; 2.; 3. |] in
  let up = f32 [| 1 |] [| 5. |] in
  let idx = i32 [| 1; 1 |] [| 0 |] in
  let fn a =
    match a with
    | [ o; u ] ->
        [
          C.bind1
            (T.Scatter_max { dimension_numbers = scalar_sdnums })
            [ o; idx; u ];
        ]
    | _ -> assert false
  in
  match Ad.jvp fn [ op; up ] [ op; up ] with
  | _ -> Alcotest.fail "expected scatter_max jvp to raise"
  | exception Failure _ -> ()

let () =
  Alcotest.run "lax_gather_scatter"
    [
      ( "impl",
        [
          Alcotest.test_case "gather" `Quick gather_impl;
          Alcotest.test_case "scatter" `Quick scatter_impls;
        ] );
      ("jvp", [ Alcotest.test_case "gather" `Quick jvp_gather ]);
      ( "grad",
        [
          Alcotest.test_case "gather" `Quick grad_gather;
          Alcotest.test_case "scatter_add" `Quick grad_scatter_add;
          Alcotest.test_case "scatter_set" `Quick grad_scatter_set;
          Alcotest.test_case "scatter_mul" `Quick grad_scatter_mul;
          Alcotest.test_case "scatter_min_max_jvp_raises" `Quick
            scatter_min_max_jvp_raises;
        ] );
      ( "vmap",
        [
          Alcotest.test_case "gather_operand" `Quick vmap_gather_operand;
          Alcotest.test_case "gather_indices" `Quick vmap_gather_indices;
          Alcotest.test_case "gather_both" `Quick vmap_gather_both;
          Alcotest.test_case "scatter_add" `Quick vmap_scatter_add;
        ] );
    ]
