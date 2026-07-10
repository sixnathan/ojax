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
let i32 x = cval D.I32 [||] [| float_of_int x |]

module Slicing = struct
  let slice ?strides operand ~start_indices ~limit_indices =
    C.bind1 (T.Slice { start_indices; limit_indices; strides }) [ operand ]

  let dynamic_slice operand ~start_indices ~slice_sizes =
    C.bind1 (T.Dynamic_slice { slice_sizes }) (operand :: start_indices)

  let dynamic_update_slice operand ~update ~start_indices =
    C.bind1 T.Dynamic_update_slice (operand :: update :: start_indices)
end

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

let slice_impl () =
  let x = f32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let y = Slicing.slice x ~start_indices:[| 1; 0 |] ~limit_indices:[| 3; 2 |] in
  iarr "slice shape" [| 2; 2 |] (shape_of y);
  farr "slice vals" [| 4.; 5.; 8.; 9. |] (out_floats y);
  let z =
    Slicing.slice x ~start_indices:[| 0; 0 |] ~limit_indices:[| 3; 4 |]
      ~strides:[| 1; 2 |]
  in
  iarr "strided shape" [| 3; 2 |] (shape_of z);
  farr "strided vals" [| 0.; 2.; 4.; 6.; 8.; 10. |] (out_floats z)

let dynamic_slice_impl () =
  let x = f32 [| 3; 4 |] (Array.init 12 float_of_int) in
  let y =
    Slicing.dynamic_slice x
      ~start_indices:[ i32 1; i32 1 ]
      ~slice_sizes:[| 2; 3 |]
  in
  iarr "ds shape" [| 2; 3 |] (shape_of y);
  farr "ds vals" [| 5.; 6.; 7.; 9.; 10.; 11. |] (out_floats y);
  let clamped =
    Slicing.dynamic_slice x
      ~start_indices:[ i32 1; i32 9 ]
      ~slice_sizes:[| 2; 4 |]
  in
  farr "ds clamp vals"
    [| 4.; 5.; 6.; 7.; 8.; 9.; 10.; 11. |]
    (out_floats clamped)

let dynamic_update_slice_impl () =
  let x = f32 [| 6 |] (Array.make 6 0.0) in
  let u = f32 [| 3 |] [| 1.; 1.; 1. |] in
  let y = Slicing.dynamic_update_slice x ~update:u ~start_indices:[ i32 2 ] in
  farr "dus vals" [| 0.; 0.; 1.; 1.; 1.; 0. |] (out_floats y);
  let clamped =
    Slicing.dynamic_update_slice x ~update:u ~start_indices:[ i32 5 ]
  in
  farr "dus clamp" [| 0.; 0.; 0.; 1.; 1.; 1. |] (out_floats clamped)

let jvp_slice () =
  let x = f32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let t = f32 [| 4 |] [| 10.; 20.; 30.; 40. |] in
  let fn a =
    [ Slicing.slice (List.hd a) ~start_indices:[| 1 |] ~limit_indices:[| 3 |] ]
  in
  let po, to_ = Ad.jvp fn [ x ] [ t ] in
  farr "jvp primal" [| 2.; 3. |] (out_floats (List.hd po));
  farr "jvp tangent" [| 20.; 30. |] (out_floats (List.hd to_))

let jvp_dynamic_slice () =
  let x = f32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let t = f32 [| 5 |] [| 10.; 20.; 30.; 40.; 50. |] in
  let fn a =
    [
      Slicing.dynamic_slice (List.hd a)
        ~start_indices:[ i32 1 ]
        ~slice_sizes:[| 2 |];
    ]
  in
  let po, to_ = Ad.jvp fn [ x ] [ t ] in
  farr "jvp ds primal" [| 2.; 3. |] (out_floats (List.hd po));
  farr "jvp ds tangent" [| 20.; 30. |] (out_floats (List.hd to_))

let jvp_dynamic_update_slice () =
  let x = f32 [| 5 |] [| 0.; 0.; 0.; 0.; 0. |] in
  let u = f32 [| 2 |] [| 1.; 1. |] in
  let tx = f32 [| 5 |] [| 5.; 5.; 5.; 5.; 5. |] in
  let tu = f32 [| 2 |] [| 100.; 200. |] in
  let fn a =
    match a with
    | [ o; up ] ->
        [ Slicing.dynamic_update_slice o ~update:up ~start_indices:[ i32 1 ] ]
    | _ -> assert false
  in
  let po, to_ = Ad.jvp fn [ x; u ] [ tx; tu ] in
  farr "jvp dus primal" [| 0.; 1.; 1.; 0.; 0. |] (out_floats (List.hd po));
  farr "jvp dus tangent" [| 5.; 100.; 200.; 5.; 5. |] (out_floats (List.hd to_))

let grad_slice () =
  let x = f32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  let fn a =
    List.hd
      (Ad.grad
         (fun xs ->
           List.hd
             [
               C.bind1 (T.Reduce_sum [| 0 |])
                 [
                   Slicing.slice (List.hd xs) ~start_indices:[| 1 |]
                     ~limit_indices:[| 3 |];
                 ];
             ])
         a)
  in
  farr "grad slice" [| 0.; 1.; 1.; 0. |] (out_floats (fn [ x ]))

let grad_dynamic_slice () =
  let x = f32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let g =
    List.hd
      (Ad.grad
         (fun xs ->
           C.bind1 (T.Reduce_sum [| 0 |])
             [
               Slicing.dynamic_slice (List.hd xs)
                 ~start_indices:[ i32 1 ]
                 ~slice_sizes:[| 2 |];
             ])
         [ x ])
  in
  farr "grad ds" [| 0.; 1.; 1.; 0.; 0. |] (out_floats g)

let grad_dynamic_update_slice () =
  let x = f32 [| 5 |] [| 1.; 2.; 3.; 4.; 5. |] in
  let u = f32 [| 2 |] [| 9.; 9. |] in
  let gs =
    Ad.grad
      (fun a ->
        match a with
        | [ o; up ] ->
            C.bind1 (T.Reduce_sum [| 0 |])
              [
                Slicing.dynamic_update_slice o ~update:up
                  ~start_indices:[ i32 1 ];
              ]
        | _ -> assert false)
      [ x; u ]
  in
  match gs with
  | [ go; gu ] ->
      farr "grad dus operand" [| 1.; 0.; 0.; 1.; 1. |] (out_floats go);
      farr "grad dus update" [| 1.; 1. |] (out_floats gu)
  | _ -> Alcotest.fail "expected two grads"

let vmap_slice () =
  let x = f32 [| 2; 4 |] (Array.init 8 float_of_int) in
  let fn a =
    [ Slicing.slice (List.hd a) ~start_indices:[| 1 |] ~limit_indices:[| 3 |] ]
  in
  let out = List.hd (Batching.vmap fn [ Some 0 ] [ x ]) in
  iarr "vmap slice shape" [| 2; 2 |] (shape_of out);
  farr "vmap slice vals" [| 1.; 2.; 5.; 6. |] (out_floats out)

let vmap_dynamic_slice () =
  let x = f32 [| 2; 4 |] (Array.init 8 float_of_int) in
  let fn a =
    [
      Slicing.dynamic_slice (List.hd a)
        ~start_indices:[ i32 1 ]
        ~slice_sizes:[| 2 |];
    ]
  in
  let out = List.hd (Batching.vmap fn [ Some 0 ] [ x ]) in
  iarr "vmap ds shape" [| 2; 2 |] (shape_of out);
  farr "vmap ds vals" [| 1.; 2.; 5.; 6. |] (out_floats out)

let vmap_dynamic_update_slice () =
  let x = f32 [| 2; 4 |] (Array.make 8 0.0) in
  let u = f32 [| 2; 2 |] [| 1.; 1.; 2.; 2. |] in
  let fn a =
    match a with
    | [ o; up ] ->
        [ Slicing.dynamic_update_slice o ~update:up ~start_indices:[ i32 1 ] ]
    | _ -> assert false
  in
  let out = List.hd (Batching.vmap fn [ Some 0; Some 0 ] [ x; u ]) in
  iarr "vmap dus shape" [| 2; 4 |] (shape_of out);
  farr "vmap dus vals" [| 0.; 1.; 1.; 0.; 0.; 2.; 2.; 0. |] (out_floats out)

let vmap_batched_indices_raises () =
  let x = f32 [| 2; 4 |] (Array.init 8 float_of_int) in
  let idx = cval D.I32 [| 2 |] [| 0.; 1. |] in
  let fn a =
    match a with
    | [ o; i ] ->
        [ Slicing.dynamic_slice o ~start_indices:[ i ] ~slice_sizes:[| 2 |] ]
    | _ -> assert false
  in
  match Batching.vmap fn [ Some 0; Some 0 ] [ x; idx ] with
  | _ -> Alcotest.fail "expected gather failure"
  | exception Failure _ -> ()

let () =
  Alcotest.run "lax_slicing"
    [
      ( "impl",
        [
          Alcotest.test_case "slice" `Quick slice_impl;
          Alcotest.test_case "dynamic_slice" `Quick dynamic_slice_impl;
          Alcotest.test_case "dynamic_update_slice" `Quick
            dynamic_update_slice_impl;
        ] );
      ( "jvp",
        [
          Alcotest.test_case "slice" `Quick jvp_slice;
          Alcotest.test_case "dynamic_slice" `Quick jvp_dynamic_slice;
          Alcotest.test_case "dynamic_update_slice" `Quick
            jvp_dynamic_update_slice;
        ] );
      ( "grad",
        [
          Alcotest.test_case "slice" `Quick grad_slice;
          Alcotest.test_case "dynamic_slice" `Quick grad_dynamic_slice;
          Alcotest.test_case "dynamic_update_slice" `Quick
            grad_dynamic_update_slice;
        ] );
      ( "vmap",
        [
          Alcotest.test_case "slice" `Quick vmap_slice;
          Alcotest.test_case "dynamic_slice" `Quick vmap_dynamic_slice;
          Alcotest.test_case "dynamic_update_slice" `Quick
            vmap_dynamic_update_slice;
          Alcotest.test_case "batched_indices_raises" `Quick
            vmap_batched_indices_raises;
        ] );
    ]
