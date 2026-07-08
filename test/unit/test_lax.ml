module L = Ojax.Lax
module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype

let () = L.install ()
let nd dtype shape xs = Nd.of_floats dtype shape xs
let cval dtype shape xs = T.Concrete (nd dtype shape xs)

let concrete_of = function
  | T.Concrete a -> a
  | T.Tracer _ -> failwith "not concrete"

let out_floats v =
  let a = concrete_of v in
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

let flist = Alcotest.(list (float 1e-6))
let ilist = Alcotest.(array int)
let bind1 prim args = C.bind1 prim args

let matmul () =
  let lhs = cval D.F32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let rhs = cval D.F32 [| 3; 2 |] [| 1.; 0.; 0.; 1.; 1.; 1. |] in
  let dd =
    {
      T.lhs_contract = [| 1 |];
      rhs_contract = [| 0 |];
      lhs_batch = [||];
      rhs_batch = [||];
    }
  in
  let out = bind1 (T.Dot_general dd) [ lhs; rhs ] in
  Alcotest.(check ilist) "shape" [| 2; 2 |] (Nd.shape (concrete_of out));
  Alcotest.check flist "matmul" [ 4.; 5.; 10.; 11. ]
    (Array.to_list (out_floats out))

let reduce () =
  let x = cval D.F32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let a0 = bind1 (T.Reduce_sum [| 0 |]) [ x ] in
  Alcotest.check flist "axis0" [ 5.; 7.; 9. ] (Array.to_list (out_floats a0));
  let a1 = bind1 (T.Reduce_sum [| 1 |]) [ x ] in
  Alcotest.check flist "axis1" [ 6.; 15. ] (Array.to_list (out_floats a1))

let broadcast () =
  let x = cval D.F32 [| 3 |] [| 10.; 20.; 30. |] in
  let out =
    bind1 (T.Broadcast_in_dim { shape = [| 2; 3 |]; dims = [| 1 |] }) [ x ]
  in
  Alcotest.(check ilist) "shape" [| 2; 3 |] (Nd.shape (concrete_of out));
  Alcotest.check flist "bcast"
    [ 10.; 20.; 30.; 10.; 20.; 30. ]
    (Array.to_list (out_floats out))

let select () =
  let which = cval D.I32 [| 3 |] [| 1.; 0.; 1. |] in
  let c0 = cval D.F32 [| 3 |] [| 10.; 20.; 30. |] in
  let c1 = cval D.F32 [| 3 |] [| 100.; 200.; 300. |] in
  let out = bind1 T.Select_n [ which; c0; c1 ] in
  Alcotest.check flist "select_n" [ 100.; 20.; 300. ]
    (Array.to_list (out_floats out))

let int_div () =
  let a = cval D.I32 [| 3 |] [| 7.; 9.; 8. |] in
  let b = cval D.I32 [| 3 |] [| 2.; 4.; 3. |] in
  let out = bind1 T.Div [ a; b ] in
  Alcotest.(check (list (float 0.)))
    "trunc div" [ 3.; 2.; 2. ]
    (Array.to_list (out_floats out))

let compare_bool () =
  let a = cval D.F32 [| 3 |] [| 1.; 5.; 3. |] in
  let b = cval D.F32 [| 3 |] [| 2.; 2.; 3. |] in
  let out = bind1 T.Lt [ a; b ] in
  Alcotest.(check bool) "dtype bool" true (Nd.dtype (concrete_of out) = D.Bool);
  Alcotest.(check (list (float 0.)))
    "lt" [ 1.; 0.; 0. ]
    (Array.to_list (out_floats out))

let abstract_shapes () =
  let dd =
    {
      T.lhs_contract = [| 1 |];
      rhs_contract = [| 0 |];
      lhs_batch = [||];
      rhs_batch = [||];
    }
  in
  let l = { T.shape = [| 2; 3 |]; dtype = D.F32; weak_type = false } in
  let r = { T.shape = [| 3; 4 |]; dtype = D.F32; weak_type = false } in
  (match L.abstract_eval (T.Dot_general dd) [ l; r ] with
  | [ o ] -> Alcotest.(check ilist) "dot shape" [| 2; 4 |] o.T.shape
  | _ -> Alcotest.fail "arity");
  match L.abstract_eval T.Eq [ l; l ] with
  | [ o ] -> Alcotest.(check bool) "eq bool" true (o.T.dtype = D.Bool)
  | _ -> Alcotest.fail "arity"

let () =
  Alcotest.run "lax"
    [
      ( "impl",
        [
          Alcotest.test_case "matmul" `Quick matmul;
          Alcotest.test_case "reduce_sum" `Quick reduce;
          Alcotest.test_case "broadcast_in_dim" `Quick broadcast;
          Alcotest.test_case "select_n" `Quick select;
          Alcotest.test_case "int div trunc" `Quick int_div;
          Alcotest.test_case "compare bool" `Quick compare_bool;
        ] );
      ("abstract_eval", [ Alcotest.test_case "shapes" `Quick abstract_shapes ]);
    ]
