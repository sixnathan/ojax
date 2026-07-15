module Cb = Ojax.Custom_batching
module B = Ojax.Interpreters.Batching
module C = Ojax.Core
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray

let () = Ojax.Lax.install ()
let b1 = C.bind1
let value shape data = T.Concrete (Nd.of_floats D.F32 shape data)
let scalar x = value [||] [| x |]

let concrete = function
  | T.Concrete nd -> nd
  | _ -> Alcotest.fail "expected concrete value"

let close a b = Float.abs (a -. b) <= 1e-5

let check_array name expected v =
  let nd = concrete v in
  let shape = Nd.shape nd in
  let expected_shape = [| Array.length expected |] in
  Alcotest.(check bool) (name ^ " shape") true (shape = expected_shape);
  Array.iteri
    (fun i x ->
      Alcotest.(check bool)
        (Printf.sprintf "%s[%d]" name i)
        true
        (close x (Nd.get_f nd [| i |])))
    expected

let custom_mul args =
  let primal a = [ b1 T.Add a ] in
  let vmap_rule ~axis_size ~in_batched values =
    if axis_size <> 3 then Alcotest.fail "axis size";
    if in_batched <> [ true; true ] then Alcotest.fail "in_batched";
    ([ b1 T.Mul values ], [ true ])
  in
  Cb.custom_vmap ~primal ~vmap_rule args

let test_eval_uses_primal () =
  let out = custom_mul [ scalar 2.0; scalar 3.0 ] in
  let got = Nd.get_f (concrete (List.hd out)) [||] in
  Alcotest.(check bool) "primal add" true (close got 5.0)

let test_vmap_uses_rule () =
  let x = value [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let y = value [| 3 |] [| 4.0; 5.0; 6.0 |] in
  let out = B.vmap custom_mul [ Some 0; Some 0 ] [ x; y ] in
  check_array "custom multiply" [| 4.0; 10.0; 18.0 |] (List.hd out)

let custom_reduce args =
  let primal a = a in
  let vmap_rule ~axis_size:_ ~in_batched:_ values =
    ([ b1 (T.Reduce_sum [| 0 |]) values ], [ false ])
  in
  Cb.custom_vmap ~primal ~vmap_rule args

let test_unbatched_output_broadcasts () =
  let x = value [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let out = B.vmap custom_reduce [ Some 0 ] [ x ] in
  check_array "broadcast total" [| 6.0; 6.0; 6.0 |] (List.hd out)

let custom_square args =
  let primal a = a in
  let vmap_rule ~axis_size:_ ~in_batched:_ values =
    match values with
    | [ x ] -> ([ b1 T.Mul [ x; x ] ], [ true ])
    | _ -> Alcotest.fail "square arity"
  in
  Cb.custom_vmap ~primal ~vmap_rule args

let test_nonzero_axis_moves_to_front () =
  let x = value [| 2; 3 |] [| 1.0; 2.0; 3.0; 4.0; 5.0; 6.0 |] in
  let out = B.vmap custom_square [ Some 1 ] [ x ] in
  let nd = concrete (List.hd out) in
  Alcotest.(check bool) "shape" true (Nd.shape nd = [| 3; 2 |]);
  let expected = [| 1.0; 16.0; 4.0; 25.0; 9.0; 36.0 |] in
  Array.iteri
    (fun i x ->
      Alcotest.(check bool)
        (Printf.sprintf "moved[%d]" i)
        true
        (close x (Nd.get_f nd [| i / 2; i mod 2 |])))
    expected

let test_sequential_vmap () =
  let f = function
    | [ x ] -> [ b1 T.Add [ b1 T.Sin [ x ]; scalar 1.0 ] ]
    | _ -> Alcotest.fail "sequential arity"
  in
  let x = value [| 3 |] [| 0.0; 0.5; 1.0 |] in
  let out = B.vmap (Cb.sequential_vmap f) [ Some 0 ] [ x ] in
  check_array "sequential"
    [| 1.0; 1.0 +. sin 0.5; 1.0 +. sin 1.0 |]
    (List.hd out)

let test_rule_arity_check () =
  let f args =
    let primal a = a in
    let vmap_rule ~axis_size:_ ~in_batched:_ values = (values, []) in
    Cb.custom_vmap ~primal ~vmap_rule args
  in
  let x = value [| 2 |] [| 1.0; 2.0 |] in
  Alcotest.check_raises "arity"
    (Invalid_argument
       "custom_vmap rule output values and batching specification must have \
        the same length") (fun () -> ignore (B.vmap f [ Some 0 ] [ x ]))

let () =
  Alcotest.run "custom_batching"
    [
      ( "custom_vmap",
        [
          Alcotest.test_case "eval uses primal" `Quick test_eval_uses_primal;
          Alcotest.test_case "vmap uses rule" `Quick test_vmap_uses_rule;
          Alcotest.test_case "unbatched output" `Quick
            test_unbatched_output_broadcasts;
          Alcotest.test_case "axis normalization" `Quick
            test_nonzero_axis_moves_to_front;
          Alcotest.test_case "rule arity" `Quick test_rule_arity_check;
        ] );
      ( "sequential_vmap",
        [ Alcotest.test_case "scan" `Quick test_sequential_vmap ] );
    ]
