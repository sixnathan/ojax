module T = Ojax.Types
module C = Ojax.Core
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module Lax = Ojax.Lax
module Ad = Ojax.Interpreters.Ad
module Batching = Ojax.Interpreters.Batching

let () = Ojax.Lax.install ()
let b1 = C.bind1
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)
let sc x = T.Concrete (Nd.of_floats D.F32 [||] [| x |])

let read v =
  match v with
  | T.Concrete nd ->
      let n = Array.fold_left ( * ) 1 (Nd.shape nd) in
      let a = Array.make n 0.0 in
      let _ =
        Nd.fold
          (fun i x ->
            a.(i) <- x;
            i + 1)
          0 nd
      in
      a
  | T.Tracer _ -> failwith "expected concrete"

let shape_of v = match v with T.Concrete nd -> Nd.shape nd | _ -> [||]

let approx name a b =
  Array.iter2
    (fun x y ->
      if Float.abs (x -. y) > 1e-5 then Alcotest.failf "%s: %f <> %f" name x y)
    a b

let cumsum l =
  match l with
  | [ c; x ] ->
      let s = b1 T.Add [ c; x ] in
      [ s; s ]
  | _ -> assert false

let test_eval () =
  let outs = Lax.scan cumsum [ sc 0.0 ] [ f32 [| 4 |] [| 1.; 2.; 3.; 4. |] ] in
  match outs with
  | [ final; ys ] ->
      approx "final" (read final) [| 10.0 |];
      approx "ys" (read ys) [| 1.0; 3.0; 6.0; 10.0 |]
  | _ -> Alcotest.fail "scan arity"

let test_eval_reverse () =
  let outs =
    Lax.scan ~reverse:true cumsum
      [ sc 0.0 ]
      [ f32 [| 4 |] [| 1.; 2.; 3.; 4. |] ]
  in
  match outs with
  | [ final; ys ] ->
      approx "final" (read final) [| 10.0 |];
      approx "ys" (read ys) [| 10.0; 9.0; 7.0; 4.0 |]
  | _ -> Alcotest.fail "scan arity"

let test_jvp () =
  let wrapped inputs =
    match inputs with
    | [ i; x ] -> Lax.scan cumsum [ i ] [ x ]
    | _ -> assert false
  in
  let _, to_ =
    Ad.jvp wrapped
      [ sc 0.0; f32 [| 4 |] [| 1.; 2.; 3.; 4. |] ]
      [ sc 1.0; f32 [| 4 |] [| 0.5; 0.5; 0.5; 0.5 |] ]
  in
  match to_ with
  | [ final_t; ys_t ] ->
      approx "final_t" (read final_t) [| 3.0 |];
      approx "ys_t" (read ys_t) [| 1.5; 2.0; 2.5; 3.0 |]
  | _ -> Alcotest.fail "scan jvp arity"

let test_vmap () =
  let wrapped inputs =
    match inputs with
    | [ i; x ] -> Lax.scan cumsum [ i ] [ x ]
    | _ -> assert false
  in
  let xs = f32 [| 2; 3 |] [| 1.; 2.; 3.; 10.; 20.; 30. |] in
  let outs =
    Batching.vmap wrapped [ Some 0; Some 0 ] [ f32 [| 2 |] [| 0.; 0. |]; xs ]
  in
  match outs with
  | [ final; ys ] ->
      approx "vmap final" (read final) [| 6.0; 60.0 |];
      if shape_of ys <> [| 2; 3 |] then Alcotest.fail "vmap ys shape";
      approx "vmap ys" (read ys) [| 1.0; 3.0; 6.0; 10.0; 30.0; 60.0 |]
  | _ -> Alcotest.fail "scan vmap arity"

let test_transpose_deferred () =
  let wrapped inputs =
    match inputs with
    | [ i; x ] -> [ List.hd (Lax.scan cumsum [ i ] [ x ]) ]
    | _ -> assert false
  in
  match
    Ad.grad
      (fun ops -> List.hd (wrapped ops))
      [ sc 0.0; f32 [| 4 |] [| 1.; 2.; 3.; 4. |] ]
  with
  | _ -> Alcotest.fail "expected scan transpose to be deferred"
  | exception Failure _ -> ()

let () =
  Alcotest.run "lax_loops"
    [
      ( "scan",
        [
          Alcotest.test_case "eval" `Quick test_eval;
          Alcotest.test_case "eval_reverse" `Quick test_eval_reverse;
          Alcotest.test_case "jvp" `Quick test_jvp;
          Alcotest.test_case "vmap" `Quick test_vmap;
          Alcotest.test_case "transpose_deferred" `Quick test_transpose_deferred;
        ] );
    ]
