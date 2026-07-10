module T = Ojax.Types
module C = Ojax.Core
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module L = Ojax.Lax
module Ad = Ojax.Interpreters.Ad

let () = Ojax.Lax.install ()
let b1 = C.bind1
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)

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
  | T.Device _ -> failwith "expected concrete"

let approx name a b =
  Array.iter2
    (fun x y ->
      if Float.abs (x -. y) > 1e-5 then Alcotest.failf "%s: %f <> %f" name x y)
    a b

let diag_setup d =
  let matvec xs =
    match xs with [ x ] -> [ b1 T.Mul [ x; d ] ] | _ -> assert false
  in
  let solve _mv bs =
    match bs with [ bb ] -> [ b1 T.Div [ bb; d ] ] | _ -> assert false
  in
  (matvec, solve)

let test_eval () =
  let d = f32 [| 3 |] [| 2.0; 4.0; 5.0 |] in
  let b = f32 [| 3 |] [| 6.0; 8.0; 10.0 |] in
  let matvec, solve = diag_setup d in
  let out = L.custom_linear_solve ~symmetric:true matvec [ b ] solve in
  approx "x=b/d" (read (List.hd out)) [| 3.0; 2.0; 2.0 |]

let test_jvp () =
  let d = f32 [| 3 |] [| 2.0; 4.0; 5.0 |] in
  let b = f32 [| 3 |] [| 6.0; 8.0; 10.0 |] in
  let bdot = f32 [| 3 |] [| 1.0; 1.0; 1.0 |] in
  let matvec, solve = diag_setup d in
  let wrapped bs = L.custom_linear_solve ~symmetric:true matvec bs solve in
  let _, to_ = Ad.jvp wrapped [ b ] [ bdot ] in
  approx "x_dot=bdot/d" (read (List.hd to_)) [| 0.5; 0.25; 0.2 |]

let test_grad () =
  let d = f32 [| 3 |] [| 2.0; 4.0; 5.0 |] in
  let b = f32 [| 3 |] [| 6.0; 8.0; 10.0 |] in
  let matvec, solve = diag_setup d in
  let g =
    Ad.grad
      (fun bs ->
        let x = L.custom_linear_solve ~symmetric:true matvec bs solve in
        b1 (T.Reduce_sum [| 0 |]) [ List.hd x ])
      [ b ]
  in
  approx "grad sum = 1/d" (read (List.hd g)) [| 0.5; 0.25; 0.2 |]

let test_grad_requires_transpose_solve () =
  let d = f32 [| 3 |] [| 2.0; 4.0; 5.0 |] in
  let b = f32 [| 3 |] [| 6.0; 8.0; 10.0 |] in
  let matvec, solve = diag_setup d in
  match
    Ad.grad
      (fun bs ->
        let x = L.custom_linear_solve matvec bs solve in
        b1 (T.Reduce_sum [| 0 |]) [ List.hd x ])
      [ b ]
  with
  | _ -> Alcotest.fail "expected transpose_solve-required failure"
  | exception Failure _ -> ()

let () =
  Alcotest.run "lax_solves"
    [
      ( "solves",
        [
          Alcotest.test_case "eval" `Quick test_eval;
          Alcotest.test_case "jvp" `Quick test_jvp;
          Alcotest.test_case "grad" `Quick test_grad;
          Alcotest.test_case "grad_requires_transpose_solve" `Quick
            test_grad_requires_transpose_solve;
        ] );
    ]
