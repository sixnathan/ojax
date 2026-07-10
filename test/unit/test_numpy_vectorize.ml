module T = Ojax.Types
module Nd = Ojax.Ndarray
module Dt = Ojax.Dtype
module V = Ojax.Numpy.Vectorize
module TC = Ojax.Numpy.Tensor_contractions

let () = Ojax.Lax.install ()
let v shape data = T.Concrete (Nd.of_floats Dt.F32 shape data)
let nd = function T.Concrete n -> n | _ -> failwith "expected concrete"

let flat value =
  let n = nd value in
  let sz = Array.fold_left ( * ) 1 (Nd.shape n) in
  let a = Array.make sz 0.0 in
  ignore
    (Nd.fold
       (fun i x ->
         a.(i) <- x;
         i + 1)
       0 n);
  a

let check name shape want got =
  Alcotest.(check (array int)) (name ^ ":shape") shape (Nd.shape (nd got));
  let g = flat got in
  Array.iteri
    (fun i w ->
      Alcotest.(check bool)
        (Printf.sprintf "%s[%d]" name i)
        true
        (abs_float (g.(i) -. w) <= 1e-5))
    want

let matvec = function [ m; x ] -> TC.matmul m x | _ -> assert false

let cross3 = function
  | [ a; b ] ->
      let fa = flat a and fb = flat b in
      v [| 3 |]
        [|
          (fa.(1) *. fb.(2)) -. (fa.(2) *. fb.(1));
          (fa.(2) *. fb.(0)) -. (fa.(0) *. fb.(2));
          (fa.(0) *. fb.(1)) -. (fa.(1) *. fb.(0));
        |]
  | _ -> assert false

let mv m x = V.vectorize ~signature:"(n,m),(m)->(n)" matvec [ m; x ]
let cr a b = V.vectorize ~signature:"(k),(k)->(k)" cross3 [ a; b ]
let m1 () = v [| 2; 3 |] [| 0.; 1.; 2.; 3.; 4.; 5. |]
let x1 () = v [| 3 |] [| 1.; 0.; -1. |]
let t_mv1 () = check "mv1" [| 2 |] [| -2.; -2. |] (mv (m1 ()) (x1 ()))

let t_mv2 () =
  let m2 = v [| 4; 2; 3 |] (Array.init 24 float_of_int) in
  check "mv2" [| 4; 2 |]
    [| -2.; -2.; -2.; -2.; -2.; -2.; -2.; -2. |]
    (mv m2 (x1 ()))

let t_mv3 () =
  let x3 = v [| 5; 3 |] (Array.init 15 float_of_int) in
  check "mv3" [| 5; 2 |]
    [| 5.; 14.; 14.; 50.; 23.; 86.; 32.; 122.; 41.; 158. |]
    (mv (m1 ()) x3)

let t_cr1 () =
  check "cr1" [| 3 |] [| 0.; 0.; 1. |]
    (cr (v [| 3 |] [| 1.; 0.; 0. |]) (v [| 3 |] [| 0.; 1.; 0. |]))

let t_cr2 () =
  let aa = v [| 2; 3 |] [| 1.; 0.; 0.; 0.; 1.; 0. |] in
  check "cr2" [| 2; 3 |]
    [| 0.; 0.; 1.; 0.; 0.; 0. |]
    (cr aa (v [| 3 |] [| 0.; 1.; 0. |]))

let () =
  Alcotest.run "numpy_vectorize"
    [
      ( "gufunc",
        [
          Alcotest.test_case "matvec" `Quick t_mv1;
          Alcotest.test_case "matvec_batch_lhs" `Quick t_mv2;
          Alcotest.test_case "matvec_batch_rhs" `Quick t_mv3;
          Alcotest.test_case "cross" `Quick t_cr1;
          Alcotest.test_case "cross_batch" `Quick t_cr2;
        ] );
    ]
