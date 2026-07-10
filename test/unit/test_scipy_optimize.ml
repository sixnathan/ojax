module T = Ojax.Types
module Nd = Ojax.Ndarray
module Dt = Ojax.Dtype
module Min = Ojax.Scipy.Optimize.Minimize
module LS = Ojax.Scipy.Optimize.Line_search
module R = Ojax.Numpy.Reductions
module U = Ojax.Numpy.Ufuncs
module FU = Ojax.Flatten_util
module TU = Ojax.Tree_util

let () = Ojax.Lax.install ()
let v shape data = T.Concrete (Nd.of_floats Dt.F64 shape data)
let nd = function T.Concrete n -> n | _ -> failwith "expected concrete"
let getf x idx = Nd.get_f (nd x) idx
let sphere x = R.sum (U.multiply x x)

let quad x =
  let b = v [| 3 |] [| 1.0; -2.0; 0.5 |] in
  let d = U.subtract x b in
  R.sum (U.multiply d d)

let close a b tol =
  Alcotest.(check bool) "close" true (abs_float (a -. b) <= tol)

let test_quad () =
  let x0 = v [| 3 |] [| 0.0; 0.0; 0.0 |] in
  let r = Min.minimize quad x0 ~method_:"BFGS" () in
  close (getf r.Min.x [| 0 |]) 1.0 1e-8;
  close (getf r.Min.x [| 1 |]) (-2.0) 1e-8;
  close (getf r.Min.x [| 2 |]) 0.5 1e-8;
  Alcotest.(check bool) "success" true r.Min.success

let test_sphere () =
  let x0 = v [| 3 |] [| 1.0; 2.0; 3.0 |] in
  let r = Min.minimize sphere x0 ~method_:"BFGS" () in
  close (getf r.Min.x [| 0 |]) 0.0 1e-6;
  close (getf r.Min.x [| 1 |]) 0.0 1e-6;
  close (getf r.Min.x [| 2 |]) 0.0 1e-6;
  Alcotest.(check bool) "success" true r.Min.success

let test_lbfgs_quad () =
  let x0 = v [| 3 |] [| 0.0; 0.0; 0.0 |] in
  let r =
    Min.minimize quad x0 ~method_:"l-bfgs-experimental-do-not-rely-on-this" ()
  in
  close (getf r.Min.x [| 0 |]) 1.0 1e-6;
  close (getf r.Min.x [| 1 |]) (-2.0) 1e-6;
  close (getf r.Min.x [| 2 |]) 0.5 1e-6

let test_line_search () =
  let x0 = v [| 3 |] [| 0.0; 0.0; 0.0 |] in
  let pk = v [| 3 |] [| 1.0; -2.0; 0.5 |] in
  let r = LS.line_search quad x0 pk () in
  Alcotest.(check bool) "ls ok" true (not r.LS.failed)

let test_ravel () =
  let a = v [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] in
  let b = v [| 3 |] [| 5.0; 6.0; 7.0 |] in
  let tree = TU.List [ TU.Leaf a; TU.Leaf b ] in
  let flat, unravel = FU.ravel_pytree tree in
  Alcotest.(check int)
    "flat size" 7
    (Array.fold_left ( * ) 1 (Nd.shape (nd flat)));
  close (getf flat [| 0 |]) 1.0 0.0;
  close (getf flat [| 4 |]) 5.0 0.0;
  let back = unravel flat in
  match back with
  | TU.List [ TU.Leaf a2; TU.Leaf _ ] -> close (getf a2 [| 1; 1 |]) 4.0 0.0
  | _ -> Alcotest.fail "structure"

let test_ravel_mixed () =
  Ojax.Config.set Ojax.Config.enable_x64 true;
  Fun.protect ~finally:(fun () -> Ojax.Config.set Ojax.Config.enable_x64 false)
  @@ fun () ->
  let a = T.Concrete (Nd.of_floats Dt.F32 [| 2 |] [| 1.0; 2.0 |]) in
  let b = T.Concrete (Nd.of_floats Dt.F64 [| 2 |] [| 3.0; 4.0 |]) in
  let tree = TU.List [ TU.Leaf a; TU.Leaf b ] in
  let flat, unravel = FU.ravel_pytree tree in
  Alcotest.(check bool) "flat f64" true (Nd.dtype (nd flat) = Dt.F64);
  match unravel flat with
  | TU.List [ TU.Leaf a2; TU.Leaf b2 ] ->
      Alcotest.(check bool) "a2 f32" true (Nd.dtype (nd a2) = Dt.F32);
      Alcotest.(check bool) "b2 f64" true (Nd.dtype (nd b2) = Dt.F64);
      close (getf a2 [| 0 |]) 1.0 1e-6;
      close (getf b2 [| 1 |]) 4.0 1e-12
  | _ -> Alcotest.fail "structure"

let () =
  Alcotest.run "scipy_optimize"
    [
      ( "optimize",
        [
          Alcotest.test_case "quad" `Quick test_quad;
          Alcotest.test_case "sphere" `Quick test_sphere;
          Alcotest.test_case "lbfgs_quad" `Quick test_lbfgs_quad;
          Alcotest.test_case "line_search" `Quick test_line_search;
          Alcotest.test_case "ravel" `Quick test_ravel;
          Alcotest.test_case "ravel_mixed" `Quick test_ravel_mixed;
        ] );
    ]
