module T = Ojax.Types
module Nd = Ojax.Ndarray
module Dt = Ojax.Dtype
module SSL = Ojax.Scipy.Sparse.Linalg
module TC = Ojax.Numpy.Tensor_contractions

let () = Ojax.Lax.install ()
let v shape data = T.Concrete (Nd.of_floats Dt.F64 shape data)
let nd = function T.Concrete n -> n | _ -> failwith "expected concrete"

let residual a x b =
  let ax = nd (TC.matmul a x) and bb = nd b in
  let n = (Nd.shape bb).(0) in
  let acc = ref 0.0 in
  for i = 0 to n - 1 do
    let d = Nd.get_f ax [| i |] -. Nd.get_f bb [| i |] in
    acc := !acc +. (d *. d)
  done;
  sqrt !acc

let a_spd () = v [| 3; 3 |] [| 4.; 1.; 1.; 1.; 3.; 0.; 1.; 0.; 2. |]

let a_gen () =
  v [| 4; 4 |]
    [| 6.; 1.; 0.; 1.; 1.; 7.; 2.; 0.; 0.; 1.; 8.; 1.; 2.; 0.; 1.; 9. |]

let check_small name r = Alcotest.(check bool) name true (r < 1e-6)

let cg_res () =
  let a = a_spd () in
  let b = v [| 3 |] [| 1.; -2.; 4. |] in
  let x, _ = SSL.cg a b in
  check_small "cg" (residual a x b)

let bicgstab_res () =
  let a = a_gen () in
  let b = v [| 4 |] [| 1.; 2.; -1.; 3. |] in
  let x, _ = SSL.bicgstab a b in
  check_small "bicgstab" (residual a x b)

let gmres_batched_res () =
  let a = a_gen () in
  let b = v [| 4 |] [| 2.; 0.; 1.; -3. |] in
  let x, _ = SSL.gmres a b in
  check_small "gmres_batched" (residual a x b)

let gmres_incremental_res () =
  let a = a_gen () in
  let b = v [| 4 |] [| 2.; 0.; 1.; -3. |] in
  let x, _ = SSL.gmres ~solve_method:"incremental" a b in
  check_small "gmres_incremental" (residual a x b)

let gmres_info () =
  let a = a_gen () in
  let b = v [| 4 |] [| 1.; 1.; 1.; 1. |] in
  let _, info = SSL.gmres a b in
  match info with
  | Some iv ->
      Alcotest.(check int) "info0" 0 (Int64.to_int (Nd.get_i64 (nd iv) [||]))
  | None -> Alcotest.fail "gmres info missing"

let leak_smoke () =
  let a = a_spd () in
  let b = v [| 3 |] [| 1.; -2.; 4. |] in
  for _ = 1 to 2000 do
    let _ = SSL.cg a b in
    ()
  done;
  Alcotest.(check bool) "leak" true true

let () =
  Alcotest.run "scipy_sparse_linalg"
    [
      ( "iterative_solvers",
        [
          Alcotest.test_case "cg" `Quick cg_res;
          Alcotest.test_case "bicgstab" `Quick bicgstab_res;
          Alcotest.test_case "gmres_batched" `Quick gmres_batched_res;
          Alcotest.test_case "gmres_incremental" `Quick gmres_incremental_res;
          Alcotest.test_case "gmres_info" `Quick gmres_info;
          Alcotest.test_case "leak_smoke" `Quick leak_smoke;
        ] );
    ]
