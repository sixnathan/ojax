module T = Ojax.Types
module Nd = Ojax.Ndarray
module Dt = Ojax.Dtype
module SL = Ojax.Scipy.Linalg
module TC = Ojax.Numpy.Tensor_contractions
module NL = Ojax.Numpy.Lax_numpy
module D = Ojax.Linalg.Discover

let () = Ojax.Lax.install ()
let v shape data = T.Concrete (Nd.of_floats Dt.F64 shape data)
let nd = function T.Concrete n -> n | _ -> failwith "expected concrete"

let approx name eps a b =
  Alcotest.(check bool) name true (abs_float (a -. b) <= eps)

let a_pd () = v [| 3; 3 |] [| 4.; 1.; 1.; 1.; 3.; 0.; 1.; 0.; 2. |]
let a_gen () = v [| 3; 3 |] [| 2.; 1.; 1.; 1.; 3.; 2.; 1.; 0.; 4. |]

let check_mat name eps got want =
  let g = nd got and w = nd want in
  Alcotest.(check (array int)) (name ^ ":shape") (Nd.shape w) (Nd.shape g);
  let sh = Nd.shape g in
  for i = 0 to sh.(0) - 1 do
    for j = 0 to sh.(1) - 1 do
      approx
        (Printf.sprintf "%s[%d,%d]" name i j)
        eps
        (Nd.get_f g [| i; j |])
        (Nd.get_f w [| i; j |])
    done
  done

let check_vec name eps got want =
  let g = nd got and w = nd want in
  let n = (Nd.shape g).(0) in
  for i = 0 to n - 1 do
    approx
      (Printf.sprintf "%s[%d]" name i)
      eps (Nd.get_f g [| i |]) (Nd.get_f w [| i |])
  done

let cho_reconstruct () =
  if D.available then begin
    let a = a_pd () in
    let l = SL.cholesky ~lower:true a in
    let recon = TC.matmul l (NL.matrix_transpose l) in
    check_mat "chol" 1e-9 recon a
  end

let solve_gen () =
  if D.available then begin
    let a = a_gen () in
    let b = v [| 3 |] [| 1.; 2.; 3. |] in
    let x = SL.solve a b in
    check_vec "solve_res" 1e-6 (TC.matmul a x) b
  end

let solve_pos () =
  if D.available then begin
    let a = a_pd () in
    let b = v [| 3 |] [| 1.; -2.; 4. |] in
    let x = SL.solve ~assume_a:"pos" a b in
    check_vec "solve_pos_res" 1e-6 (TC.matmul a x) b
  end

let lu_reconstruct () =
  if D.available then begin
    let a = a_gen () in
    match SL.lu a with
    | [ p; l; u ] ->
        let recon = TC.matmul (TC.matmul p l) u in
        check_mat "plu" 1e-9 recon a
    | _ -> Alcotest.fail "lu arity"
  end

let lu_solve_res () =
  if D.available then begin
    let a = a_gen () in
    let b = v [| 3 |] [| 5.; 1.; -2. |] in
    let lu, piv = SL.lu_factor a in
    let x = SL.lu_solve (lu, piv) b in
    check_vec "lu_solve_res" 1e-6 (TC.matmul a x) b
  end

let solve_triangular_res () =
  if D.available then begin
    let a = v [| 3; 3 |] [| 2.; 0.; 0.; 1.; 3.; 0.; 4.; 1.; 5. |] in
    let b = v [| 3 |] [| 2.; 5.; 6. |] in
    let x = SL.solve_triangular ~lower:true a b in
    check_vec "st_res" 1e-6 (TC.matmul a x) b
  end

let () =
  Alcotest.run "scipy_linalg"
    [
      ( "reconstruct",
        [
          Alcotest.test_case "cholesky" `Quick cho_reconstruct;
          Alcotest.test_case "lu" `Quick lu_reconstruct;
        ] );
      ( "solve",
        [
          Alcotest.test_case "solve_gen" `Quick solve_gen;
          Alcotest.test_case "solve_pos" `Quick solve_pos;
          Alcotest.test_case "lu_solve" `Quick lu_solve_res;
          Alcotest.test_case "solve_triangular" `Quick solve_triangular_res;
        ] );
    ]
