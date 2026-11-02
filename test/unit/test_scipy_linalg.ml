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

let hconj x = NL.matrix_transpose x

let expm_nilpotent () =
  let a = v [| 2; 2 |] [| 0.; 1.; 0.; 0. |] in
  let e = SL.expm a in
  check_mat "expm_nil" 1e-9 e (v [| 2; 2 |] [| 1.; 1.; 0.; 1. |])

let expm_zero () =
  let a = v [| 3; 3 |] (Array.make 9 0.) in
  check_mat "expm_zero" 1e-12 (SL.expm a)
    (v [| 3; 3 |] [| 1.; 0.; 0.; 0.; 1.; 0.; 0.; 0.; 1. |])

let block_diag_struct () =
  let out =
    SL.block_diag [ v [| 1; 1 |] [| 1. |]; v [| 2; 2 |] [| 2.; 3.; 4.; 5. |] ]
  in
  check_mat "bdiag" 0.0 out
    (v [| 3; 3 |] [| 1.; 0.; 0.; 0.; 2.; 3.; 0.; 4.; 5. |])

let toeplitz_vals () =
  let out = SL.toeplitz (v [| 3 |] [| 1.; 2.; 3. |]) in
  check_mat "toeplitz" 0.0 out
    (v [| 3; 3 |] [| 1.; 2.; 3.; 2.; 1.; 2.; 3.; 2.; 1. |])

let hessenberg_recon () =
  if D.available then
    match SL.hessenberg ~calc_q:true (a_gen ()) with
    | [ h; q ] ->
        let recon = TC.matmul (TC.matmul q h) (hconj q) in
        check_mat "hess" 1e-9 recon (a_gen ())
    | _ -> Alcotest.fail "hessenberg arity"

let polar_recon () =
  if D.available then
    match SL.polar ~method_:"svd" (a_gen ()) with
    | [ u; p ] -> check_mat "polar" 1e-6 (TC.matmul u p) (a_gen ())
    | _ -> Alcotest.fail "polar arity"

let eigh_recon () =
  if D.available then
    let a = a_pd () in
    match SL.eigh a with
    | [ w; vv ] ->
        let recon = TC.matmul a vv in
        let want = TC.matmul vv (NL.diag w) in
        check_mat "eigh" 1e-6 recon want
    | _ -> Alcotest.fail "eigh arity"

let eigtri_vs_eigh () =
  if D.available then begin
    let d = v [| 4 |] [| 1.; 2.; 3.; 4. |] in
    let e = v [| 3 |] [| 1.; 1.; 1. |] in
    let w = List.hd (SL.eigh_tridiagonal ~eigvals_only:true d e) in
    let t =
      v [| 4; 4 |]
        [| 1.; 1.; 0.; 0.; 1.; 2.; 1.; 0.; 0.; 1.; 3.; 1.; 0.; 0.; 1.; 4. |]
    in
    let w2 =
      match SL.eigh ~eigvals_only:true t with [ w ] -> w | _ -> assert false
    in
    check_vec "eigtri" 1e-6 w w2
  end

let raises name f =
  Alcotest.(check bool)
    name true
    (try
       ignore (f ());
       false
     with
    | Failure _ -> true
    | _ -> true)

let bounded_raises () =
  raises "funm" (fun () -> SL.funm (a_gen ()) (fun x -> x))

let special_sizes () =
  let module SS = Ojax.Scipy.Linalg_special in
  let shape_of x = Ojax.Core.(get_aval x).T.shape in
  let constructors =
    [
      ("hilbert", SS.hilbert, fun n -> [| n; n |]);
      ("dft", (fun n -> SS.dft n), fun n -> [| n; n |]);
      ("pascal", (fun n -> SS.pascal n), fun n -> [| n; n |]);
      ("invpascal", (fun n -> SS.invpascal n), fun n -> [| n; n |]);
      ("helmert", (fun n -> SS.helmert n), fun n -> [| n - 1; n |]);
      ("invhilbert", SS.invhilbert, fun n -> [| n; n |]);
    ]
  in
  List.iter
    (fun (name, f, want) ->
      List.iter
        (fun n ->
          Alcotest.(check (array int))
            (Printf.sprintf "%s %d" name n)
            (want n)
            (shape_of (f n)))
        [ 2; 3; 2 ])
    constructors

let sylvester_options () =
  let module SS = Ojax.Scipy.Linalg_special in
  let a = v [| 2; 2 |] [| 1.; 0.; 0.; 2. |] in
  let b = v [| 2; 2 |] [| -0.999; 0.; 0.; 3. |] in
  let c = v [| 2; 2 |] [| 1.; 1.; 1.; 1. |] in
  List.iter
    (fun (method_, tol, want_nan) ->
      let x = nd (SS.solve_sylvester ~method_ ~tol a b c) in
      Alcotest.(check bool)
        (Printf.sprintf "%s tol=%g nan" method_ tol)
        want_nan
        (Float.is_nan (Nd.get_f x [| 0; 0 |])))
    [
      ("schur", 1e-8, false);
      ("schur", 1e-2, true);
      ("schur", 1e-8, false);
      ("eigen", 1e-2, true);
      ("eigen", 1e-8, false);
    ]

let () =
  Alcotest.run "scipy_linalg"
    [
      ( "reconstruct",
        [
          Alcotest.test_case "cholesky" `Quick cho_reconstruct;
          Alcotest.test_case "lu" `Quick lu_reconstruct;
          Alcotest.test_case "hessenberg" `Quick hessenberg_recon;
          Alcotest.test_case "polar" `Quick polar_recon;
          Alcotest.test_case "eigh" `Quick eigh_recon;
        ] );
      ( "solve",
        [
          Alcotest.test_case "solve_gen" `Quick solve_gen;
          Alcotest.test_case "solve_pos" `Quick solve_pos;
          Alcotest.test_case "lu_solve" `Quick lu_solve_res;
          Alcotest.test_case "solve_triangular" `Quick solve_triangular_res;
        ] );
      ( "construct",
        [
          Alcotest.test_case "expm_nilpotent" `Quick expm_nilpotent;
          Alcotest.test_case "expm_zero" `Quick expm_zero;
          Alcotest.test_case "block_diag" `Quick block_diag_struct;
          Alcotest.test_case "toeplitz" `Quick toeplitz_vals;
          Alcotest.test_case "eigtri_vs_eigh" `Quick eigtri_vs_eigh;
          Alcotest.test_case "bounded_raises" `Quick bounded_raises;
          Alcotest.test_case "special_sizes" `Quick special_sizes;
          Alcotest.test_case "sylvester_options" `Quick sylvester_options;
        ] );
    ]
