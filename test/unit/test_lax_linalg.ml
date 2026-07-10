module T = Ojax.Types
module Nd = Ojax.Ndarray
module Dt = Ojax.Dtype
module LL = Ojax.Lax.Linalg
module D = Ojax.Linalg.Discover

let () = Ojax.Lax.install ()
let v shape data = T.Concrete (Nd.of_floats Dt.F64 shape data)
let nd = function T.Concrete n -> n | _ -> failwith "expected concrete"

let close =
  Alcotest.testable
    (Alcotest.pp Alcotest.(float 1e-9))
    (fun a b -> abs_float (a -. b) <= 1e-9)

let cholesky_known () =
  if D.available then begin
    let a = v [| 3; 3 |] [| 4.; 12.; -16.; 12.; 37.; -43.; -16.; -43.; 98. |] in
    let l = nd (LL.cholesky a) in
    Alcotest.check close "l00" 2.0 (Nd.get_f l [| 0; 0 |]);
    Alcotest.check close "l10" 6.0 (Nd.get_f l [| 1; 0 |]);
    Alcotest.check close "l20" (-8.0) (Nd.get_f l [| 2; 0 |]);
    Alcotest.check close "l22" 3.0 (Nd.get_f l [| 2; 2 |]);
    Alcotest.check close "upper01" 0.0 (Nd.get_f l [| 0; 1 |]);
    Alcotest.check close "upper12" 0.0 (Nd.get_f l [| 1; 2 |])
  end

let lu_permutation () =
  if D.available then begin
    let a = v [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |] in
    let _, pivots, permutation = LL.lu a in
    let piv = nd pivots and perm = nd permutation in
    Alcotest.(check (array int)) "pivots shape" [| 3 |] (Nd.shape piv);
    Alcotest.(check (array int)) "permutation shape" [| 3 |] (Nd.shape perm);
    let seen = Array.make 3 false in
    for i = 0 to 2 do
      let p = Int64.to_int (Nd.get_i64 perm [| i |]) in
      Alcotest.(check bool) "perm in range" true (p >= 0 && p < 3);
      seen.(p) <- true
    done;
    Alcotest.(check bool)
      "permutation is a bijection" true
      (Array.for_all (fun x -> x) seen)
  end

let triangular_solve_residual () =
  if D.available then begin
    let a = v [| 3; 3 |] [| 2.; 0.; 0.; 6.; 1.; 0.; -8.; 5.; 3. |] in
    let b = v [| 3; 1 |] [| 4.; 13.; 3. |] in
    let x = nd (LL.triangular_solve ~left_side:true ~lower:true a b) in
    let am = nd a and bm = nd b in
    for i = 0 to 2 do
      let acc = ref 0.0 in
      for j = 0 to 2 do
        acc := !acc +. (Nd.get_f am [| i; j |] *. Nd.get_f x [| j; 0 |])
      done;
      Alcotest.check close "residual" (Nd.get_f bm [| i; 0 |]) !acc
    done
  end

let tridiagonal_solve_residual () =
  if D.available then begin
    let dl = v [| 4 |] [| 0.; 1.; 1.; 1. |] in
    let d = v [| 4 |] [| 4.; 4.; 4.; 4. |] in
    let du = v [| 4 |] [| 1.; 1.; 1.; 0. |] in
    let b = v [| 4; 1 |] [| 1.; 2.; 3.; 4. |] in
    let x = nd (LL.tridiagonal_solve dl d du b) in
    let g i = Nd.get_f (nd b) [| i; 0 |] in
    let xv i = Nd.get_f x [| i; 0 |] in
    for i = 0 to 3 do
      let lo = if i > 0 then xv (i - 1) else 0.0 in
      let hi = if i < 3 then xv (i + 1) else 0.0 in
      Alcotest.check close "tridiag residual" (g i) ((4.0 *. xv i) +. lo +. hi)
    done
  end

let sym3 = [| 4.; 1.; 2.; 1.; 5.; 3.; 2.; 3.; 6. |]

let eigh_reconstruction () =
  if D.available then begin
    let a = v [| 3; 3 |] sym3 in
    let vv, ww = LL.eigh a in
    let vm = nd vv and wm = nd ww and am = nd a in
    for i = 0 to 2 do
      for j = 0 to 2 do
        let acc = ref 0.0 in
        for k = 0 to 2 do
          acc :=
            !acc
            +. Nd.get_f vm [| i; k |]
               *. Nd.get_f wm [| k |]
               *. Nd.get_f vm [| j; k |]
        done;
        Alcotest.check close "A = V diag(w) V^T" (Nd.get_f am [| i; j |]) !acc
      done
    done
  end

let svd_reconstruction () =
  if D.available then begin
    let a = v [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |] in
    let u, s, vt =
      match LL.svd a with [ u; s; vt ] -> (u, s, vt) | _ -> assert false
    in
    let um = nd u and sm = nd s and vtm = nd vt and am = nd a in
    for i = 0 to 2 do
      for j = 0 to 2 do
        let acc = ref 0.0 in
        for k = 0 to 2 do
          acc :=
            !acc
            +. Nd.get_f um [| i; k |]
               *. Nd.get_f sm [| k |]
               *. Nd.get_f vtm [| k; j |]
        done;
        Alcotest.check close "A = U diag(s) V^T" (Nd.get_f am [| i; j |]) !acc
      done
    done
  end

let hessenberg_structure () =
  if D.available then begin
    let a = v [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |] in
    let h, taus = LL.hessenberg a in
    Alcotest.(check (array int)) "hessenberg shape" [| 3; 3 |] (Nd.shape (nd h));
    Alcotest.(check (array int)) "taus shape" [| 2 |] (Nd.shape (nd taus))
  end

let leak_smoke () =
  if D.available then begin
    let before = Ojax.Pjrt.Abi.maxrss_bytes () in
    let a = v [| 3; 3 |] [| 4.; 1.; 1.; 1.; 4.; 1.; 1.; 1.; 4. |] in
    let g = v [| 3; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6.; 7.; 8.; 10. |] in
    let sym = v [| 3; 3 |] sym3 in
    let tri = v [| 3; 3 |] [| 2.; 0.; 0.; 6.; 1.; 0.; -8.; 5.; 3. |] in
    let b = v [| 3; 1 |] [| 4.; 13.; 3. |] in
    let dl = v [| 4 |] [| 0.; 1.; 1.; 1. |] in
    let d = v [| 4 |] [| 4.; 4.; 4.; 4. |] in
    let du = v [| 4 |] [| 1.; 1.; 1.; 0. |] in
    let rhs = v [| 4; 1 |] [| 1.; 2.; 3.; 4. |] in
    for _ = 1 to 2000 do
      ignore (LL.cholesky a);
      ignore (LL.lu g);
      ignore (LL.qr g);
      ignore (LL.eig g);
      ignore (LL.eigh sym);
      ignore (LL.hessenberg g);
      ignore (LL.schur g);
      ignore (LL.svd g);
      ignore (LL.tridiagonal sym);
      ignore (LL.triangular_solve ~left_side:true ~lower:true tri b);
      ignore (LL.tridiagonal_solve dl d du rhs)
    done;
    let after = Ojax.Pjrt.Abi.maxrss_bytes () in
    Alcotest.(check bool)
      "rss bounded across 2000 linalg primitive calls" true
      (after - before < 64 * 1024 * 1024)
  end

let () =
  Alcotest.run "lax_linalg"
    [
      ( "correctness",
        [
          Alcotest.test_case "cholesky known" `Quick cholesky_known;
          Alcotest.test_case "lu permutation" `Quick lu_permutation;
          Alcotest.test_case "triangular_solve residual" `Quick
            triangular_solve_residual;
          Alcotest.test_case "tridiagonal_solve residual" `Quick
            tridiagonal_solve_residual;
          Alcotest.test_case "eigh reconstruction" `Quick eigh_reconstruction;
          Alcotest.test_case "svd reconstruction" `Quick svd_reconstruction;
          Alcotest.test_case "hessenberg structure" `Quick hessenberg_structure;
        ] );
      ("ffi", [ Alcotest.test_case "leak smoke" `Slow leak_smoke ]);
    ]
