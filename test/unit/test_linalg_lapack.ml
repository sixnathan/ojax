module L = Ojax.Linalg.Lapack
module D = Ojax.Linalg.Discover

let f64 =
  Alcotest.testable
    (Alcotest.pp Alcotest.(float 1e-9))
    (fun a b -> abs_float (a -. b) <= 1e-9)

let of_list l =
  let a = L.alloc_f64 (List.length l) in
  List.iteri (fun i x -> Bigarray.Array1.set a i x) l;
  a

let get a i = Bigarray.Array1.get a i

let available () =
  Alcotest.(check bool) "accelerate available on macOS" true D.available;
  Alcotest.(check string) "backend name" "accelerate" D.backend

let abi_int () = Alcotest.(check int) "LAPACK int is 32-bit" 4 D.abi_int_size

let potrf_spd () =
  let a = of_list [ 4.; 12.; -16.; 12.; 37.; -43.; -16.; -43.; 98. ] in
  let info = L.potrf ~uplo:'L' ~n:3 ~a ~lda:3 in
  Alcotest.(check int) "potrf info" 0 info;
  Alcotest.check f64 "L00" 2.0 (get a 0);
  Alcotest.check f64 "L11" 1.0 (get a 4);
  Alcotest.check f64 "L22" 3.0 (get a 8)

let getrf_lu () =
  let a = of_list [ 1.; 2.; 3.; 4. ] in
  let ipiv = L.alloc_i32 2 in
  let info = L.getrf ~m:2 ~n:2 ~a ~lda:2 ~ipiv in
  Alcotest.(check int) "getrf info" 0 info;
  Alcotest.(check int32) "ipiv0" 2l (Bigarray.Array1.get ipiv 0)

let geqrf_orgqr_orthonormal () =
  let a = of_list [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let tau = L.alloc_f64 2 in
  let i1 = L.geqrf ~m:3 ~n:2 ~a ~lda:3 ~tau in
  Alcotest.(check int) "geqrf info" 0 i1;
  let i2 = L.orgqr ~m:3 ~n:2 ~k:2 ~a ~lda:3 ~tau in
  Alcotest.(check int) "orgqr info" 0 i2;
  let col c r = get a ((c * 3) + r) in
  let dot c1 c2 =
    (col c1 0 *. col c2 0) +. (col c1 1 *. col c2 1) +. (col c1 2 *. col c2 2)
  in
  Alcotest.check f64 "q0.q0" 1.0 (dot 0 0);
  Alcotest.check f64 "q1.q1" 1.0 (dot 1 1);
  Alcotest.check f64 "q0.q1" 0.0 (dot 0 1)

let gesdd_singular_values () =
  let a = of_list [ 1.; 2.; 3.; 4.; 5.; 6. ] in
  let s = L.alloc_f64 2 in
  let u = L.alloc_f64 6 in
  let vt = L.alloc_f64 4 in
  let info = L.gesdd ~jobz:'S' ~m:3 ~n:2 ~a ~lda:3 ~s ~u ~ldu:3 ~vt ~ldvt:2 in
  Alcotest.(check int) "gesdd info" 0 info;
  Alcotest.(check bool) "s0 positive descending" true (get s 0 > get s 1);
  Alcotest.(check bool) "s1 positive" true (get s 1 > 0.0)

let syevd_eigenvalues () =
  let a = of_list [ 2.; 0.; 0.; 0.; 3.; 0.; 0.; 0.; 5. ] in
  let w = L.alloc_f64 3 in
  let info = L.syevd ~jobz:'V' ~uplo:'L' ~n:3 ~a ~lda:3 ~w in
  Alcotest.(check int) "syevd info" 0 info;
  Alcotest.check f64 "w0" 2.0 (get w 0);
  Alcotest.check f64 "w1" 3.0 (get w 1);
  Alcotest.check f64 "w2" 5.0 (get w 2)

let geev_eigenvalues () =
  let a = of_list [ 1.; 0.; 0.; 2. ] in
  let wri = L.alloc_f64 4 in
  let vl = L.alloc_f64 1 in
  let vr = L.alloc_f64 4 in
  let info =
    L.geev ~jobvl:'N' ~jobvr:'V' ~n:2 ~a ~lda:2 ~wri ~vl ~ldvl:1 ~vr ~ldvr:2
  in
  Alcotest.(check int) "geev info" 0 info;
  let e0 = get wri 0 and e1 = get wri 1 in
  let lo = min e0 e1 and hi = max e0 e1 in
  Alcotest.check f64 "eig lo" 1.0 lo;
  Alcotest.check f64 "eig hi" 2.0 hi;
  Alcotest.check f64 "imag0" 0.0 (get wri 2);
  Alcotest.check f64 "imag1" 0.0 (get wri 3)

let gees_schur () =
  let a = of_list [ 1.; 0.; 3.; 2. ] in
  let wri = L.alloc_f64 4 in
  let vs = L.alloc_f64 4 in
  let sdim = L.alloc_i32 1 in
  let info = L.gees ~jobvs:'V' ~n:2 ~a ~lda:2 ~wri ~vs ~ldvs:2 ~sdim in
  Alcotest.(check int) "gees info" 0 info;
  let e0 = get wri 0 and e1 = get wri 1 in
  let lo = min e0 e1 and hi = max e0 e1 in
  Alcotest.check f64 "schur eig lo" 1.0 lo;
  Alcotest.check f64 "schur eig hi" 2.0 hi

let trtrs_solve () =
  let a = of_list [ 2.; 0.; 1.; 3. ] in
  let b = of_list [ 4.; 9. ] in
  let info =
    L.trtrs ~uplo:'U' ~trans:'N' ~diag:'N' ~n:2 ~nrhs:1 ~a ~lda:2 ~b ~ldb:2
  in
  Alcotest.(check int) "trtrs info" 0 info;
  Alcotest.check f64 "x0" 0.5 (get b 0);
  Alcotest.check f64 "x1" 3.0 (get b 1)

let gecon_rcond () =
  let a = of_list [ 1.; 0.; 0.; 2. ] in
  let ipiv = L.alloc_i32 2 in
  let i1 = L.getrf ~m:2 ~n:2 ~a ~lda:2 ~ipiv in
  Alcotest.(check int) "getrf info" 0 i1;
  let rcond = L.alloc_f64 1 in
  let info = L.gecon ~norm:'1' ~n:2 ~a ~lda:2 ~anorm:2.0 ~rcond in
  Alcotest.(check int) "gecon info" 0 info;
  Alcotest.check f64 "rcond" 0.5 (get rcond 0)

let leak_smoke () =
  let before = Ojax.Pjrt.Abi.maxrss_bytes () in
  for _ = 1 to 3000 do
    let a = of_list [ 4.; 12.; -16.; 12.; 37.; -43.; -16.; -43.; 98. ] in
    let _ = L.potrf ~uplo:'L' ~n:3 ~a ~lda:3 in
    let g = of_list [ 1.; 2.; 3.; 4.; 5.; 6. ] in
    let tau = L.alloc_f64 2 in
    let _ = L.geqrf ~m:3 ~n:2 ~a:g ~lda:3 ~tau in
    ()
  done;
  Gc.full_major ();
  let after = Ojax.Pjrt.Abi.maxrss_bytes () in
  Alcotest.(check bool)
    "rss bounded across 3000 factorizations" true
    (after - before < 64 * 1024 * 1024)

let unavailable_message () =
  let msg =
    try Printexc.to_string (L.Unavailable "linalg unavailable on this platform")
    with _ -> ""
  in
  Alcotest.(check bool)
    "unavailable carries a clean message" true
    (String.length msg > 0)

let () =
  Alcotest.run "linalg_lapack"
    [
      ( "seam",
        [
          Alcotest.test_case "available" `Quick available;
          Alcotest.test_case "abi int size" `Quick abi_int;
          Alcotest.test_case "unavailable message" `Quick unavailable_message;
        ] );
      ( "factorizations",
        [
          Alcotest.test_case "potrf spd" `Quick potrf_spd;
          Alcotest.test_case "getrf lu" `Quick getrf_lu;
          Alcotest.test_case "geqrf+orgqr orthonormal" `Quick
            geqrf_orgqr_orthonormal;
          Alcotest.test_case "gesdd singular values" `Quick
            gesdd_singular_values;
          Alcotest.test_case "syevd eigenvalues" `Quick syevd_eigenvalues;
          Alcotest.test_case "geev eigenvalues" `Quick geev_eigenvalues;
          Alcotest.test_case "gees schur" `Quick gees_schur;
          Alcotest.test_case "trtrs solve" `Quick trtrs_solve;
          Alcotest.test_case "gecon rcond" `Quick gecon_rcond;
        ] );
      ("ffi", [ Alcotest.test_case "leak smoke" `Slow leak_smoke ]);
    ]
