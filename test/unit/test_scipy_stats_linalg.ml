module T = Ojax.Types
module Nd = Ojax.Ndarray
module Dt = Ojax.Dtype
module KDE = Ojax.Scipy.Stats.Kde
module MVN = Ojax.Scipy.Stats.Multivariate_normal
module NL = Ojax.Numpy.Lax_numpy
module UF = Ojax.Numpy.Ufuncs

let () = Ojax.Lax.install ()
let v shape data = T.Concrete (Nd.of_floats Dt.F64 shape data)
let nd = function T.Concrete n -> n | _ -> failwith "expected concrete"
let scalar x = Nd.get_f (nd x) [||]

let approx name eps a b =
  Alcotest.(check bool) name true (abs_float (a -. b) <= eps)

let dataset2 () =
  v [| 2; 5 |] [| 0.1; -0.4; 1.2; 0.7; -0.9; 1.1; 0.3; -0.2; 0.5; -1.3 |]

let kde_pdf_positive () =
  let k = KDE.make (dataset2 ()) in
  let pts = v [| 2; 3 |] [| 0.0; 0.5; -0.5; 0.2; -0.3; 0.9 |] in
  let p = KDE.pdf k pts in
  let lp = KDE.logpdf k pts in
  let ps = nd p and ls = nd lp in
  Alcotest.(check (array int)) "pdf shape" [| 3 |] (Nd.shape ps);
  for i = 0 to 2 do
    Alcotest.(check bool)
      (Printf.sprintf "pdf[%d]>0" i)
      true
      (Nd.get_f ps [| i |] > 0.0);
    approx
      (Printf.sprintf "log==log(pdf)[%d]" i)
      1e-9 (Nd.get_f ls [| i |])
      (log (Nd.get_f ps [| i |]))
  done

let kde_accessors () =
  let k = KDE.make (dataset2 ()) in
  Alcotest.(check int) "d" 2 (KDE.d k);
  Alcotest.(check int) "n" 5 (KDE.n k);
  Alcotest.(check (array int))
    "cov shape" [| 2; 2 |]
    (Nd.shape (nd (KDE.covariance k)))

let mvn_matches_normal () =
  let x = v [||] [| 0.3 |] and mean = v [||] [| 0.1 |] in
  let cov = v [||] [| 2.0 |] in
  let got = scalar (MVN.logpdf x mean cov) in
  let want =
    (-0.5 *. ((0.3 -. 0.1) ** 2.0) /. 2.0)
    -. (0.5 *. (log (2.0 *. Float.pi) +. log 2.0))
  in
  approx "mvn scalar logpdf" 1e-9 got want

let mvn_pdf_exp () =
  let x = v [| 2 |] [| 0.4; -0.2 |] and mean = v [| 2 |] [| 0.0; 0.1 |] in
  let cov = v [| 2; 2 |] [| 2.0; 0.3; 0.3; 1.5 |] in
  approx "pdf==exp(logpdf)" 1e-9
    (scalar (MVN.pdf x mean cov))
    (exp (scalar (MVN.logpdf x mean cov)))

let integrate_box_1d_range () =
  let k = KDE.make (v [| 1; 6 |] [| 0.1; -0.3; 0.8; 0.2; -0.7; 0.5 |]) in
  let p = scalar (KDE.integrate_box_1d k Float.neg_infinity Float.infinity) in
  approx "full box integral ~ 1" 1e-6 p 1.0

let integrate_kde_symmetry () =
  let a = KDE.make (dataset2 ()) in
  let b =
    KDE.make (v [| 2; 4 |] [| 0.2; 0.6; -0.1; 0.3; 0.4; -0.5; 0.7; 0.0 |])
  in
  approx "integrate_kde symmetric" 1e-6
    (scalar (KDE.integrate_kde a b))
    (scalar (KDE.integrate_kde b a))

let resample_raises () =
  let k = KDE.make (dataset2 ()) in
  Alcotest.check_raises "resample bounded raise"
    (Failure
       "gaussian_kde.resample requires random.multivariate_normal and weighted \
        random.choice (M5 row 106); not yet available") (fun () ->
      ignore (KDE.resample k (v [| 2 |] [| 0.0; 1.0 |])))

let () =
  Alcotest.run "scipy_stats_linalg"
    [
      ( "kde",
        [
          Alcotest.test_case "pdf_logpdf" `Quick kde_pdf_positive;
          Alcotest.test_case "accessors" `Quick kde_accessors;
          Alcotest.test_case "integrate_box_1d" `Quick integrate_box_1d_range;
          Alcotest.test_case "integrate_kde" `Quick integrate_kde_symmetry;
          Alcotest.test_case "resample_raises" `Quick resample_raises;
        ] );
      ( "mvn",
        [
          Alcotest.test_case "scalar" `Quick mvn_matches_normal;
          Alcotest.test_case "pdf_exp" `Quick mvn_pdf_exp;
        ] );
    ]
