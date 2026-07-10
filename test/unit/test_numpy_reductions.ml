module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module R = Ojax.Numpy.Reductions

let () = Ojax.Lax.install ()
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)
let i32 shape xs = T.Concrete (Nd.of_floats D.I32 shape xs)
let b shape xs = T.Concrete (Nd.of_floats D.Bool shape xs)

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

let shape_of v = (Ojax.Core.get_aval v).T.shape
let dtype_of v = (Ojax.Core.get_aval v).T.dtype
let farr = Alcotest.(array (float 1e-6))

let test_sum_prod () =
  let x = i32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  Alcotest.check farr "sum all" [| 21. |] (read (R.sum x));
  Alcotest.check farr "sum axis1" [| 6.; 15. |] (read (R.sum ~axis:[| 1 |] x));
  Alcotest.(check (array int))
    "keepdims" [| 2; 1 |]
    (shape_of (R.sum ~axis:[| 1 |] ~keepdims:true x));
  Alcotest.check farr "prod axis1" [| 6.; 120. |]
    (read (R.prod ~axis:[| 1 |] x))

let test_minmax_ptp () =
  let x = f32 [| 2; 3 |] [| 1.; 5.; 3.; 4.; 2.; 6. |] in
  Alcotest.check farr "max axis1" [| 5.; 6. |] (read (R.max ~axis:[| 1 |] x));
  Alcotest.check farr "min axis1" [| 1.; 2. |] (read (R.min ~axis:[| 1 |] x));
  Alcotest.check farr "ptp axis1" [| 4.; 4. |] (read (R.ptp ~axis:[| 1 |] x))

let test_all_any () =
  let x = b [| 2; 2 |] [| 1.; 0.; 1.; 1. |] in
  Alcotest.check farr "all axis1" [| 0.; 1. |] (read (R.all ~axis:[| 1 |] x));
  Alcotest.check farr "any axis1" [| 1.; 1. |] (read (R.any ~axis:[| 1 |] x));
  Alcotest.(check bool) "all dtype bool" true (dtype_of (R.all x) = D.Bool)

let test_mean_var_std () =
  let x = f32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  Alcotest.check farr "mean" [| 2.5 |] (read (R.mean x));
  Alcotest.check farr "var" [| 1.25 |] (read (R.var x));
  Alcotest.check farr "var ddof1" [| 5. /. 3. |] (read (R.var ~ddof:1 x));
  Alcotest.check farr "std" [| sqrt 1.25 |] (read (R.std x));
  Alcotest.(check bool)
    "mean int->f32" true
    (dtype_of (R.mean (i32 [| 2 |] [| 1.; 2. |])) = D.F32)

let test_count_cumsum () =
  let x = i32 [| 2; 3 |] [| 0.; 2.; 0.; 4.; 0.; 6. |] in
  Alcotest.check farr "count_nonzero" [| 3. |] (read (R.count_nonzero x));
  Alcotest.check farr "count axis1" [| 1.; 2. |]
    (read (R.count_nonzero ~axis:[| 1 |] x));
  let y = f32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  Alcotest.check farr "cumsum flat" [| 1.; 3.; 6.; 10. |] (read (R.cumsum y));
  let z = f32 [| 2; 2 |] [| 1.; 2.; 3.; 4. |] in
  Alcotest.check farr "cumsum axis1" [| 1.; 3.; 3.; 7. |]
    (read (R.cumsum ~axis:1 z))

let test_nan_and_average () =
  let x = f32 [| 4 |] [| 1.; 2.; 3.; 4. |] in
  Alcotest.check farr "nansum eq sum" [| 10. |] (read (R.nansum x));
  Alcotest.check farr "nanmean eq mean" [| 2.5 |] (read (R.nanmean x));
  Alcotest.check farr "average" [| 2.5 |] (read (R.average x));
  let w = f32 [| 4 |] [| 4.; 3.; 2.; 1. |] in
  Alcotest.check farr "weighted average" [| 2. |]
    (read (R.average ~weights:w x))

let () =
  Alcotest.run "numpy_reductions"
    [
      ( "reductions",
        [
          Alcotest.test_case "sum_prod" `Quick test_sum_prod;
          Alcotest.test_case "minmax_ptp" `Quick test_minmax_ptp;
          Alcotest.test_case "all_any" `Quick test_all_any;
          Alcotest.test_case "mean_var_std" `Quick test_mean_var_std;
          Alcotest.test_case "count_cumsum" `Quick test_count_cumsum;
          Alcotest.test_case "nan_average" `Quick test_nan_and_average;
        ] );
    ]
