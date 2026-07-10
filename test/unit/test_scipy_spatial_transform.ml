module T = Ojax.Types
module Nd = Ojax.Ndarray
module Dt = Ojax.Dtype
module R = Ojax.Scipy.Spatial.Transform.Rotation
module Slerp = Ojax.Scipy.Spatial.Transform.Slerp

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

let check name eps got want =
  let g = flat got in
  Alcotest.(check int) (name ^ ":len") (Array.length want) (Array.length g);
  Array.iteri
    (fun i w ->
      Alcotest.(check bool)
        (Printf.sprintf "%s[%d]" name i)
        true
        (abs_float (g.(i) -. w) <= eps))
    want

let q () = v [| 4 |] [| 0.1; 0.2; 0.3; 0.9 |]
let eps = 1e-4

let t_as_quat () =
  check "as_quat" eps
    (R.as_quat (R.from_quat (q ())))
    [| 0.10259783; 0.20519567; 0.3077935; 0.9233805 |]

let t_as_matrix () =
  check "as_matrix" eps
    (R.as_matrix (R.from_quat (q ())))
    [|
      0.72631574;
      -0.52631575;
      0.44210523;
      0.61052626;
      0.78947365;
      -0.06315789;
      -0.31578946;
      0.31578946;
      0.89473677;
    |]

let t_as_rotvec () =
  check "as_rotvec" eps
    (R.as_rotvec (R.from_quat (q ())))
    [| 0.2106024; 0.4212048; 0.6318072 |]

let t_as_mrp () =
  check "as_mrp" eps
    (R.as_mrp (R.from_quat (q ())))
    [| 0.05334245; 0.1066849; 0.16002735 |]

let t_as_euler () =
  check "as_euler" eps
    (R.as_euler "xyz" (R.from_quat (q ())))
    [| 0.33929253; 0.32128835; 0.69899964 |]

let t_magnitude () =
  check "magnitude" eps (R.magnitude (R.from_quat (q ()))) [| 0.7880021 |]

let t_canonical () =
  check "canonical" eps
    (R.as_quat ~canonical:true (R.from_quat (q ())))
    [| 0.10259783; 0.20519567; 0.3077935; 0.9233805 |]

let t_inv () =
  check "inv" eps
    (R.as_quat (R.inv (R.from_quat (q ()))))
    [| -0.10259783; -0.20519567; -0.3077935; 0.9233805 |]

let t_from_euler () =
  let e = R.from_euler "z" (v [||] [| 90.0 |]) ~degrees:true in
  check "euler_z90_rotvec" eps (R.as_rotvec e) [| 0.0; 0.0; 1.5707964 |]

let t_from_mrp () =
  check "from_mrp" eps
    (R.as_quat (R.from_mrp (v [| 3 |] [| 0.1; 0.2; 0.3 |])))
    [| 0.1754386; 0.3508772; 0.5263158; 0.75438595 |]

let t_from_rotvec () =
  check "from_rotvec" eps
    (R.as_quat (R.from_rotvec (v [| 3 |] [| 0.2; -0.1; 0.4 |])))
    [| 0.0991273; -0.04956365; 0.1982546; 0.9738646 |]

let t_from_matrix () =
  let m = R.as_matrix (R.from_rotvec (v [| 3 |] [| 0.2; -0.1; 0.4 |])) in
  check "from_matrix" eps
    (R.as_quat (R.from_matrix m))
    [| 0.0991273; -0.04956365; 0.1982546; 0.9738646 |]

let t_apply () =
  check "apply" eps
    (R.apply (R.from_quat (q ())) (v [| 3 |] [| 1.0; 2.0; 3.0 |]))
    [| 0.99999994; 2.0; 2.9999998 |]

let t_compose () =
  let e = R.from_euler "z" (v [||] [| 90.0 |]) ~degrees:true in
  check "compose" eps
    (R.as_quat (R.compose (R.from_quat (q ())) e))
    [| 0.2176429; 0.07254763; 0.87057155; 0.4352858 |]

let t_mean () =
  let qs =
    v [| 3; 4 |]
      [| 0.1; 0.2; 0.3; 0.9; 0.0; 0.1; 0.0; 0.99; 0.2; 0.0; 0.1; 0.95 |]
  in
  check "mean" 1e-3
    (R.as_quat (R.mean (R.from_quat qs)))
    [| 0.10429325; 0.103174; 0.13854304; 0.9794303 |]

let t_slerp () =
  let rots =
    v [| 3; 3 |] [| 90.0; 0.0; 0.0; 0.0; 45.0; 0.0; 0.0; 0.0; -30.0 |]
  in
  let kr = R.from_euler "zxy" rots ~degrees:true in
  let sl = Slerp.init (v [| 3 |] [| 0.0; 1.0; 2.0 |]) kr in
  let out = Slerp.apply sl (v [| 5 |] [| 0.0; 0.5; 1.0; 1.5; 2.0 |]) in
  check "slerp" 1e-3 (R.as_euler "zxy" out)
    [|
      1.5707963;
      0.0;
      0.0;
      0.85309029;
      0.38711953;
      0.17768645;
      -2.38e-7;
      0.78539824;
      0.0;
      -0.056668043;
      0.39213133;
      -0.2834754;
      0.0;
      -2.38e-7;
      -0.52359891;
    |]

let () =
  Alcotest.run "scipy_spatial_transform"
    [
      ( "rotation",
        [
          Alcotest.test_case "as_quat" `Quick t_as_quat;
          Alcotest.test_case "as_matrix" `Quick t_as_matrix;
          Alcotest.test_case "as_rotvec" `Quick t_as_rotvec;
          Alcotest.test_case "as_mrp" `Quick t_as_mrp;
          Alcotest.test_case "as_euler" `Quick t_as_euler;
          Alcotest.test_case "magnitude" `Quick t_magnitude;
          Alcotest.test_case "canonical" `Quick t_canonical;
          Alcotest.test_case "inv" `Quick t_inv;
          Alcotest.test_case "from_euler" `Quick t_from_euler;
          Alcotest.test_case "from_mrp" `Quick t_from_mrp;
          Alcotest.test_case "from_rotvec" `Quick t_from_rotvec;
          Alcotest.test_case "from_matrix" `Quick t_from_matrix;
          Alcotest.test_case "apply" `Quick t_apply;
          Alcotest.test_case "compose" `Quick t_compose;
          Alcotest.test_case "mean" `Quick t_mean;
        ] );
      ("slerp", [ Alcotest.test_case "slerp" `Quick t_slerp ]);
    ]
