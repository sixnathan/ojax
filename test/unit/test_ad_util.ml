module A = Ojax.Ad_util
module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype

let scalar dtype v = Nd.of_floats dtype [| 1 |] [| v |]
let read nd = Nd.get_f nd [| 0 |]

let concrete_of = function
  | T.Concrete a -> a
  | T.Tracer _ -> failwith "not concrete"
  | T.Device _ -> failwith "not concrete"

let zeros_like_aval_shape () =
  let av = { T.shape = [| 2; 3 |]; dtype = D.F32; weak_type = false } in
  let z = A.zeros_like_aval av in
  let nd = concrete_of z in
  Alcotest.(check (array int)) "shape" [| 2; 3 |] (Nd.shape nd);
  Alcotest.(check bool) "dtype f32" true (Nd.dtype nd = D.F32);
  let all_zero = Nd.fold (fun acc x -> acc && x = 0.0) true nd in
  Alcotest.(check bool) "all zero" true all_zero

let zeros_like_value_matches () =
  let v = T.Concrete (Nd.of_floats D.I32 [| 4 |] [| 1.; 2.; 3.; 4. |]) in
  let z = concrete_of (A.zeros_like_value v) in
  Alcotest.(check (array int)) "shape" [| 4 |] (Nd.shape z);
  Alcotest.(check bool) "dtype i32" true (Nd.dtype z = D.I32)

let instantiate_zero () =
  let z =
    { A.z_aval = { T.shape = [| 2 |]; dtype = D.F64; weak_type = false } }
  in
  let nd = concrete_of (A.instantiate z) in
  Alcotest.(check (array int)) "shape" [| 2 |] (Nd.shape nd)

let add_jaxvals_binds_add () =
  C.rules.impl <-
    (fun prim inputs ->
      match (prim, inputs) with
      | T.Add, [ a; b ] -> [ Nd.map2 (Nd.dtype a) ( +. ) a b ]
      | _ -> failwith "unexpected prim");
  let out =
    A.add_jaxvals
      (T.Concrete (scalar D.F64 2.0))
      (T.Concrete (scalar D.F64 5.0))
  in
  Alcotest.(check (float 1e-12)) "2 + 5" 7.0 (read (concrete_of out))

let () =
  Alcotest.run "ad_util"
    [
      ( "zeros_like",
        [
          Alcotest.test_case "aval shape" `Quick zeros_like_aval_shape;
          Alcotest.test_case "value matches" `Quick zeros_like_value_matches;
          Alcotest.test_case "instantiate" `Quick instantiate_zero;
        ] );
      ( "add_jaxvals",
        [ Alcotest.test_case "binds Add" `Quick add_jaxvals_binds_add ] );
    ]
