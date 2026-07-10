module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module IDX = Ojax.Numpy.Indexing

let () = Ojax.Lax.install ()
let f32 shape xs = T.Concrete (Nd.of_floats D.F32 shape xs)
let i32 shape xs = T.Concrete (Nd.of_floats D.I32 shape xs)

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
let farr = Alcotest.(array (float 1e-6))

let test_take () =
  let x = f32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let ind = i32 [| 2 |] [| 2.; 0. |] in
  Alcotest.check farr "take flat" [| 3.; 1. |] (read (IDX.take x ind));
  Alcotest.check farr "take axis1" [| 3.; 1.; 6.; 4. |]
    (read (IDX.take ~axis:1 x ind));
  Alcotest.(check (array int))
    "take axis1 shape" [| 2; 2 |]
    (shape_of (IDX.take ~axis:1 x ind));
  let ind0 = i32 [| 2 |] [| 1.; 0. |] in
  Alcotest.check farr "take axis0"
    [| 4.; 5.; 6.; 1.; 2.; 3. |]
    (read (IDX.take ~axis:0 x ind0))

let test_take_along_axis () =
  let x = f32 [| 2; 3 |] [| 1.; 2.; 3.; 4.; 5.; 6. |] in
  let ind = i32 [| 2; 3 |] [| 0.; 2.; 1.; 2.; 0.; 1. |] in
  Alcotest.check farr "take_along_axis axis1"
    [| 1.; 3.; 2.; 6.; 4.; 5. |]
    (read (IDX.take_along_axis ~axis:1 x ind));
  let ind0 = i32 [| 1; 3 |] [| 1.; 0.; 1. |] in
  Alcotest.check farr "take_along_axis axis0 broadcast" [| 4.; 2.; 6. |]
    (read (IDX.take_along_axis ~axis:0 x ind0))

let test_put () =
  let a = f32 [| 5 |] [| 0.; 0.; 0.; 0.; 0. |] in
  let ind = i32 [| 3 |] [| 0.; 2.; 4. |] in
  let v = f32 [| 3 |] [| 10.; 20.; 30. |] in
  Alcotest.check farr "put" [| 10.; 0.; 20.; 0.; 30. |] (read (IDX.put a ind v));
  let a2 = f32 [| 2; 3 |] [| 0.; 0.; 0.; 0.; 0.; 0. |] in
  let ind2 = i32 [| 2 |] [| 0.; 4. |] in
  let v2 = f32 [| 2 |] [| 7.; 9. |] in
  Alcotest.check farr "put 2d"
    [| 7.; 0.; 0.; 0.; 9.; 0. |]
    (read (IDX.put a2 ind2 v2))

let test_put_along_axis () =
  let arr = i32 [| 2; 3 |] [| 10.; 30.; 20.; 60.; 40.; 50. |] in
  let ind = i32 [| 2; 1 |] [| 1.; 0. |] in
  let v = i32 [| 2; 1 |] [| 99.; 99. |] in
  Alcotest.check farr "put_along_axis"
    [| 10.; 99.; 20.; 99.; 40.; 50. |]
    (read (IDX.put_along_axis ~axis:1 arr ind v))

let () =
  Alcotest.run "numpy_indexing"
    [
      ( "indexing",
        [
          Alcotest.test_case "take" `Quick test_take;
          Alcotest.test_case "take_along_axis" `Quick test_take_along_axis;
          Alcotest.test_case "put" `Quick test_put;
          Alcotest.test_case "put_along_axis" `Quick test_put_along_axis;
        ] );
    ]
