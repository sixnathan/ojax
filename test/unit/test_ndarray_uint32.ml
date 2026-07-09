module C = Ojax.Core
module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype

let () = Ojax.Lax.install ()
let u32 shape xs = T.Concrete (Nd.of_floats D.Uint32 shape xs)

let out_floats v =
  match v with
  | T.Concrete a ->
      let n = Array.fold_left ( * ) 1 (Nd.shape a) in
      let arr = Array.make n 0.0 in
      let _ =
        Nd.fold
          (fun i x ->
            arr.(i) <- x;
            i + 1)
          0 a
      in
      arr
  | T.Tracer _ -> Alcotest.fail "not concrete"

let bind1 prim a =
  match C.bind prim [ a ] with [ r ] -> r | _ -> Alcotest.fail "arity"

let bind2 prim a b =
  match C.bind prim [ a; b ] with [ r ] -> r | _ -> Alcotest.fail "arity"

let feq = Alcotest.(check (float 0.0))
let flist = Alcotest.(check (list (float 0.0)))
let max_u32 = 4294967295.0

let roundtrip () =
  let a = Nd.of_floats D.Uint32 [| 1 |] [| max_u32 |] in
  feq "get_f reads full-range uint32" max_u32 (Nd.get_f a [| 0 |]);
  Alcotest.(check int64)
    "get_i64 reads full-range uint32" 4294967295L (Nd.get_i64 a [| 0 |])

let wrap_on_store () =
  let a = Nd.of_floats D.Uint32 [| 2 |] [| 4294967296.0; 4294967301.0 |] in
  flist "store masks mod 2^32" [ 0.0; 5.0 ]
    (Array.to_list (out_floats (T.Concrete a)));
  let c = Nd.canonicalize D.Uint32 a in
  flist "canonicalize masks mod 2^32" [ 0.0; 5.0 ]
    (Array.to_list (out_floats (T.Concrete c)))

let arithmetic_wraps () =
  let x = u32 [| 1 |] [| max_u32 |] in
  let one = u32 [| 1 |] [| 1.0 |] in
  flist "add wraps (2^32-1)+1 = 0" [ 0.0 ]
    (Array.to_list (out_floats (bind2 T.Add x one)));
  let a = u32 [| 1 |] [| 1.0 |] and b = u32 [| 1 |] [| 2.0 |] in
  flist "sub wraps 1-2 = 2^32-1" [ max_u32 ]
    (Array.to_list (out_floats (bind2 T.Sub a b)));
  flist "mul wraps (2^32-1)*(2^32-1) = 1" [ 1.0 ]
    (Array.to_list (out_floats (bind2 T.Mul x x)))

let bitwise () =
  let a = u32 [| 1 |] [| 4278255360.0 |] in
  let b = u32 [| 1 |] [| 16711935.0 |] in
  flist "and" [ 0.0 ] (Array.to_list (out_floats (bind2 T.And a b)));
  flist "or" [ max_u32 ] (Array.to_list (out_floats (bind2 T.Or a b)));
  flist "xor" [ max_u32 ] (Array.to_list (out_floats (bind2 T.Xor a b)))

let shifts () =
  let x = u32 [| 1 |] [| max_u32 |] in
  let four = u32 [| 1 |] [| 4.0 |] in
  flist "shift_left masks high bits" [ 4294967280.0 ]
    (Array.to_list (out_floats (bind2 T.Shift_left x four)));
  flist "shift_right_logical is unsigned" [ 268435455.0 ]
    (Array.to_list (out_floats (bind2 T.Shift_right_logical x four)))

let rotl () =
  let x = u32 [| 1 |] [| 305419896.0 |] in
  let n = u32 [| 1 |] [| 8.0 |] in
  let comp = u32 [| 1 |] [| 24.0 |] in
  let left = bind2 T.Shift_left x n in
  let right = bind2 T.Shift_right_logical x comp in
  let rotated = bind2 T.Or left right in
  flist "rotl = shift_left | shift_right_logical" [ 878082066.0 ]
    (Array.to_list (out_floats rotated))

let convert () =
  let neg = T.Concrete (Nd.of_floats D.I32 [| 1 |] [| -1.0 |]) in
  flist "convert int32 -1 -> uint32 = 2^32-1" [ max_u32 ]
    (Array.to_list (out_floats (bind1 (T.Convert_element_type D.Uint32) neg)))

let iota () =
  let r =
    match
      C.bind (T.Iota { dtype = D.Uint32; shape = [| 4 |]; dimension = 0 }) []
    with
    | [ r ] -> r
    | _ -> Alcotest.fail "arity"
  in
  flist "iota uint32" [ 0.0; 1.0; 2.0; 3.0 ] (Array.to_list (out_floats r))

let shape_ops () =
  let a = u32 [| 2; 3 |] [| 1.0; 2.0; 3.0; max_u32; 5.0; 6.0 |] in
  flist "reshape preserves uint32 values"
    [ 1.0; 2.0; 3.0; max_u32; 5.0; 6.0 ]
    (Array.to_list (out_floats (bind1 (T.Reshape [| 6 |]) a)));
  let v = u32 [| 3 |] [| 7.0; max_u32; 9.0 |] in
  flist "broadcast_in_dim preserves uint32 values"
    [ 7.0; max_u32; 9.0; 7.0; max_u32; 9.0 ]
    (Array.to_list
       (out_floats
          (bind1 (T.Broadcast_in_dim { shape = [| 2; 3 |]; dims = [| 1 |] }) v)))

let () =
  Alcotest.run "ndarray_uint32"
    [
      ( "abstraction",
        [
          Alcotest.test_case "full-range round-trip" `Quick roundtrip;
          Alcotest.test_case "store/canonicalize wrap" `Quick wrap_on_store;
        ] );
      ( "lax-uint32",
        [
          Alcotest.test_case "arithmetic wraps mod 2^32" `Quick arithmetic_wraps;
          Alcotest.test_case "bitwise" `Quick bitwise;
          Alcotest.test_case "shifts" `Quick shifts;
          Alcotest.test_case "rotl identity" `Quick rotl;
          Alcotest.test_case "convert into uint32" `Quick convert;
          Alcotest.test_case "iota" `Quick iota;
          Alcotest.test_case "reshape/broadcast" `Quick shape_ops;
        ] );
    ]
