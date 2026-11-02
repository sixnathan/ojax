open Ojax
open Ojax.Types
module P = Random.Prng
module TF = Random.Threefry

let () = Unix.putenv "OJAX_BACKEND" "interpreter"

let u32 shape values =
  Concrete (Ndarray.of_floats Dtype.Uint32 shape (Array.map float_of_int values))

let nd = function
  | Concrete a -> a
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ -> Alcotest.fail "expected concrete"

let flat v =
  let a = nd v in
  let shape = Ndarray.shape a in
  let n = Array.fold_left ( * ) 1 shape in
  let rank = Array.length shape in
  List.init n (fun f ->
      let idx = Array.make rank 0 in
      let rest = ref f in
      for d = rank - 1 downto 0 do
        idx.(d) <- !rest mod shape.(d);
        rest := !rest / shape.(d)
      done;
      Printf.sprintf "%Lu" (Ndarray.get_i64 a idx))

let check name expected_shape expected v =
  Alcotest.(check (array int))
    (name ^ " shape") expected_shape
    (Ndarray.shape (nd v));
  Alcotest.(check (list string))
    name
    (List.map string_of_int expected)
    (flat v)

let key = u32 [| 2 |] [| 0; 42 |]

let host_is_running () =
  Alcotest.(check bool) "interpreter" false (Backend.multi_device_available ())

let split () =
  check "split" [| 3; 2 |]
    [ 1832780943; 270669613; 64467757; 2916123636; 2465931498; 255383827 ]
    (P.random_split key [| 3 |]);
  check "split nested" [| 2; 2; 2 |]
    [
      1832780943;
      270669613;
      64467757;
      2916123636;
      2465931498;
      255383827;
      3134548294;
      894150801;
    ]
    (P.random_split key [| 2; 2 |])

let fold_in () =
  check "fold_in" [| 2 |] [ 2547012911; 1371500959 ]
    (P.random_fold_in key
       (Concrete (Ndarray.of_floats Dtype.Uint32 [||] [| 7.0 |])))

let bits () =
  check "bits 32" [| 5 |]
    [ 2098992034; 2919706841; 2646866425; 2409546199; 1935504149 ]
    (P.random_bits key ~bit_width:32 ~shape:[| 5 |]);
  check "bits 16" [| 4 |] [ 5026; 12505; 63993; 49623 ]
    (P.random_bits key ~bit_width:16 ~shape:[| 4 |]);
  check "bits 8" [| 4 |] [ 162; 217; 249; 215 ]
    (P.random_bits key ~bit_width:8 ~shape:[| 4 |])

let bits_64 () =
  let v = P.random_bits key ~bit_width:64 ~shape:[| 3 |] in
  Alcotest.(check (list string))
    "bits 64"
    [ "7871734211187709741"; "276886910877598708"; "10591095138341673235" ]
    (flat v)

let hash () =
  check "2x32" [| 4 |]
    [ 2587201270; 994767269; 5467140; 774776150 ]
    (TF.threefry_2x32
       (u32 [| 2 |] [| 0x13198a2e; 0x03707344 |])
       (u32 [| 4 |] [| 0x243f6a88; 0x85a308d3; 1; 2 |]));
  check "2x32 odd" [| 3 |]
    [ 3679230171; 843694964; 267815257 ]
    (TF.threefry_2x32 key (u32 [| 3 |] [| 1; 2; 3 |]))

let seed () =
  check "seed" [| 2 |] [ 0; 42 ]
    (P.random_seed (Concrete (Ndarray.of_floats Dtype.I32 [||] [| 42.0 |])))

let () =
  Alcotest.run "threefry_host"
    [
      ( "reference vectors from JAX 0.10.2",
        [
          Alcotest.test_case "host path" `Quick host_is_running;
          Alcotest.test_case "seed" `Quick seed;
          Alcotest.test_case "split" `Quick split;
          Alcotest.test_case "fold_in" `Quick fold_in;
          Alcotest.test_case "bits" `Quick bits;
          Alcotest.test_case "bits 64" `Quick bits_64;
          Alcotest.test_case "threefry_2x32" `Quick hash;
        ] );
    ]
