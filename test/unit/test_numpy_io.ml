module T = Ojax.Types
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module IO = Ojax.Numpy.Io
module X = Ojax.Numpy.Extras

let () = Ojax.Lax.install ()

let unhex s =
  String.init
    (String.length s / 2)
    (fun i -> Char.chr (int_of_string ("0x" ^ String.sub s (2 * i) 2)))

let temp suffix = Filename.temp_file "ojax_io" suffix

let write path data =
  let oc = open_out_bin path in
  output_string oc data;
  close_out oc

let read path =
  let ic = open_in_bin path in
  let s = really_input_string ic (in_channel_length ic) in
  close_in ic;
  s

let nd = function T.Concrete n -> n | _ -> Alcotest.fail "expected concrete"

let np_fixture =
  List.to_seq
    [
      ( "f32",
        "934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028322c2032292c207d202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a0000c03f00000040000050c000008040"
      );
      ( "i64",
        "934e554d5059010076007b276465736372273a20273c6938272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028332c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a0100000000000000feffffffffffffff0000000000010000"
      );
      ( "c64",
        "934e554d5059010076007b276465736372273a20273c6338272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028322c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a0000803f0000004000000080000040c0"
      );
      ( "b1",
        "934e554d5059010076007b276465736372273a20277c6231272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028332c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a010001"
      );
      ( "u8",
        "934e554d5059010076007b276465736372273a20277c7531272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028322c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200aff00"
      );
      ( "scalar",
        "934e554d5059010076007b276465736372273a20273c6638272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028292c207d20202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a0000000000000440"
      );
      ( "npz",
        "504b03042d0000000000000021009acaabe7ffffffffffffffff05001400782e6e70790100100088000000000000008800000000000000934e554d5059010076007b276465736372273a20273c6634272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028322c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a0000803f00000040504b03042d000000000000002100aaa1e44affffffffffffffff05001400792e6e70790100100088000000000000008800000000000000934e554d5059010076007b276465736372273a20273c6934272c2027666f727472616e5f6f72646572273a2046616c73652c20277368617065273a2028322c292c207d2020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020202020200a0300000004000000504b01022d032d0000000000000021009acaabe78800000088000000050000000000000000000000800100000000782e6e7079504b01022d032d000000000000002100aaa1e44a88000000880000000500000000000000000000008001bf000000792e6e7079504b05060000000002000200660000007e0100000000"
      );
    ]
  |> Seq.map (fun (name, hex) -> (name, unhex hex))
  |> Hashtbl.of_seq

let load_fixture name =
  let path = temp ".npy" in
  write path (Hashtbl.find np_fixture name);
  nd (IO.load path)

let reads_numpy () =
  let a = load_fixture "f32" in
  Alcotest.(check bool) "dtype" true (Nd.dtype a = D.F32);
  Alcotest.(check (array int)) "shape" [| 2; 2 |] (Nd.shape a);
  Alcotest.(check (float 0.0)) "cell" (-3.25) (Nd.get_f a [| 1; 0 |]);
  let i =
    Ojax.Config.with_value Ojax.Config.enable_x64 true (fun () ->
        load_fixture "i64")
  in
  Alcotest.(check int64) "wide int" 1099511627776L (Nd.get_i64 i [| 2 |]);
  let c = load_fixture "c64" in
  Alcotest.(check (float 0.0)) "imag" (-3.0) (Nd.get_c c [| 1 |]).Complex.im;
  let s = load_fixture "scalar" in
  Alcotest.(check (array int)) "rank 0" [||] (Nd.shape s);
  let b = load_fixture "b1" in
  Alcotest.(check bool) "bool dtype" true (Nd.dtype b = D.Bool);
  let u = load_fixture "u8" in
  Alcotest.(check (float 0.0)) "u8" 255.0 (Nd.get_f u [| 0 |])

let writes_numpy_bytes () =
  Ojax.Config.with_value Ojax.Config.enable_x64 true (fun () ->
      List.iter
        (fun name ->
          let path = temp ".npy" in
          let source = load_fixture name in
          IO.save path (T.Concrete source);
          Alcotest.(check string)
            name
            (Hashtbl.find np_fixture name)
            (read path))
        [ "f32"; "i64"; "c64"; "b1"; "u8"; "scalar" ])

let npz_round_trip () =
  let path = temp ".npz" in
  write path (Hashtbl.find np_fixture "npz");
  let members = IO.load_npz path in
  Alcotest.(check (list string)) "names" [ "x"; "y" ] (List.map fst members);
  let y = nd (List.assoc "y" members) in
  Alcotest.(check (float 0.0)) "numpy member" 4.0 (Nd.get_f y [| 1 |]);
  let mine = temp ".npz" in
  IO.savez mine members;
  let again = IO.load_npz mine in
  let x = nd (List.assoc "x" again) in
  Alcotest.(check (float 0.0)) "own member" 2.0 (Nd.get_f x [| 1 |])

let parsers () =
  let v = nd (IO.fromstring ~dtype:D.F32 "1 2\n3") in
  Alcotest.(check (array int)) "fromstring" [| 3 |] (Nd.shape v);
  let c = nd (IO.fromstring ~sep:"," "1.5,2.5") in
  Alcotest.(check (float 0.0)) "sep" 2.5 (Nd.get_f c [| 1 |]);
  let raw = nd (IO.frombuffer ~dtype:D.Uint8 ~offset:1 "\001\002\003") in
  Alcotest.(check (float 0.0)) "frombuffer" 3.0 (Nd.get_f raw [| 1 |]);
  let it =
    nd (IO.fromiter ~dtype:D.I32 ~count:2 (List.to_seq [ 4.; 5.; 6. ]))
  in
  Alcotest.(check (array int)) "fromiter" [| 2 |] (Nd.shape it)

let printing () =
  let v =
    T.Concrete (Nd.of_floats D.F64 [| 3 |] [| 1.0; 2.5; 3.14159265358979 |])
  in
  Alcotest.(check string) "default" "[1. 2.5 3.1415927]" (X.array_str v);
  let few = X.with_printoptions ~precision:3 (fun () -> X.array_str v) in
  Alcotest.(check string) "precision" "[1. 2.5 3.14]" few;
  Alcotest.(check string) "restored" "[1. 2.5 3.1415927]" (X.array_str v);
  let big =
    T.Concrete (Nd.of_floats D.I32 [| 10 |] (Array.init 10 float_of_int))
  in
  Alcotest.(check string)
    "summarized" "[0 1 ... 8 9]"
    (X.with_printoptions ~threshold:4 ~edgeitems:2 (fun () -> X.array_str big))

let jax_repr () =
  let check name expected v =
    Alcotest.(check string) name expected (X.array_repr v)
  in
  check "aligned" "Array([1.       , 2.5      , 3.1415927], dtype=float64)"
    (T.Concrete (Nd.of_floats D.F64 [| 3 |] [| 1.0; 2.5; 3.14159265358979 |]));
  check "matrix" "Array([[ 1.5, -2. ],\n       [ 3. ,  4. ]], dtype=float32)"
    (T.Concrete (Nd.of_floats D.F32 [| 2; 2 |] [| 1.5; -2.0; 3.0; 4.0 |]));
  check "ints" "Array([ 1, 20], dtype=int32)"
    (T.Concrete (Nd.of_floats D.I32 [| 2 |] [| 1.0; 20.0 |]));
  check "bools" "Array([ True, False], dtype=bool)"
    (T.Concrete (Nd.of_floats D.Bool [| 2 |] [| 1.0; 0.0 |]));
  check "empty" "Array([], shape=(0, 3), dtype=float32)"
    (T.Concrete (Nd.of_floats D.F32 [| 0; 3 |] [||]))

let () =
  Alcotest.run "numpy_io"
    [
      ("repr", [ Alcotest.test_case "jax repr" `Quick jax_repr ]);
      ( "files",
        [
          Alcotest.test_case "reads numpy files" `Quick reads_numpy;
          Alcotest.test_case "writes numpy bytes" `Quick writes_numpy_bytes;
          Alcotest.test_case "npz" `Quick npz_round_trip;
        ] );
      ( "text",
        [
          Alcotest.test_case "parsers" `Quick parsers;
          Alcotest.test_case "printing" `Quick printing;
        ] );
    ]
