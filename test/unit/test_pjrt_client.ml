module A = Ojax.Pjrt.Abi
module C = Ojax.Pjrt.Client
module B = Ojax.Pjrt.Buffer
module D = Ojax.Pjrt.Discover
module Dt = Ojax.Dtype
module Nd = Ojax.Ndarray

let real_plugin_path () =
  match Sys.getenv_opt D.env_var with
  | Some p when (not (Filename.is_relative p)) && Sys.file_exists p -> Some p
  | _ -> None

let shared =
  lazy
    (match real_plugin_path () with
    | None -> None
    | Some path ->
        let plugin = A.open_plugin path in
        Some (plugin, C.create plugin))

let get_client () =
  match Lazy.force shared with Some (_, c) -> Some c | None -> None

let unravel shape flat =
  let r = Array.length shape in
  let idx = Array.make r 0 in
  let rem = ref flat in
  for i = r - 1 downto 0 do
    idx.(i) <- !rem mod shape.(i);
    rem := !rem / shape.(i)
  done;
  idx

let flat_floats nd =
  let shape = Nd.shape nd in
  let n = Array.fold_left ( * ) 1 shape in
  Array.init n (fun i -> Nd.get_f nd (unravel shape i))

let check_roundtrip client dtype shape floats =
  let nd = Nd.canonicalize dtype (Nd.of_floats dtype shape floats) in
  let buf = B.of_host client nd in
  Fun.protect
    ~finally:(fun () -> B.destroy buf)
    (fun () ->
      Alcotest.(check (array int)) "dimensions" shape (B.dimensions buf);
      Alcotest.(check bool) "element_type" true (B.element_type buf = dtype);
      let back = B.to_host buf in
      Alcotest.(check bool) "dtype" true (Nd.dtype back = dtype);
      Alcotest.(check (array int)) "shape" shape (Nd.shape back);
      Alcotest.(check (array (float 0.)))
        "values" (flat_floats nd) (flat_floats back))

let roundtrip dtype shape floats () =
  match get_client () with
  | None -> ()
  | Some client -> check_roundtrip client dtype shape floats

let f32 = roundtrip Dt.F32 [| 2; 3 |] [| 1.5; -2.25; 3.0; 0.0; 100.5; -0.5 |]
let f64 = roundtrip Dt.F64 [| 2; 3 |] [| 1.5; -2.25; 3.0; 1e300; -1e-12; 42.0 |]

let i32 =
  roundtrip Dt.I32 [| 2; 3 |]
    [| 1.0; -2.0; 3.0; 2147483647.0; -2147483648.0; 0.0 |]

let i64 = roundtrip Dt.I64 [| 2; 3 |] [| 1.0; -2.0; 3.0; 1000000.0; -5.0; 0.0 |]
let bool_ = roundtrip Dt.Bool [| 2; 3 |] [| 1.0; 0.0; 1.0; 1.0; 0.0; 0.0 |]

let uint32 =
  roundtrip Dt.Uint32 [| 2; 3 |]
    [| 0.0; 1.0; 4294967295.0; 2147483648.0; 100.0; 7.0 |]

let scalar = roundtrip Dt.F32 [||] [| 3.14 |]
let vector = roundtrip Dt.I32 [| 5 |] [| 10.0; 20.0; 30.0; 40.0; 50.0 |]

let leak_smoke () =
  match get_client () with
  | None -> ()
  | Some client ->
      let nd = Nd.of_floats Dt.F32 [| 4 |] [| 1.0; 2.0; 3.0; 4.0 |] in
      Gc.full_major ();
      let before = A.maxrss_bytes () in
      for i = 1 to 1000 do
        let b = B.of_host client nd in
        let _ = B.to_host b in
        B.destroy b;
        if i mod 100 = 0 then Gc.full_major ()
      done;
      Gc.full_major ();
      let after = A.maxrss_bytes () in
      let growth = after - before in
      Alcotest.(check bool)
        (Printf.sprintf "rss growth %d bytes bounded" growth)
        true
        (growth < 64 * 1024 * 1024)

let () =
  Alcotest.run "pjrt_client"
    [
      ( "roundtrip",
        [
          Alcotest.test_case "f32" `Quick f32;
          Alcotest.test_case "f64" `Quick f64;
          Alcotest.test_case "i32" `Quick i32;
          Alcotest.test_case "i64" `Quick i64;
          Alcotest.test_case "bool" `Quick bool_;
          Alcotest.test_case "uint32" `Quick uint32;
          Alcotest.test_case "scalar" `Quick scalar;
          Alcotest.test_case "vector" `Quick vector;
        ] );
      ("leak", [ Alcotest.test_case "buffer 1000x" `Slow leak_smoke ]);
    ]
