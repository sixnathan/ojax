module A = Ojax.Pjrt.Abi
module D = Ojax.Pjrt.Discover
module Dt = Ojax.Dtype

let real_plugin_path () =
  match Sys.getenv_opt D.env_var with
  | Some p when (not (Filename.is_relative p)) && Sys.file_exists p -> Some p
  | _ -> None

let pin_major () = Alcotest.(check int) "api major" 0 A.pjrt_api_major
let pin_minor () = Alcotest.(check int) "api minor" 81 A.pjrt_api_minor

let pin_struct_size () =
  Alcotest.(check int) "struct size" 984 A.pjrt_api_struct_size

let all_dtypes = [ Dt.F32; Dt.F64; Dt.I32; Dt.I64; Dt.Bool; Dt.Uint32 ]

let buffer_type_map () =
  Alcotest.(check int) "F32" 11 (A.buffer_type Dt.F32);
  Alcotest.(check int) "F64" 12 (A.buffer_type Dt.F64);
  Alcotest.(check int) "I32" 4 (A.buffer_type Dt.I32);
  Alcotest.(check int) "I64" 5 (A.buffer_type Dt.I64);
  Alcotest.(check int) "Bool" 1 (A.buffer_type Dt.Bool);
  Alcotest.(check int) "Uint32" 8 (A.buffer_type Dt.Uint32)

let buffer_type_roundtrip () =
  List.iter
    (fun d ->
      match A.dtype_of_buffer_type (A.buffer_type d) with
      | Some d' when d' = d -> ()
      | _ -> Alcotest.failf "roundtrip failed for %s" (Dt.short_name d))
    all_dtypes

let buffer_type_unknown () =
  Alcotest.(check bool)
    "invalid enum is None" true
    (A.dtype_of_buffer_type 0 = None && A.dtype_of_buffer_type 99 = None)

let open_bad_path () =
  match A.open_plugin "/nonexistent/ojax/does_not_exist.so" with
  | _ -> Alcotest.fail "expected Abi.Error on bad path"
  | exception A.Error _ -> ()

let real_version () =
  match real_plugin_path () with
  | None -> ()
  | Some path ->
      let p = A.open_plugin path in
      Fun.protect
        ~finally:(fun () -> A.close p)
        (fun () ->
          let major, minor = A.api_version p in
          Alcotest.(check int) "plugin major" 0 major;
          Alcotest.(check int) "plugin minor" 81 minor;
          Alcotest.(check int) "plugin struct size" 984 (A.struct_size p))

let real_close_idempotent () =
  match real_plugin_path () with
  | None -> ()
  | Some path ->
      let p = A.open_plugin path in
      A.close p;
      A.close p

let leak_smoke () =
  match real_plugin_path () with
  | None -> ()
  | Some path ->
      let keep = A.open_plugin path in
      ignore (A.api_version keep);
      Gc.full_major ();
      let before = A.maxrss_bytes () in
      for i = 1 to 1000 do
        let p = A.open_plugin path in
        ignore (A.api_version p);
        A.close p;
        if i mod 100 = 0 then Gc.full_major ()
      done;
      Gc.full_major ();
      let after = A.maxrss_bytes () in
      A.close keep;
      let growth = after - before in
      Alcotest.(check bool)
        (Printf.sprintf "rss growth %d bytes bounded" growth)
        true
        (growth < 64 * 1024 * 1024)

let () =
  Alcotest.run "pjrt_abi"
    [
      ( "pins",
        [
          Alcotest.test_case "major" `Quick pin_major;
          Alcotest.test_case "minor" `Quick pin_minor;
          Alcotest.test_case "struct size" `Quick pin_struct_size;
        ] );
      ( "buffer_type",
        [
          Alcotest.test_case "map" `Quick buffer_type_map;
          Alcotest.test_case "roundtrip" `Quick buffer_type_roundtrip;
          Alcotest.test_case "unknown" `Quick buffer_type_unknown;
        ] );
      ( "plugin",
        [
          Alcotest.test_case "open bad path errors" `Quick open_bad_path;
          Alcotest.test_case "real version" `Quick real_version;
          Alcotest.test_case "close idempotent" `Quick real_close_idempotent;
          Alcotest.test_case "leak smoke 1000x" `Slow leak_smoke;
        ] );
    ]
