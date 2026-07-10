module D = Ojax.Pjrt.Discover

let expect_error name f =
  match f () with
  | _ -> Alcotest.failf "%s: expected Discover.Error but got a value" name
  | exception D.Error _ -> ()

let sha256_empty () =
  Alcotest.(check string)
    "sha256 empty"
    "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    (D.sha256_hex "")

let sha256_abc () =
  Alcotest.(check string)
    "sha256 abc"
    "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    (D.sha256_hex "abc")

let sha256_two_block () =
  Alcotest.(check string)
    "sha256 448-bit message"
    "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    (D.sha256_hex "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq")

let expected_is_hex () =
  Alcotest.(check int) "length" 64 (String.length D.expected_sha256);
  let is_lower_hex c = (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') in
  Alcotest.(check bool)
    "lowercase hex" true
    (String.for_all is_lower_hex D.expected_sha256)

let api_minor () = Alcotest.(check int) "pjrt api minor" 81 D.pjrt_api_minor

let env_var_name () =
  Alcotest.(check string) "env var" "OJAX_PJRT_PLUGIN" D.env_var

let validate_unset () = expect_error "unset" (fun () -> D.validate_path None)

let validate_empty () =
  expect_error "empty" (fun () -> D.validate_path (Some ""))

let validate_relative () =
  expect_error "relative" (fun () ->
      D.validate_path (Some "vendor/pjrt/pjrt_c_api_cpu_plugin.so"))

let validate_absolute () =
  let p = "/opt/ojax/pjrt_c_api_cpu_plugin.so" in
  Alcotest.(check string) "absolute passthrough" p (D.validate_path (Some p))

let verify_missing () =
  expect_error "missing" (fun () -> D.verify_at "/nonexistent/ojax/plugin.so")

let with_temp_file content f =
  let path = Filename.temp_file "ojax_pjrt_" ".bin" in
  let oc = open_out_bin path in
  output_string oc content;
  close_out oc;
  Fun.protect ~finally:(fun () -> Sys.remove path) (fun () -> f path)

let file_matches_hex () =
  with_temp_file "hello ojax pjrt preflight" (fun path ->
      Alcotest.(check string)
        "file hash == string hash"
        (D.sha256_hex "hello ojax pjrt preflight")
        (D.sha256_file path))

let verify_mismatch () =
  with_temp_file "not the plugin" (fun path ->
      expect_error "mismatch" (fun () -> D.verify_at path))

let real_plugin () =
  match Sys.getenv_opt D.env_var with
  | Some path when (not (Filename.is_relative path)) && Sys.file_exists path ->
      Alcotest.(check string)
        "preflight returns resolved path" path (D.preflight ())
  | _ -> ()

let () =
  Alcotest.run "pjrt_discover"
    [
      ( "sha256",
        [
          Alcotest.test_case "empty" `Quick sha256_empty;
          Alcotest.test_case "abc" `Quick sha256_abc;
          Alcotest.test_case "two-block" `Quick sha256_two_block;
          Alcotest.test_case "file matches string" `Quick file_matches_hex;
        ] );
      ( "pins",
        [
          Alcotest.test_case "expected sha is hex" `Quick expected_is_hex;
          Alcotest.test_case "api minor" `Quick api_minor;
          Alcotest.test_case "env var name" `Quick env_var_name;
        ] );
      ( "resolve",
        [
          Alcotest.test_case "unset errors" `Quick validate_unset;
          Alcotest.test_case "empty errors" `Quick validate_empty;
          Alcotest.test_case "relative errors" `Quick validate_relative;
          Alcotest.test_case "absolute ok" `Quick validate_absolute;
        ] );
      ( "preflight",
        [
          Alcotest.test_case "missing errors" `Quick verify_missing;
          Alcotest.test_case "mismatch errors" `Quick verify_mismatch;
          Alcotest.test_case "real plugin" `Quick real_plugin;
        ] );
    ]
