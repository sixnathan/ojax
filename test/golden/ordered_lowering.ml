module U = Yojson.Safe.Util

let check_row ~x64 row () =
  let id = U.to_string (U.member "id" row) in
  let expected = Docs_p2.expected_of row in
  let actual =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
        Docs_p2.run_case
          (Docs_p2.xform_thunk row
             (U.to_string (U.member "case" row))
             "lowered"))
  in
  let suite = "ordered_lowering:" ^ if x64 then "x64_on" else "x64_off" in
  let ok = Docs_p2.same expected actual in
  Docs_p2.record suite id ok;
  if not ok then
    Printf.eprintf "ORDERED_LOWERING DIFFER %s %s\n  jax:  %s\n  ojax: %s\n%!"
      suite id (Docs_p2.show expected) (Docs_p2.show actual)

let suite_for x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat
      (Filename.concat Docs_p2.goldens_root "ordered_lowering")
      (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  ( "ordered_lowering_" ^ tag,
    List.map
      (fun row ->
        Alcotest.test_case
          (U.to_string (U.member "id" row))
          `Quick (check_row ~x64 row))
      rows )
