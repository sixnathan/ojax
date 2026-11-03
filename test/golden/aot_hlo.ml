module U = Yojson.Safe.Util
module T = Ojax.Types

let goldens_root =
  match Sys.getenv_opt "OJAX_GOLDENS" with
  | Some d -> d
  | None -> Filename.concat (Filename.concat ".." "..") "goldens"

let aval_of_arg a : T.aval =
  {
    shape = Array.of_list (List.map U.to_int (U.to_list (U.member "shape" a)));
    dtype =
      (match Ojax.Dtype.of_numpy_name (U.to_string (U.member "dtype" a)) with
      | Some d -> d
      | None -> failwith "aot_hlo: unknown dtype");
    weak_type = false;
  }

let actual_text row =
  let args = U.to_list (U.member "args" row) in
  let names = List.map (fun a -> U.to_string (U.member "name" a)) args in
  let avals = List.map aval_of_arg args in
  let op = U.to_string (U.member "op" row) in
  let params = U.member "params" row in
  let lookup values name =
    let rec go ns vs =
      match (ns, vs) with
      | n :: _, v :: _ when n = name -> v
      | _ :: nt, _ :: vt -> go nt vt
      | _ -> failwith ("aot_hlo: missing input " ^ name)
    in
    go names values
  in
  let run =
    match U.to_string (U.member "suite" row) with
    | "coverage" -> Coverage_cases.run
    | "apicov" -> Apicov_cases.run
    | _ -> Parity_cases.run
  in
  let lowered =
    Ojax.Stages.lower (fun values -> run op params (lookup values)) avals
  in
  Ojax.Stages.compiled_as_text
    (Ojax.Stages.compile { lowered with module_name = "call" })

let record suite name same =
  Compare.bump suite same;
  Compare.bump (Compare.raw_prefix ^ suite) same;
  Compare.note_row suite name ~bit:same ~raw:same;
  match Lazy.force Compare.exact_log with
  | None -> ()
  | Some oc -> (
      try
        Printf.fprintf oc "%s\t%d\t%d\t%g\t%s\t%s\n%!" name 1
          (if same then 0 else 1)
          0.0 suite
          (if same then Compare.raw_identical else "differ")
      with Sys_error _ -> ())

let check_row ~x64 row () =
  let id =
    U.to_string (U.member "suite" row)
    ^ "/"
    ^ U.to_string (U.member "case_id" row)
  in
  let expected = U.to_string (U.member "hlo" row) in
  let actual =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
        try actual_text row with e -> "raise " ^ Printexc.to_string e)
  in
  let suite = "aot_hlo:" ^ if x64 then "x64_on" else "x64_off" in
  let ok = expected = actual in
  record suite id ok;
  if not ok then
    Printf.eprintf "AOT_HLO DIFFER %s %s tb0=%b\n%!" suite id
      (U.to_string (U.member "hlo_tb0" row) = actual)

let suite_for x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat (Filename.concat goldens_root "aot_hlo") (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
      |> List.filter (fun r -> U.member "error" r = `Null)
    else []
  in
  ( "aot_hlo_" ^ tag,
    List.map
      (fun row ->
        Alcotest.test_case
          (U.to_string (U.member "case_id" row))
          `Quick (check_row ~x64 row))
      rows )
