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
      | None -> failwith "aot_cost: unknown dtype");
    weak_type = false;
  }

let fmt_float x = Yojson.Safe.to_string (`Float x)

let pairs_text ~mem pairs =
  String.concat "\n"
    (List.map
       (fun p ->
         match U.to_list p with
         | [ k; v ] ->
             U.to_string k ^ "="
             ^
             if mem then string_of_int (U.to_int v)
             else fmt_float (U.to_number v)
         | _ -> failwith "aot_cost: bad pair")
       (U.to_list pairs))

let mem_text compiled =
  let m = Ojax.Stages.memory_analysis compiled in
  let module E = Ojax.Pjrt.Executable in
  String.concat "\n"
    (List.map
       (fun (k, v) -> k ^ "=" ^ string_of_int v)
       (List.sort compare
          [
            ("generated_code_size_in_bytes", m.E.generated_code_size_in_bytes);
            ("argument_size_in_bytes", m.E.argument_size_in_bytes);
            ("output_size_in_bytes", m.E.output_size_in_bytes);
            ("alias_size_in_bytes", m.E.alias_size_in_bytes);
            ("temp_size_in_bytes", m.E.temp_size_in_bytes);
            ( "host_generated_code_size_in_bytes",
              m.E.host_generated_code_size_in_bytes );
            ("host_argument_size_in_bytes", m.E.host_argument_size_in_bytes);
            ("host_output_size_in_bytes", m.E.host_output_size_in_bytes);
            ("host_alias_size_in_bytes", m.E.host_alias_size_in_bytes);
            ("host_temp_size_in_bytes", m.E.host_temp_size_in_bytes);
            ("peak_memory_in_bytes", m.E.peak_memory_in_bytes);
          ]))

let actual_text ~mem row =
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
      | _ -> failwith ("aot_cost: missing input " ^ name)
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
  let compiled = Ojax.Stages.compile lowered in
  if mem then mem_text compiled
  else
    String.concat "\n"
      (List.map
         (fun (k, v) -> k ^ "=" ^ fmt_float v)
         (List.sort compare (Ojax.Stages.cost_analysis compiled)))

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

let check_row ~x64 ~mem row () =
  let id =
    U.to_string (U.member "suite" row)
    ^ "/"
    ^ U.to_string (U.member "case_id" row)
  in
  let expected =
    pairs_text ~mem (U.member (if mem then "mem" else "cost") row)
  in
  let actual =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
        try actual_text ~mem row with e -> "raise " ^ Printexc.to_string e)
  in
  let suite =
    (if mem then "aot_mem:" else "aot_cost:")
    ^ if x64 then "x64_on" else "x64_off"
  in
  let ok = expected = actual in
  record suite id ok;
  if not ok then
    Printf.eprintf "AOT DIFFER %s %s\n  jax:  %S\n  ojax: %S\n%!" suite id
      expected actual

let suite_for x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat (Filename.concat goldens_root "aot_cost") (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
      |> List.filter (fun r -> U.member "error" r = `Null)
    else []
  in
  ( "aot_cost_" ^ tag,
    List.concat_map
      (fun row ->
        let name = U.to_string (U.member "case_id" row) in
        [
          Alcotest.test_case name `Quick (check_row ~x64 ~mem:false row);
          Alcotest.test_case (name ^ ":mem") `Quick
            (check_row ~x64 ~mem:true row);
        ])
      rows )
