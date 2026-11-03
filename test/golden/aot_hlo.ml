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
    ~arg_names:(List.mapi (fun i _ -> Printf.sprintf "vals[%d]" i) names)
    (Ojax.Stages.compile { lowered with module_name = "call" })

let record ?(defect = false) suite name same =
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
          (if same then Compare.raw_identical
           else if defect then "jax_defect"
           else "differ")
      with Sys_error _ -> ())

let find_from text sub from =
  let n = String.length text and m = String.length sub in
  let rec go i =
    if i + m > n then None
    else if String.sub text i m = sub then Some i
    else go (i + 1)
  in
  go from

let remove_all ?(closing = false) text sub_start =
  let b = Buffer.create (String.length text) in
  let rec go from =
    match find_from text sub_start from with
    | None ->
        Buffer.add_string b (String.sub text from (String.length text - from))
    | Some i ->
        Buffer.add_string b (String.sub text from (i - from));
        let j = ref (i + String.length sub_start) in
        while !j < String.length text && text.[!j] >= '0' && text.[!j] <= '9' do
          incr j
        done;
        let j =
          if closing && !j < String.length text && text.[!j] = '}' then !j + 1
          else !j
        in
        go j
  in
  go 0;
  Buffer.contents b

let strip_frames text =
  let text =
    match
      (find_from text "\n\nFileNames\n" 0, find_from text "\n\nStackFrames\n" 0)
    with
    | Some i, Some j when j > i ->
        let k = ref (j + String.length "\n\nStackFrames\n") in
        while !k < String.length text && text.[!k] >= '0' && text.[!k] <= '9' do
          match String.index_from_opt text !k '\n' with
          | Some e -> k := e + 1
          | None -> k := String.length text
        done;
        String.sub text 0 i ^ "\n"
        ^ String.sub text (!k + 1) (String.length text - !k - 1)
    | _ -> text
  in
  remove_all
    (remove_all ~closing:true text ", metadata={stack_frame_id=")
    " stack_frame_id="

let noframe_rows :
    (string, (string * string, Yojson.Safe.t) Hashtbl.t) Hashtbl.t =
  Hashtbl.create 2

let noframe_row tag suite case =
  let table =
    match Hashtbl.find_opt noframe_rows tag with
    | Some t -> t
    | None ->
        let t = Hashtbl.create 1024 in
        let path =
          Filename.concat
            (Filename.concat goldens_root "aot_hlo_noframe")
            (tag ^ ".json")
        in
        if Sys.file_exists path then
          List.iter
            (fun r ->
              Hashtbl.replace t
                ( U.to_string (U.member "suite" r),
                  U.to_string (U.member "case_id" r) )
                r)
            (U.to_list (U.member "rows" (Yojson.Safe.from_file path)));
        Hashtbl.replace noframe_rows tag t;
        t
  in
  Hashtbl.find_opt table (suite, case)

let check_row ~x64 row () =
  let suite_name = U.to_string (U.member "suite" row) in
  let case = U.to_string (U.member "case_id" row) in
  let id = suite_name ^ "/" ^ case in
  let tag = if x64 then "x64_on" else "x64_off" in
  let error = U.member "error" row in
  let expected =
    match error with
    | `String cls -> "raise " ^ cls
    | _ -> U.to_string (U.member "hlo" row)
  in
  let actual =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
        try actual_text row with
        | e when error <> `Null ->
            "raise "
            ^ Option.value ~default:(Error_types.ojax_class e)
                (Ojax.Py_exn.class_name e)
        | e -> "raise " ^ Printexc.to_string e)
  in
  let suite = "aot_hlo:" ^ tag in
  let ok = expected = actual in
  record ~defect:((not ok) && strip_frames expected = actual) suite id ok;
  (match noframe_row tag suite_name case with
  | Some r ->
      let noframe =
        match U.member "error" r with
        | `String cls -> "raise " ^ cls
        | _ -> U.to_string (U.member "hlo" r)
      in
      let same = noframe = actual in
      record ("aot_hlo_noframe:" ^ tag) id same;
      if not same then Printf.eprintf "AOT_HLO_NOFRAME DIFFER %s %s\n%!" tag id
  | None -> ());
  if not ok then Printf.eprintf "AOT_HLO DIFFER %s %s\n%!" suite id

let suite_for x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat (Filename.concat goldens_root "aot_hlo") (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  ( "aot_hlo_" ^ tag,
    List.map
      (fun row ->
        Alcotest.test_case
          (U.to_string (U.member "case_id" row))
          `Quick (check_row ~x64 row))
      rows )
