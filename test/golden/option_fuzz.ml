module U = Yojson.Safe.Util

let text_suite suite = "text_" ^ suite
let same_message ~jax ~ojax = String.trim jax = String.trim ojax

let log_text ~suite ~row ~jax e =
  match jax with
  | `String jax_text ->
      let ojax = Error_types.message_of e in
      let same = same_message ~jax:jax_text ~ojax in
      Fuzz.log_row
        ~detail:(String.escaped jax_text ^ "\t" ^ String.escaped ojax)
        ~suite:(text_suite suite) (row ^ ":error_text") 0
        (if same then 0 else 1)
        0.0
        (if same then Compare.raw_identical else "error_text")
  | _ -> ()

let check_case ~suite ~set_dir ~x64 ~nd_of_npz ~string_of_dtype ~concrete case
    () =
  Ojax.Config.set Ojax.Config.enable_x64 x64;
  Fun.protect ~finally:(fun () -> Ojax.Config.set Ojax.Config.enable_x64 false)
  @@ fun () ->
  let case_id = U.to_string (U.member "case_id" case) in
  let op = U.to_string (U.member "op" case) in
  let positional = Array.of_list (U.to_list (U.member "positional" case)) in
  let options = U.member "options" case in
  let names =
    List.map
      (fun a -> U.to_string (U.member "name" a))
      (U.to_list (U.member "args" case))
  in
  let b, table =
    Fuzz.bundle (Filename.concat (Filename.concat set_dir "ops") (op ^ ".npz"))
  in
  let input name =
    let m = Hashtbl.find table (case_id ^ "/in_" ^ name) in
    nd_of_npz (Npz.decode b m.Fuzz.start m.Fuzz.descr m.Fuzz.shape)
  in
  let values = List.map (fun n -> Ojax.Types.Concrete (input n)) names in
  let lookup args name =
    let rec go ns vs =
      match (ns, vs) with
      | n :: _, v :: _ when n = name -> v
      | _ :: nt, _ :: vt -> go nt vt
      | _ -> Option_args.no ("input:" ^ name)
    in
    go names args
  in
  let run args =
    let arr = lookup args in
    match Option_manual.run op positional options arr with
    | Some result -> result
    | None ->
        Option_cases.run op positional
          (Option_manual.without_false_flags options)
          arr
  in
  let jax_error key =
    match U.member key case with `String name -> name | _ -> "unknown"
  in
  let mode mode_name call =
    let prefix = case_id ^ ":" ^ mode_name in
    match
      try
        ignore (call ());
        None
      with
      | Option_args.No_binding name -> Some name
      | _ -> None
    with
    | Some name ->
        Fuzz.log_row ~suite (prefix ^ ":no_binding") 0 (-9) 0.0
          ("no_binding:" ^ name)
    | None -> (
        Fuzz.check_mode ~suite ~case_id ~mode:mode_name ~string_of_dtype
          ~concrete
          ~jax_error:(jax_error (mode_name ^ "_error"))
          b table
          (Fuzz.output_names case mode_name)
          call;
        match
          try
            ignore (call ());
            None
          with e -> Some e
        with
        | Some e when U.member (mode_name ^ "_error") case <> `Null ->
            log_text ~suite ~row:prefix
              ~jax:(U.member (mode_name ^ "_error_text") case)
              e
        | _ -> ())
  in
  mode "eager" (fun () -> run values);
  mode "jit" (fun () -> Ojax.Api.jit_flat run values)

let suites ~goldens_root ~nd_of_npz ~string_of_dtype ~concrete =
  List.filter_map
    (fun (set_name, x64) ->
      let set_dir =
        Filename.concat (Filename.concat goldens_root "option_fuzz") set_name
      in
      let path = Filename.concat set_dir "manifest.json" in
      if not (Sys.file_exists path) then None
      else
        let suite = "option_fuzz:" ^ set_name in
        let cases =
          Yojson.Safe.from_file path |> U.member "cases" |> U.to_list
        in
        Some
          ( suite,
            List.map
              (fun case ->
                Alcotest.test_case
                  (U.to_string (U.member "case_id" case))
                  `Quick
                  (check_case ~suite ~set_dir ~x64 ~nd_of_npz ~string_of_dtype
                     ~concrete case))
              cases ))
    [ ("x64_off", false); ("x64_on", true) ]
