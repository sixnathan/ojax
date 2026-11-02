module U = Yojson.Safe.Util
module T = Ojax.Types
module D = Ojax.Dtype
module C = Ojax.Core
module PP = Ojax.Pretty_printer

let dtype_of_name = function
  | "float32" -> D.F32
  | "float64" -> D.F64
  | "int32" -> D.I32
  | "int64" -> D.I64
  | "bool" -> D.Bool
  | "uint32" -> D.Uint32
  | "complex64" -> D.Complex64
  | "complex128" -> D.Complex128
  | "int8" -> D.I8
  | "int16" -> D.I16
  | "uint8" -> D.Uint8
  | "uint16" -> D.Uint16
  | "uint64" -> D.Uint64
  | "float16" -> D.F16
  | "bfloat16" -> D.Bf16
  | other -> failwith ("text_dump: dtype " ^ other)

let aval_of_arg a : T.aval =
  {
    shape = Array.of_list (List.map U.to_int (U.to_list (U.member "shape" a)));
    dtype = dtype_of_name (U.to_string (U.member "dtype" a));
    weak_type = false;
  }

let guarded f =
  try `Ok (f ()) with
  | Stack_overflow -> `Err "Stack_overflow"
  | Out_of_memory -> `Err "Out_of_memory"
  | e -> (
      match Ojax.Errors.message_of e with
      | Some m -> `Err m
      | None -> `Err (Printexc.to_string e))

let field name = function
  | `Ok s -> [ (name, `String s) ]
  | `Err e -> [ (name ^ "_error", `String e) ]

let record ?result_names oc ~set ~case traced =
  let lowered = guarded traced in
  let render f =
    match lowered with
    | `Err e -> `Err e
    | `Ok closed -> guarded (fun () -> f closed)
  in
  let fields =
    field "jaxpr" (render PP.closed_jaxpr_to_jax_string)
    @ field "structure" (render PP.primitive_structure)
    @ field "text"
        (render (Ojax.Stablehlo.Emit.emit_jax_text ?result_names ~name:"f"))
  in
  Yojson.Safe.to_channel oc
    (`Assoc ([ ("set", `String set); ("case", `String case) ] @ fields));
  output_char oc '\n';
  flush oc

let traced_debug n =
  {
    T.func_src_info = Printf.sprintf "f at %s:%d" __FILE__ __LINE__;
    traced_for = "jit";
    arg_names = Some (List.init n (Printf.sprintf "xs[%d]"));
  }

let parity_case ?(run = Parity_cases.run_cases) oc ~set case =
  let names =
    List.map
      (fun a -> U.to_string (U.member "name" a))
      (U.to_list (U.member "args" case))
  in
  let avals = List.map aval_of_arg (U.to_list (U.member "args" case)) in
  let op = U.to_string (U.member "op" case) in
  let params = U.member "params" case in
  let lookup args name =
    let rec go ns vs =
      match (ns, vs) with
      | n :: _, v :: _ when n = name -> v
      | _ :: nt, _ :: vt -> go nt vt
      | _ -> failwith ("text_dump: missing input " ^ name)
    in
    go names args
  in
  let result_names =
    match op with
    | "ap_indices_sparse" ->
        Some
          (List.init
             (List.length (U.to_list (U.member "dims" params)))
             (Printf.sprintf "result[%d]"))
    | _ -> Parity_cases.result_names op
  in
  record ?result_names oc ~set
    ~case:(U.to_string (U.member "case_id" case))
    (fun () ->
      Ojax.Jaxpr.with_trace_debug
        (traced_debug (List.length avals))
        (fun () ->
          Ojax.Jaxpr.make_jaxpr avals (fun args -> run op params (lookup args))))

let f32 shape : T.aval = { shape; dtype = D.F32; weak_type = false }
let of_dtype dtype shape : T.aval = { shape; dtype; weak_type = false }
let scalar_f32 v = T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| v |])
let one prim args = List.hd (C.bind prim args)

let jaxpr_builders =
  [
    ("sin", [ f32 [| 3 |] ], fun a -> [ one T.Sin a ]);
    ( "sin_mul",
      [ f32 [| 2; 3 |]; f32 [| 2; 3 |] ],
      function
      | [ x; y ] -> [ one T.Mul [ one T.Sin [ x ]; y ] ]
      | _ -> assert false );
    ("chain", [ f32 [| 4 |] ], fun a -> [ one T.Exp [ one T.Neg a ] ]);
    ("reduce", [ f32 [| 2; 3 |] ], fun a -> [ one (T.Reduce_sum [| 0 |]) a ]);
    ( "dot",
      [ f32 [| 2; 3 |]; f32 [| 3; 4 |] ],
      fun a ->
        [
          one
            (T.Dot_general
               {
                 lhs_contract = [| 1 |];
                 rhs_contract = [| 0 |];
                 lhs_batch = [||];
                 rhs_batch = [||];
                 untyped_dot = true;
                 dot_precision = None;
               })
            a;
        ] );
    ("reshape", [ f32 [| 2; 3 |] ], fun a -> [ one (T.Reshape [| 6 |]) a ]);
    ( "broadcast",
      [ f32 [| 3 |] ],
      fun a ->
        [ one (T.Broadcast_in_dim { shape = [| 2; 3 |]; dims = [| 1 |] }) a ] );
    ( "convert",
      [ f32 [| 3 |] ],
      fun a -> [ one (T.Convert_element_type D.I32) a ] );
    ("compare", [ f32 [| 3 |]; f32 [| 3 |] ], fun a -> [ one T.Lt a ]);
    ( "select",
      [ of_dtype D.Bool [| 3 |]; f32 [| 3 |]; f32 [| 3 |] ],
      fun a -> [ one T.Select_n a ] );
    ("lit_mul", [], fun _ -> [ one T.Mul [ scalar_f32 2.0; scalar_f32 3.0 ] ]);
    ( "nested",
      [ f32 [| 3 |]; f32 [| 3 |] ],
      function
      | [ x; y ] ->
          let m = one T.Mul [ x; y ] in
          let d = one T.Sin [ x ] in
          [ one T.Add [ m; d ] ]
      | _ -> assert false );
  ]

let cases_of path = U.to_list (U.member "cases" (Yojson.Safe.from_file path))
let in_shard ~shard ~shards i = shards <= 1 || i mod shards = shard

let nd_of_npz (a : Npz.t) =
  let dt = dtype_of_name a.Npz.dtype in
  match a.Npz.data with
  | Npz.C c -> Ojax.Ndarray.of_complex dt a.Npz.shape c
  | Npz.F f -> Ojax.Ndarray.of_floats dt a.Npz.shape f
  | Npz.I i ->
      Ojax.Ndarray.of_floats dt a.Npz.shape (Array.map Int64.to_float i)

let dump_arrays goldens out =
  let oc = open_out out in
  let set_dir = Filename.concat (Filename.concat goldens "parity") "x64_off" in
  List.iter
    (fun case ->
      let id = U.to_string (U.member "case_id" case) in
      let path =
        Filename.concat (Filename.concat set_dir "inputs") (id ^ ".npz")
      in
      List.iter
        (fun (name, arr) ->
          let row =
            try
              let v = T.Concrete (nd_of_npz arr) in
              [
                ("repr", `String (Ojax.Numpy.Extras.array_repr v));
                ("str", `String (Ojax.Numpy.Extras.array_str_jax v));
              ]
            with e -> [ ("error", `String (Printexc.to_string e)) ]
          in
          Yojson.Safe.to_channel oc
            (`Assoc ([ ("case", `String id); ("arg", `String name) ] @ row));
          output_char oc '\n')
        (Npz.read path))
    (cases_of (Filename.concat set_dir "manifest.json"));
  close_out oc

let dump_fuzz oc ~goldens ~shard ~shards =
  List.iter
    (fun (set, x64) ->
      Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
          let manifest =
            Filename.concat
              (Filename.concat (Filename.concat goldens "fuzz") set)
              "manifest.json"
          in
          List.iteri
            (fun i case ->
              if
                in_shard ~shard ~shards i
                && String.starts_with ~prefix:"shape"
                     (U.to_string (U.member "variant" case))
              then parity_case oc ~set:("fuzz/" ^ set) case)
            (cases_of manifest)))
    [ ("x64_off", false); ("x64_on", true) ]

let dump_fixed oc ~goldens ~shard ~shards =
  List.iter
    (fun (set, x64) ->
      Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
          let manifest =
            Filename.concat
              (Filename.concat (Filename.concat goldens "parity") set)
              "manifest.json"
          in
          List.iteri
            (fun i case ->
              if in_shard ~shard ~shards i then
                parity_case oc ~set:("parity/" ^ set) case)
            (cases_of manifest);
          let coverage =
            Filename.concat
              (Filename.concat (Filename.concat goldens "coverage") set)
              "manifest.json"
          in
          if Sys.file_exists coverage then
            List.iteri
              (fun i case ->
                if in_shard ~shard ~shards i then
                  parity_case ~run:Coverage_cases.run oc
                    ~set:("coverage/" ^ set) case)
              (cases_of coverage);
          let apicov =
            Filename.concat
              (Filename.concat (Filename.concat goldens "apicov") set)
              "manifest.json"
          in
          if Sys.file_exists apicov then
            List.iteri
              (fun i case ->
                if in_shard ~shard ~shards i then
                  parity_case ~run:Apicov_cases.run oc ~set:("apicov/" ^ set)
                    case)
              (cases_of apicov);
          if shard = 0 then
            List.iter
              (fun (case, avals, f) ->
                record oc ~set:("jaxpr/" ^ set) ~case (fun () ->
                    Ojax.Jaxpr.make_jaxpr avals f))
              jaxpr_builders))
    [ ("x64_off", false); ("x64_on", true) ]

let load_rows path =
  let table = Hashtbl.create 1024 in
  In_channel.with_open_text path (fun ic ->
      let rec go () =
        match In_channel.input_line ic with
        | None -> ()
        | Some line ->
            let row = Yojson.Safe.from_string line in
            Hashtbl.replace table
              ( U.to_string (U.member "set" row),
                U.to_string (U.member "case" row) )
              row;
            go ()
      in
      go ());
  table

let verdict reference ours field =
  match (U.member field reference, Option.map (U.member field) ours) with
  | _, None -> "-"
  | `Null, Some `Null -> "b"
  | a, Some b when a <> `Null && a = b -> "i"
  | _ -> "-"

let error_verdict reference ours field =
  match (U.member field reference, Option.map (U.member field) ours) with
  | `Null, _ -> "n"
  | a, Some b when a = b -> "i"
  | _ -> "-"

let check_text_baseline ?(fields = [ ("text", verdict); ("jaxpr", verdict) ])
    ~path ~references ~ours () =
  let failures = ref [] in
  In_channel.with_open_text path (fun ic ->
      let rec go () =
        match In_channel.input_line ic with
        | None -> ()
        | Some line ->
            (match String.split_on_char '\t' line with
            | set :: case :: wanted when List.length wanted = List.length fields
              ->
                let got = Hashtbl.find_opt ours (set, case) in
                let check (field, verdict) wanted =
                  if wanted <> "-" && wanted <> "n" then
                    let have =
                      match
                        List.find_map
                          (fun table -> Hashtbl.find_opt table (set, case))
                          references
                      with
                      | Some reference -> verdict reference got field
                      | None -> "-"
                    in
                    if have <> wanted then
                      failures :=
                        Printf.sprintf "%s %s %s: was %s, now %s" set case field
                          wanted have
                        :: !failures
                in
                List.iter2 check fields wanted
            | _ -> failwith ("text ratchet: bad line in " ^ path));
            go ()
      in
      go ());
  List.rev !failures

let ratchet goldens baseline_dir =
  let path name = Filename.concat goldens name in
  let reference_files =
    [
      path "text/reference.jsonl";
      path "coverage/text_ref.jsonl";
      path "apicov/text_ref.jsonl";
      path "fuzz/text/reference.jsonl";
    ]
  in
  List.iter
    (fun file ->
      if not (Sys.file_exists file) then
        failwith ("text ratchet: reference " ^ file ^ " is missing"))
    reference_files;
  let out = Filename.temp_file "text_ratchet" ".jsonl" in
  let oc = open_out out in
  dump_fixed oc ~goldens ~shard:0 ~shards:1;
  dump_fuzz oc ~goldens ~shard:0 ~shards:1;
  close_out oc;
  let ours = load_rows out in
  Sys.remove out;
  let references = List.map load_rows reference_files in
  let failures =
    List.concat_map
      (fun (name, fields) ->
        let file = Filename.concat baseline_dir (name ^ ".tsv") in
        if Sys.file_exists file then
          check_text_baseline ?fields ~path:file ~references ~ours ()
        else [ "no baseline " ^ file ])
      [
        ("text_parity", None);
        ("text_coverage", None);
        ("text_apicov", None);
        ("text_fuzz", None);
        ( "text_errors",
          Some
            [
              ("text_error", error_verdict);
              ("jaxpr_error", error_verdict);
              ("structure_error", error_verdict);
            ] );
        ("text_structure", Some [ ("structure", verdict) ]);
      ]
  in
  if failures <> [] then begin
    List.iteri
      (fun i line -> if i < 40 then prerr_endline ("TEXT RATCHET " ^ line))
      failures;
    Printf.eprintf "text ratchet: %d rows fell\n" (List.length failures);
    exit 1
  end

let () =
  Ojax.Lax.install ();
  Ojax.Random.Prng.install ();
  Ojax.Random.Threefry.install ();
  let goldens = Sys.argv.(1) in
  let out = Sys.argv.(2) in
  if Sys.argv.(3) = "arrays" then begin
    dump_arrays goldens out;
    exit 0
  end;
  if Sys.argv.(3) = "ratchet" then begin
    ratchet goldens Sys.argv.(4);
    exit 0
  end;
  let fuzz = Array.length Sys.argv > 5 && Sys.argv.(5) = "fuzz" in
  let shard = int_of_string Sys.argv.(3) in
  let shards = int_of_string Sys.argv.(4) in
  let oc = open_out out in
  if fuzz then dump_fuzz oc ~goldens ~shard ~shards
  else dump_fixed oc ~goldens ~shard ~shards;
  close_out oc
