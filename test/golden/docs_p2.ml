module U = Yojson.Safe.Util
module T = Ojax.Types
module F = Ojax.Numpy.Ufuncs
module G = Ojax.Numpy.Getitem

let goldens_root =
  match Sys.getenv_opt "OJAX_GOLDENS" with
  | Some d -> d
  | None -> Filename.concat (Filename.concat ".." "..") "goldens"

type result = {
  kind : string;
  cls : string;
  message : string;
  text : string;
  stdout : string;
  warnings : (string * string) list;
}

let leaf v = Ojax.Tree_util.Leaf v

let float_dt () =
  if Ojax.Config.x64_enabled () then Ojax.Dtype.F64 else Ojax.Dtype.F32

let py_float x = Ojax.Core.py_scalar (float_dt ()) x

let py_int x =
  Ojax.Core.py_scalar
    (if Ojax.Config.x64_enabled () then Ojax.Dtype.I64 else Ojax.Dtype.I32)
    x

let run_case thunk =
  let out = Buffer.create 64 in
  let saved = !Ojax.Debugging.print_sink in
  (Ojax.Debugging.print_sink :=
     fun s ->
       Buffer.add_string out s;
       Buffer.add_char out '\n');
  let outcome, warnings =
    Fun.protect
      ~finally:(fun () -> Ojax.Debugging.print_sink := saved)
      (fun () ->
        Ojax.Py_warnings.catch (fun () ->
            match thunk () with t -> Ok t | exception e -> Error e))
  in
  let base =
    {
      kind = "ok";
      cls = "";
      message = "";
      text = "";
      stdout = Buffer.contents out;
      warnings;
    }
  in
  match outcome with
  | Ok text -> { base with text }
  | Error e -> (
      match (Ojax.Py_exn.class_name e, Ojax.Py_exn.message_of e) with
      | Some cls, Some message -> { base with kind = "raise"; cls; message }
      | _ ->
          {
            base with
            kind = "raise";
            cls = "OCaml";
            message = Printexc.to_string e;
          })

let idx k = [ G.Int k ]
let at v k = G.take v (idx k)

let dprint_body name x =
  let p = Ojax.Debugging.print in
  let i = Ojax.Numpy.Lax_numpy.arange ~dtype:Ojax.Dtype.I32 3.0 in
  (match name with
  | "kw_pos" ->
      p ~kwargs:[ ("x", x) ] "x={x} y={}" [ F.multiply x (py_float 2.0) ]
  | "spec_ordered" -> p ~ordered:true "v {:.2f} {}" [ at x 0; x ]
  | "no_placeholders" -> p "hello" [ x ]
  | "indexed" -> p "{1} then {0}" [ x; F.add x (py_float 1.0) ]
  | "int_spec" -> p "n={:d} m={:5d}" [ at i 1; at i 2 ]
  | "escape" -> p "{{literal}} {}" [ x ]
  | "bool_arr" -> p "b={}" [ F.greater x (py_int 1.0) ]
  | "int_arr" -> p "i={}" [ F.multiply i (py_int 7.0) ]
  | "mat" ->
      p
        ~kwargs:
          [
            ( "m",
              F.true_divide
                (Ojax.Numpy.Tensor_contractions.outer x x)
                (py_int 3.0) );
          ]
        "m=\n{m}" []
  | "scalar_default" ->
      p "s={}" [ F.true_divide (Ojax.Numpy.Reductions.sum x) (py_int 7.0) ]
  | "exp_spec" -> p "e={:.3e}" [ F.multiply (at x 2) (py_float 1234.5) ]
  | "align_spec" ->
      p "[{:>8.3f}] [{:<6}]" [ F.true_divide (at x 1) (py_int 3.0); at i 2 ]
  | "plain_text" -> p "just text" []
  | "two_kw" -> p ~kwargs:[ ("a", x); ("b", i) ] "{b} {a} {b}" []
  | "pct_spec" -> p "{:.1%}" [ F.true_divide (at x 1) (py_int 4.0) ]
  | "g_spec" ->
      p "{:g} {:.3g}"
        [
          F.true_divide (at x 2) (py_int 3.0);
          F.multiply (at x 2) (py_float 1e7);
        ]
  | "neg_scalar" -> p "{} {:+.2f}" [ F.negative (at x 1); F.negative (at x 2) ]
  | "weak_scalar" -> p "{}" [ F.add (at x 0) (py_float 0.5) ]
  | "f64_vals" -> p "{}" [ F.true_divide x (py_int 3.0) ]
  | "nan_inf" -> p "{}" [ F.log (F.subtract x (py_int 1.0)) ]
  | "missing_kw" -> p "{y}" [ x ]
  | "missing_pos" -> p "{} {}" [ x ]
  | "spec_on_array" -> p "{:.2f}" [ x ]
  | "mixed_numbering" -> p "{} {0}" [ x; x ]
  | other -> failwith ("docs_p2: unknown dprint case " ^ other));
  F.add x (py_int 1.0)

let one f = function
  | [ Ojax.Tree_util.Leaf a ] -> f a
  | _ -> failwith "docs_p2: arity"

let dprint_thunk name mode () =
  let x = Ojax.Numpy.Lax_numpy.arange ~dtype:(float_dt ()) 3.0 in
  let body = one (fun a -> leaf (dprint_body name a)) in
  match mode with
  | "eager" ->
      ignore (dprint_body name x);
      ""
  | "jit" ->
      ignore (Ojax.Api.jit body [ leaf x ]);
      ""
  | "jaxpr" ->
      Ojax.Pretty_printer.closed_jaxpr_to_jax_string
        (Ojax.Api.make_jaxpr body [ leaf x ])
  | "lowered" ->
      let wrapped = Ojax.Stages.jit ~name:"body" body in
      Ojax.Stages.as_text (wrapped.Ojax.Stages.wrapped_lower [ leaf x ])
  | other -> failwith ("docs_p2: unknown mode " ^ other)

let expected_of row =
  let str k = match U.member k row with `String s -> s | _ -> "" in
  {
    kind = str "kind";
    cls = str "class";
    message = str "message";
    text = str "text";
    stdout = str "stdout";
    warnings =
      (match U.member "warnings" row with
      | `List l ->
          List.map
            (fun w ->
              match U.to_list w with
              | [ c; m ] -> (U.to_string c, U.to_string m)
              | _ -> failwith "docs_p2: warning entry")
            l
      | _ -> []);
  }

let pairs_text j =
  String.concat "\n"
    (List.map
       (fun p ->
         match U.to_list p with
         | [ k; v ] -> U.to_string k ^ "=" ^ Yojson.Safe.to_string v
         | _ -> failwith "docs_p2: pair")
       (U.to_list j))

let f32_range shape div =
  let n = Array.fold_left ( * ) 1 shape in
  let f v = Int32.float_of_bits (Int32.bits_of_float v) in
  T.Concrete
    (Ojax.Ndarray.of_floats Ojax.Dtype.F32 shape
       (Array.init n (fun k -> f (float_of_int k /. div))))

let aot_fn name =
  let x34 = f32_range [| 3; 4 |] 7.0
  and y42 = f32_range [| 4; 2 |] 5.0
  and v = f32_range [| 16 |] 3.0
  and i =
    T.Concrete
      (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [| 6 |]
         (Array.init 6 float_of_int))
  in
  let mm a b = Ojax.Numpy.Tensor_contractions.matmul a b in
  let two f = function
    | [ Ojax.Tree_util.Leaf a; Ojax.Tree_util.Leaf b ] -> f a b
    | _ -> failwith "docs_p2: arity"
  in
  match name with
  | "sin_dot" ->
      ( two (fun a b -> leaf (F.add (mm (F.sin a) b) (py_float 1.0))),
        [ x34; y42 ] )
  | "add" -> (two (fun a b -> leaf (F.add a b)), [ v; v ])
  | "reduce" ->
      (one (fun a -> leaf (Ojax.Numpy.Reductions.sum (F.multiply a a))), [ v ])
  | "exp_log" -> (one (fun a -> leaf (F.log1p (F.exp a))), [ x34 ])
  | "matmul" -> (two (fun a b -> leaf (mm a b)), [ x34; y42 ])
  | "cumsum" -> (one (fun a -> leaf (Ojax.Numpy.Reductions.cumsum a)), [ v ])
  | "where" ->
      ( one (fun a ->
            leaf
              (Ojax.Numpy.Lax_numpy.where_
                 (F.greater a (py_float 2.0))
                 a (F.negative a))),
        [ v ] )
  | "int_mul" ->
      (one (fun a -> leaf (F.add (F.multiply a a) (py_int 3.0))), [ i ])
  | "two_out" ->
      ( two (fun a b ->
            Ojax.Tree_util.Tuple
              [
                leaf (F.add a (py_float 1.0));
                leaf (F.multiply b (py_float 2.0));
              ]),
        [ x34; y42 ] )
  | "tanh_big" -> (one (fun a -> leaf (F.multiply (F.tanh a) (F.cos a))), [ v ])
  | other -> failwith ("docs_p2: unknown aot case " ^ other)

let fmt_float x = Printf.sprintf "%s" (Yojson.Safe.to_string (`Float x))

let aot_text row =
  let name = U.to_string (U.member "case" row) in
  let f, args = aot_fn name in
  let wrapped = Ojax.Stages.jit f in
  let compiled =
    Ojax.Stages.compile (wrapped.Ojax.Stages.wrapped_lower (List.map leaf args))
  in
  match U.to_string (U.member "group" row) with
  | "aot_cost" ->
      String.concat "\n"
        (List.map
           (fun (k, v) -> k ^ "=" ^ fmt_float v)
           (List.sort compare (Ojax.Stages.cost_analysis compiled)))
  | _ ->
      let m = Ojax.Stages.memory_analysis compiled in
      let module E = Ojax.Pjrt.Executable in
      String.concat "\n"
        (List.map
           (fun (k, v) -> k ^ "=" ^ string_of_int v)
           (List.sort compare
              [
                ( "generated_code_size_in_bytes",
                  m.E.generated_code_size_in_bytes );
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

let actual_of row =
  match U.to_string (U.member "group" row) with
  | "dprint" ->
      run_case
        (dprint_thunk
           (U.to_string (U.member "case" row))
           (U.to_string (U.member "mode" row)))
  | "aot_cost" | "aot_mem" -> run_case (fun () -> aot_text row)
  | g -> failwith ("docs_p2: unknown group " ^ g)

let expected_text row =
  match U.to_string (U.member "group" row) with
  | "aot_cost" ->
      { (expected_of row) with text = pairs_text (U.member "cost" row) }
  | "aot_mem" ->
      { (expected_of row) with text = pairs_text (U.member "mem" row) }
  | _ -> expected_of row

let same e a =
  e.kind = a.kind && e.cls = a.cls && e.message = a.message && e.text = a.text
  && e.stdout = a.stdout && e.warnings = a.warnings

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

let show r =
  Printf.sprintf "%s %s %S text=%S stdout=%S" r.kind r.cls r.message r.text
    r.stdout

let check_row ~x64 row () =
  let id = U.to_string (U.member "id" row) in
  let expected = expected_text row in
  let actual =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () -> actual_of row)
  in
  let suite = "docs_p2:" ^ if x64 then "x64_on" else "x64_off" in
  let ok = same expected actual in
  record suite id ok;
  if not ok then
    Printf.eprintf "DOCS_P2 DIFFER %s %s\n  jax:  %s\n  ojax: %s\n%!" suite id
      (show expected) (show actual)

let suite_for x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat (Filename.concat goldens_root "docs_p2") (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  ( "docs_p2_" ^ tag,
    List.map
      (fun row ->
        Alcotest.test_case
          (U.to_string (U.member "id" row))
          `Quick (check_row ~x64 row))
      rows )
