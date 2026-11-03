module U = Yojson.Safe.Util
module UF = Ojax.Numpy.Ufuncs
module R = Ojax.Numpy.Reductions
module G = Ojax.Numpy.Getitem
module Nd = Ojax.Ndarray
module D = Ojax.Dtype
module CK = Ojax.Checkify

let name = "checkify"
let arr dt xs = Ojax.Types.Concrete (Nd.of_floats dt [| 3 |] xs)
let pi = Ojax.Numpy.Jnp_util.py_int
let at x i = G.take x [ G.Int i ]

let pos = function
  | [ x ] ->
      let p = UF.greater (at x 0) (pi 0) in
      let a = at x 0 in
      CK.check p "x[0] must be positive, got {}" [ a ];
      [ UF.sin x ]
  | _ -> failwith "arity"

let two = function
  | [ x ] ->
      let p = UF.greater (at x 0) (pi 0) in
      let a = at x 0 in
      CK.check p "first {}" [ a ];
      let y = UF.multiply x (pi 2) in
      let p = UF.less (at x 1) (pi 3) in
      let a = at x 1 in
      let b = at x 2 in
      CK.check p "second {} and {}" [ a; b ];
      let p = UF.less (at x 2) (pi 3) in
      let a = at x 2 in
      CK.check p "third {}" [ a ];
      [ UF.add y (pi 1) ]
  | _ -> failwith "arity"

let kw = function
  | [ x ] ->
      let p = R.all (UF.greater x (pi 0)) in
      let t = R.sum x in
      CK.check
        ~kwargs:[ ("x", x); ("t", t) ]
        p "{x} needs to be positive, total {t}" [];
      [ UF.multiply x x ]
  | _ -> failwith "arity"

let plain = function
  | [ x ] ->
      let p = UF.greater (at x 0) (pi 0) in
      CK.check p "no args" [];
      [ UF.cos x ]
  | _ -> failwith "arity"

let nonscalar = function
  | [ x ] ->
      CK.check (UF.greater x (pi 0)) "bad" [];
      [ x ]
  | _ -> failwith "arity"

let funs = [ ("pos", pos); ("two", two); ("kw", kw); ("plain", plain) ]

let logdiv = function
  | [ x ] -> [ UF.true_divide (UF.log x) (at x 1) ]
  | _ -> failwith "arity"

let sqrtsum = function
  | [ x ] ->
      let y = UF.sqrt x in
      let p = UF.greater (at y 1) (pi 1) in
      let a = at y 1 in
      CK.check p "y[1] small {}" [ a ];
      [ UF.multiply (R.sum y) x ]
  | _ -> failwith "arity"

let intdiv = function
  | [ a; b ] -> [ UF.floor_divide a b ]
  | _ -> failwith "arity"

let intrem = function [ a; b ] -> [ UF.remainder a b ] | _ -> failwith "arity"
let nan_funs = [ ("logdiv", logdiv); ("sqrtsum", sqrtsum) ]
let int_funs = [ ("intdiv", intdiv); ("intrem", intrem) ]

let errsets =
  [
    ("user", CK.user_checks);
    ("nan", CK.nan_checks);
    ("div", CK.div_checks);
    ("float", CK.float_checks);
    ("all", CK.all_checks);
  ]

let ints xs = Ojax.Types.Concrete (Nd.of_floats D.I32 [| 3 |] xs)

let inputs dt =
  [
    ("ok", arr dt [| 1.0; 2.0; 3.0 |]);
    ("neg", arr dt [| -1.0; 2.0; 3.0 |]);
    ("big", arr dt [| 1.0; 4.0; 5.0 |]);
    ("neg_big", arr dt [| -2.0; 4.0; 2.0 |]);
  ]

let jit_checkify f x =
  let shape = ref CK.init_error in
  let leaves =
    Ojax.Api.jit_flat
      (fun xs ->
        let e, o = CK.checkify f xs in
        shape := e;
        CK.error_leaves e @ o)
      [ x ]
  in
  let n = List.length (CK.error_leaves !shape) in
  ( CK.with_leaves !shape (List.filteri (fun i _ -> i < n) leaves),
    List.filteri (fun i _ -> i >= n) leaves )

let value_cases dt =
  let xs = inputs dt in
  List.concat_map
    (fun (fname, f) ->
      List.concat_map
        (fun (xname, x) ->
          [
            ( Printf.sprintf "%s_%s" fname xname,
              `Err (fun () -> CK.checkify f [ x ]) );
            ( Printf.sprintf "jit_%s_%s" fname xname,
              `Err (fun () -> jit_checkify f x) );
            ( Printf.sprintf "throw_%s_%s" fname xname,
              `Err (fun () -> CK.checkify f [ x ]) );
          ])
        xs
      @ [
          ( "eager_" ^ fname ^ "_neg",
            `Plain (fun () -> f [ List.assoc "neg" xs ]) );
          ("eager_" ^ fname ^ "_ok", `Plain (fun () -> f [ List.assoc "ok" xs ]));
          ( "nan_errors_" ^ fname ^ "_neg",
            `Err
              (fun () ->
                CK.checkify ~errors:CK.nan_checks f [ List.assoc "neg" xs ]) );
          ( "all_errors_" ^ fname ^ "_neg",
            `Err
              (fun () ->
                CK.checkify ~errors:CK.all_checks f [ List.assoc "neg" xs ]) );
        ])
    funs
  @ [
      ( "nonscalar",
        `Err (fun () -> CK.checkify nonscalar [ List.assoc "ok" xs ]) );
    ]
  @ List.concat_map
      (fun (fname, f) ->
        List.concat_map
          (fun (xname, x) ->
            List.map
              (fun (ename, es) ->
                ( Printf.sprintf "%s_%s_%s" fname xname ename,
                  `Err (fun () -> CK.checkify ~errors:es f [ x ]) ))
              errsets)
          [
            ("pos", arr dt [| 1.0; 4.0; 9.0 |]);
            ("negnan", arr dt [| -1.0; 4.0; 0.0 |]);
            ("zero", arr dt [| 1.0; 0.0; 2.0 |]);
          ])
      nan_funs
  @ List.concat_map
      (fun (fname, f) ->
        List.concat_map
          (fun (bname, b) ->
            let ia = ints [| 7.0; -7.0; 9.0 |] in
            List.map
              (fun (ename, es) ->
                ( Printf.sprintf "%s_%s_%s" fname bname ename,
                  `Err (fun () -> CK.checkify ~errors:es f [ ia; b ]) ))
              errsets
            @ [
                ( Printf.sprintf "throw_%s_%s" fname bname,
                  `Err (fun () -> CK.checkify ~errors:CK.all_checks f [ ia; b ])
                );
              ])
          [ ("nz", ints [| 2.0; 3.0; -4.0 |]); ("z", ints [| 2.0; 0.0; 5.0 |]) ])
      int_funs

let value_cases id =
  let tag dname dt =
    List.map (fun (n, f) -> (n ^ "_" ^ dname, f)) (value_cases dt)
  in
  match List.assoc_opt id (tag "f32" D.F32) with
  | Some f -> Some f
  | None -> List.assoc_opt id (tag "f64" D.F64)

let last_paths = ref []

let traced_list =
  List.concat_map
    (fun (fname, f) ->
      [
        ( "jaxpr_checkify_" ^ fname,
          fun xs ->
            let e, o = CK.checkify f xs in
            last_paths := CK.result_paths e;
            CK.error_leaves e @ o );
        ("jaxpr_raw_" ^ fname, f);
      ])
    funs
  @ List.concat_map
      (fun (fname, f) ->
        List.map
          (fun (ename, es) ->
            ( Printf.sprintf "jaxpr_%s_%s" fname ename,
              fun xs ->
                let e, o = CK.checkify ~errors:es f xs in
                last_paths := CK.result_paths e;
                CK.error_leaves e @ o ))
          errsets)
      (nan_funs @ int_funs)

let traced id =
  let strip s =
    let n = String.length s in
    if n > 4 then String.sub s 0 (n - 4) else s
  in
  List.assoc_opt (strip id) traced_list

let aval_of_arg a : Ojax.Types.aval =
  match U.to_list a with
  | [ dt; shape ] ->
      {
        dtype =
          (match D.of_numpy_name (U.to_string dt) with
          | Some d -> d
          | None -> failwith "checkify: dtype");
        shape = Array.of_list (List.map U.to_int (U.to_list shape));
        weak_type = false;
      }
  | _ -> failwith "checkify: arg"

let read_rows path =
  let ic = open_in path in
  let rec loop acc =
    match input_line ic with
    | line -> loop (Yojson.Safe.from_string line :: acc)
    | exception End_of_file ->
        close_in ic;
        List.rev acc
  in
  loop []

let str_opt row field =
  match U.member field row with `String s -> Some s | _ -> None

let host = function
  | Ojax.Types.Concrete nd -> nd
  | Ojax.Types.Device b -> Ojax.Pjrt.Buffer.to_host b
  | Ojax.Types.Tracer _ -> failwith "checkify: tracer result"

let hex s =
  String.concat ""
    (List.init (String.length s) (fun i ->
         Printf.sprintf "%02x" (Char.code s.[i])))

let error_detail e =
  match Ojax.Py_exn.class_name e with
  | Some c -> c ^ ": " ^ Error_types.message_of e
  | None -> Error_types.ojax_class e ^ ": " ^ Error_types.message_of e

let out_text v =
  let raw = Npz.raw_of_nd (host v) in
  raw.Npz.raw_dtype ^ "["
  ^ String.concat ","
      (Array.to_list (Array.map string_of_int (Nd.shape (host v))))
  ^ "]" ^ hex raw.Npz.bytes

let jax_out o =
  U.to_string (U.member "dtype" o)
  ^ "["
  ^ String.concat ","
      (List.map
         (fun s -> string_of_int (U.to_int s))
         (U.to_list (U.member "shape" o)))
  ^ "]"
  ^ U.to_string (U.member "hex" o)

let check_row ~suite row () =
  let id = U.to_string (U.member "id" row) in
  let x64 = U.to_bool (U.member "x64" row) in
  Ojax.Config.set Ojax.Config.enable_x64 x64;
  Fun.protect ~finally:(fun () -> Ojax.Config.set Ojax.Config.enable_x64 false)
  @@ fun () ->
  let log ?(name = id) jax ojax =
    let same = String.equal jax ojax in
    Fuzz.log_row
      ~detail:(String.escaped jax ^ "\t" ^ String.escaped ojax)
      ~suite name 1
      (if same then 0 else 1)
      0.0
      (if same then Compare.raw_identical else "differ")
  in
  let expected_error =
    match (str_opt row "error_class", str_opt row "message") with
    | Some c, Some m -> Some (c ^ ": " ^ m)
    | _ -> None
  in
  match U.member "args" row with
  | `List args -> (
      let avals = List.map aval_of_arg args in
      let closed =
        match traced id with
        | None -> Error "no_binding"
        | Some f -> (
            CK.reset_codes ();
            try Ok (Ojax.Jaxpr.make_jaxpr avals f)
            with e -> Error (error_detail e))
      in
      match (expected_error, closed) with
      | Some jax, Error ojax -> log jax ojax
      | Some jax, Ok _ -> log jax "returned"
      | None, Error ojax ->
          log ~name:(id ^ ":jaxpr")
            (Option.value (str_opt row "jaxpr") ~default:"")
            ojax;
          log ~name:(id ^ ":hlo")
            (Option.value (str_opt row "hlo") ~default:"")
            ojax
      | None, Ok c -> (
          let text f = try f c with e -> error_detail e in
          log ~name:(id ^ ":jaxpr")
            (Option.value (str_opt row "jaxpr") ~default:"")
            (text Ojax.Pretty_printer.closed_jaxpr_to_jax_string);
          match str_opt row "hlo" with
          | Some h ->
              log ~name:(id ^ ":hlo") h
                (text
                   (Ojax.Stablehlo.Emit.emit_jax_text
                      ~result_names:
                        (List.map
                           (fun p -> "result" ^ p)
                           (!last_paths @ [ "[1]" ]))
                      ~name:"_lambda"))
          | None -> ()))
  | _ -> (
      let mode = Option.value (str_opt row "mode") ~default:"get" in
      let got =
        match value_cases id with
        | None -> Error "no_binding"
        | Some f -> (
            CK.reset_codes ();
            try
              match f with
              | `Plain g -> Ok (List.map out_text (g ()), None)
              | `Err g ->
                  let e, o = g () in
                  let extra =
                    if mode = "throw" then
                      match CK.throw e with
                      | () -> "None"
                      | exception ex -> error_detail ex
                    else Option.value (CK.get e) ~default:"None"
                  in
                  Ok (List.map out_text (CK.error_leaves e @ o), Some extra)
            with e -> Error (error_detail e))
      in
      match (expected_error, got) with
      | Some jax, Error ojax -> log jax ojax
      | Some jax, Ok _ -> log jax "returned"
      | None, Error ojax ->
          List.iteri
            (fun i o ->
              log ~name:(Printf.sprintf "%s:out%d" id i) (jax_out o) ojax)
            (U.to_list (U.member "outs" row))
      | None, Ok (outs, extra) ->
          (match extra with
          | Some x ->
              let field = if mode = "throw" then "thrown" else "get" in
              log
                ~name:(id ^ ":" ^ field)
                (Option.value (str_opt row field) ~default:"None")
                x
          | None -> ());
          let jax = List.map jax_out (U.to_list (U.member "outs" row)) in
          if List.compare_lengths jax outs <> 0 then
            log ~name:(id ^ ":count")
              (string_of_int (List.length jax))
              (string_of_int (List.length outs))
          else
            List.iteri
              (fun i (j, o) -> log ~name:(Printf.sprintf "%s:out%d" id i) j o)
              (List.combine jax outs))

let path ~goldens_root =
  Filename.concat (Filename.concat goldens_root name) "cases.jsonl"

let suites ~goldens_root =
  if not (Sys.file_exists (path ~goldens_root)) then []
  else
    let rows = read_rows (path ~goldens_root) in
    List.map
      (fun mode ->
        let suite = name ^ ":" ^ mode in
        ( suite,
          List.filter_map
            (fun row ->
              let x64 = U.to_bool (U.member "x64" row) in
              if mode = "x64_on" = x64 then
                Some
                  (Alcotest.test_case
                     (U.to_string (U.member "id" row))
                     `Quick (check_row ~suite row))
              else None)
            rows ))
      [ "x64_off"; "x64_on" ]
