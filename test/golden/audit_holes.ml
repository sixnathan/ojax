module U = Yojson.Safe.Util
module Cd = Ojax.Custom_derivatives
module AC = Ojax.Numpy.Array_creation
module UF = Ojax.Numpy.Ufuncs
module R = Ojax.Numpy.Reductions
module Ix = Ojax.Numpy.Indexing
module Ad = Ojax.Interpreters.Ad
module Nd = Ojax.Ndarray
module D = Ojax.Dtype

let name = "audit_holes"
let mul = UF.multiply
let ones dt n = AC.ones ~dtype:dt n
let x3 () = Ojax.Numpy.Lax_numpy.arange ~dtype:D.F32 3.0
let y3 () = AC.linspace ~num:3 ~dtype:D.F32 1.0 2.0

let bwd_of kind res g =
  match (res, g) with
  | [ x; y ], [ g ] -> (
      match kind with
      | "ok" -> [ mul g y; mul g x ]
      | "shape" -> [ ones D.F32 [| 2 |]; x ]
      | "scalar" -> [ ones D.F32 [||]; x ]
      | "f64" -> [ ones D.F64 [| 3 |]; x ]
      | "i32" -> [ ones D.I32 [| 3 |]; x ]
      | "c64" -> [ ones D.Complex64 [| 3 |]; x ]
      | "both" -> [ ones D.I32 [| 2 |]; x ]
      | "second" -> [ g; ones D.F32 [| 3; 1 |] ]
      | "count" -> [ g ]
      | _ -> failwith "audit_holes: kind")
  | _ -> failwith "audit_holes: arity"

let grad2 f = Ad.grad (fun xs -> R.sum (List.hd (f xs))) [ x3 (); y3 () ]
let primal = function [ x; y ] -> [ mul x y ] | _ -> failwith "arity"

let vjp_plain kind () =
  let f =
    Cd.custom_vjp primal
      ~fwd:(fun xs -> (primal xs, xs))
      ~bwd:(fun res g -> bwd_of kind res g)
  in
  grad2 f

let vjp_sym kind () =
  let f =
    Cd.custom_vjp_symbolic primal
      ~fwd:(fun ps ->
        let xs = List.map (fun p -> p.Cd.value) ps in
        (primal xs, xs))
      ~bwd:(fun res g ->
        List.map
          (fun v -> Some (Cd.Tangent v))
          (bwd_of kind res (List.map Cd.instantiate_tangent g)))
  in
  grad2 f

let defjvps () =
  Cd.defjvps
    (function [ x; y ] -> [ mul (UF.sin x) y ] | _ -> failwith "arity")
    [
      Some
        (fun g _ ps ->
          match ps with
          | [ x; y ] -> [ mul (mul g (UF.cos x)) y ]
          | _ -> failwith "arity");
      Some
        (fun g _ ps ->
          match ps with
          | [ x; _ ] -> [ mul g (UF.sin x) ]
          | _ -> failwith "arity");
    ]

let lin a b = AC.linspace ~num:4 ~dtype:D.F32 a b
let primals () = [ lin 0.1 1.3; lin 2.0 (-1.0) ]
let tangents () = [ lin 0.5 1.5; lin (-1.0) 3.0 ]

let defjvps_jvp () =
  let o, t = Ad.jvp (defjvps ()) (primals ()) (tangents ()) in
  o @ t

let defjvps_grad () =
  let f = defjvps () in
  Ad.grad (fun xs -> R.sum (List.hd (f xs))) (primals ())

let defjvps_jit_jvp () =
  let f = defjvps () in
  Ojax.Api.jit_flat
    (function
      | [ x; y; tx; ty ] ->
          let o, t = Ad.jvp f [ x; y ] [ tx; ty ] in
          o @ t
      | _ -> failwith "arity")
    (primals () @ tangents ())

let kinds =
  [ "ok"; "shape"; "scalar"; "f64"; "i32"; "c64"; "both"; "second"; "count" ]

let value_cases =
  List.map (fun k -> ("vjp_plain_" ^ k, vjp_plain k)) kinds
  @ List.map (fun k -> ("vjp_sym_" ^ k, vjp_sym k)) kinds
  @ [
      ("defjvps_jvp", defjvps_jvp);
      ("defjvps_grad", defjvps_grad);
      ("defjvps_jit_jvp", defjvps_jit_jvp);
    ]

let put mode = function
  | [ a; i; v ] -> [ Ix.put ?mode ~inplace:false a i v ]
  | _ -> failwith "arity"

let traced =
  List.concat_map
    (fun (tag, mode) ->
      [
        ("put_" ^ tag, put mode);
        ("put_" ^ tag ^ "_1d_int", put mode);
        ("put_" ^ tag ^ "_tiled", put mode);
      ])
    [ ("none", None); ("clip", Some "clip"); ("wrap", Some "wrap") ]

let compress ?axis ?size ?fill_value () = function
  | [ c; x ] -> [ Ojax.Numpy.Extras.compress ?axis ?size ?fill_value c x ]
  | _ -> failwith "arity"

let extract ?size () = function
  | [ c; x ] -> [ Ojax.Numpy.Extras.extract ?size c x ]
  | _ -> failwith "arity"

let traced =
  traced
  @ [
      ("compress_size", compress ~size:4 ~fill_value:(-5.0) ());
      ("compress_size_long_cond", compress ~size:3 ());
      ("compress_size_short_cond", compress ~size:3 ());
      ("compress_size_axis1", compress ~axis:1 ~size:2 ~fill_value:7.0 ());
      ("compress_size_int", compress ~size:2 ~fill_value:(-1.0) ());
      ("extract_size", extract ~size:5 ());
    ]

let aval_of_arg a : Ojax.Types.aval =
  match U.to_list a with
  | [ dt; shape ] ->
      {
        dtype =
          (match D.of_numpy_name (U.to_string dt) with
          | Some d -> d
          | None -> failwith "audit_holes: dtype");
        shape = Array.of_list (List.map U.to_int (U.to_list shape));
        weak_type = false;
      }
  | _ -> failwith "audit_holes: arg"

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
  | Ojax.Types.Tracer _ -> failwith "audit_holes: tracer result"

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
        match List.assoc_opt id traced with
        | None -> Error "no_binding"
        | Some f -> (
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
      | None, Ok c ->
          let text f = try f c with e -> error_detail e in
          log ~name:(id ^ ":jaxpr")
            (Option.value (str_opt row "jaxpr") ~default:"")
            (text Ojax.Pretty_printer.closed_jaxpr_to_jax_string);
          log ~name:(id ^ ":hlo")
            (Option.value (str_opt row "hlo") ~default:"")
            (text (Ojax.Stablehlo.Emit.emit_jax_text ~name:"f")))
  | _ -> (
      let got =
        match List.assoc_opt id value_cases with
        | None -> Error "no_binding"
        | Some f -> (
            try Ok (List.map out_text (f ())) with e -> Error (error_detail e))
      in
      match (expected_error, got) with
      | Some jax, Error ojax -> log jax ojax
      | Some jax, Ok _ -> log jax "returned"
      | None, Error ojax ->
          List.iteri
            (fun i o ->
              log ~name:(Printf.sprintf "%s:out%d" id i) (jax_out o) ojax)
            (U.to_list (U.member "outs" row))
      | None, Ok outs ->
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
