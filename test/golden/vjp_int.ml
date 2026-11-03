module U = Yojson.Safe.Util
module Cd = Ojax.Custom_derivatives
module AC = Ojax.Numpy.Array_creation
module UF = Ojax.Numpy.Ufuncs
module R = Ojax.Numpy.Reductions
module JU = Ojax.Numpy.Jnp_util
module TU = Ojax.Tree_util
module D = Ojax.Dtype

let name = "vjp_int"

let second_dtype = function
  | "i32" -> D.I32
  | "bool" -> D.Bool
  | "u8" -> D.Uint8
  | _ -> failwith "vjp_int: dtype"

let seconds = [ "i32"; "bool"; "u8" ]
let kinds = [ "f32"; "f32_shape"; "same"; "scalar" ]
let lin a b = AC.linspace ~num:3 ~dtype:D.F32 a b
let stored = ref `Null
let floats j = Array.of_list (List.map U.to_number (U.to_list j))

let inputs second =
  match !stored with
  | `Null ->
      [
        lin 1.0 2.0;
        Ojax.Types.Concrete
          (Ojax.Ndarray.of_floats (second_dtype second) [| 3 |]
             [| 1.0; 0.0; 3.0 |]);
      ]
  | j ->
      [
        Ojax.Types.Concrete
          (Ojax.Ndarray.of_floats D.F32 [| 3 |] (floats (U.member "x" j)));
        Ojax.Types.Concrete
          (Ojax.Ndarray.of_floats (second_dtype second) [| 3 |]
             (floats (U.member second (U.member "n" j))));
      ]

let primal = function
  | [ x; n ] ->
      [
        UF.add
          (UF.multiply x (JU.py_float 2.0))
          (R.sum (Ojax.Numpy.Lax_numpy.astype n D.F32));
      ]
  | _ -> failwith "vjp_int: arity"

let bwd_of kind res g =
  match (res, g) with
  | [ _; n ], [ g ] -> (
      let gx = UF.multiply g (JU.py_float 2.0) in
      match kind with
      | "f32" -> [ gx; AC.ones ~dtype:D.F32 [| 3 |] ]
      | "f32_shape" -> [ gx; AC.ones ~dtype:D.F32 [| 7 |] ]
      | "same" -> [ gx; n ]
      | "scalar" ->
          [
            gx;
            Ojax.Types.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 4.0 |]);
          ]
      | _ -> failwith "vjp_int: kind")
  | _ -> failwith "vjp_int: arity"

let make kind sym =
  if sym then
    Cd.custom_vjp_symbolic primal
      ~fwd:(fun ps ->
        let xs = List.map (fun p -> p.Cd.value) ps in
        (primal xs, xs))
      ~bwd:(fun res g ->
        List.map
          (fun v -> Some (Cd.Tangent v))
          (bwd_of kind res (List.map Cd.instantiate_tangent g)))
  else
    Cd.custom_vjp primal
      ~fwd:(fun xs -> (primal xs, xs))
      ~bwd:(fun res g -> bwd_of kind res g)

let pulled f args =
  let tree_f = function
    | [ TU.Leaf x; TU.Leaf n ] -> TU.Leaf (List.hd (f [ x; n ]))
    | _ -> failwith "vjp_int: tree"
  in
  let _, pull = Ojax.Api.vjp tree_f (List.map (fun v -> TU.Leaf v) args) in
  List.concat_map TU.tree_leaves (pull (TU.Leaf (lin 0.5 1.5)))

let out_text v =
  let text = Audit_holes.out_text v in
  match Ojax.Core.get_aval v with
  | { dtype = D.Float0; _ } ->
      "[('float0', 'V')]" ^ String.sub text 6 (String.length text - 6)
  | _ -> text

let tag second kind sym =
  Printf.sprintf "%s_%s_%s" second kind (if sym then "sym" else "plain")

let grad_int f args =
  match
    Ojax.Api.grad ~allow_int:true
      (function
        | [ TU.Tuple [ TU.Leaf x; TU.Leaf n ] ] ->
            TU.Leaf (R.sum (List.hd (f [ x; n ])))
        | _ -> failwith "vjp_int: tree")
      [ TU.Tuple (List.map (fun v -> TU.Leaf v) args) ]
  with
  | tree -> TU.tree_leaves tree

let derived f = function
  | [ x ] ->
      Ojax.Interpreters.Ad.grad
        (fun xs ->
          match xs with
          | [ x ] ->
              R.sum
                (List.hd
                   (f
                      [
                        x;
                        Ojax.Numpy.Lax_numpy.astype
                          (UF.multiply x (JU.py_float 3.0))
                          D.I32;
                      ]))
          | _ -> failwith "vjp_int: arity")
        [ x ]
  | _ -> failwith "vjp_int: arity"

let through_jit f = function
  | [ x ] ->
      let inner =
        Ojax.Api.jit_flat ~name:"<lambda>" (function
          | [ y ] ->
              [
                UF.add
                  (Ojax.Numpy.Lax_numpy.astype
                     (UF.multiply y (JU.py_float 3.0))
                     D.I32)
                  (JU.py_int 1);
              ]
          | _ -> failwith "vjp_int: arity")
      in
      Ojax.Interpreters.Ad.grad
        (fun xs ->
          match xs with
          | [ x ] -> R.sum (List.hd (f [ x; List.hd (inner [ x ]) ]))
          | _ -> failwith "vjp_int: arity")
        [ x ]
  | _ -> failwith "vjp_int: arity"

let value_cases =
  List.concat_map
    (fun second ->
      List.concat_map
        (fun kind ->
          List.concat_map
            (fun sym ->
              let t = tag second kind sym in
              [
                ("vjp_" ^ t, fun () -> pulled (make kind sym) (inputs second));
                ( "jit_vjp_" ^ t,
                  fun () ->
                    Ojax.Api.jit_flat (pulled (make kind sym)) (inputs second)
                );
              ])
            [ false; true ])
        kinds
      @ List.map
          (fun kind ->
            ( Printf.sprintf "grad_int_%s_%s" second kind,
              fun () -> grad_int (make kind false) (inputs second) ))
          [ "f32"; "same" ])
    seconds
  @ List.concat_map
      (fun kind ->
        List.map
          (fun sym ->
            ( Printf.sprintf "grad_derived_%s_%s" kind
                (if sym then "sym" else "plain"),
              fun () -> derived (make kind sym) [ List.hd (inputs "i32") ] ))
          [ false; true ])
      kinds

let traced =
  List.concat_map
    (fun second ->
      List.concat_map
        (fun kind ->
          List.map
            (fun sym ->
              ("trace_vjp_" ^ tag second kind sym, pulled (make kind sym)))
            [ false; true ])
        [ "f32"; "f32_shape" ])
    [ "i32"; "bool" ]
  @ List.concat_map
      (fun kind ->
        List.map
          (fun sym ->
            ( Printf.sprintf "trace_derived_%s_%s" kind
                (if sym then "sym" else "plain"),
              derived (make kind sym) ))
          [ false; true ])
      [ "f32"; "f32_shape" ]
  @ List.concat_map
      (fun kind ->
        List.map
          (fun sym ->
            ( Printf.sprintf "trace_jit_%s_%s" kind
                (if sym then "sym" else "plain"),
              through_jit (make kind sym) ))
          [ false; true ])
      [ "f32"; "f32_shape" ]

let check_row ~suite row () =
  let id = U.to_string (U.member "id" row) in
  let x64 = U.to_bool (U.member "x64" row) in
  Ojax.Config.set Ojax.Config.enable_x64 x64;
  stored := U.member "inputs" row;
  Fun.protect ~finally:(fun () ->
      stored := `Null;
      Ojax.Config.set Ojax.Config.enable_x64 false)
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
    match
      (Audit_holes.str_opt row "error_class", Audit_holes.str_opt row "message")
    with
    | Some c, Some m -> Some (c ^ ": " ^ m)
    | _ -> None
  in
  match U.member "args" row with
  | `List args -> (
      let avals = List.map Audit_holes.aval_of_arg args in
      let closed =
        match List.assoc_opt id traced with
        | None -> Error "no_binding"
        | Some f -> (
            try Ok (Ojax.Jaxpr.make_jaxpr avals f)
            with e -> Error (Audit_holes.error_detail e))
      in
      let field f = Option.value (Audit_holes.str_opt row f) ~default:"" in
      match (expected_error, closed) with
      | Some jax, Error ojax -> log jax ojax
      | Some jax, Ok _ -> log jax "returned"
      | None, Error ojax ->
          log ~name:(id ^ ":jaxpr") (field "jaxpr") ojax;
          log ~name:(id ^ ":hlo") (field "hlo") ojax
      | None, Ok c ->
          let text f = try f c with e -> Audit_holes.error_detail e in
          log ~name:(id ^ ":jaxpr") (field "jaxpr")
            (text Ojax.Pretty_printer.closed_jaxpr_to_jax_string);
          log ~name:(id ^ ":hlo") (field "hlo")
            (text (Ojax.Stablehlo.Emit.emit_jax_text ~name:"f")))
  | _ -> (
      let got =
        match List.assoc_opt id value_cases with
        | None -> Error "no_binding"
        | Some f -> (
            try Ok (List.map out_text (f ()))
            with e -> Error (Audit_holes.error_detail e))
      in
      let jax =
        List.map Audit_holes.jax_out (U.to_list (U.member "outs" row))
      in
      match (expected_error, got) with
      | Some jax, Error ojax -> log jax ojax
      | Some jax, Ok _ -> log jax "returned"
      | None, Error ojax ->
          List.iteri
            (fun i j -> log ~name:(Printf.sprintf "%s:out%d" id i) j ojax)
            jax
      | None, Ok outs ->
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
    let rows = Audit_holes.read_rows (path ~goldens_root) in
    List.map
      (fun mode ->
        let suite = name ^ ":" ^ mode in
        ( suite,
          List.filter_map
            (fun row ->
              if mode = "x64_on" = U.to_bool (U.member "x64" row) then
                Some
                  (Alcotest.test_case
                     (U.to_string (U.member "id" row))
                     `Quick (check_row ~suite row))
              else None)
            rows ))
      [ "x64_off"; "x64_on" ]
