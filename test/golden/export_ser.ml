module U = Yojson.Safe.Util
module ES = Ojax.Export_serialization
module T = Ojax.Tree_util
module Nd = Ojax.Ndarray
module D = Ojax.Dtype

let name = "export_ser"
let suite = "export_ser:x64_off"

let of_hex h =
  String.init
    (String.length h / 2)
    (fun i -> Char.chr (int_of_string ("0x" ^ String.sub h (2 * i) 2)))

let to_hex s =
  String.concat ""
    (List.init (String.length s) (fun i ->
         Printf.sprintf "%02x" (Char.code s.[i])))

let input_of j =
  let dt =
    match D.of_numpy_name (U.to_string (U.member "dtype" j)) with
    | Some d -> d
    | None -> failwith "export_ser: dtype"
  in
  let shape =
    Array.of_list (List.map U.to_int (U.to_list (U.member "shape" j)))
  in
  let values =
    Array.of_list (List.map U.to_number (U.to_list (U.member "values" j)))
  in
  Ojax.Types.Concrete (Nd.of_floats dt shape values)

let show_array j =
  Printf.sprintf "%s%s %s"
    (U.to_string (U.member "dtype" j))
    (String.concat ","
       (List.map
          (fun d -> string_of_int (U.to_int d))
          (U.to_list (U.member "shape" j))))
    (U.to_string (U.member "hex" j))

let show_nd nd =
  let raw = Npz.raw_of_nd nd in
  Printf.sprintf "%s%s %s"
    (D.numpy_name (Nd.dtype nd))
    (String.concat "," (Array.to_list (Array.map string_of_int (Nd.shape nd))))
    (to_hex raw.Npz.bytes)

let expected_call c =
  match U.to_string (U.member "kind" c) with
  | "ok" ->
      "ok "
      ^ String.concat " | "
          (List.map show_array (U.to_list (U.member "outputs" c)))
  | _ ->
      Printf.sprintf "raise %s %s"
        (U.to_string (U.member "class" c))
        (U.to_string (U.member "message" c))

let args_of id inputs =
  match (id, inputs) with
  | "dict_in", [ a; b ] ->
      T.Tuple [ T.Dict [ ("a", T.Leaf a); ("b", T.Leaf b) ] ]
  | _ -> T.Tuple (List.map (fun x -> T.Leaf x) inputs)

let actual_call e id c =
  let inputs = List.map input_of (U.to_list (U.member "inputs" c)) in
  match ES.call e (args_of id inputs) with
  | out ->
      "ok "
      ^ String.concat " | "
          (List.map
             (fun v ->
               match Ojax.Backend.to_host_value v with
               | Ojax.Types.Concrete nd -> show_nd nd
               | _ -> "tracer")
             (T.tree_leaves out))
  | exception e -> (
      match (Ojax.Py_exn.class_name e, Ojax.Py_exn.message_of e) with
      | Some cls, Some message -> Printf.sprintf "raise %s %s" cls message
      | _ -> "raise " ^ Printexc.to_string e)

let check id expected actual () =
  let ok = expected = actual in
  Docs_p2.record suite id ok;
  if not ok then
    Printf.eprintf "EXPORT_SER DIFFER %s\n  jax:  %s\n  ojax: %s\n%!" id
      expected actual

let guarded f = try f () with e -> "raise " ^ Printexc.to_string e

module UF = Ojax.Numpy.Ufuncs

let av dtype shape = { Ojax.Types.shape; dtype; weak_type = false }

let flat_case = function
  | "sin_f32_3" ->
      Some ("f_sin", (fun xs -> List.map UF.sin xs), [ av D.F32 [| 3 |] ])
  | "sin_f32_2x4" ->
      Some ("f_sin", (fun xs -> List.map UF.sin xs), [ av D.F32 [| 2; 4 |] ])
  | ("add_mul" | "add_mul_i32") as id ->
      let a =
        if id = "add_mul" then av D.F32 [| 3 |] else av D.I32 [| 2; 2 |]
      in
      Some
        ( "f_add_mul",
          (function
          | [ x; y ] -> [ UF.add x y; UF.multiply x y ]
          | _ -> assert false),
          [ a; a ] )
  | "matmul" ->
      Some
        ( "f_matmul",
          (function
          | [ x; y ] -> [ Ojax.Numpy.Tensor_contractions.matmul x y ]
          | _ -> assert false),
          [ av D.F32 [| 2; 3 |]; av D.F32 [| 3; 4 |] ] )
  | "unused_arg" ->
      Some
        ( "f_unused",
          (function [ _; y ] -> [ UF.exp y ] | _ -> assert false),
          [ av D.F32 [| 2 |]; av D.F32 [| 3 |] ] )
  | _ -> None

let ojax_export id =
  match flat_case id with
  | Some (fun_name, f, avals) ->
      Some
        (Ojax.Export.export_tree ~fun_name
           (fun args ->
             match f (T.tree_leaves args) with
             | [ r ] -> T.Leaf r
             | rs -> T.Tuple (List.map (fun r -> T.Leaf r) rs))
           (T.Tuple (List.map (fun a -> T.Leaf a) avals)))
  | None when id = "dict_in" ->
      let a = av D.F32 [| 2 |] in
      Some
        (Ojax.Export.export_tree ~fun_name:"f_dict"
           (fun args ->
             match args with
             | T.Tuple [ T.Dict [ ("a", T.Leaf x); ("b", T.Leaf y) ] ] ->
                 let s = UF.subtract x y in
                 T.Dict
                   [
                     ("p", T.List [ T.Leaf x; T.Leaf (UF.cos y) ]);
                     ("s", T.Leaf s);
                   ]
             | _ -> failwith "f_dict: args")
           (T.Tuple [ T.Dict [ ("a", T.Leaf a); ("b", T.Leaf a) ] ]))
  | None -> None

let ojax_rows row id =
  match ojax_export id with
  | exception e ->
      let m = "raise " ^ Printexc.to_string e in
      [
        (id ^ ":ojax_text", U.to_string (U.member "text" row), fun () -> m);
        (id ^ ":ojax_fields", U.to_string (U.member "fields" row), fun () -> m);
      ]
  | None -> []
  | Some e ->
      [
        ( id ^ ":ojax_text",
          U.to_string (U.member "text" row),
          fun () -> e.Ojax.Export.mlir_module );
        ( id ^ ":ojax_fields",
          U.to_string (U.member "fields" row),
          fun () ->
            guarded (fun () ->
                ES.describe (ES.deserialize (ES.serialize (ES.of_exported e))))
        );
      ]

let rows_of row =
  let id = U.to_string (U.member "id" row) in
  let bytes = of_hex (U.to_string (U.member "serialized" row)) in
  let e = lazy (ES.deserialize bytes) in
  let base =
    [
      ( id ^ ":fields",
        U.to_string (U.member "fields" row),
        fun () -> guarded (fun () -> ES.describe (Lazy.force e)) );
      ( id ^ ":reserialize",
        to_hex bytes,
        fun () -> guarded (fun () -> to_hex (ES.serialize (Lazy.force e))) );
    ]
  in
  let vjp =
    match U.member "serialized_vjp1" row with
    | `String h ->
        let b = of_hex h in
        [
          ( id ^ ":reserialize_vjp1",
            h,
            fun () ->
              guarded (fun () ->
                  to_hex (ES.serialize ~vjp_order:1 (ES.deserialize b))) );
        ]
    | _ -> []
  in
  let calls =
    List.mapi
      (fun k c ->
        ( Printf.sprintf "%s:call%d" id k,
          expected_call c,
          fun () -> guarded (fun () -> actual_call (Lazy.force e) id c) ))
      (U.to_list (U.member "calls" row))
  in
  base @ vjp @ calls @ ojax_rows row id

let suites ~goldens_root =
  let path = Filename.concat (Filename.concat goldens_root name) "cases.json" in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  [
    ( "export_ser_x64_off",
      List.concat_map
        (fun row ->
          List.map
            (fun (id, expected, actual) ->
              Alcotest.test_case id `Quick (fun () ->
                  check id expected (actual ()) ()))
            (rows_of row))
        rows );
  ]
