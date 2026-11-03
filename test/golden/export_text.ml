module U = Yojson.Safe.Util
module T = Ojax.Tree_util
module L = Ojax.Lax_api
module JU = Ojax.Numpy.Jnp_util

let name = "export_text"

let maxpool x =
  L.reduce_window x
    (JU.py_float Float.neg_infinity)
    L.max ~window_dimensions:[| 2; 2 |] ~window_strides:[| 1; 1 |]
    [| (0, 0); (0, 0) |]

let one f = function [ x ] -> f x | _ -> failwith "export_text: arity"
let two f = function [ x; y ] -> f x y | _ -> failwith "export_text: arity"

let fn = function
  | "conv_same" ->
      Some (two (fun x w -> L.conv x w ~window_strides:[| 1; 1 |] L.Same))
  | "conv_valid_stride" ->
      Some (two (fun x w -> L.conv x w ~window_strides:[| 2; 1 |] L.Valid))
  | "conv_valid" ->
      Some (two (fun x w -> L.conv x w ~window_strides:[| 1; 1 |] L.Valid))
  | "cumsum1" -> Some (one (fun x -> Ojax.Numpy.Reductions.cumsum ~axis:1 x))
  | "maxpool" -> Some (one maxpool)
  | "sumpool_stride" ->
      Some
        (one (fun x ->
             L.reduce_window x (JU.py_float 0.0) L.add
               ~window_dimensions:[| 2; 2 |] ~window_strides:[| 2; 1 |]
               [| (0, 0); (0, 0) |]))
  | "sumpool_pad" ->
      Some
        (one (fun x ->
             L.reduce_window x (JU.py_float 0.0) L.add
               ~window_dimensions:[| 2; 2 |] ~window_strides:[| 1; 1 |]
               [| (1, 0); (0, 0) |]))
  | "maxpool_grad" ->
      Some
        (one (fun x ->
             match
               Ojax.Interpreters.Ad.grad
                 (function
                   | [ y ] -> Ojax.Numpy.Reductions.sum (maxpool y)
                   | _ -> failwith "export_text: arity")
                 [ x ]
             with
             | [ g ] -> g
             | _ -> failwith "export_text: grad arity"))
  | "batch_dot" ->
      Some (two (fun a b -> Ojax.Numpy.Einsum.einsum "bij,bjk->bik" [ a; b ]))
  | "matmul" -> Some (two Ojax.Numpy.Tensor_contractions.matmul)
  | "dot_highest" ->
      Some
        (two
           (Ojax.Numpy.Tensor_contractions.dot
              ~precision:
                (Ojax.Lax_config.precision_of_level
                   Ojax.Lax_config.Precision_highest)))
  | _ -> None

let actual row =
  let id = U.to_string (U.member "id" row) in
  let dtype =
    match Ojax.Dtype.of_numpy_name (U.to_string (U.member "dtype" row)) with
    | Some d -> d
    | None -> failwith "export_text: dtype"
  in
  let avals =
    List.map
      (fun s ->
        T.Leaf
          {
            Ojax.Types.shape = Array.of_list (List.map U.to_int (U.to_list s));
            dtype;
            weak_type = false;
          })
      (U.to_list (U.member "shapes" row))
  in
  match fn id with
  | None -> "no_binding"
  | Some f -> (
      try
        (Ojax.Export.export_tree ~fun_name:id
           (fun args -> T.Leaf (f (T.tree_leaves args)))
           (T.Tuple avals))
          .Ojax.Export.mlir_module
      with e -> (
        match (Ojax.Py_exn.class_name e, Ojax.Py_exn.message_of e) with
        | Some c, Some m -> "raise " ^ c ^ " " ^ m
        | _ -> "raise " ^ Printexc.to_string e))

let expected row =
  match U.member "text" row with
  | `String t -> t
  | _ ->
      "raise "
      ^ U.to_string (U.member "error_class" row)
      ^ " "
      ^ U.to_string (U.member "message" row)

let suite_for goldens_root x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat (Filename.concat goldens_root name) (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  let suite = name ^ ":" ^ tag in
  ( name ^ "_" ^ tag,
    List.map
      (fun row ->
        let id =
          U.to_string (U.member "id" row)
          ^ "_"
          ^ U.to_string (U.member "dtype" row)
        in
        Alcotest.test_case id `Quick (fun () ->
            Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
                let jax = expected row and ojax = actual row in
                let same = jax = ojax in
                Docs_p2.record suite id same;
                if not same then
                  Printf.eprintf
                    "EXPORT_TEXT DIFFER %s\n  jax:  %s\n  ojax: %s\n%!" id jax
                    ojax)))
      rows )

let suites ~goldens_root =
  [ suite_for goldens_root false; suite_for goldens_root true ]
