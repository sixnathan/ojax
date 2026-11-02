module T = Ojax.Types
module D = Ojax.Dtype
module S = Ojax.Sharding
module Tr = Ojax.Tree_util
module NL = Ojax.Numpy.Lax_numpy
module U = Ojax.Numpy.Ufuncs
module R = Ojax.Numpy.Reductions
module L = Ojax.Lax_api

let leaf = function Tr.Leaf v -> v | _ -> failwith "expected a leaf"
let host v = Ojax.Backend.to_host_value v

let values v =
  match host v with
  | T.Concrete nd ->
      Yojson.Safe.(
        `List
          (List.map
             (fun f -> `String (Int64.to_string (Int64.bits_of_float f)))
             (List.rev (Ojax.Ndarray.fold (fun acc x -> x :: acc) [] nd))))
  | _ -> failwith "expected a host array"

let arange16 =
  T.Concrete
    (Ojax.Ndarray.of_floats D.F32 [| 4; 4 |] (Array.init 16 float_of_int))

let arange8 =
  T.Concrete
    (Ojax.Ndarray.of_floats D.F32 [| 4; 2 |] (Array.init 8 float_of_int))

let vector8 =
  T.Concrete (Ojax.Ndarray.of_floats D.F32 [| 8 |] (Array.init 8 float_of_int))

let mesh22 = lazy (S.make_mesh [ 2; 2 ] [ "a"; "b" ])
let mesh2 = lazy (S.make_mesh [ 2 ] [ "x" ])

let mesh4 =
  lazy (Ojax.Spmd.make_mesh ~axis_types:[ Ojax.Spmd.Auto ] [ 4 ] [ "i" ])

let mesh_tail =
  lazy
    (Ojax.Spmd.make_mesh
       ~devices:(Array.sub (Ojax.Spmd.local_devices ()) 4 4)
       [ 4 ] [ "i" ])

let on mesh spec x =
  leaf (S.device_put (Tr.Leaf x) (S.named_sharding (Lazy.force mesh) spec))

let f32_aval shape : T.aval = { shape; dtype = D.F32; weak_type = false }
let two = T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 2.0 |])
let body x = U.add (U.multiply (U.sin x) two) (R.sum x)
let unary f = function [ Tr.Leaf a ] -> Tr.Leaf (f a) | _ -> failwith "arity"
let seen : Yojson.Safe.t list list ref = ref []

let cases () =
  let ab = on mesh22 [ S.Named "a"; S.Named "b" ] arange16 in
  let x_only = on mesh2 [ S.Named "x"; S.Unconstrained ] arange8 in
  let i_only = on mesh4 [ S.Named "i" ] vector8 in
  let tail_only = on mesh_tail [ S.Named "i" ] vector8 in
  [
    ("sin_sum", unary body, [ ab ], None);
    ( "out_sharded",
      unary body,
      [ ab ],
      Some
        [
          S.named_sharding (Lazy.force mesh22) [ S.Named "a"; S.Unconstrained ];
        ] );
    ( "two_results",
      (function
      | [ Tr.Leaf x; Tr.Leaf y ] ->
          Tr.Tuple
            [
              Tr.Leaf
                (Ojax.Numpy.Tensor_contractions.dot x (NL.matrix_transpose y));
              Tr.Leaf (U.exp y);
            ]
      | _ -> failwith "arity"),
      [ x_only; x_only ],
      None );
    ( "unused_argument",
      (function
      | [ Tr.Leaf x; Tr.Leaf _ ] -> Tr.Leaf (U.multiply x two)
      | _ -> failwith "arity"),
      [ ab; ab ],
      None );
    ( "constraint",
      unary (fun x ->
          leaf
            (S.with_sharding_constraint
               (Tr.Leaf (U.multiply x two))
               (S.named_sharding (Lazy.force mesh22)
                  [ S.Unconstrained; S.Named "a" ]))),
      [ ab ],
      None );
    ( "scan_in_shard_map",
      unary (fun x ->
          leaf
            (S.shard_map ~mesh:(Lazy.force mesh4) ~in_specs:[ [ S.Named "i" ] ]
               ~out_specs:[ [] ]
               (unary (fun y ->
                    let body = function
                      | [ carry; elt ] ->
                          [
                            L.add carry elt;
                            Ojax.Parallel.psum elt ~axis_name:"i";
                          ]
                      | _ -> failwith "arity"
                    in
                    match
                      L.scan body
                        [
                          T.Concrete
                            (Ojax.Ndarray.of_floats D.F32 [||] [| 0.0 |]);
                        ]
                        [ y ]
                    with
                    | [ _; ys ] -> ys
                    | _ -> failwith "arity"))
               [ Tr.Leaf x ])),
      [ i_only ],
      None );
    ( "shard_unused_input",
      (function
      | [ Tr.Leaf x; Tr.Leaf y ] ->
          Tr.Leaf
            (leaf
               (S.shard_map ~mesh:(Lazy.force mesh4)
                  ~in_specs:[ [ S.Named "i" ]; [ S.Named "i" ] ]
                  ~out_specs:[ [ S.Named "i" ] ]
                  (function
                    | [ Tr.Leaf a; Tr.Leaf _ ] -> Tr.Leaf (U.sin a)
                    | _ -> failwith "arity")
                  [ Tr.Leaf x; Tr.Leaf y ]))
      | _ -> failwith "arity"),
      [ i_only; i_only ],
      None );
    ( "shard_unused_output",
      unary (fun x ->
          match
            S.shard_map ~mesh:(Lazy.force mesh4) ~in_specs:[ [ S.Named "i" ] ]
              ~out_specs:[ [ S.Named "i" ]; [ S.Named "i" ] ]
              (function
                | [ Tr.Leaf a ] ->
                    Tr.Tuple [ Tr.Leaf (U.sin a); Tr.Leaf (U.cos a) ]
                | _ -> failwith "arity")
              [ Tr.Leaf x ]
          with
          | Tr.Tuple [ _; Tr.Leaf c ] -> c
          | _ -> failwith "arity"),
      [ i_only ],
      None );
    ( "shard_dead_eqns",
      unary (fun x ->
          leaf
            (S.shard_map ~mesh:(Lazy.force mesh4) ~in_specs:[ [ S.Named "i" ] ]
               ~out_specs:[ [ S.Named "i" ] ]
               (unary (fun a ->
                    ignore (U.cos a);
                    U.sin a))
               [ Tr.Leaf x ])),
      [ i_only ],
      None );
    ( "shard_all_dead",
      unary (fun x ->
          ignore
            (S.shard_map ~mesh:(Lazy.force mesh4) ~in_specs:[ [ S.Named "i" ] ]
               ~out_specs:[ [ S.Named "i" ] ] (unary U.sin) [ Tr.Leaf x ]);
          U.cos x),
      [ i_only ],
      None );
    ( "callback_pure",
      unary (fun x ->
          let doubled =
            match
              Ojax.Custom_derivatives.pure_callback
                (function
                  | [ a ] -> [ U.multiply a two ] | _ -> failwith "arity")
                ~result_avals:[ f32_aval [| 8 |] ]
                [ x ]
            with
            | [ y ] -> y
            | _ -> failwith "arity"
          in
          U.add doubled
            (T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 1.0 |]))),
      [ i_only ],
      None );
    ( "callback_debug",
      unary (fun x ->
          Ojax.Debugging.callback
            (fun vs -> seen := List.map values vs :: !seen)
            [ x ];
          U.multiply x two),
      [ i_only ],
      None );
    ( "callback_tail_mesh",
      unary (fun x ->
          Ojax.Debugging.callback (fun _ -> ()) [ x ];
          U.multiply x two),
      [ tail_only ],
      None );
    ( "shard_map",
      unary (fun x ->
          leaf
            (S.shard_map ~mesh:(Lazy.force mesh4) ~in_specs:[ [ S.Named "i" ] ]
               ~out_specs:[ [ S.Named "i" ] ]
               (unary (fun y ->
                    U.add (U.sin y) (Ojax.Parallel.psum y ~axis_name:"i")))
               [ Tr.Leaf x ])),
      [ i_only ],
      None );
  ]

let staged x =
  let a = U.add (U.multiply (U.sin x) two) two in
  let b =
    NL.where_
      (U.greater x (T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 0.0 |])))
      x
      (Ojax.Numpy.Jnp_util.py_float 1.0)
  in
  let start = U.add a b in
  let three = T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 3.0 |]) in
  let one = T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 1.0 |]) in
  match
    L.scan ~length:2
      (function
        | [ c ] -> [ U.add (U.multiply c three) one ] | _ -> failwith "arity")
      [ start ] []
  with
  | [ final ] -> final
  | _ -> failwith "arity"

let staged_where x =
  let a = U.add (U.multiply (U.sin x) two) two in
  let weak =
    NL.where_
      (U.greater x (T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 0.0 |])))
      x
      (Ojax.Numpy.Jnp_util.py_float 1.0)
  in
  U.add a weak

let vector3 =
  T.Concrete (Ojax.Ndarray.of_floats D.F32 [| 3 |] [| 1.0; 2.0; -3.0 |])

let single_device oc =
  let write name text =
    Yojson.Safe.to_channel oc
      (`Assoc
         [
           ("case", `String name); ("text", `String text); ("values", `List []);
         ]);
    output_char oc '\n'
  in
  let flat = function x :: _ -> [ staged x ] | [] -> failwith "arity" in
  let avals = [ f32_aval [| 3 |]; f32_aval [| 3 |] ] in
  let lowered = Ojax.Stages.lower flat avals in
  write "stages_as_text" (Ojax.Stages.as_text lowered);
  let exported = Ojax.Export.export ~fun_name:"f" flat avals in
  write "export_scan" exported.Ojax.Export.mlir_module;
  let plain = function
    | x :: _ -> [ staged_where x ]
    | [] -> failwith "arity"
  in
  write "stages_as_text_where"
    (Ojax.Stages.as_text (Ojax.Stages.lower plain avals));
  write "export_where"
    (Ojax.Export.export ~fun_name:"f" plain avals).Ojax.Export.mlir_module;
  write "export_kept"
    (String.concat ","
       (List.map string_of_int exported.Ojax.Export.module_kept_var_idx));
  let platforms = [ "cpu"; "cuda" ] in
  let sin_first = function x :: _ -> [ U.sin x ] | [] -> failwith "arity" in
  write "export_multi_sin"
    (Ojax.Export.export ~fun_name:"h" ~platforms sin_first [ f32_aval [| 3 |] ])
      .Ojax.Export.mlir_module;
  let times c =
    let k = T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| c |]) in
    fun xs -> [ U.multiply (List.hd xs) k ]
  in
  let dependent = function
    | x :: _ ->
        L.platform_dependent_args ~cpu:(times 2.0) ~default:(times 3.0) [ x ]
    | [] -> failwith "arity"
  in
  let z3 = [ f32_aval [| 3 |] ] in
  write "export_multi_dependent"
    (Ojax.Export.export ~fun_name:"h2" ~platforms dependent z3)
      .Ojax.Export.mlir_module;
  write "export_single_dependent"
    (Ojax.Export.export ~fun_name:"h2" ~platforms:[ "cpu" ] dependent z3)
      .Ojax.Export.mlir_module;
  write "jaxpr_dependent"
    (Ojax.Pretty_printer.closed_jaxpr_to_jax_string
       (Ojax.Jaxpr.make_jaxpr [ f32_aval [| 8 |] ] dependent));
  let jaxpr_of f =
    let flat, tree = Tr.tree_flatten (Tr.Tuple [ Tr.Leaf vector8 ]) in
    let wrapped, _ = Ojax.Api_util.flatten_fun f tree in
    Ojax.Pretty_printer.closed_jaxpr_to_jax_string
      (Ojax.Jaxpr.make_jaxpr
         (List.map Ojax.Core.get_aval flat)
         (Ojax.Linear_util.call_wrapped wrapped))
  in
  write "jaxpr_shard_map"
    (jaxpr_of
       (unary (fun x ->
            leaf
              (S.shard_map ~mesh:(Lazy.force mesh4)
                 ~in_specs:[ [ S.Named "i" ] ] ~out_specs:[ [ S.Named "i" ] ]
                 (unary U.sin) [ Tr.Leaf x ]))));
  write "jaxpr_shard_psum"
    (jaxpr_of
       (unary (fun x ->
            leaf
              (S.shard_map ~mesh:(Lazy.force mesh4)
                 ~in_specs:[ [ S.Named "i" ] ] ~out_specs:[ [ S.Named "i" ] ]
                 (unary (fun y ->
                      U.add (U.sin y) (Ojax.Parallel.psum y ~axis_name:"i")))
                 [ Tr.Leaf x ]))));
  write "jaxpr_constraint"
    (jaxpr_of
       (unary (fun x ->
            leaf
              (S.with_sharding_constraint (Tr.Leaf x)
                 (S.named_sharding (Lazy.force mesh4) [ S.Named "i" ])))));
  write "lower_dependent"
    (Ojax.Stages.as_jax_text ~name:"h2" (Ojax.Stages.lower dependent z3));
  let scan_where = function x :: _ -> [ staged x ] | [] -> failwith "arity" in
  let e =
    Ojax.Export.export ~fun_name:"f" ~platforms scan_where
      [ f32_aval [| 3 |]; f32_aval [| 3 |] ]
  in
  write "export_multi_scan" e.Ojax.Export.mlir_module;
  let nested_scan = function
    | x :: _ -> (
        let three = T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 3.0 |]) in
        let body = function
          | [ c; e ] ->
              let inner =
                L.scan ~length:2
                  (function [ d ] -> [ U.add d e ] | _ -> failwith "arity")
                  [ c ] []
              in
              [ List.hd inner; U.multiply e three ]
          | _ -> failwith "arity"
        in
        match
          L.scan body
            [ T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 0.0 |]) ]
            [ x ]
        with
        | [ c; ys ] -> [ c; ys ]
        | _ -> failwith "arity")
    | [] -> failwith "arity"
  in
  write "stages_nested_scan"
    (Ojax.Stages.as_jax_text (Ojax.Stages.lower nested_scan z3));
  write "export_nested_scan"
    (Ojax.Export.export ~fun_name:"f" nested_scan z3).Ojax.Export.mlir_module;
  let scan_cond = function
    | x :: _ -> (
        let zero = T.Concrete (Ojax.Ndarray.of_floats D.F32 [||] [| 0.0 |]) in
        let body = function
          | [ c; e ] ->
              let r =
                L.cond (U.greater e zero)
                  (function [ a ] -> [ U.sin a ] | _ -> failwith "arity")
                  (function [ a ] -> [ U.cos a ] | _ -> failwith "arity")
                  [ e ]
              in
              [ U.add c (List.hd r) ]
          | _ -> failwith "arity"
        in
        match L.scan body [ zero ] [ x ] with
        | [ c ] -> [ c ]
        | _ -> failwith "arity")
    | [] -> failwith "arity"
  in
  write "stages_scan_cond"
    (Ojax.Stages.as_jax_text (Ojax.Stages.lower scan_cond z3));
  write "export_scan_cond"
    (Ojax.Export.export ~fun_name:"f" scan_cond z3).Ojax.Export.mlir_module;
  write "export_multi_scan_cond"
    (Ojax.Export.export ~fun_name:"f" ~platforms scan_cond z3)
      .Ojax.Export.mlir_module;
  let while_body = function
    | x :: _ -> (
        let limit = T.Concrete (Ojax.Ndarray.of_floats D.I32 [||] [| 3.0 |]) in
        let one = T.Concrete (Ojax.Ndarray.of_floats D.I32 [||] [| 1.0 |]) in
        match
          L.while_loop
            (function [ i; _ ] -> L.lt i limit | _ -> failwith "arity")
            (function
              | [ i; v ] -> [ L.add i one; U.sin v ] | _ -> failwith "arity")
            [ T.Concrete (Ojax.Ndarray.of_floats D.I32 [||] [| 0.0 |]); x ]
        with
        | [ _; v ] -> [ v ]
        | _ -> failwith "arity")
    | [] -> failwith "arity"
  in
  write "stages_while"
    (Ojax.Stages.as_jax_text (Ojax.Stages.lower while_body z3));
  write "export_while"
    (Ojax.Export.export ~fun_name:"f" while_body z3).Ojax.Export.mlir_module;
  write "export_multi_while"
    (Ojax.Export.export ~fun_name:"f" ~platforms while_body z3)
      .Ojax.Export.mlir_module;
  let tree_lowered =
    (Ojax.Stages.jit ~name:"f" (function
      | [ Tr.Leaf x ] ->
          Tr.Dict
            [
              ("a", Tr.Leaf (U.sin x));
              ("b", Tr.Tuple [ Tr.Leaf (U.cos x); Tr.Leaf x ]);
            ]
      | _ -> failwith "arity"))
      .Ojax.Stages.wrapped_lower
      [
        Tr.Leaf
          (T.Concrete (Ojax.Ndarray.of_floats D.F32 [| 3 |] [| 0.; 0.; 0. |]));
      ]
  in
  write "stages_tree_paths" (Ojax.Stages.as_text tree_lowered);
  Tr.register_attribute_keys "QRResult" [ "Q"; "R" ];
  let named_lowered =
    (Ojax.Stages.jit ~name:"f" (function
      | [ Tr.Leaf x ] -> (
          match Ojax.Numpy.Linalg.qr x with
          | [ q; r ] -> Tr.Custom ("QRResult", "", [ Tr.Leaf q; Tr.Leaf r ])
          | _ -> failwith "arity")
      | _ -> failwith "arity"))
      .Ojax.Stages.wrapped_lower
      [
        Tr.Leaf
          (T.Concrete
             (Ojax.Ndarray.of_floats D.F32 [| 2; 2 |] [| 2.; 1.; 1.; 3. |]));
      ]
  in
  write "stages_namedtuple_paths" (Ojax.Stages.as_text named_lowered);
  let called = Ojax.Export.call e [ vector3; vector3 ] in
  Yojson.Safe.to_channel oc
    (`Assoc
       [
         ("case", `String "export_multi_call");
         ("text", `String "");
         ("values", `List (List.map values called));
       ]);
  output_char oc '\n'

let () =
  let oc = open_out Sys.argv.(1) in
  single_device oc;
  List.iter
    (fun (name, f, args, out_shardings) ->
      let row =
        try
          let text =
            S.lower_text ?out_shardings ~name:"f" f
              (List.map (fun v -> Tr.Leaf v) args)
          in
          let result =
            S.jit ?out_shardings f (List.map (fun v -> Tr.Leaf v) args)
          in
          let leaves, _ = Tr.tree_flatten result in
          [ ("text", `String text); ("values", `List (List.map values leaves)) ]
        with e -> [ ("error", `String (Printexc.to_string e)) ]
      in
      Yojson.Safe.to_channel oc (`Assoc (("case", `String name) :: row));
      output_char oc '\n')
    (cases ());
  Yojson.Safe.to_channel oc
    (`Assoc
       [
         ("case", `String "callback_seen");
         ("text", `String "");
         ("values", `List (List.map (fun row -> `List row) (List.rev !seen)));
       ]);
  output_char oc '\n';
  close_out oc
