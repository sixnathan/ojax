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
let mesh4 = lazy (S.make_mesh [ 4 ] [ "i" ])

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
          Ojax.Debugging.callback (fun _ -> ()) [ x ];
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
  write "lower_dependent"
    (Ojax.Stages.as_jax_text ~name:"h2" (Ojax.Stages.lower dependent z3));
  let scan_where = function x :: _ -> [ staged x ] | [] -> failwith "arity" in
  let e =
    Ojax.Export.export ~fun_name:"f" ~platforms scan_where
      [ f32_aval [| 3 |]; f32_aval [| 3 |] ]
  in
  write "export_multi_scan" e.Ojax.Export.mlir_module;
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
  close_out oc
