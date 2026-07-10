module Emit = Ojax.Stablehlo.Emit
module J = Ojax.Jaxpr
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray
module U = Yojson.Safe.Util

let spec_path =
  match Sys.getenv_opt "OJAX_SPEC" with
  | Some d -> Filename.concat d "stablehlo_emit.cases.json"
  | None ->
      Filename.concat
        (Filename.concat (Filename.concat ".." "..") "spec")
        "stablehlo_emit.cases.json"

let goldens =
  lazy
    (Yojson.Safe.from_file spec_path
    |> U.member "cases" |> U.to_list
    |> List.map (fun c ->
        (U.member "name" c |> U.to_string, U.member "text" c |> U.to_string)))

let golden name =
  match List.assoc_opt name (Lazy.force goldens) with
  | Some t -> t
  | None -> Alcotest.failf "no golden for %s" name

let av shape dtype : T.aval = { shape; dtype; weak_type = false }
let cst dtype shape data = T.Concrete (Nd.of_floats dtype shape data)

let builders : (string * (unit -> T.closed_jaxpr)) list =
  [
    ("identity_vec", fun () -> J.make_jaxpr [ av [| 3 |] D.F32 ] (fun a -> a));
    ("identity_scalar", fun () -> J.make_jaxpr [ av [||] D.F32 ] (fun a -> a));
    ( "multi_out",
      fun () ->
        J.make_jaxpr
          [ av [| 2 |] D.F32; av [| 3 |] D.I32 ]
          (fun args ->
            match args with [ x; y ] -> [ x; y ] | _ -> assert false) );
    ( "dup_out",
      fun () ->
        J.make_jaxpr
          [ av [| 2 |] D.F32 ]
          (fun args -> match args with [ x ] -> [ x; x ] | _ -> assert false) );
    ( "const_scalar_f32",
      fun () -> J.make_jaxpr [] (fun _ -> [ cst D.F32 [||] [| 2.0 |] ]) );
    ( "const_scalar_i32",
      fun () -> J.make_jaxpr [] (fun _ -> [ cst D.I32 [||] [| 7.0 |] ]) );
    ( "const_scalar_bool",
      fun () -> J.make_jaxpr [] (fun _ -> [ cst D.Bool [||] [| 1.0 |] ]) );
    ( "const_vec_i32",
      fun () ->
        J.make_jaxpr [] (fun _ -> [ cst D.I32 [| 3 |] [| 1.0; 2.0; 3.0 |] ]) );
    ( "const_vec_f32",
      fun () ->
        J.make_jaxpr [] (fun _ -> [ cst D.F32 [| 3 |] [| 1.0; 2.0; 3.0 |] ]) );
    ( "const_mat_f32",
      fun () ->
        J.make_jaxpr [] (fun _ ->
            [ cst D.F32 [| 2; 2 |] [| 1.0; 2.0; 3.0; 4.0 |] ]) );
    ( "const_splat_i32",
      fun () ->
        J.make_jaxpr [] (fun _ -> [ cst D.I32 [| 3 |] [| 5.0; 5.0; 5.0 |] ]) );
    ( "const_splat_f32",
      fun () -> J.make_jaxpr [] (fun _ -> [ cst D.F32 [| 2 |] [| 2.0; 2.0 |] ])
    );
    ( "const_neg_zero",
      fun () ->
        J.make_jaxpr [] (fun _ -> [ cst D.F32 [| 3 |] [| -0.0; 1.5; -2.25 |] ])
    );
    ( "const_and_arg",
      fun () ->
        J.make_jaxpr
          [ av [| 2 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] -> [ cst D.F32 [||] [| 2.0 |]; x ]
            | _ -> assert false) );
  ]

let check_case name build () =
  Alcotest.(check string) name (golden name) (Emit.emit_closed_jaxpr (build ()))

let coverage () =
  let have = List.map fst builders in
  List.iter
    (fun (name, _) ->
      if not (List.mem name have) then
        Alcotest.failf "golden %s has no builder" name)
    (Lazy.force goldens);
  Alcotest.(check int)
    "case count"
    (List.length (Lazy.force goldens))
    (List.length builders)

let () =
  Alcotest.run "stablehlo_emit"
    [
      ( "emitter-core",
        Alcotest.test_case "coverage" `Quick coverage
        :: List.map
             (fun (name, build) ->
               Alcotest.test_case name `Quick (check_case name build))
             builders );
    ]
