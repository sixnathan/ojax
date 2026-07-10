module Emit = Ojax.Stablehlo.Emit
module J = Ojax.Jaxpr
module C = Ojax.Core
module L = Ojax.Lax
module T = Ojax.Types
module D = Ojax.Dtype
module Nd = Ojax.Ndarray
module U = Yojson.Safe.Util

let () = L.install ()

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

let unary prim dtype =
  (fun () ->
     J.make_jaxpr
       [ av [| 3 |] dtype ]
       (fun args ->
         match args with [ x ] -> C.bind prim [ x ] | _ -> assert false)
    : unit -> T.closed_jaxpr)

let binary prim dtype =
  (fun () ->
     J.make_jaxpr
       [ av [| 3 |] dtype; av [| 3 |] dtype ]
       (fun args ->
         match args with [ x; y ] -> C.bind prim [ x; y ] | _ -> assert false)
    : unit -> T.closed_jaxpr)

let ternary prim dtype =
  (fun () ->
     J.make_jaxpr
       [ av [| 3 |] dtype; av [| 3 |] dtype; av [| 3 |] dtype ]
       (fun args ->
         match args with
         | [ a; b; c ] -> C.bind prim [ a; b; c ]
         | _ -> assert false)
    : unit -> T.closed_jaxpr)

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
    ("unary_abs", unary T.Abs D.F32);
    ("unary_acos", unary T.Acos D.F32);
    ("unary_acosh", unary T.Acosh D.F32);
    ("unary_asin", unary T.Asin D.F32);
    ("unary_asinh", unary T.Asinh D.F32);
    ("unary_atan", unary T.Atan D.F32);
    ("unary_atanh", unary T.Atanh D.F32);
    ("unary_cbrt", unary T.Cbrt D.F32);
    ("unary_ceil", unary T.Ceil D.F32);
    ("unary_clz", unary T.Clz D.I32);
    ("unary_copy", unary T.Copy D.F32);
    ("unary_cos", unary T.Cos D.F32);
    ("unary_cosh", unary T.Cosh D.F32);
    ("unary_exp", unary T.Exp D.F32);
    ("unary_exp2", unary T.Exp2 D.F32);
    ("unary_expm1", unary T.Expm1 D.F32);
    ("unary_floor", unary T.Floor D.F32);
    ("unary_integer_pow", unary (T.Integer_pow 3) D.F32);
    ("unary_is_finite", unary T.Is_finite D.F32);
    ("unary_log", unary T.Log D.F32);
    ("unary_log1p", unary T.Log1p D.F32);
    ("unary_logistic", unary T.Logistic D.F32);
    ("unary_neg", unary T.Neg D.F32);
    ("unary_not", unary T.Not D.I32);
    ("unary_population_count", unary T.Population_count D.I32);
    ("unary_round", unary T.Round D.F32);
    ("unary_rsqrt", unary T.Rsqrt D.F32);
    ("unary_sign", unary T.Sign D.F32);
    ("unary_sin", unary T.Sin D.F32);
    ("unary_sinh", unary T.Sinh D.F32);
    ("unary_sqrt", unary T.Sqrt D.F32);
    ("unary_square", unary T.Square D.F32);
    ("unary_tan", unary T.Tan D.F32);
    ("unary_tanh", unary T.Tanh D.F32);
    ("binary_add", binary T.Add D.F32);
    ("binary_and", binary T.And D.I32);
    ("binary_atan2", binary T.Atan2 D.F32);
    ("binary_div", binary T.Div D.F32);
    ("binary_max", binary T.Max D.F32);
    ("binary_min", binary T.Min D.F32);
    ("binary_mul", binary T.Mul D.F32);
    ("binary_mulhi", binary T.Mulhi D.I32);
    ("binary_nextafter", binary T.Nextafter D.F32);
    ("binary_or", binary T.Or D.I32);
    ("binary_pow", binary T.Pow D.F32);
    ("binary_rem", binary T.Rem D.F32);
    ("binary_shift_left", binary T.Shift_left D.I32);
    ("binary_shift_right_arithmetic", binary T.Shift_right_arithmetic D.I32);
    ("binary_shift_right_logical", binary T.Shift_right_logical D.I32);
    ("binary_sub", binary T.Sub D.F32);
    ("binary_xor", binary T.Xor D.I32);
    ("compare_eq", binary T.Eq D.F32);
    ("compare_ne", binary T.Ne D.F32);
    ("compare_ge", binary T.Ge D.F32);
    ("compare_gt", binary T.Gt D.F32);
    ("compare_le", binary T.Le D.F32);
    ("compare_lt", binary T.Lt D.F32);
    ("compare_eq_i32", binary T.Eq D.I32);
    ("compare_eq_bool", binary T.Eq D.Bool);
    ("compare_eq_to", binary T.Eq_to D.F32);
    ("compare_le_to", binary T.Le_to D.F32);
    ("compare_lt_to", binary T.Lt_to D.F32);
    ("clamp", ternary T.Clamp D.F32);
    ( "select_n2",
      fun () ->
        J.make_jaxpr
          [ av [| 3 |] D.Bool; av [| 3 |] D.F32; av [| 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ p; a; b ] -> C.bind T.Select_n [ p; a; b ]
            | _ -> assert false) );
    ( "select_n3",
      fun () ->
        J.make_jaxpr
          [
            av [| 3 |] D.I32;
            av [| 3 |] D.F32;
            av [| 3 |] D.F32;
            av [| 3 |] D.F32;
          ]
          (fun args ->
            match args with
            | [ p; a; b; c ] -> C.bind T.Select_n [ p; a; b; c ]
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
