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

let gdn : T.gather_dims =
  {
    offset_dims = [| 1 |];
    collapsed_slice_dims = [| 0 |];
    start_index_map = [| 0 |];
    g_operand_batching_dims = [||];
    g_start_indices_batching_dims = [||];
  }

let sdn : T.scatter_dims =
  {
    update_window_dims = [| 1 |];
    inserted_window_dims = [| 0 |];
    scatter_dims_to_operand_dims = [| 0 |];
    s_operand_batching_dims = [||];
    s_scatter_indices_batching_dims = [||];
  }

let unary5 prim dtype =
  (fun () ->
     J.make_jaxpr
       [ av [| 5 |] dtype ]
       (fun args ->
         match args with [ x ] -> C.bind prim [ x ] | _ -> assert false)
    : unit -> T.closed_jaxpr)

let window1 : T.window_dims =
  {
    window_dimensions = [| 2 |];
    window_strides = [| 1 |];
    w_padding = [| (0, 0) |];
    base_dilation = [| 1 |];
    window_dilation = [| 1 |];
  }

let scatter_case prim () =
  J.make_jaxpr
    [ av [| 5; 3 |] D.F32; av [| 2; 1 |] D.I32; av [| 2; 3 |] D.F32 ]
    (fun args ->
      match args with
      | [ o; i; u ] -> C.bind prim [ o; i; u ]
      | _ -> assert false)

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
    ("convert_f32_to_i32", unary (T.Convert_element_type D.I32) D.F32);
    ("convert_i32_to_f32", unary (T.Convert_element_type D.F32) D.I32);
    ("convert_bool_to_i32", unary (T.Convert_element_type D.I32) D.Bool);
    ("convert_f32_to_bool", unary (T.Convert_element_type D.Bool) D.F32);
    ("bitcast_f32_to_i32", unary (T.Bitcast_convert_type D.I32) D.F32);
    ("optimization_barrier", unary T.Optimization_barrier D.F32);
    ( "reduce_precision",
      unary (T.Reduce_precision { exponent_bits = 8; mantissa_bits = 10 }) D.F32
    );
    ("tie", binary T.Tie D.F32);
    ( "empty",
      fun () ->
        J.make_jaxpr [] (fun _ ->
            C.bind (T.Empty { shape = [| 3 |]; dtype = D.F32 }) []) );
    ( "platform_index",
      fun () ->
        J.make_jaxpr [] (fun _ ->
            C.bind (T.Platform_index [| Some [| "cpu" |] |]) []) );
    ( "shape_broadcast_in_dim",
      fun () ->
        J.make_jaxpr
          [ av [| 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] ->
                C.bind
                  (T.Broadcast_in_dim { shape = [| 2; 3 |]; dims = [| 1 |] })
                  [ x ]
            | _ -> assert false) );
    ( "shape_concatenate",
      fun () ->
        J.make_jaxpr
          [ av [| 2 |] D.F32; av [| 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ x; y ] -> C.bind (T.Concatenate 0) [ x; y ]
            | _ -> assert false) );
    ( "shape_iota",
      fun () ->
        J.make_jaxpr [] (fun _ ->
            C.bind
              (T.Iota { dtype = D.I32; shape = [| 2; 3 |]; dimension = 1 })
              []) );
    ( "shape_pad",
      fun () ->
        J.make_jaxpr
          [ av [| 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] ->
                C.bind (T.Pad [| (1, 2, 0) |]) [ x; cst D.F32 [||] [| 0.0 |] ]
            | _ -> assert false) );
    ( "shape_pad_interior",
      fun () ->
        J.make_jaxpr
          [ av [| 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] ->
                C.bind (T.Pad [| (0, 0, 1) |]) [ x; cst D.F32 [||] [| 0.0 |] ]
            | _ -> assert false) );
    ( "shape_reshape",
      fun () ->
        J.make_jaxpr
          [ av [| 6 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] -> C.bind (T.Reshape [| 2; 3 |]) [ x ]
            | _ -> assert false) );
    ( "shape_rev",
      fun () ->
        J.make_jaxpr
          [ av [| 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] -> C.bind (T.Rev [| 0 |]) [ x ]
            | _ -> assert false) );
    ( "shape_split",
      fun () ->
        J.make_jaxpr
          [ av [| 5 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] -> C.bind (T.Split { sizes = [| 2; 3 |]; axis = 0 }) [ x ]
            | _ -> assert false) );
    ( "shape_squeeze",
      fun () ->
        J.make_jaxpr
          [ av [| 1; 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] -> C.bind (T.Squeeze [| 0 |]) [ x ]
            | _ -> assert false) );
    ( "shape_stack",
      fun () ->
        J.make_jaxpr
          [ av [| 3 |] D.F32; av [| 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ x; y ] -> C.bind (T.Stack 0) [ x; y ]
            | _ -> assert false) );
    ( "shape_tile",
      fun () ->
        J.make_jaxpr
          [ av [| 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] -> C.bind (T.Tile [| 2 |]) [ x ]
            | _ -> assert false) );
    ( "shape_transpose",
      fun () ->
        J.make_jaxpr
          [ av [| 2; 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] -> C.bind (T.Transpose [| 1; 0 |]) [ x ]
            | _ -> assert false) );
    ( "shape_unstack",
      fun () ->
        J.make_jaxpr
          [ av [| 2; 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] -> C.bind (T.Unstack 0) [ x ]
            | _ -> assert false) );
    ("reduce_sum", unary (T.Reduce_sum [| 0 |]) D.F32);
    ("reduce_max", unary (T.Reduce_max [| 0 |]) D.F32);
    ("reduce_min", unary (T.Reduce_min [| 0 |]) D.F32);
    ("reduce_prod", unary (T.Reduce_prod [| 0 |]) D.F32);
    ("reduce_and", unary (T.Reduce_and [| 0 |]) D.I32);
    ("reduce_or", unary (T.Reduce_or [| 0 |]) D.I32);
    ("reduce_xor", unary (T.Reduce_xor [| 0 |]) D.I32);
    ("argmax", unary (T.Argmax { axis = 0; index_dtype = D.I32 }) D.F32);
    ("argmin", unary (T.Argmin { axis = 0; index_dtype = D.I32 }) D.F32);
    ("cumsum", unary (T.Cumsum { axis = 0; reverse = false }) D.F32);
    ("cumprod", unary (T.Cumprod { axis = 0; reverse = false }) D.F32);
    ("cummax", unary (T.Cummax { axis = 0; reverse = false }) D.F32);
    ("cummin", unary (T.Cummin { axis = 0; reverse = false }) D.F32);
    ("cumlogsumexp", unary (T.Cumlogsumexp { axis = 0; reverse = false }) D.F32);
    ( "slice",
      fun () ->
        J.make_jaxpr
          [ av [| 6 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] ->
                C.bind
                  (T.Slice
                     {
                       start_indices = [| 1 |];
                       limit_indices = [| 4 |];
                       strides = None;
                     })
                  [ x ]
            | _ -> assert false) );
    ( "slice_strided",
      fun () ->
        J.make_jaxpr
          [ av [| 3; 5 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] ->
                C.bind
                  (T.Slice
                     {
                       start_indices = [| 0; 1 |];
                       limit_indices = [| 2; 4 |];
                       strides = Some [| 1; 2 |];
                     })
                  [ x ]
            | _ -> assert false) );
    ( "dynamic_slice",
      fun () ->
        J.make_jaxpr
          [ av [| 6 |] D.F32; av [||] D.I32 ]
          (fun args ->
            match args with
            | [ x; i ] ->
                C.bind (T.Dynamic_slice { slice_sizes = [| 3 |] }) [ x; i ]
            | _ -> assert false) );
    ( "dynamic_update_slice",
      fun () ->
        J.make_jaxpr
          [ av [| 6 |] D.F32; av [| 2 |] D.F32; av [||] D.I32 ]
          (fun args ->
            match args with
            | [ x; u; i ] -> C.bind T.Dynamic_update_slice [ x; u; i ]
            | _ -> assert false) );
    ( "gather",
      fun () ->
        J.make_jaxpr
          [ av [| 5; 3 |] D.F32; av [| 2; 1 |] D.I32 ]
          (fun args ->
            match args with
            | [ o; i ] ->
                C.bind
                  (T.Gather
                     { dimension_numbers = gdn; slice_sizes = [| 1; 3 |] })
                  [ o; i ]
            | _ -> assert false) );
    ( "scatter",
      scatter_case
        (T.Scatter { dimension_numbers = sdn; unique_indices = false }) );
    ("scatter_add", scatter_case (T.Scatter_add { dimension_numbers = sdn }));
    ("scatter_sub", scatter_case (T.Scatter_sub { dimension_numbers = sdn }));
    ( "scatter_mul",
      scatter_case
        (T.Scatter_mul { dimension_numbers = sdn; unique_indices = false }) );
    ("scatter_min", scatter_case (T.Scatter_min { dimension_numbers = sdn }));
    ("scatter_max", scatter_case (T.Scatter_max { dimension_numbers = sdn }));
    ( "dot_general",
      fun () ->
        J.make_jaxpr
          [ av [| 2; 3 |] D.F32; av [| 3; 4 |] D.F32 ]
          (fun args ->
            match args with
            | [ a; b ] ->
                C.bind
                  (T.Dot_general
                     {
                       lhs_contract = [| 1 |];
                       rhs_contract = [| 0 |];
                       lhs_batch = [||];
                       rhs_batch = [||];
                     })
                  [ a; b ]
            | _ -> assert false) );
    ( "dot_general_batch",
      fun () ->
        J.make_jaxpr
          [ av [| 2; 3; 4 |] D.F32; av [| 2; 4; 5 |] D.F32 ]
          (fun args ->
            match args with
            | [ a; b ] ->
                C.bind
                  (T.Dot_general
                     {
                       lhs_contract = [| 2 |];
                       rhs_contract = [| 1 |];
                       lhs_batch = [| 0 |];
                       rhs_batch = [| 0 |];
                     })
                  [ a; b ]
            | _ -> assert false) );
    ( "conv",
      fun () ->
        J.make_jaxpr
          [ av [| 1; 1; 5 |] D.F32; av [| 1; 1; 3 |] D.F32 ]
          (fun args ->
            match args with
            | [ l; r ] ->
                C.bind
                  (T.Conv_general_dilated
                     {
                       window_strides = [| 1 |];
                       padding = [| (0, 0) |];
                       lhs_dilation = [| 1 |];
                       rhs_dilation = [| 1 |];
                       dimension_numbers =
                         {
                           lhs_spec = [| 0; 1; 2 |];
                           rhs_spec = [| 0; 1; 2 |];
                           out_spec = [| 0; 1; 2 |];
                         };
                       feature_group_count = 1;
                       batch_group_count = 1;
                     })
                  [ l; r ]
            | _ -> assert false) );
    ("reduce_window_sum", unary5 (T.Reduce_window_sum window1) D.F32);
    ("reduce_window_max", unary5 (T.Reduce_window_max window1) D.F32);
    ("reduce_window_min", unary5 (T.Reduce_window_min window1) D.F32);
    ( "select_and_scatter_add",
      fun () ->
        J.make_jaxpr
          [ av [| 4 |] D.F32; av [| 5 |] D.F32 ]
          (fun args ->
            match args with
            | [ s; o ] ->
                C.bind
                  (T.Select_and_scatter_add { select = T.Wge; window = window1 })
                  [ s; o ]
            | _ -> assert false) );
    ( "select_and_gather_add",
      fun () ->
        J.make_jaxpr
          [ av [| 5 |] D.F32; av [| 5 |] D.F32 ]
          (fun args ->
            match args with
            | [ t; o ] ->
                C.bind
                  (T.Select_and_gather_add { select = T.Wge; window = window1 })
                  [ t; o ]
            | _ -> assert false) );
    ( "sort",
      unary5 (T.Sort { dimension = 0; is_stable = true; num_keys = 1 }) D.F32 );
    ( "top_k",
      fun () ->
        J.make_jaxpr
          [ av [| 5 |] D.F32 ]
          (fun args ->
            match args with
            | [ x ] -> C.bind (T.Top_k { k = 2; axis = 0 }) [ x ]
            | _ -> assert false) );
    ("special_bessel_i1e", unary T.Bessel_i1e D.F32);
    ("special_digamma", unary T.Digamma D.F32);
    ("special_erf", unary T.Erf D.F32);
    ("special_erf_inv", unary T.Erf_inv D.F32);
    ("special_erfc", unary T.Erfc D.F32);
    ("special_lgamma", unary T.Lgamma D.F32);
    ("special_polygamma", binary T.Polygamma D.F32);
    ("special_zeta", binary T.Zeta D.F32);
    ( "rng_uniform",
      fun () ->
        let a : T.var = { vid = 0; vaval = av [||] D.F32 } in
        let b : T.var = { vid = 1; vaval = av [||] D.F32 } in
        let o : T.var = { vid = 2; vaval = av [| 3 |] D.F32 } in
        let eqn : T.eqn =
          {
            prim = T.Rng_uniform;
            inputs = [ T.A_var a; T.A_var b ];
            outs = [ o ];
            multiple_results = false;
          }
        in
        {
          T.jid = 0;
          jaxpr =
            { T.in_binders = [ a; b ]; eqns = [ eqn ]; outs = [ T.A_var o ] };
          consts = [];
        } );
    ( "region_cond",
      fun () ->
        let idx : T.var = { vid = 0; vaval = av [||] D.I32 } in
        let x : T.var = { vid = 1; vaval = av [| 3 |] D.F32 } in
        let y : T.var = { vid = 2; vaval = av [| 3 |] D.F32 } in
        let o : T.var = { vid = 3; vaval = av [| 3 |] D.F32 } in
        let branch prim =
          J.make_jaxpr
            [ av [| 3 |] D.F32; av [| 3 |] D.F32 ]
            (fun args ->
              match args with
              | [ p; q ] -> C.bind prim [ p; q ]
              | _ -> assert false)
        in
        let eqn : T.eqn =
          {
            prim = T.Cond { t = branch T.Add; f = branch T.Sub };
            inputs = [ T.A_var idx; T.A_var x; T.A_var y ];
            outs = [ o ];
            multiple_results = false;
          }
        in
        {
          T.jid = 0;
          jaxpr =
            {
              T.in_binders = [ idx; x; y ];
              eqns = [ eqn ];
              outs = [ T.A_var o ];
            };
          consts = [];
        } );
    ( "region_reduce",
      fun () ->
        let x : T.var = { vid = 0; vaval = av [| 3 |] D.F32 } in
        let o : T.var = { vid = 1; vaval = av [||] D.F32 } in
        let reducer =
          J.make_jaxpr
            [ av [||] D.F32; av [||] D.F32 ]
            (fun args ->
              match args with
              | [ p; q ] -> (
                  match C.bind T.Mul [ p; q ] with
                  | [ pr ] -> C.bind T.Add [ pr; p ]
                  | _ -> assert false)
              | _ -> assert false)
        in
        let eqn : T.eqn =
          {
            prim = T.Reduce { jaxpr = reducer; dimensions = [| 0 |] };
            inputs = [ T.A_var x; T.A_lit (Nd.of_floats D.F32 [||] [| 1.0 |]) ];
            outs = [ o ];
            multiple_results = false;
          }
        in
        {
          T.jid = 0;
          jaxpr = { T.in_binders = [ x ]; eqns = [ eqn ]; outs = [ T.A_var o ] };
          consts = [];
        } );
    ( "region_reduce_window",
      fun () ->
        let x : T.var = { vid = 0; vaval = av [| 5 |] D.F32 } in
        let o : T.var = { vid = 1; vaval = av [| 4 |] D.F32 } in
        let reducer =
          J.make_jaxpr
            [ av [||] D.F32; av [||] D.F32 ]
            (fun args ->
              match args with
              | [ p; q ] -> C.bind T.Add [ p; q ]
              | _ -> assert false)
        in
        let eqn : T.eqn =
          {
            prim = T.Reduce_window { reducer; window = window1 };
            inputs = [ T.A_var x; T.A_lit (Nd.of_floats D.F32 [||] [| 0.0 |]) ];
            outs = [ o ];
            multiple_results = false;
          }
        in
        {
          T.jid = 0;
          jaxpr = { T.in_binders = [ x ]; eqns = [ eqn ]; outs = [ T.A_var o ] };
          consts = [];
        } );
    ( "region_while",
      fun () ->
        let init : T.var = { vid = 0; vaval = av [||] D.F32 } in
        let o : T.var = { vid = 1; vaval = av [||] D.F32 } in
        let cond =
          J.make_jaxpr
            [ av [||] D.F32 ]
            (fun args ->
              match args with
              | [ v ] -> C.bind T.Lt [ v; cst D.F32 [||] [| 10.0 |] ]
              | _ -> assert false)
        in
        let body =
          J.make_jaxpr
            [ av [||] D.F32 ]
            (fun args ->
              match args with
              | [ v ] -> C.bind T.Add [ v; cst D.F32 [||] [| 1.0 |] ]
              | _ -> assert false)
        in
        let eqn : T.eqn =
          {
            prim = T.While { cond; body };
            inputs = [ T.A_var init ];
            outs = [ o ];
            multiple_results = false;
          }
        in
        {
          T.jid = 0;
          jaxpr =
            { T.in_binders = [ init ]; eqns = [ eqn ]; outs = [ T.A_var o ] };
          consts = [];
        } );
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
