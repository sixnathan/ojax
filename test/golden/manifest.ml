module U = Yojson.Safe.Util

let goldens_root =
  match Sys.getenv_opt "OJAX_GOLDENS" with
  | Some d -> d
  | None -> Filename.concat (Filename.concat ".." "..") "goldens"

type arg = { name : string; shape : int array; dtype : string }

type out = {
  oname : string;
  oshape : int array;
  odtype : string;
  ocompare : string option;
  oatol : float option;
  ortol : float option;
  otreason : string option;
}

type case = {
  case_id : string;
  op : string;
  params : Yojson.Safe.t;
  compare : string;
  atol : float;
  rtol : float;
  treason : string option;
  args : arg list;
  outs : out list;
}

let to_shape j = Array.of_list (List.map U.to_int (U.to_list j))

let parse_arg j =
  {
    name = U.member "name" j |> U.to_string;
    shape = U.member "shape" j |> to_shape;
    dtype = U.member "dtype" j |> U.to_string;
  }

let parse_out j =
  let ocompare =
    match U.member "compare" j with `Null -> None | v -> Some (U.to_string v)
  in
  let oatol, ortol =
    match U.member "tol" j with
    | `Null -> (None, None)
    | tol ->
        ( Some (U.member "atol" tol |> U.to_number),
          Some (U.member "rtol" tol |> U.to_number) )
  in
  let otreason =
    match U.member "tol_reason" j with
    | `Null -> None
    | v -> Some (U.to_string v)
  in
  {
    oname = U.member "name" j |> U.to_string;
    oshape = U.member "shape" j |> to_shape;
    odtype = U.member "dtype" j |> U.to_string;
    ocompare;
    oatol;
    ortol;
    otreason;
  }

let parse_case j =
  let tol = U.member "tol" j in
  let treason =
    match U.member "tol_reason" j with
    | `Null -> None
    | v -> Some (U.to_string v)
  in
  {
    case_id = U.member "case_id" j |> U.to_string;
    op = U.member "op" j |> U.to_string;
    params = U.member "params" j;
    compare = U.member "compare" j |> U.to_string;
    atol = U.member "atol" tol |> U.to_number;
    rtol = U.member "rtol" tol |> U.to_number;
    treason;
    args = U.member "args" j |> U.to_list |> List.map parse_arg;
    outs = U.member "outputs" j |> U.to_list |> List.map parse_out;
  }

let load_manifest path =
  let j = Yojson.Safe.from_file path in
  ( U.member "x64" j |> U.to_bool,
    U.member "cases" j |> U.to_list |> List.map parse_case )

let find_member members name =
  match List.assoc_opt name members with
  | Some a -> a
  | None -> failwith ("missing npz member " ^ name)

let check_case ~set_dir ~x64 c () =
  let canon d = if x64 then d else Compare.canonical_dtype_x64_off d in
  let inputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "inputs") (c.case_id ^ ".npz"))
  in
  let outputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "outputs") (c.case_id ^ ".npz"))
  in
  List.iter
    (fun a ->
      let arr = find_member inputs a.name in
      if not (Compare.shapes_equal arr.Npz.shape a.shape) then
        Alcotest.failf "%s: input %s shape mismatch" c.case_id a.name;
      if canon arr.Npz.dtype <> canon a.dtype then
        Alcotest.failf "%s: input %s dtype %s != %s" c.case_id a.name
          arr.Npz.dtype a.dtype)
    c.args;
  List.iter
    (fun o ->
      let arr = find_member outputs o.oname in
      if not (Compare.shapes_equal arr.Npz.shape o.oshape) then
        Alcotest.failf "%s: output %s shape mismatch" c.case_id o.oname;
      if canon arr.Npz.dtype <> canon o.odtype then
        Alcotest.failf "%s: output %s dtype %s != %s" c.case_id o.oname
          arr.Npz.dtype o.odtype;
      let reason =
        match o.otreason with Some _ as r -> r | None -> c.treason
      in
      Compare.assert_tol_widened o.odtype c.atol c.rtol reason)
    c.outs

let dir_case_ids dir =
  Sys.readdir dir |> Array.to_list
  |> List.filter (fun f -> Filename.check_suffix f ".npz")
  |> List.map Filename.remove_extension
  |> List.sort String.compare

let check_coverage ~set_dir cases () =
  let expected =
    List.map (fun c -> c.case_id) cases |> List.sort String.compare
  in
  List.iter
    (fun sub ->
      let got = dir_case_ids (Filename.concat set_dir sub) in
      if got <> expected then begin
        let extra = List.filter (fun x -> not (List.mem x expected)) got in
        let missing = List.filter (fun x -> not (List.mem x got)) expected in
        Alcotest.failf "%s: coverage mismatch missing=[%s] extra=[%s]" sub
          (String.concat ";" missing)
          (String.concat ";" extra)
      end)
    [ "inputs"; "outputs" ]

let suite_for module_ set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root module_) set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick (check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  (module_ ^ ":" ^ set_name, coverage :: case_tests)

module Nd = Ojax.Ndarray
module C = Ojax.Core
module T = Ojax.Types
module D = Ojax.Dtype

let dtype_of_string = function
  | "float32" -> D.F32
  | "float64" -> D.F64
  | "int32" -> D.I32
  | "int64" -> D.I64
  | "bool" -> D.Bool
  | s -> failwith ("lax golden: unsupported dtype " ^ s)

let string_of_dtype = function
  | D.F32 -> "float32"
  | D.F64 -> "float64"
  | D.I32 -> "int32"
  | D.I64 -> "int64"
  | D.Bool -> "bool"

let nd_of_npz (a : Npz.t) =
  let floats =
    match a.Npz.data with
    | Npz.F f -> f
    | Npz.I i -> Array.map Int64.to_float i
    | Npz.C _ -> failwith "lax golden: complex operand unsupported"
  in
  Nd.of_floats (dtype_of_string a.Npz.dtype) a.Npz.shape floats

let read_nd nd =
  let n = Array.fold_left ( * ) 1 (Nd.shape nd) in
  let arr = Array.make n 0.0 in
  let _ =
    Nd.fold
      (fun i x ->
        arr.(i) <- x;
        i + 1)
      0 nd
  in
  arr

let ia j = Array.of_list (List.map U.to_int (U.to_list j))

let pairs j =
  Array.of_list
    (List.map
       (fun t ->
         match U.to_list t with
         | [ a; b ] -> (U.to_int a, U.to_int b)
         | _ -> failwith "lax golden: bad padding pair")
       (U.to_list j))

let window_of params : T.window_dims =
  let member name = U.member name params in
  {
    window_dimensions = ia (member "window_dimensions");
    window_strides = ia (member "window_strides");
    w_padding = pairs (member "padding");
    base_dilation = ia (member "base_dilation");
    window_dilation = ia (member "window_dilation");
  }

let window_select params =
  match U.to_string (U.member "select" params) with
  | "ge" -> T.Wge
  | "le" -> T.Wle
  | s -> failwith ("lax golden: bad window select " ^ s)

let prim_of op params : T.primitive =
  let member name = U.member name params in
  match op with
  | "neg" -> T.Neg
  | "sin" -> T.Sin
  | "cos" -> T.Cos
  | "exp" -> T.Exp
  | "log" -> T.Log
  | "tanh" -> T.Tanh
  | "abs" -> T.Abs
  | "sign" -> T.Sign
  | "add" -> T.Add
  | "sub" -> T.Sub
  | "mul" -> T.Mul
  | "div" -> T.Div
  | "max" -> T.Max
  | "min" -> T.Min
  | "pow" -> T.Pow
  | "eq" -> T.Eq
  | "lt" -> T.Lt
  | "gt" -> T.Gt
  | "select_n" -> T.Select_n
  | "acos" -> T.Acos
  | "acosh" -> T.Acosh
  | "asin" -> T.Asin
  | "asinh" -> T.Asinh
  | "atan" -> T.Atan
  | "atanh" -> T.Atanh
  | "cbrt" -> T.Cbrt
  | "ceil" -> T.Ceil
  | "clz" -> T.Clz
  | "conj" -> T.Conj
  | "copy" -> T.Copy
  | "cosh" -> T.Cosh
  | "exp2" -> T.Exp2
  | "expm1" -> T.Expm1
  | "floor" -> T.Floor
  | "imag" -> T.Imag
  | "integer_pow" -> T.Integer_pow (U.to_int (member "y"))
  | "is_finite" -> T.Is_finite
  | "log1p" -> T.Log1p
  | "logistic" -> T.Logistic
  | "not" -> T.Not
  | "population_count" -> T.Population_count
  | "real" -> T.Real
  | "round" -> T.Round
  | "rsqrt" -> T.Rsqrt
  | "sinh" -> T.Sinh
  | "sqrt" -> T.Sqrt
  | "square" -> T.Square
  | "tan" -> T.Tan
  | "and" -> T.And
  | "atan2" -> T.Atan2
  | "complex" -> T.Complex
  | "eq_to" -> T.Eq_to
  | "ge" -> T.Ge
  | "le" -> T.Le
  | "le_to" -> T.Le_to
  | "lt_to" -> T.Lt_to
  | "mulhi" -> T.Mulhi
  | "ne" -> T.Ne
  | "nextafter" -> T.Nextafter
  | "or" -> T.Or
  | "rem" -> T.Rem
  | "shift_left" -> T.Shift_left
  | "shift_right_arithmetic" -> T.Shift_right_arithmetic
  | "shift_right_logical" -> T.Shift_right_logical
  | "xor" -> T.Xor
  | "concatenate" -> T.Concatenate (U.to_int (member "dimension"))
  | "pad" ->
      let cfg =
        Array.of_list
          (List.map
             (fun t ->
               match U.to_list t with
               | [ a; b; c ] -> (U.to_int a, U.to_int b, U.to_int c)
               | _ -> failwith "lax golden: bad padding_config")
             (U.to_list (member "padding_config")))
      in
      T.Pad cfg
  | "rev" -> T.Rev (ia (member "dimensions"))
  | "split" ->
      T.Split { sizes = ia (member "sizes"); axis = U.to_int (member "axis") }
  | "squeeze" -> T.Squeeze (ia (member "dimensions"))
  | "stack" -> T.Stack (U.to_int (member "axis"))
  | "tile" -> T.Tile (ia (member "reps"))
  | "transpose" -> T.Transpose (ia (member "permutation"))
  | "unstack" -> T.Unstack (U.to_int (member "axis"))
  | "reduce_max" -> T.Reduce_max (ia (member "axes"))
  | "reduce_min" -> T.Reduce_min (ia (member "axes"))
  | "reduce_prod" -> T.Reduce_prod (ia (member "axes"))
  | "reduce_and" -> T.Reduce_and (ia (member "axes"))
  | "reduce_or" -> T.Reduce_or (ia (member "axes"))
  | "reduce_xor" -> T.Reduce_xor (ia (member "axes"))
  | "argmax" ->
      T.Argmax
        {
          axis = U.to_int (member "axis");
          index_dtype = dtype_of_string (U.to_string (member "index_dtype"));
        }
  | "argmin" ->
      T.Argmin
        {
          axis = U.to_int (member "axis");
          index_dtype = dtype_of_string (U.to_string (member "index_dtype"));
        }
  | "reduce" ->
      let sc = { T.shape = [||]; dtype = D.F32; weak_type = false } in
      let reducer =
        Ojax.Jaxpr.make_jaxpr [ sc; sc ] (fun args -> [ C.bind1 T.Add args ])
      in
      T.Reduce { jaxpr = reducer; dimensions = ia (member "dimensions") }
  | "clamp" -> T.Clamp
  | "bitcast_convert_type" ->
      T.Bitcast_convert_type
        (dtype_of_string (U.to_string (member "new_dtype")))
  | "iota" ->
      T.Iota
        {
          dtype = dtype_of_string (U.to_string (member "dtype"));
          shape = ia (member "shape");
          dimension = U.to_int (member "dimension");
        }
  | "optimization_barrier" -> T.Optimization_barrier
  | "reduce_precision" ->
      T.Reduce_precision
        {
          exponent_bits = U.to_int (member "exponent_bits");
          mantissa_bits = U.to_int (member "mantissa_bits");
        }
  | "sort" ->
      T.Sort
        {
          dimension = U.to_int (member "dimension");
          is_stable = U.to_bool (member "is_stable");
          num_keys = U.to_int (member "num_keys");
        }
  | "tie" -> T.Tie
  | "top_k" ->
      T.Top_k { k = U.to_int (member "k"); axis = U.to_int (member "axis") }
  | "slice" ->
      let strides =
        match member "strides" with `Null -> None | j -> Some (ia j)
      in
      T.Slice
        {
          start_indices = ia (member "start_indices");
          limit_indices = ia (member "limit_indices");
          strides;
        }
  | "dynamic_slice" ->
      T.Dynamic_slice { slice_sizes = ia (member "slice_sizes") }
  | "dynamic_update_slice" -> T.Dynamic_update_slice
  | "gather" ->
      let opt name = match member name with `Null -> [||] | j -> ia j in
      T.Gather
        {
          dimension_numbers =
            {
              offset_dims = ia (member "offset_dims");
              collapsed_slice_dims = ia (member "collapsed_slice_dims");
              start_index_map = ia (member "start_index_map");
              g_operand_batching_dims = opt "operand_batching_dims";
              g_start_indices_batching_dims = opt "start_indices_batching_dims";
            };
          slice_sizes = ia (member "slice_sizes");
        }
  | "scatter" | "scatter_add" | "scatter_sub" | "scatter_mul" | "scatter_min"
  | "scatter_max" -> (
      let opt name = match member name with `Null -> [||] | j -> ia j in
      let sd : T.scatter_dims =
        {
          update_window_dims = ia (member "update_window_dims");
          inserted_window_dims = ia (member "inserted_window_dims");
          scatter_dims_to_operand_dims =
            ia (member "scatter_dims_to_operand_dims");
          s_operand_batching_dims = opt "operand_batching_dims";
          s_scatter_indices_batching_dims = opt "scatter_indices_batching_dims";
        }
      in
      let unique =
        match member "unique_indices" with `Bool b -> b | _ -> true
      in
      match op with
      | "scatter" ->
          T.Scatter { dimension_numbers = sd; unique_indices = unique }
      | "scatter_add" -> T.Scatter_add { dimension_numbers = sd }
      | "scatter_sub" -> T.Scatter_sub { dimension_numbers = sd }
      | "scatter_mul" ->
          T.Scatter_mul { dimension_numbers = sd; unique_indices = unique }
      | "scatter_min" -> T.Scatter_min { dimension_numbers = sd }
      | _ -> T.Scatter_max { dimension_numbers = sd })
  | "convert_element_type" ->
      T.Convert_element_type
        (dtype_of_string (U.to_string (member "new_dtype")))
  | "broadcast_in_dim" ->
      T.Broadcast_in_dim
        { shape = ia (member "shape"); dims = ia (member "dims") }
  | "reshape" -> T.Reshape (ia (member "new_sizes"))
  | "reduce_sum" -> T.Reduce_sum (ia (member "axes"))
  | "dot_general" ->
      let dn = U.to_list (member "dimension_numbers") in
      let contracting = U.to_list (List.nth dn 0) in
      let batch = U.to_list (List.nth dn 1) in
      T.Dot_general
        {
          lhs_contract = ia (List.nth contracting 0);
          rhs_contract = ia (List.nth contracting 1);
          lhs_batch = ia (List.nth batch 0);
          rhs_batch = ia (List.nth batch 1);
        }
  | "conv_general_dilated" ->
      let dn = U.to_list (member "dimension_numbers") in
      let padding =
        Array.of_list
          (List.map
             (fun t ->
               match U.to_list t with
               | [ a; b ] -> (U.to_int a, U.to_int b)
               | _ -> failwith "lax golden: bad conv padding")
             (U.to_list (member "padding")))
      in
      T.Conv_general_dilated
        {
          window_strides = ia (member "window_strides");
          padding;
          lhs_dilation = ia (member "lhs_dilation");
          rhs_dilation = ia (member "rhs_dilation");
          dimension_numbers =
            {
              lhs_spec = ia (List.nth dn 0);
              rhs_spec = ia (List.nth dn 1);
              out_spec = ia (List.nth dn 2);
            };
          feature_group_count = U.to_int (member "feature_group_count");
          batch_group_count = U.to_int (member "batch_group_count");
        }
  | "reduce_window_sum" -> T.Reduce_window_sum (window_of params)
  | "reduce_window_max" -> T.Reduce_window_max (window_of params)
  | "reduce_window_min" -> T.Reduce_window_min (window_of params)
  | "reduce_window" ->
      let sc = { T.shape = [||]; dtype = D.F32; weak_type = false } in
      let reducer =
        Ojax.Jaxpr.make_jaxpr [ sc; sc ] (fun args -> [ C.bind1 T.Mul args ])
      in
      T.Reduce_window { reducer; window = window_of params }
  | "select_and_gather_add" ->
      T.Select_and_gather_add
        { select = window_select params; window = window_of params }
  | "select_and_scatter_add" ->
      T.Select_and_scatter_add
        { select = window_select params; window = window_of params }
  | "bessel_i0e" -> T.Bessel_i0e
  | "bessel_i1e" -> T.Bessel_i1e
  | "digamma" -> T.Digamma
  | "erf" -> T.Erf
  | "erf_inv" -> T.Erf_inv
  | "erfc" -> T.Erfc
  | "igamma" -> T.Igamma
  | "igamma_grad_a" -> T.Igamma_grad_a
  | "igammac" -> T.Igammac
  | "lgamma" -> T.Lgamma
  | "polygamma" -> T.Polygamma
  | "regularized_incomplete_beta" -> T.Regularized_incomplete_beta
  | "zeta" -> T.Zeta
  | _ -> failwith ("lax golden: unknown op " ^ op)

let concrete = function
  | T.Concrete n -> n
  | T.Tracer _ -> failwith "lax golden: expected concrete result"

let lax_check_case ~set_dir ~x64 c () =
  let canon d = if x64 then d else Compare.canonical_dtype_x64_off d in
  let inputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "inputs") (c.case_id ^ ".npz"))
  in
  let outputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "outputs") (c.case_id ^ ".npz"))
  in
  let operands =
    List.map
      (fun a -> T.Concrete (nd_of_npz (find_member inputs a.name)))
      c.args
  in
  let results = C.bind (prim_of c.op c.params) operands in
  let paired =
    try List.combine c.outs results
    with Invalid_argument _ ->
      Alcotest.failf "%s: output arity mismatch" c.case_id
  in
  List.iter
    (fun (o, v) ->
      let ocompare = match o.ocompare with Some s -> s | None -> c.compare in
      let oatol = match o.oatol with Some t -> t | None -> c.atol in
      let ortol = match o.ortol with Some t -> t | None -> c.rtol in
      let oreason =
        match o.otreason with Some _ as r -> r | None -> c.treason
      in
      let nd = concrete v in
      if not (Compare.shapes_equal (Nd.shape nd) o.oshape) then
        Alcotest.failf "%s: output %s shape mismatch" c.case_id o.oname;
      if canon (string_of_dtype (Nd.dtype nd)) <> canon o.odtype then
        Alcotest.failf "%s: output %s dtype %s != %s" c.case_id o.oname
          (string_of_dtype (Nd.dtype nd))
          o.odtype;
      let golden = find_member outputs o.oname in
      let floats = read_nd nd in
      let data =
        if ocompare = "exact" then Npz.I (Array.map Int64.of_float floats)
        else Npz.F floats
      in
      let actual =
        { Npz.dtype = golden.Npz.dtype; shape = Nd.shape nd; data }
      in
      Compare.assert_tol_widened o.odtype oatol ortol oreason;
      Compare.check
        ~name:(c.case_id ^ ":" ^ o.oname)
        ~compare:ocompare ~atol:oatol ~rtol:ortol ~expected:golden ~actual)
    paired

let lax_suite_for set_name =
  let set_dir = Filename.concat (Filename.concat goldens_root "lax") set_name in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick (lax_check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("lax:" ^ set_name, coverage :: case_tests)

module J = Ojax.Jaxpr
module PP = Ojax.Pretty_printer

let av shape dtype : T.aval = { shape; dtype; weak_type = false }
let jb1 = C.bind1
let scalar_f32 x = T.Concrete (Nd.of_floats D.F32 [||] [| x |])

let jaxpr_builders :
    (string * T.aval list * (T.value list -> T.value list)) list =
  [
    ("sin", [ av [| 3 |] D.F32 ], fun a -> [ jb1 T.Sin a ]);
    ( "sin_mul",
      [ av [| 2; 3 |] D.F32; av [| 2; 3 |] D.F32 ],
      fun args ->
        match args with
        | [ x; y ] ->
            let c = jb1 T.Sin [ x ] in
            [ jb1 T.Mul [ c; y ] ]
        | _ -> assert false );
    ( "chain",
      [ av [| 4 |] D.F32 ],
      fun args ->
        match args with
        | [ x ] ->
            let n = jb1 T.Neg [ x ] in
            [ jb1 T.Exp [ n ] ]
        | _ -> assert false );
    ( "reduce",
      [ av [| 2; 3 |] D.F32 ],
      fun args -> [ jb1 (T.Reduce_sum [| 0 |]) args ] );
    ( "dot",
      [ av [| 2; 3 |] D.F32; av [| 3; 4 |] D.F32 ],
      fun args ->
        [
          jb1
            (T.Dot_general
               {
                 lhs_contract = [| 1 |];
                 rhs_contract = [| 0 |];
                 lhs_batch = [||];
                 rhs_batch = [||];
               })
            args;
        ] );
    ( "reshape",
      [ av [| 2; 3 |] D.F32 ],
      fun args -> [ jb1 (T.Reshape [| 6 |]) args ] );
    ( "broadcast",
      [ av [| 3 |] D.F32 ],
      fun args ->
        [ jb1 (T.Broadcast_in_dim { shape = [| 2; 3 |]; dims = [| 1 |] }) args ]
    );
    ( "convert",
      [ av [| 3 |] D.F32 ],
      fun args -> [ jb1 (T.Convert_element_type D.I32) args ] );
    ( "compare",
      [ av [| 3 |] D.F32; av [| 3 |] D.F32 ],
      fun args -> [ jb1 T.Lt args ] );
    ( "select",
      [ av [| 3 |] D.Bool; av [| 3 |] D.F32; av [| 3 |] D.F32 ],
      fun args -> [ jb1 T.Select_n args ] );
    ("lit_mul", [], fun _ -> [ jb1 T.Mul [ scalar_f32 2.0; scalar_f32 3.0 ] ]);
    ( "nested",
      [ av [| 3 |] D.F32; av [| 3 |] D.F32 ],
      fun args ->
        match args with
        | [ x; y ] ->
            let m = jb1 T.Mul [ x; y ] in
            let s = jb1 T.Sin [ x ] in
            [ jb1 T.Add [ m; s ] ]
        | _ -> assert false );
  ]

let load_jaxpr_manifest path =
  let j = Yojson.Safe.from_file path in
  U.member "cases" j |> U.to_list
  |> List.map (fun c ->
      (U.member "case_id" c |> U.to_string, U.member "text" c |> U.to_string))

let jaxpr_check_case want (_case_id, avals, f) () =
  let got = PP.closed_jaxpr_to_string (J.make_jaxpr avals f) in
  Alcotest.(check string) "jaxpr text" want got

let jaxpr_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "jaxpr") set_name
  in
  let cases = load_jaxpr_manifest (Filename.concat set_dir "manifest.json") in
  let builder_ids =
    List.map (fun (id, _, _) -> id) jaxpr_builders |> List.sort String.compare
  in
  let manifest_ids = List.map fst cases |> List.sort String.compare in
  let coverage () =
    if builder_ids <> manifest_ids then
      Alcotest.failf "jaxpr:%s coverage mismatch" set_name
  in
  let case_tests =
    List.map
      (fun (case_id, want) ->
        let builder =
          List.find (fun (id, _, _) -> id = case_id) jaxpr_builders
        in
        Alcotest.test_case case_id `Quick (jaxpr_check_case want builder))
      cases
  in
  ( "jaxpr:" ^ set_name,
    Alcotest.test_case "coverage" `Quick coverage :: case_tests )

module Ad = Ojax.Interpreters.Ad

let ad_builders : (string * (T.value list -> T.value list)) list =
  [
    ("sin", fun a -> [ jb1 T.Sin a ]);
    ( "sin_mul",
      fun a ->
        match a with
        | [ x; y ] -> [ jb1 T.Mul [ jb1 T.Sin [ x ]; y ] ]
        | _ -> assert false );
    ("exp_neg", fun a -> [ jb1 T.Exp [ jb1 T.Neg a ] ]);
    ("tanh", fun a -> [ jb1 T.Tanh a ]);
    ( "cubic",
      fun a ->
        match a with
        | [ x ] -> [ jb1 T.Sub [ jb1 T.Mul [ x; jb1 T.Mul [ x; x ] ]; x ] ]
        | _ -> assert false );
    ("div2", fun a -> [ jb1 T.Div a ]);
    ( "dot",
      fun a ->
        [
          jb1
            (T.Dot_general
               {
                 lhs_contract = [| 1 |];
                 rhs_contract = [| 0 |];
                 lhs_batch = [||];
                 rhs_batch = [||];
               })
            a;
        ] );
    ("max2", fun a -> [ jb1 T.Max a ]);
    ("reduce", fun a -> [ jb1 (T.Reduce_sum [| 0 |]) a ]);
    ( "bcast",
      fun a ->
        [ jb1 (T.Broadcast_in_dim { shape = [| 2; 3 |]; dims = [| 1 |] }) a ] );
    ("reshape_fn", fun a -> [ jb1 (T.Reshape [| 6 |]) a ]);
    ("sum_sin", fun a -> [ jb1 (T.Reduce_sum [| 0 |]) [ jb1 T.Sin a ] ]);
    ( "sum_cubic",
      fun a ->
        match a with
        | [ x ] ->
            [
              jb1 (T.Reduce_sum [| 0 |])
                [ jb1 T.Sub [ jb1 T.Mul [ x; jb1 T.Mul [ x; x ] ]; x ] ];
            ]
        | _ -> assert false );
    ("sum_max", fun a -> [ jb1 (T.Reduce_sum [| 0 |]) [ jb1 T.Max a ] ]);
    ( "bcast_sum",
      fun a ->
        [
          jb1
            (T.Reduce_sum [| 0; 1 |])
            [
              jb1 (T.Broadcast_in_dim { shape = [| 2; 3 |]; dims = [| 1 |] }) a;
            ];
        ] );
    ( "reshape_sum",
      fun a -> [ jb1 (T.Reduce_sum [| 0 |]) [ jb1 (T.Reshape [| 6 |]) a ] ] );
  ]

let ad_run mode fn primals tangents =
  match mode with
  | "jvp" ->
      let po, to_ = Ad.jvp fn primals tangents in
      po @ to_
  | "grad" -> [ List.hd (Ad.grad (fun a -> List.hd (fn a)) primals) ]
  | "grad2" ->
      [
        List.hd
          (Ad.grad
             (fun xs -> List.hd (Ad.grad (fun a -> List.hd (fn a)) xs))
             primals);
      ]
  | _ -> failwith ("ad golden: unknown mode " ^ mode)

type ad_case = {
  a_id : string;
  a_fn : string;
  a_mode : string;
  a_args : arg list;
  a_tans : arg list;
  a_outs : out list;
  a_compare : string;
  a_atol : float;
  a_rtol : float;
}

let parse_ad_case j =
  let tol = U.member "tol" j in
  {
    a_id = U.member "case_id" j |> U.to_string;
    a_fn = U.member "fn" j |> U.to_string;
    a_mode = U.member "mode" j |> U.to_string;
    a_args = U.member "args" j |> U.to_list |> List.map parse_arg;
    a_tans = U.member "tangents" j |> U.to_list |> List.map parse_arg;
    a_outs = U.member "outputs" j |> U.to_list |> List.map parse_out;
    a_compare = U.member "compare" j |> U.to_string;
    a_atol = U.member "atol" tol |> U.to_number;
    a_rtol = U.member "rtol" tol |> U.to_number;
  }

let load_ad_manifest path =
  let j = Yojson.Safe.from_file path in
  ( U.member "x64" j |> U.to_bool,
    U.member "cases" j |> U.to_list |> List.map parse_ad_case )

let ad_check_case ~set_dir ~x64 c () =
  let canon d = if x64 then d else Compare.canonical_dtype_x64_off d in
  let inputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "inputs") (c.a_id ^ ".npz"))
  in
  let outputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "outputs") (c.a_id ^ ".npz"))
  in
  let read_operand (a : arg) =
    T.Concrete (nd_of_npz (find_member inputs a.name))
  in
  let primals = List.map read_operand c.a_args in
  let tangents = List.map read_operand c.a_tans in
  let fn =
    match List.assoc_opt c.a_fn ad_builders with
    | Some f -> f
    | None -> Alcotest.failf "%s: unknown fn %s" c.a_id c.a_fn
  in
  let results = ad_run c.a_mode fn primals tangents in
  let paired =
    try List.combine c.a_outs results
    with Invalid_argument _ ->
      Alcotest.failf "%s: output arity mismatch" c.a_id
  in
  List.iter
    (fun (o, v) ->
      let nd = concrete v in
      if not (Compare.shapes_equal (Nd.shape nd) o.oshape) then
        Alcotest.failf "%s: output %s shape mismatch" c.a_id o.oname;
      if canon (string_of_dtype (Nd.dtype nd)) <> canon o.odtype then
        Alcotest.failf "%s: output %s dtype %s != %s" c.a_id o.oname
          (string_of_dtype (Nd.dtype nd))
          o.odtype;
      let golden = find_member outputs o.oname in
      let actual =
        {
          Npz.dtype = golden.Npz.dtype;
          shape = Nd.shape nd;
          data = Npz.F (read_nd nd);
        }
      in
      Compare.assert_tol o.odtype c.a_atol c.a_rtol;
      Compare.check
        ~name:(c.a_id ^ ":" ^ o.oname)
        ~compare:c.a_compare ~atol:c.a_atol ~rtol:c.a_rtol ~expected:golden
        ~actual)
    paired

let ad_check_coverage ~set_dir cases () =
  let expected = List.map (fun c -> c.a_id) cases |> List.sort String.compare in
  List.iter
    (fun sub ->
      let got = dir_case_ids (Filename.concat set_dir sub) in
      if got <> expected then Alcotest.failf "%s: ad coverage mismatch" sub)
    [ "inputs"; "outputs" ]

let ad_suite_for set_name =
  let set_dir = Filename.concat (Filename.concat goldens_root "ad") set_name in
  let x64, cases = load_ad_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.a_id `Quick (ad_check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (ad_check_coverage ~set_dir cases)
  in
  ("ad:" ^ set_name, coverage :: case_tests)

module Batching = Ojax.Interpreters.Batching

let cubic_fn a =
  match a with
  | [ z ] -> [ jb1 T.Sub [ jb1 T.Mul [ z; jb1 T.Mul [ z; z ] ]; z ] ]
  | _ -> assert false

let sum_sin_fn a = [ jb1 (T.Reduce_sum [| 0 |]) [ jb1 T.Sin a ] ]

let jvp_tangent g a =
  match a with
  | [ x; t ] ->
      let _, to_ = Ad.jvp g [ x ] [ t ] in
      [ List.hd to_ ]
  | _ -> assert false

let batch_builders : (string * (T.value list -> T.value list)) list =
  [
    ("sin", fun a -> [ jb1 T.Sin a ]);
    ("neg", fun a -> [ jb1 T.Neg a ]);
    ("exp", fun a -> [ jb1 T.Exp a ]);
    ("tanh", fun a -> [ jb1 T.Tanh a ]);
    ("add", fun a -> [ jb1 T.Add a ]);
    ("mul", fun a -> [ jb1 T.Mul a ]);
    ("sub", fun a -> [ jb1 T.Sub a ]);
    ("div", fun a -> [ jb1 T.Div a ]);
    ("max", fun a -> [ jb1 T.Max a ]);
    ("min", fun a -> [ jb1 T.Min a ]);
    ("sum_sin", sum_sin_fn);
    ( "bcast",
      fun a ->
        [ jb1 (T.Broadcast_in_dim { shape = [| 2; 3 |]; dims = [| 1 |] }) a ] );
    ("reshape_fn", fun a -> [ jb1 (T.Reshape [| 6 |]) a ]);
    ("convert", fun a -> [ jb1 (T.Convert_element_type D.I32) a ]);
    ("select", fun a -> [ jb1 T.Select_n a ]);
    ("jvp_sin", jvp_tangent (fun z -> [ jb1 T.Sin z ]));
    ("jvp_cubic", jvp_tangent cubic_fn);
    ("jvp_sum_sin", jvp_tangent sum_sin_fn);
  ]

type batch_case = {
  b_id : string;
  b_fn : string;
  b_in_axes : int option list;
  b_args : arg list;
  b_outs : out list;
  b_compare : string;
  b_atol : float;
  b_rtol : float;
}

let parse_in_axis j = match j with `Null -> None | _ -> Some (U.to_int j)

let parse_batch_case j =
  let tol = U.member "tol" j in
  {
    b_id = U.member "case_id" j |> U.to_string;
    b_fn = U.member "fn" j |> U.to_string;
    b_in_axes = U.member "in_axes" j |> U.to_list |> List.map parse_in_axis;
    b_args = U.member "args" j |> U.to_list |> List.map parse_arg;
    b_outs = U.member "outputs" j |> U.to_list |> List.map parse_out;
    b_compare = U.member "compare" j |> U.to_string;
    b_atol = U.member "atol" tol |> U.to_number;
    b_rtol = U.member "rtol" tol |> U.to_number;
  }

let load_batch_manifest path =
  let j = Yojson.Safe.from_file path in
  ( U.member "x64" j |> U.to_bool,
    U.member "cases" j |> U.to_list |> List.map parse_batch_case )

let batch_check_case ~set_dir ~x64 c () =
  let canon d = if x64 then d else Compare.canonical_dtype_x64_off d in
  let inputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "inputs") (c.b_id ^ ".npz"))
  in
  let outputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "outputs") (c.b_id ^ ".npz"))
  in
  let operands =
    List.map
      (fun a -> T.Concrete (nd_of_npz (find_member inputs a.name)))
      c.b_args
  in
  let fn =
    match List.assoc_opt c.b_fn batch_builders with
    | Some f -> f
    | None -> Alcotest.failf "%s: unknown fn %s" c.b_id c.b_fn
  in
  let results = Batching.vmap fn c.b_in_axes operands in
  let paired =
    try List.combine c.b_outs results
    with Invalid_argument _ ->
      Alcotest.failf "%s: output arity mismatch" c.b_id
  in
  List.iter
    (fun (o, v) ->
      let nd = concrete v in
      if not (Compare.shapes_equal (Nd.shape nd) o.oshape) then
        Alcotest.failf "%s: output %s shape mismatch" c.b_id o.oname;
      if canon (string_of_dtype (Nd.dtype nd)) <> canon o.odtype then
        Alcotest.failf "%s: output %s dtype %s != %s" c.b_id o.oname
          (string_of_dtype (Nd.dtype nd))
          o.odtype;
      let golden = find_member outputs o.oname in
      let floats = read_nd nd in
      let data =
        if c.b_compare = "exact" then Npz.I (Array.map Int64.of_float floats)
        else Npz.F floats
      in
      let actual =
        { Npz.dtype = golden.Npz.dtype; shape = Nd.shape nd; data }
      in
      Compare.assert_tol o.odtype c.b_atol c.b_rtol;
      Compare.check
        ~name:(c.b_id ^ ":" ^ o.oname)
        ~compare:c.b_compare ~atol:c.b_atol ~rtol:c.b_rtol ~expected:golden
        ~actual)
    paired

let batch_check_coverage ~set_dir cases () =
  let expected = List.map (fun c -> c.b_id) cases |> List.sort String.compare in
  List.iter
    (fun sub ->
      let got = dir_case_ids (Filename.concat set_dir sub) in
      if got <> expected then
        Alcotest.failf "%s: batching coverage mismatch" sub)
    [ "inputs"; "outputs" ]

let batch_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "batching") set_name
  in
  let x64, cases =
    load_batch_manifest (Filename.concat set_dir "manifest.json")
  in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.b_id `Quick (batch_check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (batch_check_coverage ~set_dir cases)
  in
  ("batching:" ^ set_name, coverage :: case_tests)

module ApiM = Ojax.Api

let api_add a b = jb1 T.Add [ a; b ]
let api_mul a b = jb1 T.Mul [ a; b ]
let api_sin a = jb1 T.Sin [ a ]

let scalar_like (v : T.value) (x : float) : T.value =
  let dt = (C.get_aval v).T.dtype in
  T.Concrete (Nd.of_floats dt [||] [| x |])

let api_builders : (string * (T.value list -> T.value list)) list =
  [
    ("sin", fun a -> [ jb1 T.Sin a ]);
    ( "cubic",
      fun a ->
        match a with
        | [ x ] -> [ jb1 T.Sub [ jb1 T.Mul [ x; jb1 T.Mul [ x; x ] ]; x ] ]
        | _ -> assert false );
    ("exp_neg", fun a -> [ jb1 T.Exp [ jb1 T.Neg a ] ]);
    ("tanh", fun a -> [ jb1 T.Tanh a ]);
    ( "sin_mul",
      fun a ->
        match a with
        | [ x; y ] -> [ jb1 T.Mul [ jb1 T.Sin [ x ]; y ] ]
        | _ -> assert false );
    ("sum_sin", fun a -> [ jb1 (T.Reduce_sum [| 0 |]) [ jb1 T.Sin a ] ]);
  ]

let nested_calls_2 (x : T.value) : T.value =
  let one = scalar_like x 1.0 in
  let bar (y : T.value) : T.value =
    let baz (w : T.value) : T.value =
      let q = List.hd (ApiM.call (fun _ -> [ y ]) [ x ]) in
      let q = api_add q (List.hd (ApiM.call (fun _ -> [ y ]) [])) in
      let q =
        api_add q
          (List.hd (ApiM.call (fun a -> [ api_add w (List.hd a) ]) [ y ]))
      in
      let inner =
        List.hd
          (ApiM.call
             (fun _ ->
               [
                 api_mul
                   (List.hd
                      (ApiM.call (fun b -> [ api_sin (List.hd b) ]) [ x ]))
                   y;
               ])
             [ one ])
      in
      api_add inner q
    in
    let p, t = Ad.jvp (fun a -> [ baz (List.hd a) ]) [ api_add x one ] [ y ] in
    api_add (List.hd t) (api_mul x (List.hd p))
  in
  List.hd (ApiM.call (fun a -> [ bar (List.hd a) ]) [ x ])

let api_run mode fn in_axes primals tangents =
  match mode with
  | "jit" -> ApiM.jit_flat fn primals
  | "jvp_jit" ->
      let po, to_ = Ad.jvp (ApiM.jit_flat fn) primals tangents in
      po @ to_
  | "vmap_jit" -> Batching.vmap (ApiM.jit_flat fn) in_axes primals
  | "grad_jit" -> Ad.grad (fun a -> List.hd (ApiM.jit_flat fn a)) primals
  | "nested2_jit" ->
      ApiM.jit_flat (fun a -> [ nested_calls_2 (List.hd a) ]) primals
  | "nested2_vmap" ->
      Batching.vmap (fun a -> [ nested_calls_2 (List.hd a) ]) in_axes primals
  | _ -> failwith ("api golden: unknown mode " ^ mode)

type api_case = {
  ap_id : string;
  ap_fn : string;
  ap_mode : string;
  ap_in_axes : int option list;
  ap_args : arg list;
  ap_tans : arg list;
  ap_outs : out list;
  ap_compare : string;
  ap_atol : float;
  ap_rtol : float;
}

let parse_api_case j =
  let tol = U.member "tol" j in
  let in_axes_j = U.member "in_axes" j in
  {
    ap_id = U.member "case_id" j |> U.to_string;
    ap_fn = U.member "fn" j |> U.to_string;
    ap_mode = U.member "mode" j |> U.to_string;
    ap_in_axes =
      (match in_axes_j with
      | `Null -> []
      | _ -> U.to_list in_axes_j |> List.map parse_in_axis);
    ap_args = U.member "args" j |> U.to_list |> List.map parse_arg;
    ap_tans = U.member "tangents" j |> U.to_list |> List.map parse_arg;
    ap_outs = U.member "outputs" j |> U.to_list |> List.map parse_out;
    ap_compare = U.member "compare" j |> U.to_string;
    ap_atol = U.member "atol" tol |> U.to_number;
    ap_rtol = U.member "rtol" tol |> U.to_number;
  }

let load_api_manifest path =
  let j = Yojson.Safe.from_file path in
  ( U.member "x64" j |> U.to_bool,
    U.member "cases" j |> U.to_list |> List.map parse_api_case )

let api_check_case ~set_dir ~x64 c () =
  let canon d = if x64 then d else Compare.canonical_dtype_x64_off d in
  let inputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "inputs") (c.ap_id ^ ".npz"))
  in
  let outputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "outputs") (c.ap_id ^ ".npz"))
  in
  let read_operand (a : arg) =
    T.Concrete (nd_of_npz (find_member inputs a.name))
  in
  let primals = List.map read_operand c.ap_args in
  let tangents = List.map read_operand c.ap_tans in
  let fn =
    match List.assoc_opt c.ap_fn api_builders with
    | Some f -> f
    | None -> fun _ -> assert false
  in
  let results = api_run c.ap_mode fn c.ap_in_axes primals tangents in
  let paired =
    try List.combine c.ap_outs results
    with Invalid_argument _ ->
      Alcotest.failf "%s: output arity mismatch" c.ap_id
  in
  List.iter
    (fun (o, v) ->
      let nd = concrete v in
      if not (Compare.shapes_equal (Nd.shape nd) o.oshape) then
        Alcotest.failf "%s: output %s shape mismatch" c.ap_id o.oname;
      if canon (string_of_dtype (Nd.dtype nd)) <> canon o.odtype then
        Alcotest.failf "%s: output %s dtype %s != %s" c.ap_id o.oname
          (string_of_dtype (Nd.dtype nd))
          o.odtype;
      let golden = find_member outputs o.oname in
      let actual =
        {
          Npz.dtype = golden.Npz.dtype;
          shape = Nd.shape nd;
          data = Npz.F (read_nd nd);
        }
      in
      Compare.assert_tol o.odtype c.ap_atol c.ap_rtol;
      Compare.check
        ~name:(c.ap_id ^ ":" ^ o.oname)
        ~compare:c.ap_compare ~atol:c.ap_atol ~rtol:c.ap_rtol ~expected:golden
        ~actual)
    paired

let api_check_coverage ~set_dir cases () =
  let expected =
    List.map (fun c -> c.ap_id) cases |> List.sort String.compare
  in
  List.iter
    (fun sub ->
      let got = dir_case_ids (Filename.concat set_dir sub) in
      if got <> expected then Alcotest.failf "%s: api coverage mismatch" sub)
    [ "inputs"; "outputs" ]

let api_suite_for set_name =
  let set_dir = Filename.concat (Filename.concat goldens_root "api") set_name in
  let x64, cases =
    load_api_manifest (Filename.concat set_dir "manifest.json")
  in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.ap_id `Quick (api_check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (api_check_coverage ~set_dir cases)
  in
  ("api:" ^ set_name, coverage :: case_tests)

let cond_branches :
    (string * (T.value list -> T.value list) * (T.value list -> T.value list))
    list =
  [
    ("sin_cos", (fun a -> [ jb1 T.Sin a ]), fun a -> [ jb1 T.Cos a ]);
    ("mul_add", (fun a -> [ jb1 T.Mul a ]), fun a -> [ jb1 T.Add a ]);
    ( "sq_neg",
      (fun a ->
        match a with [ x ] -> [ jb1 T.Mul [ x; x ] ] | _ -> assert false),
      fun a -> [ jb1 T.Neg a ] );
  ]

type cond_case = {
  cn_id : string;
  cn_fn : string;
  cn_mode : string;
  cn_pred : int option;
  cn_platforms : string array option array;
  cn_in_axes : int option list;
  cn_args : arg list;
  cn_tans : arg list;
  cn_outs : out list;
  cn_compare : string;
  cn_atol : float;
  cn_rtol : float;
}

let parse_platforms j =
  match j with
  | `Null -> [||]
  | _ ->
      Array.of_list
        (List.map
           (fun p ->
             match p with
             | `Null -> None
             | _ -> Some (Array.of_list (List.map U.to_string (U.to_list p))))
           (U.to_list j))

let parse_cond_case j =
  let tol = U.member "tol" j in
  let in_axes_j = U.member "in_axes" j in
  {
    cn_id = U.member "case_id" j |> U.to_string;
    cn_fn = U.member "fn" j |> U.to_string;
    cn_mode = U.member "mode" j |> U.to_string;
    cn_pred =
      (match U.member "pred" j with `Null -> None | v -> Some (U.to_int v));
    cn_platforms = parse_platforms (U.member "platforms" j);
    cn_in_axes =
      (match in_axes_j with
      | `Null -> []
      | _ -> U.to_list in_axes_j |> List.map parse_in_axis);
    cn_args = U.member "args" j |> U.to_list |> List.map parse_arg;
    cn_tans = U.member "tangents" j |> U.to_list |> List.map parse_arg;
    cn_outs = U.member "outputs" j |> U.to_list |> List.map parse_out;
    cn_compare = U.member "compare" j |> U.to_string;
    cn_atol = U.member "atol" tol |> U.to_number;
    cn_rtol = U.member "rtol" tol |> U.to_number;
  }

let load_cond_manifest path =
  let j = Yojson.Safe.from_file path in
  ( U.member "x64" j |> U.to_bool,
    U.member "cases" j |> U.to_list |> List.map parse_cond_case )

let cond_pred_value pred =
  T.Concrete (Nd.of_floats D.Bool [||] [| (if pred <> 0 then 1.0 else 0.0) |])

let cond_run c primals tangents =
  match c.cn_mode with
  | "platform" -> [ Ojax.Lax.platform_index ~platforms:c.cn_platforms ]
  | _ -> (
      let _, tf, ff =
        match List.find_opt (fun (n, _, _) -> n = c.cn_fn) cond_branches with
        | Some x -> x
        | None -> Alcotest.failf "%s: unknown cond fn %s" c.cn_id c.cn_fn
      in
      let pred = cond_pred_value (Option.get c.cn_pred) in
      let wrapped ops = Ojax.Lax.cond pred tf ff ops in
      match c.cn_mode with
      | "eval" -> wrapped primals
      | "jvp" ->
          let po, to_ = Ad.jvp wrapped primals tangents in
          po @ to_
      | "grad" -> Ad.grad (fun a -> List.hd (wrapped a)) primals
      | "vmap" -> Batching.vmap wrapped c.cn_in_axes primals
      | m -> failwith ("conditionals: unknown mode " ^ m))

let cond_check_case ~set_dir ~x64 c () =
  let canon d = if x64 then d else Compare.canonical_dtype_x64_off d in
  let inputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "inputs") (c.cn_id ^ ".npz"))
  in
  let outputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "outputs") (c.cn_id ^ ".npz"))
  in
  let read_operand (a : arg) =
    T.Concrete (nd_of_npz (find_member inputs a.name))
  in
  let primals = List.map read_operand c.cn_args in
  let tangents = List.map read_operand c.cn_tans in
  let results = cond_run c primals tangents in
  let paired =
    try List.combine c.cn_outs results
    with Invalid_argument _ ->
      Alcotest.failf "%s: output arity mismatch" c.cn_id
  in
  List.iter
    (fun (o, v) ->
      let nd = concrete v in
      if not (Compare.shapes_equal (Nd.shape nd) o.oshape) then
        Alcotest.failf "%s: output %s shape mismatch" c.cn_id o.oname;
      if canon (string_of_dtype (Nd.dtype nd)) <> canon o.odtype then
        Alcotest.failf "%s: output %s dtype %s != %s" c.cn_id o.oname
          (string_of_dtype (Nd.dtype nd))
          o.odtype;
      let golden = find_member outputs o.oname in
      let floats = read_nd nd in
      let data =
        if c.cn_compare = "exact" then Npz.I (Array.map Int64.of_float floats)
        else Npz.F floats
      in
      let actual =
        { Npz.dtype = golden.Npz.dtype; shape = Nd.shape nd; data }
      in
      Compare.assert_tol o.odtype c.cn_atol c.cn_rtol;
      Compare.check
        ~name:(c.cn_id ^ ":" ^ o.oname)
        ~compare:c.cn_compare ~atol:c.cn_atol ~rtol:c.cn_rtol ~expected:golden
        ~actual)
    paired

let cond_check_coverage ~set_dir cases () =
  let expected =
    List.map (fun c -> c.cn_id) cases |> List.sort String.compare
  in
  List.iter
    (fun sub ->
      let got = dir_case_ids (Filename.concat set_dir sub) in
      if got <> expected then
        Alcotest.failf "%s: conditionals coverage mismatch" sub)
    [ "inputs"; "outputs" ]

let cond_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "conditionals") set_name
  in
  let x64, cases =
    load_cond_manifest (Filename.concat set_dir "manifest.json")
  in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.cn_id `Quick (cond_check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (cond_check_coverage ~set_dir cases)
  in
  ("conditionals:" ^ set_name, coverage :: case_tests)

let loops_bodies : (string * int * (T.value list -> T.value list)) list =
  [
    ( "cumsum",
      1,
      fun l ->
        match l with
        | [ c; x ] ->
            let s = jb1 T.Add [ c; x ] in
            [ s; s ]
        | _ -> assert false );
    ( "cumprod",
      1,
      fun l ->
        match l with [ c; x ] -> [ jb1 T.Mul [ c; x ]; c ] | _ -> assert false
    );
    ( "lin",
      1,
      fun l ->
        match l with
        | [ c; x ] -> [ jb1 T.Add [ c; x ]; jb1 T.Sin [ c ] ]
        | _ -> assert false );
    ( "twocarry",
      2,
      fun l ->
        match l with
        | [ a; b; x ] -> [ jb1 T.Add [ a; x ]; jb1 T.Mul [ b; x ]; a ]
        | _ -> assert false );
  ]

let loops_while :
    (string * int * (T.value list -> T.value) * (T.value list -> T.value list))
    list =
  [
    ( "wdouble",
      1,
      (fun l ->
        match l with
        | [ v ] -> jb1 T.Lt [ v; scalar_like v 8.0 ]
        | _ -> assert false),
      fun l ->
        match l with [ v ] -> [ jb1 T.Add [ v; v ] ] | _ -> assert false );
    ( "wtwo",
      2,
      (fun l ->
        match l with
        | [ a; _ ] -> jb1 T.Lt [ a; scalar_like a 20.0 ]
        | _ -> assert false),
      fun l ->
        match l with [ a; b ] -> [ jb1 T.Add [ a; b ]; b ] | _ -> assert false
    );
  ]

type loops_case = {
  lp_id : string;
  lp_fn : string;
  lp_kind : string;
  lp_mode : string;
  lp_reverse : bool;
  lp_axis : int;
  lp_num_carry : int;
  lp_in_axes : int option list;
  lp_args : arg list;
  lp_tans : arg list;
  lp_outs : out list;
  lp_compare : string;
  lp_atol : float;
  lp_rtol : float;
}

let parse_loops_case j =
  let tol = U.member "tol" j in
  let in_axes_j = U.member "in_axes" j in
  {
    lp_id = U.member "case_id" j |> U.to_string;
    lp_fn = U.member "fn" j |> U.to_string;
    lp_kind =
      (match U.member "kind" j with `Null -> "scan" | k -> U.to_string k);
    lp_mode = U.member "mode" j |> U.to_string;
    lp_reverse = U.member "reverse" j |> U.to_bool;
    lp_axis = (match U.member "axis" j with `Null -> 0 | a -> U.to_int a);
    lp_num_carry = U.member "num_carry" j |> U.to_int;
    lp_in_axes =
      (match in_axes_j with
      | `Null -> []
      | _ -> U.to_list in_axes_j |> List.map parse_in_axis);
    lp_args = U.member "args" j |> U.to_list |> List.map parse_arg;
    lp_tans = U.member "tangents" j |> U.to_list |> List.map parse_arg;
    lp_outs = U.member "outputs" j |> U.to_list |> List.map parse_out;
    lp_compare = U.member "compare" j |> U.to_string;
    lp_atol = U.member "atol" tol |> U.to_number;
    lp_rtol = U.member "rtol" tol |> U.to_number;
  }

let load_loops_manifest path =
  let j = Yojson.Safe.from_file path in
  ( U.member "x64" j |> U.to_bool,
    U.member "cases" j |> U.to_list |> List.map parse_loops_case )

let loops_split_at n l =
  let rec go n l acc =
    if n = 0 then (List.rev acc, l)
    else
      match l with
      | x :: tl -> go (n - 1) tl (x :: acc)
      | [] -> (List.rev acc, [])
  in
  go n l []

let loops_run c primals tangents =
  let wrapped =
    if c.lp_kind = "cumulative" then
      let f =
        match c.lp_fn with
        | "cumsum" -> Ojax.Lax.cumsum
        | "cumprod" -> Ojax.Lax.cumprod
        | "cummax" -> Ojax.Lax.cummax
        | "cummin" -> Ojax.Lax.cummin
        | "cumlogsumexp" -> Ojax.Lax.cumlogsumexp
        | _ -> Alcotest.failf "%s: unknown cumulative fn %s" c.lp_id c.lp_fn
      in
      fun inputs ->
        match inputs with
        | [ x ] -> [ f ~axis:c.lp_axis ~reverse:c.lp_reverse x ]
        | _ -> Alcotest.failf "%s: cumulative expects 1 operand" c.lp_id
    else if c.lp_kind = "while" then
      match List.find_opt (fun (n, _, _, _) -> n = c.lp_fn) loops_while with
      | Some (_, _, cond_f, body_f) ->
          fun inputs -> Ojax.Lax.while_loop cond_f body_f inputs
      | None -> Alcotest.failf "%s: unknown while fn %s" c.lp_id c.lp_fn
    else
      let _, num_carry, body =
        match List.find_opt (fun (n, _, _) -> n = c.lp_fn) loops_bodies with
        | Some x -> x
        | None -> Alcotest.failf "%s: unknown loops fn %s" c.lp_id c.lp_fn
      in
      let reverse = c.lp_reverse in
      fun inputs ->
        let init, xs = loops_split_at num_carry inputs in
        Ojax.Lax.scan ~reverse body init xs
  in
  match c.lp_mode with
  | "eval" -> wrapped primals
  | "jvp" ->
      let po, to_ = Ad.jvp wrapped primals tangents in
      po @ to_
  | "vmap" -> Batching.vmap wrapped c.lp_in_axes primals
  | m -> failwith ("loops: unknown mode " ^ m)

let loops_check_case ~set_dir ~x64 c () =
  let canon d = if x64 then d else Compare.canonical_dtype_x64_off d in
  let inputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "inputs") (c.lp_id ^ ".npz"))
  in
  let outputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "outputs") (c.lp_id ^ ".npz"))
  in
  let read_operand (a : arg) =
    T.Concrete (nd_of_npz (find_member inputs a.name))
  in
  let primals = List.map read_operand c.lp_args in
  let tangents = List.map read_operand c.lp_tans in
  let results = loops_run c primals tangents in
  let paired =
    try List.combine c.lp_outs results
    with Invalid_argument _ ->
      Alcotest.failf "%s: output arity mismatch" c.lp_id
  in
  List.iter
    (fun (o, v) ->
      let nd = concrete v in
      if not (Compare.shapes_equal (Nd.shape nd) o.oshape) then
        Alcotest.failf "%s: output %s shape mismatch" c.lp_id o.oname;
      if canon (string_of_dtype (Nd.dtype nd)) <> canon o.odtype then
        Alcotest.failf "%s: output %s dtype %s != %s" c.lp_id o.oname
          (string_of_dtype (Nd.dtype nd))
          o.odtype;
      let golden = find_member outputs o.oname in
      let floats = read_nd nd in
      let data =
        if c.lp_compare = "exact" then Npz.I (Array.map Int64.of_float floats)
        else Npz.F floats
      in
      let actual =
        { Npz.dtype = golden.Npz.dtype; shape = Nd.shape nd; data }
      in
      Compare.assert_tol o.odtype c.lp_atol c.lp_rtol;
      Compare.check
        ~name:(c.lp_id ^ ":" ^ o.oname)
        ~compare:c.lp_compare ~atol:c.lp_atol ~rtol:c.lp_rtol ~expected:golden
        ~actual)
    paired

let loops_check_coverage ~set_dir cases () =
  let expected =
    List.map (fun c -> c.lp_id) cases |> List.sort String.compare
  in
  List.iter
    (fun sub ->
      let got = dir_case_ids (Filename.concat set_dir sub) in
      if got <> expected then Alcotest.failf "%s: loops coverage mismatch" sub)
    [ "inputs"; "outputs" ]

let loops_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "loops") set_name
  in
  let x64, cases =
    load_loops_manifest (Filename.concat set_dir "manifest.json")
  in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.lp_id `Quick (loops_check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (loops_check_coverage ~set_dir cases)
  in
  ("loops:" ^ set_name, coverage :: case_tests)

let must_fail msg f =
  match f () with
  | () -> Alcotest.failf "expected failure: %s" msg
  | exception Failure _ -> ()

let f32 xs =
  { Npz.dtype = "float32"; shape = [| Array.length xs |]; data = Npz.F xs }

let i32 xs =
  { Npz.dtype = "int32"; shape = [| Array.length xs |]; data = Npz.I xs }

let chk ~compare ~atol ~rtol a b =
  Compare.check ~name:"t" ~compare ~atol ~rtol ~expected:a ~actual:b

let compare_tests () =
  chk ~compare:"allclose" ~atol:1e-6 ~rtol:1e-6
    (f32 [| 1.0; 2.0 |])
    (f32 [| 1.0; 2.0000001 |]);
  must_fail "beyond tol" (fun () ->
      chk ~compare:"allclose" ~atol:1e-6 ~rtol:1e-6 (f32 [| 1.0 |])
        (f32 [| 1.1 |]));
  chk ~compare:"allclose" ~atol:0.0 ~rtol:0.0 (f32 [| Float.nan |])
    (f32 [| Float.nan |]);
  must_fail "nan vs value" (fun () ->
      chk ~compare:"allclose" ~atol:0.0 ~rtol:0.0 (f32 [| Float.nan |])
        (f32 [| 1.0 |]));
  chk ~compare:"allclose" ~atol:0.0 ~rtol:0.0 (f32 [| Float.infinity |])
    (f32 [| Float.infinity |]);
  must_fail "inf sign" (fun () ->
      chk ~compare:"allclose" ~atol:0.0 ~rtol:0.0 (f32 [| Float.infinity |])
        (f32 [| Float.neg_infinity |]));
  chk ~compare:"exact" ~atol:0.0 ~rtol:0.0 (i32 [| 1L; 2L |]) (i32 [| 1L; 2L |]);
  must_fail "int mismatch" (fun () ->
      chk ~compare:"exact" ~atol:0.0 ~rtol:0.0 (i32 [| 1L |]) (i32 [| 2L |]));
  must_fail "shape mismatch" (fun () ->
      chk ~compare:"exact" ~atol:0.0 ~rtol:0.0 (i32 [| 1L |]) (i32 [| 1L; 1L |]));
  must_fail "tol table" (fun () -> Compare.assert_tol "float32" 1e-3 1e-3)

let () =
  Ojax.Lax.install ();
  Alcotest.run "goldens"
    [
      suite_for "dtypes" "x64_off";
      suite_for "dtypes" "x64_on";
      lax_suite_for "x64_off";
      lax_suite_for "x64_on";
      jaxpr_suite_for "x64_off";
      jaxpr_suite_for "x64_on";
      ad_suite_for "x64_off";
      ad_suite_for "x64_on";
      batch_suite_for "x64_off";
      batch_suite_for "x64_on";
      api_suite_for "x64_off";
      api_suite_for "x64_on";
      cond_suite_for "x64_off";
      cond_suite_for "x64_on";
      loops_suite_for "x64_off";
      loops_suite_for "x64_on";
      ("compare", [ Alcotest.test_case "semantics" `Quick compare_tests ]);
    ]
