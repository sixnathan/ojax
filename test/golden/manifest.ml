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
  primitive : string;
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
    primitive = U.member "primitive" j |> U.to_string;
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
  | "uint32" -> D.Uint32
  | "complex64" -> D.Complex64
  | "complex128" -> D.Complex128
  | s -> failwith ("lax golden: unsupported dtype " ^ s)

let string_of_dtype = function
  | D.F32 -> "float32"
  | D.F64 -> "float64"
  | D.I32 -> "int32"
  | D.I64 -> "int64"
  | D.Bool -> "bool"
  | D.Uint32 -> "uint32"
  | D.Complex64 -> "complex64"
  | D.Complex128 -> "complex128"

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
  | T.Device _ -> failwith "lax golden: expected concrete result"

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

module NL = Ojax.Numpy.Lax_numpy
module UF = Ojax.Numpy.Ufuncs
module RED = Ojax.Numpy.Reductions
module IDX = Ojax.Numpy.Indexing
module AM = Ojax.Numpy.Array_methods

let opt_ia params name =
  match U.member name params with `Null -> None | j -> Some (ia j)

let opt_f params name =
  match U.member name params with `Null -> None | j -> Some (U.to_number j)

let opt_i params name =
  match U.member name params with `Null -> None | j -> Some (U.to_int j)

let opt_b params name =
  match U.member name params with `Null -> None | j -> Some (U.to_bool j)

let opt_dt params name =
  match U.member name params with
  | `Null -> None
  | j -> Some (dtype_of_string (U.to_string j))

let sections_of params =
  match U.member "indices" params with
  | `Null -> NL.Count (U.to_int (U.member "sections" params))
  | j -> NL.Indices (ia j)

let int_or_ia params name =
  match U.member name params with `List _ as j -> ia j | j -> [| U.to_int j |]

let opt_int_or_ia params name =
  match U.member name params with
  | `Null -> None
  | `List _ as j -> Some (ia j)
  | j -> Some [| U.to_int j |]

let numpy_fn op params operands : T.value list =
  let member name = U.member name params in
  let one v = [ v ] in
  match (op, operands) with
  | "transpose", [ x ] -> one (NL.transpose ?axes:(opt_ia params "axes") x)
  | "permute_dims", [ x ] -> one (NL.permute_dims x (ia (member "axes")))
  | "matrix_transpose", [ x ] -> one (NL.matrix_transpose x)
  | "flip", [ x ] -> one (NL.flip ?axis:(opt_ia params "axis") x)
  | "fliplr", [ x ] -> one (NL.fliplr x)
  | "flipud", [ x ] -> one (NL.flipud x)
  | "reshape", [ x ] -> one (NL.reshape x (ia (member "shape")))
  | "ravel", [ x ] -> one (NL.ravel x)
  | "rot90", [ x ] ->
      let axes =
        match ia (member "axes") with
        | [| a; b |] -> (a, b)
        | _ -> failwith "numpy golden: bad rot90 axes"
      in
      one (NL.rot90 ~k:(U.to_int (member "k")) ~axes x)
  | "trunc", [ x ] -> one (NL.trunc x)
  | "fmax", [ a; b ] -> one (NL.fmax a b)
  | "fmin", [ a; b ] -> one (NL.fmin a b)
  | "diff", [ x ] ->
      one
        (NL.diff ~n:(U.to_int (member "n")) ~axis:(U.to_int (member "axis")) x)
  | "ediff1d", [ x ] -> one (NL.ediff1d x)
  | "angle", [ x ] -> one (NL.angle ~deg:(U.to_bool (member "deg")) x)
  | "iscomplex", [ x ] -> one (NL.iscomplex x)
  | "isreal", [ x ] -> one (NL.isreal x)
  | "convolve", [ a; b ] ->
      one (NL.convolve ~mode:(U.to_string (member "mode")) a b)
  | "correlate", [ a; b ] ->
      one (NL.correlate ~mode:(U.to_string (member "mode")) a b)
  | "allclose", [ a; b ] -> one (NL.allclose a b)
  | "isclose", [ a; b ] -> one (NL.isclose a b)
  | "clip", [ x ] ->
      one (NL.clip ?min:(opt_f params "min") ?max:(opt_f params "max") x)
  | "round", [ x ] -> one (NL.round ?decimals:(opt_i params "decimals") x)
  | "around", [ x ] -> one (NL.around ?decimals:(opt_i params "decimals") x)
  | "nan_to_num", [ x ] -> one (NL.nan_to_num x)
  | "expand_dims", [ x ] -> one (NL.expand_dims x (ia (member "axis")))
  | "squeeze", [ x ] -> one (NL.squeeze ?axis:(opt_ia params "axis") x)
  | "swapaxes", [ x ] ->
      one
        (NL.swapaxes (U.to_int (member "axis1")) (U.to_int (member "axis2")) x)
  | "moveaxis", [ x ] ->
      one (NL.moveaxis (ia (member "source")) (ia (member "destination")) x)
  | "broadcast_to", [ x ] -> one (NL.broadcast_to x (ia (member "shape")))
  | "broadcast_arrays", ops -> NL.broadcast_arrays ops
  | "resize", [ x ] -> one (NL.resize x (ia (member "new_shape")))
  | "unravel_index", [ x ] -> NL.unravel_index x (ia (member "shape"))
  | "unwrap", [ x ] -> one (NL.unwrap ?axis:(opt_i params "axis") x)
  | "where", [ c; x; y ] -> one (NL.where_ c x y)
  | "select", ops ->
      let k = U.to_int (member "n") in
      let conds = List.filteri (fun i _ -> i < k) ops in
      let choices = List.filteri (fun i _ -> i >= k) ops in
      one (NL.select conds choices)
  | "split", [ x ] ->
      NL.split ~axis:(U.to_int (member "axis")) x (sections_of params)
  | "array_split", [ x ] ->
      NL.array_split ~axis:(U.to_int (member "axis")) x (sections_of params)
  | "vsplit", [ x ] -> NL.vsplit x (sections_of params)
  | "hsplit", [ x ] -> NL.hsplit x (sections_of params)
  | "dsplit", [ x ] -> NL.dsplit x (sections_of params)
  | "astype", [ x ] ->
      one (NL.astype x (dtype_of_string (U.to_string (member "dtype"))))
  | "copy", [ x ] -> one (NL.copy x)
  | "atleast_1d", [ x ] -> one (NL.atleast_1d x)
  | "atleast_2d", [ x ] -> one (NL.atleast_2d x)
  | "atleast_3d", [ x ] -> one (NL.atleast_3d x)
  | "concatenate", ops -> one (NL.concatenate ?axis:(opt_i params "axis") ops)
  | "concat", ops -> one (NL.concat ?axis:(opt_i params "axis") ops)
  | "stack", ops -> one (NL.stack ?axis:(opt_i params "axis") ops)
  | "unstack", [ x ] -> NL.unstack ?axis:(opt_i params "axis") x
  | "vstack", ops -> one (NL.vstack ops)
  | "hstack", ops -> one (NL.hstack ops)
  | "dstack", ops -> one (NL.dstack ops)
  | "column_stack", ops -> one (NL.column_stack ops)
  | "tile", [ x ] -> one (NL.tile x (ia (member "reps")))
  | "pad", [ x ] ->
      let cval =
        match opt_f params "constant_values" with Some v -> v | None -> 0.0
      in
      one (NL.pad x (pairs (member "pad_width")) cval)
  | "i0", [ x ] -> one (NL.i0 x)
  | "array_equal", [ a; b ] ->
      one (NL.array_equal ?equal_nan:(opt_b params "equal_nan") a b)
  | "array_equiv", [ a; b ] -> one (NL.array_equiv a b)
  | "arange", [] ->
      one
        (NL.arange ?start:(opt_f params "start") ?step:(opt_f params "step")
           ~dtype:(dtype_of_string (U.to_string (member "dtype")))
           (U.to_number (member "stop")))
  | "eye", [] ->
      one
        (NL.eye ?m:(opt_i params "m") ?k:(opt_i params "k")
           ~dtype:(dtype_of_string (U.to_string (member "dtype")))
           (U.to_int (member "n")))
  | "identity", [] ->
      one
        (NL.identity
           ~dtype:(dtype_of_string (U.to_string (member "dtype")))
           (U.to_int (member "n")))
  | "indices", [] ->
      one
        (NL.indices
           ~dtype:(dtype_of_string (U.to_string (member "dtype")))
           (ia (member "dimensions")))
  | "meshgrid", ops ->
      let indexing =
        match U.member "indexing" params with
        | `Null -> None
        | j -> Some (U.to_string j)
      in
      NL.meshgrid ?indexing ?sparse:(opt_b params "sparse") ops
  | "ix_", ops -> NL.ix_ ops
  | "append", [ a; b ] -> one (NL.append ?axis:(opt_i params "axis") a b)
  | "argmax", [ x ] ->
      one
        (NL.argmax ?axis:(opt_i params "axis")
           ~keepdims:(U.to_bool (member "keepdims"))
           x)
  | "cross", [ a; b ] ->
      one
        (NL.cross ?axisa:(opt_i params "axisa") ?axisb:(opt_i params "axisb")
           ?axisc:(opt_i params "axisc") ?axis:(opt_i params "axis") a b)
  | "diag", [ x ] -> one (NL.diag ?k:(opt_i params "k") x)
  | "diagflat", [ x ] -> one (NL.diagflat ?k:(opt_i params "k") x)
  | "diagonal", [ x ] ->
      one
        (NL.diagonal ?offset:(opt_i params "offset")
           ?axis1:(opt_i params "axis1") ?axis2:(opt_i params "axis2") x)
  | "diag_indices", [] ->
      NL.diag_indices ?ndim:(opt_i params "ndim") (U.to_int (member "n"))
  | "diag_indices_from", [ x ] -> NL.diag_indices_from x
  | "kron", [ a; b ] -> one (NL.kron a b)
  | "repeat", [ x ] ->
      one
        (NL.repeat ?axis:(opt_i params "axis") x (U.to_int (member "repeats")))
  | "trace", [ x ] ->
      one
        (NL.trace ?offset:(opt_i params "offset") ?axis1:(opt_i params "axis1")
           ?axis2:(opt_i params "axis2") x)
  | "trapezoid", [ y ] ->
      one (NL.trapezoid ?dx:(opt_f params "dx") ?axis:(opt_i params "axis") y)
  | "trapezoid", [ y; x ] -> one (NL.trapezoid ~x ?axis:(opt_i params "axis") y)
  | "tri", [] ->
      one
        (NL.tri ?m:(opt_i params "m") ?k:(opt_i params "k")
           ~dtype:(dtype_of_string (U.to_string (member "dtype")))
           (U.to_int (member "n")))
  | "tril", [ x ] -> one (NL.tril ?k:(opt_i params "k") x)
  | "triu", [ x ] -> one (NL.triu ?k:(opt_i params "k") x)
  | "vander", [ x ] ->
      one
        (NL.vander ?n:(opt_i params "N")
           ?increasing:(opt_b params "increasing")
           x)
  | "argmin", [ x ] ->
      one
        (NL.argmin ?axis:(opt_i params "axis")
           ~keepdims:(U.to_bool (member "keepdims"))
           x)
  | "nanargmax", [ x ] ->
      one
        (NL.nanargmax ?axis:(opt_i params "axis")
           ~keepdims:(U.to_bool (member "keepdims"))
           x)
  | "nanargmin", [ x ] ->
      one
        (NL.nanargmin ?axis:(opt_i params "axis")
           ~keepdims:(U.to_bool (member "keepdims"))
           x)
  | "roll", [ x ] ->
      one
        (NL.roll
           ?axis:(opt_int_or_ia params "axis")
           x (int_or_ia params "shift"))
  | "rollaxis", [ x ] ->
      one
        (NL.rollaxis ?start:(opt_i params "start") (U.to_int (member "axis")) x)
  | "gcd", [ a; b ] -> one (NL.gcd a b)
  | "lcm", [ a; b ] -> one (NL.lcm a b)
  | "searchsorted", [ a; v ] ->
      one (NL.searchsorted ~side:(U.to_string (member "side")) a v)
  | "digitize", [ x; bins ] ->
      one (NL.digitize ~right:(U.to_bool (member "right")) x bins)
  | "cov", ops -> (
      let rowvar = opt_b params "rowvar" in
      let bias = opt_b params "bias" in
      let ddof = opt_i params "ddof" in
      match ops with
      | [ m ] -> one (NL.cov ?rowvar ?bias ?ddof m)
      | [ m; y ] -> one (NL.cov ~y ?rowvar ?bias ?ddof m)
      | _ -> failwith "numpy golden: bad cov arity")
  | "corrcoef", ops -> (
      let rowvar = opt_b params "rowvar" in
      match ops with
      | [ x ] -> one (NL.corrcoef ?rowvar x)
      | [ x; y ] -> one (NL.corrcoef ~y ?rowvar x)
      | _ -> failwith "numpy golden: bad corrcoef arity")
  | _ -> failwith ("numpy golden: unknown op " ^ op)

let ufuncs_fn op _params operands : T.value list =
  let one v = [ v ] in
  match (op, operands) with
  | "negative", [ x ] -> one (UF.negative x)
  | "positive", [ x ] -> one (UF.positive x)
  | "sign", [ x ] -> one (UF.sign x)
  | "fabs", [ x ] -> one (UF.fabs x)
  | "floor", [ x ] -> one (UF.floor x)
  | "ceil", [ x ] -> one (UF.ceil x)
  | "exp", [ x ] -> one (UF.exp x)
  | "expm1", [ x ] -> one (UF.expm1 x)
  | "log", [ x ] -> one (UF.log x)
  | "log1p", [ x ] -> one (UF.log1p x)
  | "sin", [ x ] -> one (UF.sin x)
  | "cos", [ x ] -> one (UF.cos x)
  | "tan", [ x ] -> one (UF.tan x)
  | "arcsin", [ x ] -> one (UF.arcsin x)
  | "arccos", [ x ] -> one (UF.arccos x)
  | "arctan", [ x ] -> one (UF.arctan x)
  | "sinh", [ x ] -> one (UF.sinh x)
  | "cosh", [ x ] -> one (UF.cosh x)
  | "arcsinh", [ x ] -> one (UF.arcsinh x)
  | "arccosh", [ x ] -> one (UF.arccosh x)
  | "tanh", [ x ] -> one (UF.tanh x)
  | "arctanh", [ x ] -> one (UF.arctanh x)
  | "sqrt", [ x ] -> one (UF.sqrt x)
  | "cbrt", [ x ] -> one (UF.cbrt x)
  | "bitwise_not", [ x ] -> one (UF.bitwise_not x)
  | "bitwise_invert", [ x ] -> one (UF.bitwise_invert x)
  | "invert", [ x ] -> one (UF.invert x)
  | "logical_not", [ x ] -> one (UF.logical_not x)
  | "spacing", [ x ] -> one (UF.spacing x)
  | "add", [ a; b ] -> one (UF.add a b)
  | "subtract", [ a; b ] -> one (UF.subtract a b)
  | "multiply", [ a; b ] -> one (UF.multiply a b)
  | "maximum", [ a; b ] -> one (UF.maximum a b)
  | "minimum", [ a; b ] -> one (UF.minimum a b)
  | "bitwise_and", [ a; b ] -> one (UF.bitwise_and a b)
  | "bitwise_or", [ a; b ] -> one (UF.bitwise_or a b)
  | "bitwise_xor", [ a; b ] -> one (UF.bitwise_xor a b)
  | "left_shift", [ a; b ] -> one (UF.left_shift a b)
  | "bitwise_left_shift", [ a; b ] -> one (UF.bitwise_left_shift a b)
  | "logical_and", [ a; b ] -> one (UF.logical_and a b)
  | "logical_or", [ a; b ] -> one (UF.logical_or a b)
  | "logical_xor", [ a; b ] -> one (UF.logical_xor a b)
  | "equal", [ a; b ] -> one (UF.equal a b)
  | "not_equal", [ a; b ] -> one (UF.not_equal a b)
  | "greater", [ a; b ] -> one (UF.greater a b)
  | "greater_equal", [ a; b ] -> one (UF.greater_equal a b)
  | "arctan2", [ a; b ] -> one (UF.arctan2 a b)
  | "float_power", [ a; b ] -> one (UF.float_power a b)
  | "nextafter", [ a; b ] -> one (UF.nextafter a b)
  | "abs", [ x ] -> one (UF.abs x)
  | "absolute", [ x ] -> one (UF.absolute x)
  | "acos", [ x ] -> one (UF.acos x)
  | "acosh", [ x ] -> one (UF.acosh x)
  | "asin", [ x ] -> one (UF.asin x)
  | "asinh", [ x ] -> one (UF.asinh x)
  | "atan", [ x ] -> one (UF.atan x)
  | "atanh", [ x ] -> one (UF.atanh x)
  | "deg2rad", [ x ] -> one (UF.deg2rad x)
  | "degrees", [ x ] -> one (UF.degrees x)
  | "exp2", [ x ] -> one (UF.exp2 x)
  | "isfinite", [ x ] -> one (UF.isfinite x)
  | "isinf", [ x ] -> one (UF.isinf x)
  | "isnan", [ x ] -> one (UF.isnan x)
  | "isneginf", [ x ] -> one (UF.isneginf x)
  | "isposinf", [ x ] -> one (UF.isposinf x)
  | "log10", [ x ] -> one (UF.log10 x)
  | "log2", [ x ] -> one (UF.log2 x)
  | "rad2deg", [ x ] -> one (UF.rad2deg x)
  | "radians", [ x ] -> one (UF.radians x)
  | "reciprocal", [ x ] -> one (UF.reciprocal x)
  | "rint", [ x ] -> one (UF.rint x)
  | "signbit", [ x ] -> one (UF.signbit x)
  | "sinc", [ x ] -> one (UF.sinc x)
  | "square", [ x ] -> one (UF.square x)
  | "atan2", [ a; b ] -> one (UF.atan2 a b)
  | "bitwise_right_shift", [ a; b ] -> one (UF.bitwise_right_shift a b)
  | "copysign", [ a; b ] -> one (UF.copysign a b)
  | "divide", [ a; b ] -> one (UF.divide a b)
  | "floor_divide", [ a; b ] -> one (UF.floor_divide a b)
  | "fmod", [ a; b ] -> one (UF.fmod a b)
  | "heaviside", [ a; b ] -> one (UF.heaviside a b)
  | "hypot", [ a; b ] -> one (UF.hypot a b)
  | "less", [ a; b ] -> one (UF.less a b)
  | "less_equal", [ a; b ] -> one (UF.less_equal a b)
  | "logaddexp", [ a; b ] -> one (UF.logaddexp a b)
  | "logaddexp2", [ a; b ] -> one (UF.logaddexp2 a b)
  | "mod", [ a; b ] -> one (UF.mod_ a b)
  | "power", [ a; b ] -> one (UF.power a b)
  | "pow", [ a; b ] -> one (UF.pow a b)
  | "remainder", [ a; b ] -> one (UF.remainder a b)
  | "right_shift", [ a; b ] -> one (UF.right_shift a b)
  | "true_divide", [ a; b ] -> one (UF.true_divide a b)
  | "divmod", [ a; b ] -> UF.divmod a b
  | "modf", [ x ] -> UF.modf x
  | _ -> failwith ("ufuncs golden: unknown op " ^ op)

let numpy_check_case ~fn ~set_dir ~x64 c () =
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
  let results =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () ->
        fn c.op c.params operands)
  in
  let paired =
    try List.combine c.outs results
    with Invalid_argument _ ->
      Alcotest.failf "%s: output arity mismatch" c.case_id
  in
  List.iter
    (fun (o, v) ->
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
        if c.compare = "exact" then Npz.I (Array.map Int64.of_float floats)
        else Npz.F floats
      in
      let actual =
        { Npz.dtype = golden.Npz.dtype; shape = Nd.shape nd; data }
      in
      Compare.assert_tol_widened o.odtype c.atol c.rtol c.treason;
      Compare.check
        ~name:(c.case_id ^ ":" ^ o.oname)
        ~compare:c.compare ~atol:c.atol ~rtol:c.rtol ~expected:golden ~actual)
    paired

let numpy_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "lax_numpy") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:numpy_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("lax_numpy:" ^ set_name, coverage :: case_tests)

let ufuncs_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "ufuncs") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:ufuncs_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("ufuncs:" ^ set_name, coverage :: case_tests)

let opt_s params name =
  match U.member name params with `Null -> None | j -> Some (U.to_string j)

let q_value params =
  match U.member "q" params with
  | `List _ as j ->
      let vs = List.map U.to_number (U.to_list j) in
      T.Concrete (Nd.of_floats D.F32 [| List.length vs |] (Array.of_list vs))
  | j -> T.Concrete (Nd.of_floats D.F32 [||] [| U.to_number j |])

let reductions_fn op params operands : T.value list =
  let one v = [ v ] in
  let ax = opt_int_or_ia params "axis" in
  let axi =
    match U.member "axis" params with
    | `Null -> None
    | `List (x :: _) -> Some (U.to_int x)
    | `List [] -> None
    | j -> Some (U.to_int j)
  in
  let kd = match opt_b params "keepdims" with Some b -> b | None -> false in
  let ii =
    match opt_b params "include_initial" with Some b -> b | None -> false
  in
  let meth =
    match opt_s params "method" with Some m -> m | None -> "linear"
  in
  let dopt params = match opt_i params "ddof" with Some d -> d | None -> 0 in
  match (op, operands) with
  | "cumprod", [ x ] -> one (RED.cumprod ?axis:axi x)
  | "nancumsum", [ x ] -> one (RED.nancumsum ?axis:axi x)
  | "nancumprod", [ x ] -> one (RED.nancumprod ?axis:axi x)
  | "cumulative_sum", [ x ] ->
      one (RED.cumulative_sum ?axis:axi ~include_initial:ii x)
  | "cumulative_prod", [ x ] ->
      one (RED.cumulative_prod ?axis:axi ~include_initial:ii x)
  | "median", [ x ] -> one (RED.median ?axis:axi ~keepdims:kd x)
  | "nanmedian", [ x ] -> one (RED.nanmedian ?axis:axi ~keepdims:kd x)
  | "quantile", [ x ] ->
      one (RED.quantile ?axis:axi ~keepdims:kd ~method_:meth x (q_value params))
  | "nanquantile", [ x ] ->
      one
        (RED.nanquantile ?axis:axi ~keepdims:kd ~method_:meth x (q_value params))
  | "percentile", [ x ] ->
      one
        (RED.percentile ?axis:axi ~keepdims:kd ~method_:meth x (q_value params))
  | "nanpercentile", [ x ] ->
      one
        (RED.nanpercentile ?axis:axi ~keepdims:kd ~method_:meth x
           (q_value params))
  | "sum", [ x ] -> one (RED.sum ?axis:ax ~keepdims:kd x)
  | "prod", [ x ] -> one (RED.prod ?axis:ax ~keepdims:kd x)
  | "max", [ x ] -> one (RED.max ?axis:ax ~keepdims:kd x)
  | "min", [ x ] -> one (RED.min ?axis:ax ~keepdims:kd x)
  | "amax", [ x ] -> one (RED.amax ?axis:ax ~keepdims:kd x)
  | "amin", [ x ] -> one (RED.amin ?axis:ax ~keepdims:kd x)
  | "all", [ x ] -> one (RED.all ?axis:ax ~keepdims:kd x)
  | "any", [ x ] -> one (RED.any ?axis:ax ~keepdims:kd x)
  | "mean", [ x ] -> one (RED.mean ?axis:ax ~keepdims:kd x)
  | "ptp", [ x ] -> one (RED.ptp ?axis:ax ~keepdims:kd x)
  | "count_nonzero", [ x ] -> one (RED.count_nonzero ?axis:ax ~keepdims:kd x)
  | "nansum", [ x ] -> one (RED.nansum ?axis:ax ~keepdims:kd x)
  | "nanprod", [ x ] -> one (RED.nanprod ?axis:ax ~keepdims:kd x)
  | "nanmax", [ x ] -> one (RED.nanmax ?axis:ax ~keepdims:kd x)
  | "nanmin", [ x ] -> one (RED.nanmin ?axis:ax ~keepdims:kd x)
  | "nanmean", [ x ] -> one (RED.nanmean ?axis:ax ~keepdims:kd x)
  | "var", [ x ] -> one (RED.var ?axis:ax ~keepdims:kd ~ddof:(dopt params) x)
  | "std", [ x ] -> one (RED.std ?axis:ax ~keepdims:kd ~ddof:(dopt params) x)
  | "nanvar", [ x ] ->
      one (RED.nanvar ?axis:ax ~keepdims:kd ~ddof:(dopt params) x)
  | "nanstd", [ x ] ->
      one (RED.nanstd ?axis:ax ~keepdims:kd ~ddof:(dopt params) x)
  | "cumsum", [ x ] -> one (RED.cumsum ?axis:(opt_i params "axis") x)
  | "average", [ x ] -> one (RED.average ?axis:ax ~keepdims:kd x)
  | "average", [ x; w ] -> one (RED.average ?axis:ax ~keepdims:kd ~weights:w x)
  | _ -> failwith ("reductions golden: unknown op " ^ op)

let indexing_fn op params operands : T.value list =
  match (op, operands) with
  | "take", [ a; ind ] ->
      [ IDX.take ?axis:(opt_i params "axis") ?mode:(opt_s params "mode") a ind ]
  | "take_along_axis", [ a; ind ] ->
      [ IDX.take_along_axis ?axis:(opt_i params "axis") a ind ]
  | "put", [ a; ind; v ] -> [ IDX.put ?mode:(opt_s params "mode") a ind v ]
  | "put_along_axis", [ arr; ind; v ] ->
      [ IDX.put_along_axis ?axis:(opt_i params "axis") arr ind v ]
  | _ -> failwith ("indexing golden: unknown op " ^ op)

let indexing_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "indexing") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:indexing_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("indexing:" ^ set_name, coverage :: case_tests)

let array_methods_fn op params operands : T.value list =
  let member name = U.member name params in
  let one v = [ v ] in
  match (op, operands) with
  | "all", [ x ] ->
      one
        (AM.all
           ?axis:(opt_int_or_ia params "axis")
           ?keepdims:(opt_b params "keepdims") x)
  | "any", [ x ] ->
      one
        (AM.any
           ?axis:(opt_int_or_ia params "axis")
           ?keepdims:(opt_b params "keepdims") x)
  | "sum", [ x ] ->
      one
        (AM.sum
           ?axis:(opt_int_or_ia params "axis")
           ?keepdims:(opt_b params "keepdims") x)
  | "prod", [ x ] ->
      one
        (AM.prod
           ?axis:(opt_int_or_ia params "axis")
           ?keepdims:(opt_b params "keepdims") x)
  | "max", [ x ] ->
      one
        (AM.max
           ?axis:(opt_int_or_ia params "axis")
           ?keepdims:(opt_b params "keepdims") x)
  | "min", [ x ] ->
      one
        (AM.min
           ?axis:(opt_int_or_ia params "axis")
           ?keepdims:(opt_b params "keepdims") x)
  | "mean", [ x ] ->
      one
        (AM.mean
           ?axis:(opt_int_or_ia params "axis")
           ?keepdims:(opt_b params "keepdims") x)
  | "ptp", [ x ] ->
      one
        (AM.ptp
           ?axis:(opt_int_or_ia params "axis")
           ?keepdims:(opt_b params "keepdims") x)
  | "var", [ x ] ->
      one
        (AM.var
           ?axis:(opt_int_or_ia params "axis")
           ?keepdims:(opt_b params "keepdims") ?ddof:(opt_i params "ddof") x)
  | "std", [ x ] ->
      one
        (AM.std
           ?axis:(opt_int_or_ia params "axis")
           ?keepdims:(opt_b params "keepdims") ?ddof:(opt_i params "ddof") x)
  | "cumsum", [ x ] -> one (AM.cumsum ?axis:(opt_i params "axis") x)
  | "cumprod", [ x ] -> one (AM.cumprod ?axis:(opt_i params "axis") x)
  | "argmax", [ x ] -> one (AM.argmax ?axis:(opt_i params "axis") x)
  | "argmin", [ x ] -> one (AM.argmin ?axis:(opt_i params "axis") x)
  | "reshape", [ x ] -> one (AM.reshape x (ia (member "shape")))
  | "ravel", [ x ] -> one (AM.ravel x)
  | "flatten", [ x ] -> one (AM.flatten x)
  | "copy", [ x ] -> one (AM.copy x)
  | "conj", [ x ] -> one (AM.conj x)
  | "transpose", [ x ] -> one (AM.transpose ?axes:(opt_ia params "axes") x)
  | "squeeze", [ x ] -> one (AM.squeeze ?axis:(opt_ia params "axis") x)
  | "swapaxes", [ x ] ->
      one
        (AM.swapaxes (U.to_int (member "axis1")) (U.to_int (member "axis2")) x)
  | "repeat", [ x ] ->
      one
        (AM.repeat ?axis:(opt_i params "axis") x (U.to_int (member "repeats")))
  | "astype", [ x ] ->
      one (AM.astype x (dtype_of_string (U.to_string (member "dtype"))))
  | "clip", [ x ] ->
      one (AM.clip ?min:(opt_f params "min") ?max:(opt_f params "max") x)
  | "round", [ x ] -> one (AM.round ?decimals:(opt_i params "decimals") x)
  | "diagonal", [ x ] ->
      one
        (AM.diagonal ?offset:(opt_i params "offset")
           ?axis1:(opt_i params "axis1") ?axis2:(opt_i params "axis2") x)
  | "trace", [ x ] ->
      one
        (AM.trace ?offset:(opt_i params "offset") ?axis1:(opt_i params "axis1")
           ?axis2:(opt_i params "axis2") x)
  | "searchsorted", [ a; b ] ->
      one (AM.searchsorted ?side:(opt_s params "side") a b)
  | "take", [ a; b ] ->
      one (AM.take ?axis:(opt_i params "axis") ?mode:(opt_s params "mode") a b)
  | "T", [ x ] -> one (AM.t x)
  | "mT", [ x ] -> one (AM.mt x)
  | "real", [ x ] -> one (AM.real x)
  | "imag", [ x ] -> one (AM.imag x)
  | "neg", [ x ] -> one (AM.neg x)
  | "pos", [ x ] -> one (AM.pos x)
  | "abs", [ x ] -> one (AM.abs x)
  | "invert", [ x ] -> one (AM.invert x)
  | "add", [ a; b ] -> one (AM.add a b)
  | "sub", [ a; b ] -> one (AM.sub a b)
  | "mul", [ a; b ] -> one (AM.mul a b)
  | "truediv", [ a; b ] -> one (AM.truediv a b)
  | "floordiv", [ a; b ] -> one (AM.floordiv a b)
  | "mod", [ a; b ] -> one (AM.mod_ a b)
  | "pow", [ a; b ] -> one (AM.pow a b)
  | "eq", [ a; b ] -> one (AM.eq a b)
  | "ne", [ a; b ] -> one (AM.ne a b)
  | "lt", [ a; b ] -> one (AM.lt a b)
  | "le", [ a; b ] -> one (AM.le a b)
  | "gt", [ a; b ] -> one (AM.gt a b)
  | "ge", [ a; b ] -> one (AM.ge a b)
  | "and", [ a; b ] -> one (AM.and_ a b)
  | "or", [ a; b ] -> one (AM.or_ a b)
  | "xor", [ a; b ] -> one (AM.xor a b)
  | "lshift", [ a; b ] -> one (AM.lshift a b)
  | "rshift", [ a; b ] -> one (AM.rshift a b)
  | _ -> failwith ("array_methods golden: unknown op " ^ op)

let array_methods_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "array_methods") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:array_methods_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("array_methods:" ^ set_name, coverage :: case_tests)

module AC = Ojax.Numpy.Array_creation
module ACON = Ojax.Numpy.Array_constructors
module ST = Ojax.Numpy.Scalar_types

let creation_fn op params operands : T.value list =
  let member name = U.member name params in
  let one v = [ v ] in
  match (op, operands) with
  | "zeros", [] ->
      one (AC.zeros ?dtype:(opt_dt params "dtype") (ia (member "shape")))
  | "ones", [] ->
      one (AC.ones ?dtype:(opt_dt params "dtype") (ia (member "shape")))
  | "empty", [] ->
      one (AC.empty ?dtype:(opt_dt params "dtype") (ia (member "shape")))
  | "full", [] ->
      one
        (AC.full ?dtype:(opt_dt params "dtype")
           (ia (member "shape"))
           (U.to_number (member "fill_value")))
  | "zeros_like", [ x ] ->
      one
        (AC.zeros_like ?dtype:(opt_dt params "dtype")
           ?shape:(opt_ia params "shape") x)
  | "ones_like", [ x ] ->
      one
        (AC.ones_like ?dtype:(opt_dt params "dtype")
           ?shape:(opt_ia params "shape") x)
  | "empty_like", [ x ] ->
      one
        (AC.empty_like ?dtype:(opt_dt params "dtype")
           ?shape:(opt_ia params "shape") x)
  | "full_like", [ x ] ->
      one
        (AC.full_like ?dtype:(opt_dt params "dtype")
           ?shape:(opt_ia params "shape") x
           (U.to_number (member "fill_value")))
  | "linspace", [] ->
      one
        (AC.linspace ?num:(opt_i params "num")
           ?endpoint:(opt_b params "endpoint") ?dtype:(opt_dt params "dtype")
           (U.to_number (member "start"))
           (U.to_number (member "stop")))
  | "logspace", [] ->
      one
        (AC.logspace ?num:(opt_i params "num")
           ?endpoint:(opt_b params "endpoint") ?base:(opt_f params "base")
           ?dtype:(opt_dt params "dtype")
           (U.to_number (member "start"))
           (U.to_number (member "stop")))
  | "geomspace", [] ->
      one
        (AC.geomspace ?num:(opt_i params "num")
           ?endpoint:(opt_b params "endpoint") ?dtype:(opt_dt params "dtype")
           (U.to_number (member "start"))
           (U.to_number (member "stop")))
  | "array", [ x ] ->
      one
        (ACON.array ?dtype:(opt_dt params "dtype") ?ndmin:(opt_i params "ndmin")
           x)
  | "asarray", [ x ] -> one (ACON.asarray ?dtype:(opt_dt params "dtype") x)
  | "bool_", [ x ] -> one (ST.bool_ x)
  | "int32", [ x ] -> one (ST.int32 x)
  | "int64", [ x ] -> one (ST.int64 x)
  | "float32", [ x ] -> one (ST.float32 x)
  | "float64", [ x ] -> one (ST.float64 x)
  | _ -> failwith ("creation golden: unknown op " ^ op)

let creation_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "creation") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:creation_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("creation:" ^ set_name, coverage :: case_tests)

module TC = Ojax.Numpy.Tensor_contractions
module EI = Ojax.Numpy.Einsum

let td_axes_of params =
  match U.member "axes" params with
  | `List (a :: b :: _) -> TC.Ax_pair (ia a, ia b)
  | j -> TC.Ax_int (U.to_int j)

let contractions_fn op params operands : T.value list =
  let member name = U.member name params in
  let one v = [ v ] in
  match (op, operands) with
  | "dot", [ a; b ] -> one (TC.dot a b)
  | "matmul", [ a; b ] -> one (TC.matmul a b)
  | "matvec", [ a; b ] -> one (TC.matvec a b)
  | "vecmat", [ a; b ] -> one (TC.vecmat a b)
  | "vdot", [ a; b ] -> one (TC.vdot a b)
  | "vecdot", [ a; b ] -> one (TC.vecdot ?axis:(opt_i params "axis") a b)
  | "inner", [ a; b ] -> one (TC.inner a b)
  | "outer", [ a; b ] -> one (TC.outer a b)
  | "tensordot", [ a; b ] -> one (TC.tensordot ~axes:(td_axes_of params) a b)
  | "einsum", ops -> one (EI.einsum (U.to_string (member "subscripts")) ops)
  | _ -> failwith ("contractions golden: unknown op " ^ op)

let contractions_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "contractions") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:contractions_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("contractions:" ^ set_name, coverage :: case_tests)

module WF = Ojax.Numpy.Window_functions
module SORT = Ojax.Numpy.Sorting
module SETOPS = Ojax.Numpy.Setops
module POLY = Ojax.Numpy.Polynomial
module NN = Ojax.Nn.Functions

let axis_opt params =
  match U.member "axis" params with `Null -> None | j -> Some (U.to_int j)

let setops_fn op params operands : T.value list =
  let member name = U.member name params in
  let one v = [ v ] in
  match (op, operands) with
  | "blackman", [] -> one (WF.blackman (U.to_int (member "M")))
  | "bartlett", [] -> one (WF.bartlett (U.to_int (member "M")))
  | "hamming", [] -> one (WF.hamming (U.to_int (member "M")))
  | "hanning", [] -> one (WF.hanning (U.to_int (member "M")))
  | "kaiser", [] ->
      one (WF.kaiser (U.to_int (member "M")) (U.to_number (member "beta")))
  | "sort", [ x ] ->
      one
        (SORT.sort ~axis:(axis_opt params) ?stable:(opt_b params "stable")
           ?descending:(opt_b params "descending")
           x)
  | "argsort", [ x ] ->
      one
        (SORT.argsort ~axis:(axis_opt params) ?stable:(opt_b params "stable")
           ?descending:(opt_b params "descending")
           ?dtype:(opt_dt params "dtype") x)
  | "lexsort", keys -> one (SORT.lexsort ?axis:(opt_i params "axis") keys)
  | "partition", [ x ] ->
      one
        (SORT.partition ?axis:(opt_i params "axis") x
           ~kth:(U.to_int (member "kth")))
  | "isin", [ a; b ] -> one (SETOPS.isin ?invert:(opt_b params "invert") a b)
  | _ -> failwith ("setops golden: unknown op " ^ op)

let setops_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "setops") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:setops_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("setops:" ^ set_name, coverage :: case_tests)

let poly_fn op params operands : T.value list =
  let one v = [ v ] in
  match (op, operands) with
  | "polyval", [ p; x ] -> one (POLY.polyval p x)
  | "polyadd", [ a; b ] -> one (POLY.polyadd a b)
  | "polysub", [ a; b ] -> one (POLY.polysub a b)
  | "polymul", [ a; b ] -> one (POLY.polymul a b)
  | "poly", [ s ] -> one (POLY.poly s)
  | "polyint", [ p ] -> one (POLY.polyint ?m:(opt_i params "m") p)
  | "polyder", [ p ] -> one (POLY.polyder ?m:(opt_i params "m") p)
  | _ -> failwith ("polynomial golden: unknown op " ^ op)

let polynomial_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "polynomial") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:poly_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("polynomial:" ^ set_name, coverage :: case_tests)

let reductions_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "reductions") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:reductions_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("reductions:" ^ set_name, coverage :: case_tests)

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

type solves_case = {
  sv_id : string;
  sv_mode : string;
  sv_symmetric : bool;
  sv_has_ts : bool;
  sv_args : arg list;
  sv_tans : arg list;
  sv_outs : out list;
  sv_compare : string;
  sv_atol : float;
  sv_rtol : float;
}

let parse_solves_case j =
  let tol = U.member "tol" j in
  {
    sv_id = U.member "case_id" j |> U.to_string;
    sv_mode = U.member "mode" j |> U.to_string;
    sv_symmetric = U.member "symmetric" j |> U.to_bool;
    sv_has_ts = U.member "has_ts" j |> U.to_bool;
    sv_args = U.member "args" j |> U.to_list |> List.map parse_arg;
    sv_tans = U.member "tangents" j |> U.to_list |> List.map parse_arg;
    sv_outs = U.member "outputs" j |> U.to_list |> List.map parse_out;
    sv_compare = U.member "compare" j |> U.to_string;
    sv_atol = U.member "atol" tol |> U.to_number;
    sv_rtol = U.member "rtol" tol |> U.to_number;
  }

let load_solves_manifest path =
  let j = Yojson.Safe.from_file path in
  ( U.member "x64" j |> U.to_bool,
    U.member "cases" j |> U.to_list |> List.map parse_solves_case )

let solves_run c primals tangents =
  match primals with
  | [ d; b ] -> (
      let matvec xs =
        match xs with [ x ] -> [ jb1 T.Mul [ x; d ] ] | _ -> assert false
      in
      let solve _mv bs =
        match bs with [ bb ] -> [ jb1 T.Div [ bb; d ] ] | _ -> assert false
      in
      let transpose_solve =
        if c.sv_has_ts then
          Some
            (fun _mv bs ->
              match bs with
              | [ bb ] -> [ jb1 T.Div [ bb; d ] ]
              | _ -> assert false)
        else None
      in
      let wrapped bs =
        Ojax.Lax.custom_linear_solve ~symmetric:c.sv_symmetric ?transpose_solve
          matvec bs solve
      in
      match c.sv_mode with
      | "eval" -> wrapped [ b ]
      | "jvp" ->
          let po, to_ = Ad.jvp wrapped [ b ] tangents in
          po @ to_
      | "grad" ->
          Ad.grad
            (fun bs -> jb1 (T.Reduce_sum [| 0 |]) [ List.hd (wrapped bs) ])
            [ b ]
      | m -> failwith ("solves: unknown mode " ^ m))
  | _ -> Alcotest.failf "%s: solves expects [d;b]" c.sv_id

let solves_check_case ~set_dir ~x64 c () =
  let canon d = if x64 then d else Compare.canonical_dtype_x64_off d in
  let inputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "inputs") (c.sv_id ^ ".npz"))
  in
  let outputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "outputs") (c.sv_id ^ ".npz"))
  in
  let read_operand (a : arg) =
    T.Concrete (nd_of_npz (find_member inputs a.name))
  in
  let primals = List.map read_operand c.sv_args in
  let tangents = List.map read_operand c.sv_tans in
  let results = solves_run c primals tangents in
  let paired =
    try List.combine c.sv_outs results
    with Invalid_argument _ ->
      Alcotest.failf "%s: output arity mismatch" c.sv_id
  in
  List.iter
    (fun (o, v) ->
      let nd = concrete v in
      if not (Compare.shapes_equal (Nd.shape nd) o.oshape) then
        Alcotest.failf "%s: output %s shape mismatch" c.sv_id o.oname;
      if canon (string_of_dtype (Nd.dtype nd)) <> canon o.odtype then
        Alcotest.failf "%s: output %s dtype %s != %s" c.sv_id o.oname
          (string_of_dtype (Nd.dtype nd))
          o.odtype;
      let golden = find_member outputs o.oname in
      let floats = read_nd nd in
      let data =
        if c.sv_compare = "exact" then Npz.I (Array.map Int64.of_float floats)
        else Npz.F floats
      in
      let actual =
        { Npz.dtype = golden.Npz.dtype; shape = Nd.shape nd; data }
      in
      Compare.assert_tol o.odtype c.sv_atol c.sv_rtol;
      Compare.check
        ~name:(c.sv_id ^ ":" ^ o.oname)
        ~compare:c.sv_compare ~atol:c.sv_atol ~rtol:c.sv_rtol ~expected:golden
        ~actual)
    paired

let solves_check_coverage ~set_dir cases () =
  let expected =
    List.map (fun c -> c.sv_id) cases |> List.sort String.compare
  in
  List.iter
    (fun sub ->
      let got = dir_case_ids (Filename.concat set_dir sub) in
      if got <> expected then Alcotest.failf "%s: solves coverage mismatch" sub)
    [ "inputs"; "outputs" ]

let solves_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "solves") set_name
  in
  let x64, cases =
    load_solves_manifest (Filename.concat set_dir "manifest.json")
  in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.sv_id `Quick (solves_check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (solves_check_coverage ~set_dir cases)
  in
  ("solves:" ^ set_name, coverage :: case_tests)

module PR = Ojax.Random.Prng
module TF = Ojax.Random.Threefry

let prng_fn op params operands : T.value list =
  let member name = U.member name params in
  let shape () = ia (member "shape") in
  let bw () = U.to_int (member "bit_width") in
  match (op, operands) with
  | "threefry_seed", [ s ] -> [ TF.threefry_seed s ]
  | "threefry_2x32", [ k; c ] -> [ TF.threefry_2x32 k c ]
  | "threefry_split", [ k ] -> [ TF.threefry_split k (shape ()) ]
  | "threefry_fold_in", [ k; d ] -> [ TF.threefry_fold_in k d ]
  | "threefry_random_bits", [ k ] ->
      [ TF.threefry_random_bits k (bw ()) (shape ()) ]
  | "iota_2x32_shape", [] ->
      let a, b = PR.iota_2x32_shape (shape ()) in
      [ a; b ]
  | "random_seed", [ s ] -> [ PR.random_seed s ]
  | "random_split", [ k ] -> [ PR.random_split (PR.random_wrap k) (shape ()) ]
  | "random_fold_in", [ k; m ] -> [ PR.random_fold_in (PR.random_wrap k) m ]
  | "random_bits", [ k ] ->
      [ PR.random_bits (PR.random_wrap k) ~bit_width:(bw ()) ~shape:(shape ()) ]
  | "random_wrap", [ k ] -> [ PR.random_unwrap (PR.random_wrap k) ]
  | "random_unwrap", [ k ] -> [ PR.random_unwrap (PR.random_wrap k) ]
  | _ -> failwith ("prng golden: unknown op " ^ op)

let prng_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "prng") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:prng_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("prng:" ^ set_name, coverage :: case_tests)

module RC = Ojax.Random.Core

let random_core_fn op params operands : T.value list =
  let member name = U.member name params in
  let shape () = ia (member "shape") in
  let ip name = U.to_int (member name) in
  let fp name = U.to_number (member name) in
  match (op, operands) with
  | "key", [ s ] -> [ RC.key s ]
  | "key_data", [ k ] -> [ RC.key_data (PR.random_wrap k) ]
  | "wrap_key_data", [ k ] -> [ RC.key_data (RC.wrap_key_data k) ]
  | "clone", [ k ] -> [ RC.key_data (RC.clone (PR.random_wrap k)) ]
  | "fold_in", [ k; d ] -> [ RC.fold_in k d ]
  | "split", [ k ] -> [ RC.split k (ip "num") ]
  | "bits", [ k ] -> [ RC.bits k ~shape:(shape ()) ]
  | "randint", [ k ] ->
      [
        RC.randint k ~shape:(shape ()) ~minval:(ip "minval")
          ~maxval:(ip "maxval");
      ]
  | "uniform", [ k ] ->
      [
        RC.uniform k ~shape:(shape ()) ~minval:(fp "minval")
          ~maxval:(fp "maxval");
      ]
  | "normal", [ k ] -> [ RC.normal k ~shape:(shape ()) ]
  | "truncated_normal", [ k ] ->
      [
        RC.truncated_normal k ~lower:(fp "lower") ~upper:(fp "upper")
          ~shape:(shape ());
      ]
  | "permutation", [ k ] -> [ RC.permutation k (ip "n") ]
  | "choice", [ k ] ->
      [
        RC.choice k ~n:(ip "n") ~shape:(shape ())
          ~replace:(U.to_bool (member "replace"));
      ]
  | "exponential", [ k ] -> [ RC.exponential k ~shape:(shape ()) ]
  | "cauchy", [ k ] -> [ RC.cauchy k ~shape:(shape ()) ]
  | "laplace", [ k ] -> [ RC.laplace k ~shape:(shape ()) ]
  | "logistic", [ k ] -> [ RC.logistic k ~shape:(shape ()) ]
  | "gumbel", [ k ] -> [ RC.gumbel k ~shape:(shape ()) ]
  | "pareto", [ k ] -> [ RC.pareto k ~shape:(shape ()) ~b:(fp "b") ]
  | "rayleigh", [ k ] -> [ RC.rayleigh k ~shape:(shape ()) ~scale:(fp "scale") ]
  | "weibull_min", [ k ] ->
      [
        RC.weibull_min k ~shape:(shape ()) ~scale:(fp "scale")
          ~concentration:(fp "concentration");
      ]
  | "lognormal", [ k ] ->
      [ RC.lognormal k ~shape:(shape ()) ~sigma:(fp "sigma") ]
  | "triangular", [ k ] ->
      [
        RC.triangular k ~shape:(shape ()) ~left:(fp "left") ~mode:(fp "mode")
          ~right:(fp "right");
      ]
  | "wald", [ k ] -> [ RC.wald k ~shape:(shape ()) ~mean:(fp "mean") ]
  | "geometric", [ k ] -> [ RC.geometric k ~shape:(shape ()) ~p:(fp "p") ]
  | "bernoulli", [ k ] -> [ RC.bernoulli k ~shape:(shape ()) ~p:(fp "p") ]
  | "rademacher", [ k ] -> [ RC.rademacher k ~shape:(shape ()) ]
  | "categorical", [ k; logits ] ->
      [ RC.categorical k ~logits ~axis:(ip "axis") ]
  | "gamma", [ k ] -> [ RC.gamma k ~shape:(shape ()) ~a:(fp "a") ]
  | "loggamma", [ k ] -> [ RC.loggamma k ~shape:(shape ()) ~a:(fp "a") ]
  | "beta", [ k ] -> [ RC.beta k ~shape:(shape ()) ~a:(fp "a") ~b:(fp "b") ]
  | "chisquare", [ k ] -> [ RC.chisquare k ~shape:(shape ()) ~df:(fp "df") ]
  | "t", [ k ] -> [ RC.t k ~shape:(shape ()) ~df:(fp "df") ]
  | "f", [ k ] ->
      [ RC.f k ~shape:(shape ()) ~dfnum:(fp "dfnum") ~dfden:(fp "dfden") ]
  | "generalized_normal", [ k ] ->
      [ RC.generalized_normal k ~shape:(shape ()) ~p:(fp "p") ]
  | "dirichlet", [ k; alpha ] -> [ RC.dirichlet k ~alpha ~shape:[||] ]
  | "poisson", [ k ] -> [ RC.poisson k ~shape:(shape ()) ~lam:(fp "lam") ]
  | "binomial", [ k ] ->
      [ RC.binomial k ~shape:(shape ()) ~count:(fp "n") ~prob:(fp "p") ]
  | "multinomial", [ k; p ] -> [ RC.multinomial k ~p ~n_trials:(fp "n") ]
  | _ -> failwith ("random_core golden: unknown op " ^ op)

let random_core_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "random_core") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:random_core_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("random_core:" ^ set_name, coverage :: case_tests)

let nn_fn op params operands : T.value list =
  let member name = U.member name params in
  let ip name = U.to_int (member name) in
  let fp name = U.to_number (member name) in
  let one v = [ v ] in
  match (op, operands) with
  | "identity", [ x ] -> one (NN.identity x)
  | "relu", [ x ] -> one (NN.relu x)
  | "relu6", [ x ] -> one (NN.relu6 x)
  | "softplus", [ x ] -> one (NN.softplus x)
  | "sparse_plus", [ x ] -> one (NN.sparse_plus x)
  | "soft_sign", [ x ] -> one (NN.soft_sign x)
  | "sigmoid", [ x ] -> one (NN.sigmoid x)
  | "sparse_sigmoid", [ x ] -> one (NN.sparse_sigmoid x)
  | "silu", [ x ] -> one (NN.silu x)
  | "mish", [ x ] -> one (NN.mish x)
  | "log_sigmoid", [ x ] -> one (NN.log_sigmoid x)
  | "hard_tanh", [ x ] -> one (NN.hard_tanh x)
  | "hard_sigmoid", [ x ] -> one (NN.hard_sigmoid x)
  | "hard_silu", [ x ] -> one (NN.hard_silu x)
  | "selu", [ x ] -> one (NN.selu x)
  | "log1mexp", [ x ] -> one (NN.log1mexp x)
  | "elu", [ x ] -> one (NN.elu ~alpha:(fp "alpha") x)
  | "celu", [ x ] -> one (NN.celu ~alpha:(fp "alpha") x)
  | "leaky_relu", [ x ] ->
      one (NN.leaky_relu ~negative_slope:(fp "negative_slope") x)
  | "squareplus", [ x ] -> one (NN.squareplus ~b:(fp "b") x)
  | "gelu", [ x ] ->
      one (NN.gelu ~approximate:(U.to_bool (member "approximate")) x)
  | "glu", [ x ] -> one (NN.glu ~axis:(ip "axis") x)
  | "softmax", [ x ] -> one (NN.softmax ~axis:(ip "axis") x)
  | "log_softmax", [ x ] -> one (NN.log_softmax ~axis:(ip "axis") x)
  | "standardize", [ x ] ->
      one (NN.standardize ~axis:(ip "axis") ~epsilon:(fp "epsilon") x)
  | "logmeanexp", [ x ] ->
      one
        (NN.logmeanexp
           ?axis:(Option.map (fun a -> [| a |]) (opt_i params "axis"))
           ~keepdims:(Option.value ~default:false (opt_b params "keepdims"))
           x)
  | "one_hot", [ x ] ->
      one (NN.one_hot ~num_classes:(ip "num_classes") ~axis:(ip "axis") x)
  | "scaled_dot_general", [ a; b ] ->
      one
        (NN.scaled_dot_general
           ~lhs_contract:(ia (member "lhs_contract"))
           ~rhs_contract:(ia (member "rhs_contract"))
           ~lhs_batch:(ia (member "lhs_batch"))
           ~rhs_batch:(ia (member "rhs_batch"))
           a b)
  | _ -> failwith ("nn golden: unknown op " ^ op)

let nn_suite_for set_name =
  let set_dir = Filename.concat (Filename.concat goldens_root "nn") set_name in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:nn_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("nn:" ^ set_name, coverage :: case_tests)

module INIT = Ojax.Nn.Initializers

let init_mode_of j =
  match U.to_string j with
  | "fan_in" -> INIT.Fan_in
  | "fan_out" -> INIT.Fan_out
  | "fan_avg" -> INIT.Fan_avg
  | "fan_geo_avg" -> INIT.Fan_geo_avg
  | s -> failwith ("initializers golden: unknown mode " ^ s)

let init_dist_of j =
  match U.to_string j with
  | "truncated_normal" -> INIT.Truncated_normal
  | "normal" -> INIT.Normal
  | "uniform" -> INIT.Uniform
  | s -> failwith ("initializers golden: unknown distribution " ^ s)

let initializers_fn op params operands : T.value list =
  let member name = U.member name params in
  let fp name = U.to_number (member name) in
  let shape () = ia (member "shape") in
  match (op, operands) with
  | "zeros", [ k ] -> [ INIT.zeros k ~shape:(shape ()) ]
  | "ones", [ k ] -> [ INIT.ones k ~shape:(shape ()) ]
  | "constant", [ k ] -> [ INIT.constant (fp "value") k ~shape:(shape ()) ]
  | "uniform", [ k ] ->
      [ INIT.uniform ~scale:(fp "scale") () k ~shape:(shape ()) ]
  | "normal", [ k ] ->
      [ INIT.normal ~stddev:(fp "stddev") () k ~shape:(shape ()) ]
  | "truncated_normal", [ k ] ->
      [
        INIT.truncated_normal ~stddev:(fp "stddev") ~lower:(fp "lower")
          ~upper:(fp "upper") () k ~shape:(shape ());
      ]
  | "variance_scaling", [ k ] ->
      [
        INIT.variance_scaling (fp "scale")
          (init_mode_of (member "mode"))
          (init_dist_of (member "distribution"))
          () k ~shape:(shape ());
      ]
  | "glorot_uniform", [ k ] -> [ INIT.glorot_uniform () k ~shape:(shape ()) ]
  | "glorot_normal", [ k ] -> [ INIT.glorot_normal () k ~shape:(shape ()) ]
  | "lecun_uniform", [ k ] -> [ INIT.lecun_uniform () k ~shape:(shape ()) ]
  | "lecun_normal", [ k ] -> [ INIT.lecun_normal () k ~shape:(shape ()) ]
  | "he_uniform", [ k ] -> [ INIT.he_uniform () k ~shape:(shape ()) ]
  | "he_normal", [ k ] -> [ INIT.he_normal () k ~shape:(shape ()) ]
  | _ -> failwith ("initializers golden: unknown op " ^ op)

let initializers_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "initializers") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:initializers_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("initializers:" ^ set_name, coverage :: case_tests)

module SEG = Ojax.Ops.Segment
module SPC = Ojax.Ops.Special

let segment_fn op params operands : T.value list =
  let ns = opt_i params "num_segments" in
  match (op, operands) with
  | "segment_sum", [ data; seg ] ->
      [ SEG.segment_sum ?num_segments:ns data seg ]
  | "segment_prod", [ data; seg ] ->
      [ SEG.segment_prod ?num_segments:ns data seg ]
  | "segment_max", [ data; seg ] ->
      [ SEG.segment_max ?num_segments:ns data seg ]
  | "segment_min", [ data; seg ] ->
      [ SEG.segment_min ?num_segments:ns data seg ]
  | _ -> failwith ("segment golden: unknown op " ^ op)

let segment_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "segment") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:segment_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("segment:" ^ set_name, coverage :: case_tests)

let special_fn op params operands : T.value list =
  let axis = opt_ia params "axis" in
  let keepdims = Option.value ~default:false (opt_b params "keepdims") in
  match (op, operands) with
  | "logsumexp", [ a ] -> [ SPC.logsumexp ?axis ~keepdims a ]
  | "logsumexp", [ a; b ] -> [ SPC.logsumexp ?axis ~b ~keepdims a ]
  | _ -> failwith ("special golden: unknown op " ^ op)

let special_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "special") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:special_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("special:" ^ set_name, coverage :: case_tests)

module SC = Ojax.Image.Scale

let fa params name =
  Array.of_list (List.map U.to_number (U.to_list (U.member name params)))

let image_fn op params operands : T.value list =
  let meth = SC.from_string (U.to_string (U.member "method" params)) in
  let antialias = Option.value ~default:true (opt_b params "antialias") in
  match (op, operands) with
  | "resize", [ img ] ->
      let shape = ia (U.member "shape" params) in
      [ SC.resize img ~shape ~method_:meth ~antialias () ]
  | "scale_and_translate", [ img ] ->
      let shape = ia (U.member "shape" params) in
      let spatial_dims = ia (U.member "spatial_dims" params) in
      let scale = fa params "scale" in
      let translation = fa params "translation" in
      [
        SC.scale_and_translate img ~shape ~spatial_dims ~scale ~translation
          ~method_:meth ~antialias ();
      ]
  | "compute_weight_mat", [] ->
      let input_size = U.to_int (U.member "input_size" params) in
      let output_size = U.to_int (U.member "output_size" params) in
      let scale = U.to_number (U.member "scale" params) in
      let translation = U.to_number (U.member "translation" params) in
      let _, kernel = SC.kernels meth in
      [
        SC.compute_weight_mat ~input_size ~output_size ~scale ~translation
          ~kernel ~antialias ~edge_padding:false ~radius:0;
      ]
  | _ -> failwith ("image golden: unknown op " ^ op)

let image_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "image") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:image_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("image:" ^ set_name, coverage :: case_tests)

module SP = Ojax.Scipy.Special
module SS_bernoulli = Ojax.Scipy.Stats.Bernoulli
module SS_beta = Ojax.Scipy.Stats.Beta
module SS_betabinom = Ojax.Scipy.Stats.Betabinom
module SS_binom = Ojax.Scipy.Stats.Binom
module SS_cauchy = Ojax.Scipy.Stats.Cauchy
module SS_chi2 = Ojax.Scipy.Stats.Chi2
module SS_dirichlet = Ojax.Scipy.Stats.Dirichlet
module SS_expon = Ojax.Scipy.Stats.Expon
module SS_gamma = Ojax.Scipy.Stats.Gamma
module SS_gennorm = Ojax.Scipy.Stats.Gennorm
module SS_geom = Ojax.Scipy.Stats.Geom
module SS_gumbel_l = Ojax.Scipy.Stats.Gumbel_l
module SS_gumbel_r = Ojax.Scipy.Stats.Gumbel_r
module SS_laplace = Ojax.Scipy.Stats.Laplace
module SS_logistic = Ojax.Scipy.Stats.Logistic
module SS_multinomial = Ojax.Scipy.Stats.Multinomial
module SS_nbinom = Ojax.Scipy.Stats.Nbinom
module SS_norm = Ojax.Scipy.Stats.Norm
module SS_pareto = Ojax.Scipy.Stats.Pareto
module SS_poisson = Ojax.Scipy.Stats.Poisson
module SS_t = Ojax.Scipy.Stats.T
module SS_truncnorm = Ojax.Scipy.Stats.Truncnorm
module SS_uniform = Ojax.Scipy.Stats.Uniform
module SS_vonmises = Ojax.Scipy.Stats.Vonmises
module SS_wrapcauchy = Ojax.Scipy.Stats.Wrapcauchy
module SS_core = Ojax.Scipy.Stats.Stats_core

let scipy_special_fn op params operands : T.value list =
  let one v = [ v ] in
  match (op, operands) with
  | "gammaln", [ x ] -> one (SP.gammaln x)
  | "gammasgn", [ x ] -> one (SP.gammasgn x)
  | "loggamma", [ x ] -> one (SP.loggamma x)
  | "gamma", [ x ] -> one (SP.gamma x)
  | "digamma", [ x ] -> one (SP.digamma x)
  | "erf", [ x ] -> one (SP.erf x)
  | "erfc", [ x ] -> one (SP.erfc x)
  | "erfinv", [ x ] -> one (SP.erfinv x)
  | "erfcx", [ x ] -> one (SP.erfcx x)
  | "dawsn", [ x ] -> one (SP.dawsn x)
  | "expit", [ x ] -> one (SP.expit x)
  | "logit", [ x ] -> one (SP.logit x)
  | "entr", [ x ] -> one (SP.entr x)
  | "i0", [ x ] -> one (SP.i0 x)
  | "i0e", [ x ] -> one (SP.i0e x)
  | "i1", [ x ] -> one (SP.i1 x)
  | "i1e", [ x ] -> one (SP.i1e x)
  | "ndtr", [ x ] -> one (SP.ndtr x)
  | "ndtri", [ x ] -> one (SP.ndtri x)
  | "log_ndtr", [ x ] -> one (SP.log_ndtr x)
  | "factorial", [ x ] -> one (SP.factorial x)
  | "betaln", [ a; b ] -> one (SP.betaln a b)
  | "beta", [ a; b ] -> one (SP.beta a b)
  | "comb", [ a; b ] -> one (SP.comb a b)
  | "gammainc", [ a; b ] -> one (SP.gammainc a b)
  | "gammaincc", [ a; b ] -> one (SP.gammaincc a b)
  | "xlogy", [ a; b ] -> one (SP.xlogy a b)
  | "xlog1py", [ a; b ] -> one (SP.xlog1py a b)
  | "boxcox", [ a; b ] -> one (SP.boxcox a b)
  | "boxcox1p", [ a; b ] -> one (SP.boxcox1p a b)
  | "rel_entr", [ a; b ] -> one (SP.rel_entr a b)
  | "kl_div", [ a; b ] -> one (SP.kl_div a b)
  | "zeta", [ x; q ] -> one (SP.zeta ~q x)
  | "polygamma", [ n; x ] -> one (SP.polygamma n x)
  | "betainc", [ a; b; x ] -> one (SP.betainc a b x)
  | "multigammaln", [ a ] ->
      one (SP.multigammaln a (U.to_int (U.member "d" params)))
  | "softmax", [ x ] -> (
      match U.member "axis" params with
      | `Null -> one (SP.softmax x)
      | j -> one (SP.softmax ~axis:(U.to_int j) x))
  | "log_softmax", [ x ] -> (
      match U.member "axis" params with
      | `Null -> one (SP.log_softmax x)
      | j -> one (SP.log_softmax ~axis:(U.to_int j) x))
  | "poch", [ z; m ] -> one (SP.poch z m)
  | "spence", [ x ] -> one (SP.spence x)
  | "sici", [ x ] -> SP.sici x
  | "owens_t", [ h; a ] -> one (SP.owens_t h a)
  | "bernoulli", [] -> one (SP.bernoulli (U.to_int (U.member "n" params)))
  | "bessel_jn", [ z ] ->
      one (SP.bessel_jn ~v:(U.to_int (U.member "v" params)) z)
  | "hyp1f1", [ a; b; x ] -> one (SP.hyp1f1 a b x)
  | "exp1", [ x ] -> one (SP.exp1 x)
  | "expn", [ n; x ] -> one (SP.expn n x)
  | _ -> failwith ("scipy_special golden: unknown op " ^ op)

let scipy_special_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "scipy_special") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:scipy_special_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("scipy_special:" ^ set_name, coverage :: case_tests)

let scipy_stats_fn op _params operands : T.value list =
  let one v = [ v ] in
  match (op, operands) with
  | "bernoulli.logpmf", [ k; p ] -> one (SS_bernoulli.logpmf k p)
  | "bernoulli.pmf", [ k; p ] -> one (SS_bernoulli.pmf k p)
  | "bernoulli.cdf", [ k; p ] -> one (SS_bernoulli.cdf k p)
  | "bernoulli.ppf", [ q; p ] -> one (SS_bernoulli.ppf q p)
  | "beta.logpdf", [ x; a; b ] -> one (SS_beta.logpdf x a b)
  | "beta.pdf", [ x; a; b ] -> one (SS_beta.pdf x a b)
  | "beta.cdf", [ x; a; b ] -> one (SS_beta.cdf x a b)
  | "beta.logcdf", [ x; a; b ] -> one (SS_beta.logcdf x a b)
  | "beta.sf", [ x; a; b ] -> one (SS_beta.sf x a b)
  | "beta.logsf", [ x; a; b ] -> one (SS_beta.logsf x a b)
  | "betabinom.logpmf", [ k; n; a; b ] -> one (SS_betabinom.logpmf k n a b)
  | "betabinom.pmf", [ k; n; a; b ] -> one (SS_betabinom.pmf k n a b)
  | "binom.logpmf", [ k; n; p ] -> one (SS_binom.logpmf k n p)
  | "binom.pmf", [ k; n; p ] -> one (SS_binom.pmf k n p)
  | "cauchy.logpdf", [ x ] -> one (SS_cauchy.logpdf x)
  | "cauchy.pdf", [ x ] -> one (SS_cauchy.pdf x)
  | "cauchy.cdf", [ x ] -> one (SS_cauchy.cdf x)
  | "cauchy.logcdf", [ x ] -> one (SS_cauchy.logcdf x)
  | "cauchy.sf", [ x ] -> one (SS_cauchy.sf x)
  | "cauchy.logsf", [ x ] -> one (SS_cauchy.logsf x)
  | "cauchy.isf", [ q ] -> one (SS_cauchy.isf q)
  | "cauchy.ppf", [ q ] -> one (SS_cauchy.ppf q)
  | "chi2.logpdf", [ x; df ] -> one (SS_chi2.logpdf x df)
  | "chi2.pdf", [ x; df ] -> one (SS_chi2.pdf x df)
  | "chi2.cdf", [ x; df ] -> one (SS_chi2.cdf x df)
  | "chi2.logcdf", [ x; df ] -> one (SS_chi2.logcdf x df)
  | "chi2.sf", [ x; df ] -> one (SS_chi2.sf x df)
  | "chi2.logsf", [ x; df ] -> one (SS_chi2.logsf x df)
  | "dirichlet.logpdf", [ x; alpha ] -> one (SS_dirichlet.logpdf x alpha)
  | "dirichlet.pdf", [ x; alpha ] -> one (SS_dirichlet.pdf x alpha)
  | "expon.logpdf", [ x ] -> one (SS_expon.logpdf x)
  | "expon.pdf", [ x ] -> one (SS_expon.pdf x)
  | "expon.cdf", [ x ] -> one (SS_expon.cdf x)
  | "expon.logcdf", [ x ] -> one (SS_expon.logcdf x)
  | "expon.sf", [ x ] -> one (SS_expon.sf x)
  | "expon.logsf", [ x ] -> one (SS_expon.logsf x)
  | "expon.ppf", [ q ] -> one (SS_expon.ppf q)
  | "gamma.logpdf", [ x; a ] -> one (SS_gamma.logpdf x a)
  | "gamma.pdf", [ x; a ] -> one (SS_gamma.pdf x a)
  | "gamma.cdf", [ x; a ] -> one (SS_gamma.cdf x a)
  | "gamma.logcdf", [ x; a ] -> one (SS_gamma.logcdf x a)
  | "gamma.sf", [ x; a ] -> one (SS_gamma.sf x a)
  | "gamma.logsf", [ x; a ] -> one (SS_gamma.logsf x a)
  | "gennorm.logpdf", [ x; beta ] -> one (SS_gennorm.logpdf x beta)
  | "gennorm.pdf", [ x; beta ] -> one (SS_gennorm.pdf x beta)
  | "gennorm.cdf", [ x; beta ] -> one (SS_gennorm.cdf x beta)
  | "geom.logpmf", [ k; p ] -> one (SS_geom.logpmf k p)
  | "geom.pmf", [ k; p ] -> one (SS_geom.pmf k p)
  | "gumbel_l.logpdf", [ x ] -> one (SS_gumbel_l.logpdf x)
  | "gumbel_l.pdf", [ x ] -> one (SS_gumbel_l.pdf x)
  | "gumbel_l.logcdf", [ x ] -> one (SS_gumbel_l.logcdf x)
  | "gumbel_l.cdf", [ x ] -> one (SS_gumbel_l.cdf x)
  | "gumbel_l.ppf", [ p ] -> one (SS_gumbel_l.ppf p)
  | "gumbel_l.logsf", [ x ] -> one (SS_gumbel_l.logsf x)
  | "gumbel_l.sf", [ x ] -> one (SS_gumbel_l.sf x)
  | "gumbel_r.logpdf", [ x ] -> one (SS_gumbel_r.logpdf x)
  | "gumbel_r.pdf", [ x ] -> one (SS_gumbel_r.pdf x)
  | "gumbel_r.logcdf", [ x ] -> one (SS_gumbel_r.logcdf x)
  | "gumbel_r.cdf", [ x ] -> one (SS_gumbel_r.cdf x)
  | "gumbel_r.ppf", [ p ] -> one (SS_gumbel_r.ppf p)
  | "gumbel_r.sf", [ x ] -> one (SS_gumbel_r.sf x)
  | "gumbel_r.logsf", [ x ] -> one (SS_gumbel_r.logsf x)
  | "laplace.logpdf", [ x ] -> one (SS_laplace.logpdf x)
  | "laplace.pdf", [ x ] -> one (SS_laplace.pdf x)
  | "laplace.cdf", [ x ] -> one (SS_laplace.cdf x)
  | "logistic.logpdf", [ x ] -> one (SS_logistic.logpdf x)
  | "logistic.pdf", [ x ] -> one (SS_logistic.pdf x)
  | "logistic.cdf", [ x ] -> one (SS_logistic.cdf x)
  | "logistic.sf", [ x ] -> one (SS_logistic.sf x)
  | "logistic.ppf", [ x ] -> one (SS_logistic.ppf x)
  | "logistic.isf", [ x ] -> one (SS_logistic.isf x)
  | "multinomial.logpmf", [ x; n; p ] -> one (SS_multinomial.logpmf x n p)
  | "multinomial.pmf", [ x; n; p ] -> one (SS_multinomial.pmf x n p)
  | "nbinom.logpmf", [ k; n; p ] -> one (SS_nbinom.logpmf k n p)
  | "nbinom.pmf", [ k; n; p ] -> one (SS_nbinom.pmf k n p)
  | "norm.logpdf", [ x ] -> one (SS_norm.logpdf x)
  | "norm.pdf", [ x ] -> one (SS_norm.pdf x)
  | "norm.cdf", [ x ] -> one (SS_norm.cdf x)
  | "norm.logcdf", [ x ] -> one (SS_norm.logcdf x)
  | "norm.ppf", [ q ] -> one (SS_norm.ppf q)
  | "norm.logsf", [ x ] -> one (SS_norm.logsf x)
  | "norm.sf", [ x ] -> one (SS_norm.sf x)
  | "norm.isf", [ q ] -> one (SS_norm.isf q)
  | "pareto.logpdf", [ x; b ] -> one (SS_pareto.logpdf x b)
  | "pareto.pdf", [ x; b ] -> one (SS_pareto.pdf x b)
  | "pareto.cdf", [ x; b ] -> one (SS_pareto.cdf x b)
  | "pareto.logcdf", [ x; b ] -> one (SS_pareto.logcdf x b)
  | "pareto.logsf", [ x; b ] -> one (SS_pareto.logsf x b)
  | "pareto.sf", [ x; b ] -> one (SS_pareto.sf x b)
  | "pareto.ppf", [ q; b ] -> one (SS_pareto.ppf q b)
  | "poisson.logpmf", [ k; mu ] -> one (SS_poisson.logpmf k mu)
  | "poisson.pmf", [ k; mu ] -> one (SS_poisson.pmf k mu)
  | "poisson.cdf", [ k; mu ] -> one (SS_poisson.cdf k mu)
  | "poisson.entropy", [ mu ] -> one (SS_poisson.entropy mu)
  | "t.logpdf", [ x; df ] -> one (SS_t.logpdf x df)
  | "t.pdf", [ x; df ] -> one (SS_t.pdf x df)
  | "truncnorm.logpdf", [ x; a; b ] -> one (SS_truncnorm.logpdf x a b)
  | "truncnorm.pdf", [ x; a; b ] -> one (SS_truncnorm.pdf x a b)
  | "truncnorm.logcdf", [ x; a; b ] -> one (SS_truncnorm.logcdf x a b)
  | "truncnorm.cdf", [ x; a; b ] -> one (SS_truncnorm.cdf x a b)
  | "truncnorm.logsf", [ x; a; b ] -> one (SS_truncnorm.logsf x a b)
  | "truncnorm.sf", [ x; a; b ] -> one (SS_truncnorm.sf x a b)
  | "uniform.logpdf", [ x ] -> one (SS_uniform.logpdf x)
  | "uniform.pdf", [ x ] -> one (SS_uniform.pdf x)
  | "uniform.cdf", [ x ] -> one (SS_uniform.cdf x)
  | "uniform.ppf", [ q ] -> one (SS_uniform.ppf q)
  | "vonmises.logpdf", [ x; kappa ] -> one (SS_vonmises.logpdf x kappa)
  | "vonmises.pdf", [ x; kappa ] -> one (SS_vonmises.pdf x kappa)
  | "wrapcauchy.logpdf", [ x; c ] -> one (SS_wrapcauchy.logpdf x c)
  | "wrapcauchy.pdf", [ x; c ] -> one (SS_wrapcauchy.pdf x c)
  | "core.sem", [ a ] -> one (SS_core.sem a)
  | "core.invert_permutation", [ i ] -> one (SS_core.invert_permutation i)
  | _ -> failwith ("scipy_stats golden: unknown op " ^ op)

let scipy_stats_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "scipy_stats") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:scipy_stats_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("scipy_stats:" ^ set_name, coverage :: case_tests)

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

module SI = Ojax.Scipy.Integrate
module SN = Ojax.Scipy.Ndimage

let scipy_integrate_fn op params operands : T.value list =
  let dx = opt_f params "dx" in
  let axis = opt_i params "axis" in
  match (op, operands) with
  | "trapezoid", [ y ] -> [ SI.trapezoid ?dx ?axis y ]
  | "trapezoid", [ y; x ] -> [ SI.trapezoid ~x ?dx ?axis y ]
  | _ -> failwith ("scipy_integrate golden: unknown op " ^ op)

let scipy_integrate_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "scipy_integrate") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:scipy_integrate_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("scipy_integrate:" ^ set_name, coverage :: case_tests)

let scipy_ndimage_fn op params operands : T.value list =
  let order = U.to_int (U.member "order" params) in
  let mode = Option.value ~default:"constant" (opt_s params "mode") in
  let cval = Option.value ~default:0.0 (opt_f params "cval") in
  match (op, operands) with
  | "map_coordinates", input :: coords ->
      [ SN.map_coordinates ~mode ~cval input coords ~order ]
  | _ -> failwith ("scipy_ndimage golden: unknown op " ^ op)

let scipy_ndimage_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "scipy_ndimage") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick
          (numpy_check_case ~fn:scipy_ndimage_fn ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("scipy_ndimage:" ^ set_name, coverage :: case_tests)

let flat_decode flat shape =
  let n = Array.length shape in
  let idx = Array.make n 0 in
  let f = ref flat in
  for d = n - 1 downto 0 do
    idx.(d) <- !f mod shape.(d);
    f := !f / shape.(d)
  done;
  idx

let nd_of_npz_any (a : Npz.t) =
  let dt = dtype_of_string a.Npz.dtype in
  match a.Npz.data with
  | Npz.C c -> Nd.of_complex dt a.Npz.shape c
  | Npz.F f -> Nd.of_floats dt a.Npz.shape f
  | Npz.I i -> Nd.of_floats dt a.Npz.shape (Array.map Int64.to_float i)

let npz_of_nd nd golden_dtype compare =
  let sh = Nd.shape nd in
  match Nd.dtype nd with
  | D.Complex64 | D.Complex128 ->
      let n = Array.fold_left ( * ) 1 sh in
      let c = Array.init n (fun i -> Nd.get_c nd (flat_decode i sh)) in
      { Npz.dtype = golden_dtype; shape = sh; data = Npz.C c }
  | _ ->
      let floats = read_nd nd in
      let data =
        if compare = "exact" then Npz.I (Array.map Int64.of_float floats)
        else Npz.F floats
      in
      { Npz.dtype = golden_dtype; shape = sh; data }

let complex_fn c operands : T.value list =
  if String.length c.primitive >= 4 && String.sub c.primitive 0 4 = "lax." then
    C.bind (prim_of c.op c.params) operands
  else
    let one v = [ v ] in
    match (c.op, operands) with
    | "angle", [ x ] ->
        let deg = Option.value ~default:false (opt_b c.params "deg") in
        one (NL.angle ~deg x)
    | "conjugate", [ x ] -> one (UF.conjugate x)
    | "imag", [ x ] -> one (UF.imag x)
    | "real", [ x ] -> one (UF.real x)
    | "iscomplex", [ x ] -> one (NL.iscomplex x)
    | "isreal", [ x ] -> one (NL.isreal x)
    | _ -> failwith ("complex golden: unknown op " ^ c.op)

let complex_check_case ~set_dir ~x64 c () =
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
      (fun a -> T.Concrete (nd_of_npz_any (find_member inputs a.name)))
      c.args
  in
  let results = complex_fn c operands in
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
      let actual = npz_of_nd nd golden.Npz.dtype ocompare in
      Compare.assert_tol_widened o.odtype oatol ortol oreason;
      Compare.check
        ~name:(c.case_id ^ ":" ^ o.oname)
        ~compare:ocompare ~atol:oatol ~rtol:ortol ~expected:golden ~actual)
    paired

let complex_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "complex") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick (complex_check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("complex:" ^ set_name, coverage :: case_tests)

module LL = Ojax.Lax.Linalg

let linalg_run c inputs =
  let get name = T.Concrete (nd_of_npz (find_member inputs name)) in
  let pb name = U.member name c.params |> U.to_bool in
  let pi name = U.member name c.params |> U.to_int in
  match c.op with
  | "cholesky" -> [ LL.cholesky (get "a") ]
  | "lu" ->
      let a, b, cc = LL.lu (get "a") in
      [ a; b; cc ]
  | "qr" ->
      let q, r = LL.qr ~full_matrices:(pb "full_matrices") (get "a") in
      [ q; r ]
  | "householder_product" -> [ LL.householder_product (get "a") (get "taus") ]
  | "lu_pivots_to_permutation" ->
      [
        LL.lu_pivots_to_permutation ~permutation_size:(pi "permutation_size")
          (get "pivots");
      ]
  | "triangular_solve" ->
      [
        LL.triangular_solve ~left_side:(pb "left_side") ~lower:(pb "lower")
          ~transpose_a:(pb "transpose_a") ~unit_diagonal:(pb "unit_diagonal")
          (get "a") (get "b");
      ]
  | "tridiagonal_solve" ->
      [ LL.tridiagonal_solve (get "dl") (get "d") (get "du") (get "b") ]
  | _ -> failwith ("linalg golden: unknown op " ^ c.op)

let linalg_check_case ~set_dir ~x64 c () =
  let canon d = if x64 then d else Compare.canonical_dtype_x64_off d in
  let inputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "inputs") (c.case_id ^ ".npz"))
  in
  let outputs =
    Npz.read
      (Filename.concat (Filename.concat set_dir "outputs") (c.case_id ^ ".npz"))
  in
  let results = linalg_run c inputs in
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
      let oreason = o.otreason in
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

let linalg_suite_for set_name =
  let set_dir =
    Filename.concat (Filename.concat goldens_root "linalg") set_name
  in
  let x64, cases = load_manifest (Filename.concat set_dir "manifest.json") in
  let case_tests =
    List.map
      (fun c ->
        Alcotest.test_case c.case_id `Quick (linalg_check_case ~set_dir ~x64 c))
      cases
  in
  let coverage =
    Alcotest.test_case "coverage" `Quick (check_coverage ~set_dir cases)
  in
  ("linalg:" ^ set_name, coverage :: case_tests)

let () =
  Ojax.Lax.install ();
  Ojax.Random.Prng.install ();
  Ojax.Random.Threefry.install ();
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
      solves_suite_for "x64_off";
      solves_suite_for "x64_on";
      numpy_suite_for "x64_off";
      numpy_suite_for "x64_on";
      ufuncs_suite_for "x64_off";
      ufuncs_suite_for "x64_on";
      reductions_suite_for "x64_off";
      reductions_suite_for "x64_on";
      indexing_suite_for "x64_off";
      indexing_suite_for "x64_on";
      array_methods_suite_for "x64_off";
      array_methods_suite_for "x64_on";
      creation_suite_for "x64_off";
      creation_suite_for "x64_on";
      contractions_suite_for "x64_off";
      contractions_suite_for "x64_on";
      setops_suite_for "x64_off";
      setops_suite_for "x64_on";
      polynomial_suite_for "x64_off";
      polynomial_suite_for "x64_on";
      prng_suite_for "x64_off";
      prng_suite_for "x64_on";
      random_core_suite_for "x64_off";
      random_core_suite_for "x64_on";
      nn_suite_for "x64_off";
      nn_suite_for "x64_on";
      initializers_suite_for "x64_off";
      initializers_suite_for "x64_on";
      segment_suite_for "x64_off";
      segment_suite_for "x64_on";
      special_suite_for "x64_off";
      special_suite_for "x64_on";
      image_suite_for "x64_off";
      image_suite_for "x64_on";
      scipy_special_suite_for "x64_off";
      scipy_special_suite_for "x64_on";
      scipy_stats_suite_for "x64_off";
      scipy_stats_suite_for "x64_on";
      scipy_integrate_suite_for "x64_off";
      scipy_integrate_suite_for "x64_on";
      scipy_ndimage_suite_for "x64_off";
      scipy_ndimage_suite_for "x64_on";
      complex_suite_for "x64_off";
      complex_suite_for "x64_on";
      linalg_suite_for "x64_off";
      linalg_suite_for "x64_on";
      ("compare", [ Alcotest.test_case "semantics" `Quick compare_tests ]);
    ]
