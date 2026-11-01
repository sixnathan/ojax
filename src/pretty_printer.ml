open Types

module Doc = struct
  type t = Text of string | Concat of t list

  let text s = Text s
  let concat ts = Concat ts

  let rec flat = function
    | Text s -> s
    | Concat ts -> String.concat "" (List.map flat ts)

  let to_string d = flat d
end

let encode_var n =
  let rec go k acc =
    let acc = String.make 1 (Char.chr (97 + (k mod 26))) ^ acc in
    let k = (k / 26) - 1 in
    if k < 0 then acc else go k acc
  in
  go n ""

let python_repr_float x =
  if x <> x then "nan"
  else if x = infinity then "inf"
  else if x = neg_infinity then "-inf"
  else if x = 0.0 then if 1.0 /. x = neg_infinity then "-0.0" else "0.0"
  else begin
    let neg = x < 0.0 in
    let ax = abs_float x in
    let rec find p =
      if p >= 17 then 17
      else if float_of_string (Printf.sprintf "%.*e" (p - 1) ax) = ax then p
      else find (p + 1)
    in
    let sig_digits = find 1 in
    let s = Printf.sprintf "%.*e" (sig_digits - 1) ax in
    let mantissa, exp =
      match String.split_on_char 'e' s with
      | [ m; e ] -> (m, int_of_string e)
      | _ -> (s, 0)
    in
    let raw = String.concat "" (String.split_on_char '.' mantissa) in
    let digits =
      let n = ref (String.length raw) in
      while !n > 1 && raw.[!n - 1] = '0' do
        decr n
      done;
      String.sub raw 0 !n
    in
    let ndigits = String.length digits in
    let decpt = exp + 1 in
    let body =
      if decpt <= -4 || decpt > 16 then begin
        let m =
          if ndigits = 1 then digits
          else String.sub digits 0 1 ^ "." ^ String.sub digits 1 (ndigits - 1)
        in
        let e = decpt - 1 in
        Printf.sprintf "%se%c%02d" m (if e < 0 then '-' else '+') (abs e)
      end
      else if decpt <= 0 then "0." ^ String.make (-decpt) '0' ^ digits
      else if decpt >= ndigits then
        digits ^ String.make (decpt - ndigits) '0' ^ ".0"
      else
        String.sub digits 0 decpt ^ "."
        ^ String.sub digits decpt (ndigits - decpt)
    in
    if neg then "-" ^ body else body
  end

let shape_str shape =
  "[" ^ String.concat "," (Array.to_list (Array.map string_of_int shape)) ^ "]"

let aval_short (a : aval) = Dtype.short_name a.dtype ^ shape_str a.shape

let int_tuple arr =
  "(" ^ String.concat "," (Array.to_list (Array.map string_of_int arr)) ^ ")"

let dot_dims_str (dd : dot_dims) =
  "((" ^ int_tuple dd.lhs_contract ^ "," ^ int_tuple dd.rhs_contract ^ "),("
  ^ int_tuple dd.lhs_batch ^ "," ^ int_tuple dd.rhs_batch ^ "))"

let prim_name = function
  | Add -> "add"
  | Sub -> "sub"
  | Mul -> "mul"
  | Div -> "div"
  | Neg -> "neg"
  | Sin -> "sin"
  | Cos -> "cos"
  | Exp -> "exp"
  | Log -> "log"
  | Tanh -> "tanh"
  | Max -> "max"
  | Min -> "min"
  | Pow -> "pow"
  | Abs -> "abs"
  | Sign -> "sign"
  | Eq -> "eq"
  | Lt -> "lt"
  | Gt -> "gt"
  | Select_n -> "select_n"
  | Convert_element_type _ -> "convert_element_type"
  | Broadcast_in_dim _ -> "broadcast_in_dim"
  | Reshape _ -> "reshape"
  | Reduce_sum _ -> "reduce_sum"
  | Dot_general _ -> "dot_general"
  | Acos -> "acos"
  | Acosh -> "acosh"
  | Asin -> "asin"
  | Asinh -> "asinh"
  | Atan -> "atan"
  | Atanh -> "atanh"
  | Cbrt -> "cbrt"
  | Ceil -> "ceil"
  | Clz -> "clz"
  | Conj -> "conj"
  | Copy -> "copy"
  | Cosh -> "cosh"
  | Exp2 -> "exp2"
  | Expm1 -> "expm1"
  | Floor -> "floor"
  | Imag -> "imag"
  | Integer_pow _ -> "integer_pow"
  | Is_finite -> "is_finite"
  | Log1p -> "log1p"
  | Logistic -> "logistic"
  | Not -> "not"
  | Population_count -> "population_count"
  | Real -> "real"
  | Round | Round_nearest_even -> "round"
  | Rsqrt -> "rsqrt"
  | Sinh -> "sinh"
  | Sqrt -> "sqrt"
  | Square -> "square"
  | Tan -> "tan"
  | And -> "and"
  | Atan2 -> "atan2"
  | Complex -> "complex"
  | Eq_to -> "eq_to"
  | Ge -> "ge"
  | Le -> "le"
  | Le_to -> "le_to"
  | Lt_to -> "lt_to"
  | Mulhi -> "mulhi"
  | Ne -> "ne"
  | Nextafter -> "nextafter"
  | Or -> "or"
  | Rem -> "rem"
  | Shift_left -> "shift_left"
  | Shift_right_arithmetic -> "shift_right_arithmetic"
  | Shift_right_logical -> "shift_right_logical"
  | Xor -> "xor"
  | Concatenate _ -> "concatenate"
  | Pad _ -> "pad"
  | Rev _ -> "rev"
  | Split _ -> "split"
  | Squeeze _ -> "squeeze"
  | Stack _ -> "stack"
  | Tile _ -> "tile"
  | Transpose _ -> "transpose"
  | Unstack _ -> "unstack"
  | Argmax _ -> "argmax"
  | Argmin _ -> "argmin"
  | Reduce _ -> "reduce"
  | Reduce_and _ -> "reduce_and"
  | Reduce_max _ -> "reduce_max"
  | Reduce_min _ -> "reduce_min"
  | Reduce_or _ -> "reduce_or"
  | Reduce_prod _ -> "reduce_prod"
  | Reduce_xor _ -> "reduce_xor"
  | After_all -> "after_all"
  | Ffi_call _ -> "ffi_call"
  | Bitcast_convert_type _ -> "bitcast_convert_type"
  | Clamp -> "clamp"
  | Composite _ -> "composite"
  | Create_token -> "create_token"
  | Dce_sink -> "dce_sink"
  | Empty _ -> "empty"
  | Empty2 _ -> "empty2"
  | From_edtype _ -> "from_edtype"
  | Iota _ -> "iota"
  | Optimization_barrier -> "optimization_barrier"
  | Ragged_dot_general -> "ragged_dot_general"
  | Reduce_precision _ -> "reduce_precision"
  | Rng_bit_generator -> "rng_bit_generator"
  | Rng_uniform -> "rng_uniform"
  | Sort _ -> "sort"
  | Tie -> "tie"
  | To_edtype _ -> "to_edtype"
  | Top_k _ -> "top_k"
  | Slice _ -> "slice"
  | Dynamic_slice _ -> "dynamic_slice"
  | Dynamic_update_slice -> "dynamic_update_slice"
  | Gather _ -> "gather"
  | Scatter _ -> "scatter"
  | Scatter_add _ -> "scatter_add"
  | Scatter_sub _ -> "scatter_sub"
  | Scatter_mul _ -> "scatter_mul"
  | Scatter_min _ -> "scatter_min"
  | Scatter_max _ -> "scatter_max"
  | Conv_general_dilated _ -> "conv_general_dilated"
  | Reduce_window _ -> "reduce_window"
  | Reduce_window_max _ -> "reduce_window_max"
  | Reduce_window_min _ -> "reduce_window_min"
  | Reduce_window_sum _ -> "reduce_window_sum"
  | Select_and_gather_add _ -> "select_and_gather_add"
  | Select_and_scatter _ -> "select_and_scatter"
  | Select_and_scatter_add _ -> "select_and_scatter_add"
  | Bessel_i0e -> "bessel_i0e"
  | Bessel_i1e -> "bessel_i1e"
  | Digamma -> "digamma"
  | Erf -> "erf"
  | Erf_inv -> "erf_inv"
  | Erfc -> "erfc"
  | Igamma -> "igamma"
  | Igamma_grad_a -> "igamma_grad_a"
  | Igammac -> "igammac"
  | Lgamma -> "lgamma"
  | Polygamma -> "polygamma"
  | Regularized_incomplete_beta -> "regularized_incomplete_beta"
  | Zeta -> "zeta"
  | Platform_index _ -> "platform_index"
  | Xla_call _ -> "xla_call"
  | Collective _ -> "collective"
  | Fft _ -> "fft"
  | Custom_jvp_call _ -> "custom_jvp_call"
  | Custom_vjp_call _ -> "custom_vjp_call"
  | Custom_lin _ -> "custom_lin"
  | Checkpoint_name _ -> "name"
  | Cond _ -> "cond"
  | Scan _ -> "scan"
  | While _ -> "while"
  | Cumsum _ -> "cumsum"
  | Cumprod _ -> "cumprod"
  | Cummax _ -> "cummax"
  | Cummin _ -> "cummin"
  | Cumlogsumexp _ -> "cumlogsumexp"
  | Custom_linear_solve _ -> "custom_linear_solve"
  | Threefry2x32 -> "threefry2x32"
  | Iota_2x32_shape _ -> "iota_2x32_shape"
  | Random_seed -> "random_seed"
  | Random_split _ -> "random_split"
  | Random_fold_in -> "random_fold_in"
  | Random_bits _ -> "random_bits"
  | Random_wrap -> "random_wrap"
  | Random_unwrap -> "random_unwrap"
  | Cholesky -> "cholesky"
  | Householder_product -> "householder_product"
  | Lu -> "lu"
  | Lu_pivots_to_permutation _ -> "lu_pivots_to_permutation"
  | Qr _ -> "qr"
  | Triangular_solve _ -> "triangular_solve"
  | Tridiagonal_solve _ -> "tridiagonal_solve"
  | Eig _ -> "eig"
  | Eigh _ -> "eigh"
  | Hessenberg -> "hessenberg"
  | Schur _ -> "schur"
  | Svd _ -> "svd"
  | Tridiagonal _ -> "tridiagonal"

let primitive_name p = prim_name p

let prim_params = function
  | Convert_element_type dt -> "[new_dtype=" ^ Dtype.short_name dt ^ "]"
  | Broadcast_in_dim { shape; dims } ->
      "[broadcast_dimensions=" ^ int_tuple dims ^ " shape=" ^ int_tuple shape
      ^ "]"
  | Reshape ns -> "[new_sizes=" ^ int_tuple ns ^ "]"
  | Reduce_sum axes -> "[axes=" ^ int_tuple axes ^ "]"
  | Dot_general dd -> "[dimension_numbers=" ^ dot_dims_str dd ^ "]"
  | Integer_pow y -> "[y=" ^ string_of_int y ^ "]"
  | Cumsum { axis; reverse }
  | Cumprod { axis; reverse }
  | Cummax { axis; reverse }
  | Cummin { axis; reverse }
  | Cumlogsumexp { axis; reverse } ->
      "[axis=" ^ string_of_int axis ^ " reverse="
      ^ (if reverse then "True" else "False")
      ^ "]"
  | _ -> ""

let lit_short nd =
  Dtype.short_name (Ndarray.dtype nd) ^ shape_str (Ndarray.shape nd)

let lit_value nd =
  let scalar = [||] in
  match Ndarray.dtype nd with
  | Dtype.Bool -> if Ndarray.get_f nd scalar = 0.0 then "False" else "True"
  | Dtype.Float0 -> "(b'',)"
  | Dtype.I32 | Dtype.I64 | Dtype.Uint32 | Dtype.I8 | Dtype.I16 | Dtype.I2
  | Dtype.I4 | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4
  | Dtype.Uint64 ->
      Int64.to_string (Ndarray.get_i64 nd scalar)
  | Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn
  | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
  | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
  | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
      python_repr_float (Ndarray.get_f nd scalar)
  | Dtype.Complex64 | Dtype.Complex128 ->
      let z = Ndarray.get_c nd scalar in
      let im = python_repr_float z.Complex.im in
      let sign = if String.length im > 0 && im.[0] = '-' then "" else "+" in
      "(" ^ python_repr_float z.Complex.re ^ sign ^ im ^ "j)"

let lit_str nd = lit_value nd ^ ":" ^ lit_short nd

let jaxpr_to_doc (jx : jaxpr) =
  let names : (int, string) Hashtbl.t = Hashtbl.create 16 in
  let counter = ref 0 in
  let bind_var (v : var) =
    if not (Hashtbl.mem names v.vid) then begin
      Hashtbl.replace names v.vid (encode_var !counter);
      incr counter
    end
  in
  List.iter bind_var jx.in_binders;
  List.iter (fun (e : eqn) -> List.iter bind_var e.outs) jx.eqns;
  let var_name (v : var) =
    match Hashtbl.find_opt names v.vid with
    | Some n -> n
    | None -> invalid_arg "pretty_printer: unbound variable"
  in
  let atom_str = function
    | A_var v -> var_name v
    | A_lit nd -> lit_str nd
    | DropVar _ -> "_"
  in
  let binder_str (v : var) = var_name v ^ ":" ^ aval_short v.vaval in
  let eqn_doc (e : eqn) =
    let lhs = String.concat " " (List.map binder_str e.outs) in
    let rhs =
      prim_name e.prim ^ prim_params e.prim ^ " "
      ^ String.concat " " (List.map atom_str e.inputs)
    in
    Doc.text (lhs ^ " = " ^ rhs)
  in
  let binders = String.concat ", " (List.map binder_str jx.in_binders) in
  let eqns =
    Doc.concat
      (match jx.eqns with
      | [] -> []
      | first :: rest ->
          eqn_doc first
          :: List.concat_map (fun e -> [ Doc.text " ; "; eqn_doc e ]) rest)
  in
  let outs = String.concat ", " (List.map atom_str jx.outs) in
  Doc.concat
    [
      Doc.text ("{ lambda " ^ binders ^ " . let ");
      eqns;
      Doc.text (" in ( " ^ outs ^ " ) }");
    ]

let jaxpr_to_string jx = Doc.to_string (jaxpr_to_doc jx)
let closed_jaxpr_to_string (cj : closed_jaxpr) = jaxpr_to_string cj.jaxpr
let call_names : (int, string) Hashtbl.t = Hashtbl.create 64
let register_call_name jid name = Hashtbl.replace call_names jid name
let registered_call_name jid = Hashtbl.find_opt call_names jid

let call_name jid =
  match Hashtbl.find_opt call_names jid with Some n -> n | None -> "jit"

let custom_call :
    (int -> aval list -> (string * string * closed_jaxpr) option) ref =
  ref (fun _ _ -> None)

let atom_type = function
  | A_var (v : var) -> v.vaval
  | A_lit nd ->
      { shape = Ndarray.shape nd; dtype = Ndarray.dtype nd; weak_type = false }
  | DropVar a -> a

let custom_call_of (e : eqn) =
  match e.prim with
  | Custom_jvp_call id -> !custom_call id (List.map atom_type e.inputs)
  | _ -> None

let rec structure_of_jaxpr (jx : jaxpr) =
  String.concat " " (List.map structure_of_eqn jx.eqns)

and structure_of_eqn (e : eqn) =
  let nested =
    match e.prim with
    | Reduce { jaxpr; _ } | Scan { jaxpr; _ } -> [ jaxpr ]
    | Composite c | Xla_call c -> [ c ]
    | Reduce_window { reducer; _ } -> [ reducer ]
    | Select_and_scatter { select_jaxpr; scatter_jaxpr; _ } ->
        [ scatter_jaxpr; select_jaxpr ]
    | Cond { t; f } -> [ f; t ]
    | While { cond; body } -> [ body; cond ]
    | Custom_linear_solve { solve; transpose_solve } ->
        solve :: Option.to_list transpose_solve
    | Custom_jvp_call _ -> (
        match custom_call_of e with Some (_, _, c) -> [ c ] | None -> [])
    | _ -> []
  in
  let name = match e.prim with Xla_call _ -> "jit" | p -> prim_name p in
  match nested with
  | [] -> name
  | cs ->
      name ^ "("
      ^ String.concat " | "
          (List.map (fun (c : closed_jaxpr) -> structure_of_jaxpr c.jaxpr) cs)
      ^ ")"

let primitive_structure (cj : closed_jaxpr) = structure_of_jaxpr cj.jaxpr

module Layout = struct
  type t =
    | Nil
    | Text of string
    | Concat of t list
    | Brk of string
    | Group of t
    | Nest of int * t

  let rec join sep = function
    | [] -> Nil
    | [ d ] -> d
    | d :: rest -> Concat [ d; sep; join sep rest ]

  let fits doc width =
    let rec go width = function
      | [] -> width >= 0
      | _ when width < 0 -> false
      | Nil :: rest -> go width rest
      | Text s :: rest | Brk s :: rest -> go (width - String.length s) rest
      | Concat ds :: rest -> go width (ds @ rest)
      | (Group d | Nest (_, d)) :: rest -> go width (d :: rest)
    in
    go width [ doc ]

  let render doc =
    let out = Buffer.create 256 in
    let rec go k = function
      | [] -> ()
      | (_, _, Nil) :: rest -> go k rest
      | (_, _, Text s) :: rest ->
          Buffer.add_string out s;
          go (k + String.length s) rest
      | (i, flat, Concat ds) :: rest ->
          go k (List.map (fun d -> (i, flat, d)) ds @ rest)
      | (i, flat, Nest (n, d)) :: rest -> go k ((i + n, flat, d) :: rest)
      | (i, _, Group d) :: rest -> go k ((i, fits d (80 - k), d) :: rest)
      | (i, flat, Brk s) :: rest ->
          if flat then begin
            Buffer.add_string out s;
            go (k + String.length s) rest
          end
          else begin
            Buffer.add_char out '\n';
            Buffer.add_string out (String.make i ' ');
            go i rest
          end
    in
    go 0 [ (0, false, doc) ];
    Buffer.contents out
end

type jax_param =
  | Plain of string
  | Sub of closed_jaxpr
  | Subs of closed_jaxpr list

let py_tuple items =
  match items with
  | [] -> "()"
  | [ x ] -> "(" ^ x ^ ",)"
  | _ -> "(" ^ String.concat ", " items ^ ")"

let py_ints a = py_tuple (Array.to_list (Array.map string_of_int a))
let py_bool b = if b then "True" else "False"

let py_list a =
  "[" ^ String.concat ", " (Array.to_list (Array.map string_of_int a)) ^ "]"

let py_pairs a =
  py_tuple (Array.to_list (Array.map (fun (x, y) -> py_ints [| x; y |]) a))

let py_triples a =
  py_tuple
    (Array.to_list (Array.map (fun (x, y, z) -> py_ints [| x; y; z |]) a))

let gather_numbers (d : gather_dims) =
  Printf.sprintf
    "GatherDimensionNumbers(offset_dims=%s, collapsed_slice_dims=%s, \
     start_index_map=%s, operand_batching_dims=%s, \
     start_indices_batching_dims=%s)"
    (py_ints d.offset_dims)
    (py_ints d.collapsed_slice_dims)
    (py_ints d.start_index_map)
    (py_ints d.g_operand_batching_dims)
    (py_ints d.g_start_indices_batching_dims)

let scatter_numbers (d : scatter_dims) =
  Printf.sprintf
    "ScatterDimensionNumbers(update_window_dims=%s, inserted_window_dims=%s, \
     scatter_dims_to_operand_dims=%s, operand_batching_dims=%s, \
     scatter_indices_batching_dims=%s)"
    (py_ints d.update_window_dims)
    (py_ints d.inserted_window_dims)
    (py_ints d.scatter_dims_to_operand_dims)
    (py_ints d.s_operand_batching_dims)
    (py_ints d.s_scatter_indices_batching_dims)

let scatter_params d unique =
  [
    ("dimension_numbers", Plain (scatter_numbers d));
    ("indices_are_sorted", Plain "False");
    ("mode", Plain "GatherScatterMode.FILL_OR_DROP");
    ("unique_indices", Plain (py_bool unique));
    ("update_consts", Plain "()");
    ("update_jaxpr", Plain "None");
  ]

let window_params (w : window_dims) =
  [
    ("base_dilation", Plain (py_ints w.base_dilation));
    ("padding", Plain (py_pairs w.w_padding));
    ("window_dilation", Plain (py_ints w.window_dilation));
    ("window_dimensions", Plain (py_ints w.window_dimensions));
    ("window_strides", Plain (py_ints w.window_strides));
  ]

let jax_prim_name = function
  | Scatter_add _ -> "scatter-add"
  | Scatter_sub _ -> "scatter-sub"
  | Scatter_mul _ -> "scatter-mul"
  | Scatter_min _ -> "scatter-min"
  | Scatter_max _ -> "scatter-max"
  | Xla_call _ -> "jit"
  | p -> prim_name p

let jax_params (e : eqn) =
  let in_dtype () =
    match e.inputs with
    | A_var v :: _ -> v.vaval.dtype
    | A_lit nd :: _ -> Ndarray.dtype nd
    | DropVar a :: _ -> a.dtype
    | [] -> Dtype.F32
  in
  let out_dtype () =
    match e.outs with (v : var) :: _ -> v.vaval.dtype | [] -> Dtype.F32
  in
  let dn = Dtype.numpy_name in
  match e.prim with
  | Broadcast_in_dim { dims; _ } ->
      if Array.length dims = 0 then []
      else [ ("broadcast_dimensions", Plain (py_ints dims)) ]
  | Convert_element_type dt ->
      [ ("new_dtype", Plain (dn dt)); ("weak_type", Plain "False") ]
  | Bitcast_convert_type dt -> [ ("new_dtype", Plain (dn dt)) ]
  | Reshape ns ->
      [
        ("dimensions", Plain "None");
        ("new_sizes", Plain (py_ints ns));
        ("sharding", Plain "None");
      ]
  | Transpose perm -> [ ("permutation", Plain (py_ints perm)) ]
  | Reduce_sum axes ->
      [ ("axes", Plain (py_ints axes)); ("out_sharding", Plain "None") ]
  | Reduce_max axes
  | Reduce_min axes
  | Reduce_prod axes
  | Reduce_and axes
  | Reduce_or axes
  | Reduce_xor axes ->
      [ ("axes", Plain (py_ints axes)) ]
  | Dot_general d ->
      [
        ( "dimension_numbers",
          Plain
            (Printf.sprintf "((%s, %s), (%s, %s))" (py_list d.lhs_contract)
               (py_list d.rhs_contract) (py_list d.lhs_batch)
               (py_list d.rhs_batch)) );
        ("preferred_element_type", Plain (dn (out_dtype ())));
      ]
  | Iota { dtype; shape; dimension } ->
      [
        ("dimension", Plain (string_of_int dimension));
        ("dtype", Plain (dn dtype));
        ("shape", Plain (py_ints shape));
        ("sharding", Plain "None");
      ]
  | Slice { start_indices; limit_indices; strides } ->
      [
        ("limit_indices", Plain (py_ints limit_indices));
        ("start_indices", Plain (py_ints start_indices));
        ( "strides",
          Plain (match strides with Some s -> py_ints s | None -> "None") );
      ]
  | Squeeze dims | Rev dims -> [ ("dimensions", Plain (py_ints dims)) ]
  | Concatenate d -> [ ("dimension", Plain (string_of_int d)) ]
  | Pad cfg -> [ ("padding_config", Plain (py_triples cfg)) ]
  | Integer_pow y -> [ ("y", Plain (string_of_int y)) ]
  | Dynamic_slice { slice_sizes } ->
      [ ("slice_sizes", Plain (py_ints slice_sizes)) ]
  | Cumsum { axis; reverse }
  | Cumprod { axis; reverse }
  | Cummax { axis; reverse }
  | Cummin { axis; reverse }
  | Cumlogsumexp { axis; reverse } ->
      [
        ("axis", Plain (string_of_int axis));
        ("reverse", Plain (py_bool reverse));
      ]
  | Conj -> [ ("input_dtype", Plain (dn (in_dtype ()))) ]
  | Round -> [ ("rounding_method", Plain "0") ]
  | Round_nearest_even -> [ ("rounding_method", Plain "1") ]
  | Fft { fft_type; fft_lengths } ->
      let code =
        match fft_type with
        | Fft_forward -> "0"
        | Fft_inverse -> "1"
        | Fft_real -> "2"
        | Fft_inverse_real -> "3"
      in
      [ ("fft_lengths", Plain (py_ints fft_lengths)); ("fft_type", Plain code) ]
  | Sort { dimension; is_stable; num_keys } ->
      [
        ("dimension", Plain (string_of_int dimension));
        ("is_stable", Plain (py_bool is_stable));
        ("num_keys", Plain (string_of_int num_keys));
      ]
  | Argmax { axis; index_dtype } | Argmin { axis; index_dtype } ->
      [
        ("axes", Plain (py_ints [| axis |]));
        ("index_dtype", Plain (dn index_dtype));
      ]
  | Top_k { k; axis } ->
      [ ("axis", Plain (string_of_int axis)); ("k", Plain (string_of_int k)) ]
  | Gather { dimension_numbers; slice_sizes } ->
      [
        ("dimension_numbers", Plain (gather_numbers dimension_numbers));
        ("fill_value", Plain "None");
        ("indices_are_sorted", Plain "False");
        ("mode", Plain "GatherScatterMode.PROMISE_IN_BOUNDS");
        ("slice_sizes", Plain (py_ints slice_sizes));
        ("unique_indices", Plain "False");
      ]
  | Scatter { dimension_numbers; unique_indices }
  | Scatter_mul { dimension_numbers; unique_indices } ->
      scatter_params dimension_numbers unique_indices
  | Scatter_add { dimension_numbers }
  | Scatter_sub { dimension_numbers }
  | Scatter_min { dimension_numbers }
  | Scatter_max { dimension_numbers } ->
      scatter_params dimension_numbers false
  | Conv_general_dilated c ->
      [
        ("batch_group_count", Plain (string_of_int c.batch_group_count));
        ( "dimension_numbers",
          Plain
            (Printf.sprintf
               "ConvDimensionNumbers(lhs_spec=%s, rhs_spec=%s, out_spec=%s)"
               (py_ints c.dimension_numbers.lhs_spec)
               (py_ints c.dimension_numbers.rhs_spec)
               (py_ints c.dimension_numbers.out_spec)) );
        ("feature_group_count", Plain (string_of_int c.feature_group_count));
        ("lhs_dilation", Plain (py_ints c.lhs_dilation));
        ("out_sharding", Plain "None");
        ("padding", Plain (py_pairs c.padding));
        ("precision", Plain "None");
        ("preferred_element_type", Plain "None");
        ("rhs_dilation", Plain (py_ints c.rhs_dilation));
        ("window_strides", Plain (py_ints c.window_strides));
      ]
  | Reduce_window { reducer; window } ->
      List.sort compare
        (("consts", Plain "()") :: ("jaxpr", Sub reducer)
       :: window_params window)
  | Reduce_window_max w | Reduce_window_min w | Reduce_window_sum w ->
      window_params w
  | Reduce { jaxpr; dimensions } ->
      [
        ("computation", Plain "<function>");
        ("dimensions", Plain (py_ints dimensions));
        ("jaxpr", Sub jaxpr);
      ]
  | Triangular_solve t ->
      [
        ("conjugate_a", Plain (py_bool t.conjugate_a));
        ("left_side", Plain (py_bool t.left_side));
        ("lower", Plain (py_bool t.lower));
        ("transpose_a", Plain (py_bool t.transpose_a));
        ("unit_diagonal", Plain (py_bool t.unit_diagonal));
      ]
  | Qr full ->
      [
        ("full_matrices", Plain (py_bool full));
        ("pivoting", Plain "False");
        ("use_magma", Plain "None");
      ]
  | Eigh { lower } ->
      [
        ("algorithm", Plain "None");
        ("lower", Plain (py_bool lower));
        ("sort_eigenvalues", Plain "True");
        ("subset_by_index", Plain "None");
      ]
  | Svd { full_matrices; compute_uv; _ } ->
      [
        ("algorithm", Plain "None");
        ("compute_uv", Plain (py_bool compute_uv));
        ("full_matrices", Plain (py_bool full_matrices));
        ("subset_by_index", Plain "None");
      ]
  | Eig c ->
      [
        ("compute_left_eigenvectors", Plain (py_bool c.compute_left));
        ("compute_right_eigenvectors", Plain (py_bool c.compute_right));
        ("enable_eigvec_derivs", Plain (py_bool c.enable_eigvec_derivs));
        ("implementation", Plain "None");
      ]
  | Schur { compute_vectors } ->
      [
        ("compute_schur_vectors", Plain (py_bool compute_vectors));
        ("select_callable", Plain "None");
        ("sort_eig_vals", Plain "False");
      ]
  | Tridiagonal { lower } -> [ ("lower", Plain (py_bool lower)) ]
  | Tridiagonal_solve { perturb_singular } ->
      [ ("perturb_singular", Plain (py_bool perturb_singular)) ]
  | Lu_pivots_to_permutation n ->
      [ ("permutation_size", Plain (string_of_int n)) ]
  | Reduce_precision { exponent_bits; mantissa_bits } ->
      [
        ("exponent_bits", Plain (string_of_int exponent_bits));
        ("mantissa_bits", Plain (string_of_int mantissa_bits));
      ]
  | Tile reps -> [ ("reps", Plain (py_ints reps)) ]
  | Split { sizes; axis } ->
      [ ("axis", Plain (string_of_int axis)); ("sizes", Plain (py_ints sizes)) ]
  | Stack axis | Unstack axis -> [ ("axis", Plain (string_of_int axis)) ]
  | Random_bits { bit_width; shape } ->
      [
        ("bit_width", Plain (string_of_int bit_width));
        ("shape", Plain (py_ints shape));
      ]
  | Random_split shape -> [ ("shape", Plain (py_ints shape)) ]
  | Random_seed | Random_wrap -> [ ("impl", Plain "fry") ]
  | While { cond; body } ->
      [
        ("body_jaxpr", Sub body);
        ("body_nconsts", Plain (string_of_int (List.length body.consts)));
        ("cond_jaxpr", Sub cond);
        ("cond_nconsts", Plain (string_of_int (List.length cond.consts)));
      ]
  | Cond { t; f } -> [ ("branches", Subs [ f; t ]) ]
  | Scan { length; reverse; num_carry; jaxpr } ->
      [
        ("jaxpr", Sub jaxpr);
        ("length", Plain (string_of_int length));
        ("num_carry", Plain (string_of_int num_carry));
        ("num_consts", Plain (string_of_int (List.length jaxpr.consts)));
        ("reverse", Plain (py_bool reverse));
        ("unroll", Plain "1");
      ]
  | Xla_call c -> [ ("jaxpr", Sub c) ]
  | Select_and_scatter { select_jaxpr; scatter_jaxpr; window } ->
      List.sort compare
        (("scatter_jaxpr", Sub scatter_jaxpr)
        :: ("select_jaxpr", Sub select_jaxpr)
        :: window_params window)
  | _ -> []

let insertion_order (e : eqn) =
  match e.prim with
  | While { cond; body } -> [ cond; body ]
  | Cond { t; f } -> [ f; t ]
  | Select_and_scatter { select_jaxpr; scatter_jaxpr; _ } ->
      [ select_jaxpr; scatter_jaxpr ]
  | _ ->
      List.concat_map
        (fun (_, p) ->
          match p with Sub c -> [ c ] | Subs cs -> cs | Plain _ -> [])
        (jax_params e)

let py_complex_part x =
  let s = python_repr_float x in
  let n = String.length s in
  if n > 2 && String.sub s (n - 2) 2 = ".0" then String.sub s 0 (n - 2) else s

let py_complex (z : Complex.t) =
  if z.re = 0.0 && not (Float.sign_bit z.re) then py_complex_part z.im ^ "j"
  else
    let im = py_complex_part z.im in
    let sign = if String.length im > 0 && im.[0] = '-' then "" else "+" in
    "(" ^ py_complex_part z.re ^ sign ^ im ^ "j)"

let jax_literal nd =
  let body =
    if Array.length (Ndarray.shape nd) > 0 then "[...]"
    else
      match Ndarray.dtype nd with
      | Dtype.Complex64 | Dtype.Complex128 -> py_complex (Ndarray.get_c nd [||])
      | _ -> lit_value nd
  in
  body ^ ":" ^ lit_short nd

let alpha_name n =
  let rec go n acc =
    let acc = String.make 1 (Char.chr (97 + (n mod 26))) ^ acc in
    if n / 26 = 0 then acc else go (n / 26) acc
  in
  go n ""

type context = {
  mutable next_name : int;
  var_names : (int, string) Hashtbl.t;
  shared : (int, string) Hashtbl.t;
  shared_names : (string, unit) Hashtbl.t;
}

let rec fresh_var_name ctx =
  let name = alpha_name ctx.next_name in
  ctx.next_name <- ctx.next_name + 1;
  if Hashtbl.mem ctx.shared_names name then fresh_var_name ctx else name

let name_of_var ctx (v : var) =
  match Hashtbl.find_opt ctx.var_names v.vid with
  | Some n -> n
  | None ->
      let n = fresh_var_name ctx in
      Hashtbl.replace ctx.var_names v.vid n;
      n

let suggest_names ctx for_vars like_atoms =
  if List.length for_vars = List.length like_atoms then begin
    let used = Hashtbl.create 8 in
    List.iter2
      (fun for_atom like ->
        match (for_atom, like) with
        | A_var fv, A_var lv
          when (not (Hashtbl.mem used lv.vid))
               && not (Hashtbl.mem ctx.var_names fv.vid) ->
            Hashtbl.replace used lv.vid ();
            Hashtbl.replace ctx.var_names fv.vid (name_of_var ctx lv)
        | _ -> ())
      for_vars like_atoms
  end

let pp_atoms ctx ?(separator = "") atoms =
  let doc = function
    | A_var v -> Layout.Text (name_of_var ctx v)
    | A_lit nd -> Layout.Text (jax_literal nd)
    | DropVar _ -> Layout.Text "_"
  in
  Layout.Nest
    ( 2,
      Layout.Group
        (Layout.join
           (Layout.Concat
              [ Layout.Text separator; Layout.Group (Layout.Brk " ") ])
           (List.map doc atoms)) )

let pp_binders ctx (vars : var list) =
  let doc (v : var) =
    let name = name_of_var ctx v in
    Layout.Text (name ^ ":" ^ aval_short v.vaval)
  in
  Layout.Nest
    ( 2,
      Layout.Group
        (Layout.join (Layout.Group (Layout.Brk " ")) (List.map doc vars)) )

let rec pp_jaxpr ctx (jx : jaxpr) n_consts =
  let consts, args =
    let rec split k l =
      if k = 0 then ([], l)
      else
        match l with
        | x :: r ->
            let a, b = split (k - 1) r in
            (x :: a, b)
        | [] -> ([], [])
    in
    split n_consts jx.in_binders
  in
  let const_doc = pp_binders ctx consts in
  let arg_doc = pp_binders ctx args in
  let eqns = Layout.join (Layout.Brk "; ") (List.map (pp_eqn ctx) jx.eqns) in
  let outs =
    Layout.Concat
      [
        Layout.Text "(";
        pp_atoms ctx ~separator:"," jx.outs;
        Layout.Text (if List.length jx.outs <> 1 then ")" else ",)");
      ]
  in
  Layout.Concat
    [
      Layout.Group
        (Layout.Concat
           [
             Layout.Nest
               ( 2,
                 Layout.Concat
                   [
                     Layout.Text "{ lambda ";
                     const_doc;
                     Layout.Text "; ";
                     arg_doc;
                     Layout.Text ". let";
                     Layout.Nest (2, Layout.Concat [ Layout.Brk " "; eqns ]);
                     Layout.Brk " ";
                     Layout.Text "in ";
                     outs;
                   ] );
             Layout.Text " }";
           ]);
    ]

and pp_sub ctx (c : closed_jaxpr) =
  match Hashtbl.find_opt ctx.shared c.jid with
  | Some name -> Layout.Text name
  | None -> pp_jaxpr ctx c.jaxpr (List.length c.consts)

and pp_param ctx (key, value) =
  let body =
    match value with
    | Plain s -> Layout.Text s
    | Sub c -> pp_sub ctx c
    | Subs cs ->
        Layout.Group
          (Layout.Concat
             [
               Layout.Nest
                 ( 2,
                   Layout.Concat
                     [
                       Layout.Text "(";
                       Layout.Brk "";
                       Layout.join (Layout.Brk " ") (List.map (pp_sub ctx) cs);
                     ] );
               Layout.Brk "";
               Layout.Text ")";
             ])
  in
  Layout.Concat [ Layout.Text (key ^ "="); body ]

and pp_params ctx params =
  match params with
  | [] -> Layout.Nil
  | _ ->
      Layout.Group
        (Layout.Concat
           [
             Layout.Nest
               ( 2,
                 Layout.Concat
                   [
                     Layout.Text "[";
                     Layout.Brk "";
                     Layout.join (Layout.Brk " ")
                       (List.map (pp_param ctx) params);
                   ] );
             Layout.Brk "";
             Layout.Text "]";
           ])

and pp_eqn ctx (e : eqn) =
  let name_param =
    match e.prim with
    | Xla_call c ->
        if not (Hashtbl.mem ctx.shared c.jid) then begin
          let args =
            let rec drop k l = if k = 0 then l else drop (k - 1) (List.tl l) in
            drop (List.length c.consts) c.jaxpr.in_binders
          in
          suggest_names ctx (List.map (fun v -> A_var v) args) e.inputs;
          suggest_names ctx c.jaxpr.outs (List.map (fun v -> A_var v) e.outs)
        end;
        [ ("name", Plain (xla_call_name c)) ]
    | Custom_jvp_call _ -> (
        match custom_call_of e with
        | Some (name, jvp_name, c) ->
            [
              ("name", Plain name);
              ("call_jaxpr", Sub c);
              ("jvp", Plain jvp_name);
              ("symbolic_zeros", Plain "False");
            ]
        | None -> [])
    | _ -> []
  in
  let lhs = pp_binders ctx e.outs in
  let params = pp_params ctx (name_param @ jax_params e) in
  let rhs =
    Layout.Concat
      [
        Layout.Text (jax_prim_name e.prim);
        params;
        Layout.Text " ";
        pp_atoms ctx e.inputs;
      ]
  in
  if e.outs = [] then rhs else Layout.Concat [ lhs; Layout.Text " = "; rhs ]

and xla_call_name (c : closed_jaxpr) = call_name c.jid

let dropvars ctx (jx : jaxpr) =
  let used = Hashtbl.create 64 in
  let note = function
    | A_var (v : var) -> Hashtbl.replace used v.vid ()
    | A_lit _ | DropVar _ -> ()
  in
  List.iter note jx.outs;
  List.iter
    (fun (e : eqn) ->
      List.iter
        (fun (v : var) ->
          if not (Hashtbl.mem used v.vid) then
            Hashtbl.replace ctx.var_names v.vid "_")
        e.outs;
      List.iter note e.inputs)
    (List.rev jx.eqns)

let closed_jaxpr_to_jax_string (cj : closed_jaxpr) =
  let ctx =
    {
      next_name = 0;
      var_names = Hashtbl.create 64;
      shared = Hashtbl.create 8;
      shared_names = Hashtbl.create 8;
    }
  in
  dropvars ctx cj.jaxpr;
  let counts = Hashtbl.create 16 in
  let order = ref [] in
  let names = Hashtbl.create 16 in
  let bump (c : closed_jaxpr) n =
    match Hashtbl.find_opt counts c.jid with
    | None ->
        order := c :: !order;
        Hashtbl.replace counts c.jid n
    | Some k -> Hashtbl.replace counts c.jid (k + n)
  in
  let queue = Queue.create () in
  Queue.add (cj, true) queue;
  while not (Queue.is_empty queue) do
    let c, top = Queue.pop queue in
    bump c 1;
    if (not top) && List.length c.jaxpr.eqns > 10 then bump c 1;
    List.iter
      (fun (e : eqn) ->
        let subs = insertion_order e in
        (match e.prim with
        | Xla_call inner ->
            if not (Hashtbl.mem names inner.jid) then
              Hashtbl.replace names inner.jid (xla_call_name inner)
        | _ -> ());
        List.iter (fun s -> Queue.add (s, false) queue) subs)
      c.jaxpr.eqns
  done;
  let name_counts = Hashtbl.create 8 in
  let count name =
    Option.value (Hashtbl.find_opt name_counts name) ~default:0
  in
  let shared =
    List.filter_map
      (fun (c : closed_jaxpr) ->
        if Hashtbl.find counts c.jid = 1 then None
        else begin
          let base =
            Option.value (Hashtbl.find_opt names c.jid) ~default:"jaxpr"
          in
          let k = count base in
          let name =
            if k > 0 then begin
              Hashtbl.replace name_counts base (k + 1);
              let n = base ^ string_of_int k in
              Hashtbl.replace name_counts n (count n + 1);
              n
            end
            else begin
              Hashtbl.replace name_counts base 1;
              base
            end
          in
          Hashtbl.replace ctx.shared_names name ();
          Hashtbl.replace ctx.shared c.jid name;
          Some (name, c)
        end)
      (List.rev !order)
  in
  let docs =
    List.map
      (fun (name, (c : closed_jaxpr)) ->
        let saved = Hashtbl.find ctx.shared c.jid in
        Hashtbl.remove ctx.shared c.jid;
        let body = pp_jaxpr ctx c.jaxpr (List.length c.consts) in
        Hashtbl.replace ctx.shared c.jid saved;
        Layout.Concat
          [
            Layout.Text ("let " ^ name ^ " = ");
            body;
            Layout.Text " in";
            Layout.Brk " ";
          ])
      shared
  in
  let main = pp_jaxpr ctx cj.jaxpr (List.length cj.consts) in
  Layout.render (Layout.Concat (docs @ [ main ]))
