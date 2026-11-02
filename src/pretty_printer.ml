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

let custom_view_names : (int, string) Hashtbl.t = Hashtbl.create 8
let varying_axes : (int, string list) Hashtbl.t = Hashtbl.create 16
let aval_short (a : aval) = Dtype.short_name a.dtype ^ shape_str a.shape

let int_tuple arr =
  "(" ^ String.concat "," (Array.to_list (Array.map string_of_int arr)) ^ ")"

let dot_dims_str (dd : dot_dims) =
  "((" ^ int_tuple dd.lhs_contract ^ "," ^ int_tuple dd.rhs_contract ^ "),("
  ^ int_tuple dd.lhs_batch ^ "," ^ int_tuple dd.rhs_batch ^ "))"

let custom_lin_vjp : (int -> bool) ref = ref (fun _ -> false)

let callback_views : (int, string * (string * string) list) Hashtbl.t =
  Hashtbl.create 8

let register_callback_view id name params =
  Hashtbl.replace callback_views id (name, params)

let prim_name = function
  | Add -> "add"
  | Add_any -> "add_any"
  | Stop_gradient -> "stop_gradient"
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
  | Reshape _ | Reshape_perm _ -> "reshape"
  | Decomposable { dname; _ } -> dname
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
  | Dce_sink _ -> "dce_sink"
  | Empty _ -> "empty"
  | Empty2 _ -> "empty2"
  | From_edtype _ -> "from_edtype"
  | Iota _ -> "iota"
  | Optimization_barrier -> "optimization_barrier"
  | Ragged_dot_general -> "ragged_dot_general"
  | Reduce_precision _ -> "reduce_precision"
  | Rng_bit_generator _ -> "rng_bit_generator"
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
  | Scatter_add _ -> "scatter-add"
  | Scatter_sub _ -> "scatter-sub"
  | Scatter_mul _ -> "scatter-mul"
  | Scatter_min _ -> "scatter-min"
  | Scatter_max _ -> "scatter-max"
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
  | Collective { kind = Psum; _ } -> "psum_invariant"
  | Collective { kind = Pmax; _ } -> "pmax"
  | Collective { kind = Pmin; _ } -> "pmin"
  | Collective { kind = Axis_index; _ } -> "axis_index"
  | Collective { kind = Ppermute _; _ } -> "ppermute"
  | Collective _ -> "collective"
  | Fft _ -> "fft"
  | Custom_jvp_call id when Hashtbl.mem custom_view_names id ->
      Hashtbl.find custom_view_names id
  | Custom_jvp_call _ -> "custom_jvp_call"
  | Custom_vjp_call id when Hashtbl.mem callback_views id ->
      fst (Hashtbl.find callback_views id)
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
  | Random_seed _ -> "random_seed"
  | Random_split _ -> "random_split"
  | Random_fold_in -> "random_fold_in"
  | Random_bits _ -> "random_bits"
  | Random_wrap -> "random_wrap"
  | Random_unwrap -> "random_unwrap"
  | Random_gamma _ -> "random_gamma"
  | Cholesky -> "cholesky"
  | Cholesky_update -> "cholesky_update"
  | Symmetric_product _ -> "symmetric_product"
  | Householder_product -> "householder_product"
  | Lu -> "lu"
  | Lu_pivots_to_permutation _ -> "lu_pivots_to_permutation"
  | Qr _ | Qr_pivoted _ -> "qr"
  | Geqrf -> "geqrf"
  | Geqp3 -> "geqp3"
  | Ormqr _ -> "ormqr"
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
  | Decomposable { dparams; _ } ->
      "["
      ^ String.concat "," (List.map (fun (k, v) -> k ^ "=" ^ v) dparams)
      ^ "]"
  | Reshape_perm { new_sizes; dimensions } ->
      "[new_sizes=" ^ int_tuple new_sizes ^ ",dimensions="
      ^ int_tuple dimensions ^ "]"
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
      python_repr_float
        (match Ndarray.py_value nd with
        | Some v -> v
        | None -> Ndarray.get_f nd scalar)
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
let has_call_name jid = Hashtbl.mem call_names jid

let call_name jid =
  match Hashtbl.find_opt call_names jid with Some n -> n | None -> "jit"

let custom_call :
    (int -> int option -> aval list -> (string * string * closed_jaxpr) option)
    ref =
  ref (fun _ _ _ -> None)

let atom_type = function
  | A_var (v : var) -> v.vaval
  | A_lit nd ->
      {
        shape = Ndarray.shape nd;
        dtype = Ndarray.dtype nd;
        weak_type = Ndarray.weak nd;
      }
  | DropVar a -> a

let custom_num_consts : (eqn -> int option) ref = ref (fun _ -> None)

let custom_call_of (e : eqn) =
  match e.prim with
  | Custom_jvp_call id ->
      let vid =
        match e.outs with (first : var) :: _ -> Some first.vid | [] -> None
      in
      !custom_call id vid (List.map atom_type e.inputs)
  | _ -> None

let update_counter = ref 0

let fresh_update_id () =
  decr update_counter;
  !update_counter

let update_jaxprs = Hashtbl.create 16

let update_jaxpr prim dt =
  match Hashtbl.find_opt update_jaxprs (prim, dt) with
  | Some c -> c
  | None ->
      let var () =
        {
          vid = fresh_update_id ();
          vaval = { dtype = dt; shape = [||]; weak_type = false };
        }
      in
      let a = var () and b = var () and c = var () in
      let jaxpr =
        {
          in_binders = [ a; b ];
          eqns =
            [
              {
                prim;
                inputs = [ A_var a; A_var b ];
                outs = [ c ];
                multiple_results = false;
              };
            ];
          outs = [ A_var c ];
        }
      in
      let c = { jid = fresh_update_id (); jaxpr; consts = [] } in
      Hashtbl.replace update_jaxprs (prim, dt) c;
      c

let rec structure_of_jaxpr (jx : jaxpr) =
  String.concat " " (List.map structure_of_eqn jx.eqns)

and structure_of_eqn (e : eqn) =
  let update_of prim =
    match e.inputs with
    | operand :: _ -> [ update_jaxpr prim (atom_type operand).dtype ]
    | [] -> []
  in
  let nested =
    match e.prim with
    | Reduce { jaxpr; _ } | Scan { jaxpr; _ } -> [ jaxpr ]
    | Composite c | Xla_call c -> [ c ]
    | Reduce_window { reducer; _ } -> [ reducer ]
    | Select_and_scatter { select_jaxpr; scatter_jaxpr; _ } ->
        [ scatter_jaxpr; select_jaxpr ]
    | Cond { branches; _ } -> branches
    | While { cond; body; _ } -> [ body; cond ]
    | Custom_linear_solve { matvec; vecmat; solve; transpose_solve; _ } ->
        (matvec :: Option.to_list vecmat)
        @ (solve :: Option.to_list transpose_solve)
    | Custom_jvp_call _ -> (
        match custom_call_of e with Some (_, _, c) -> [ c ] | None -> [])
    | Scatter_add _ -> update_of Add
    | Scatter_sub _ -> update_of Sub
    | Scatter_mul _ -> update_of Mul
    | Scatter_min _ -> update_of Min
    | Scatter_max _ -> update_of Max
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

let custom_view_params : (int, (string * jax_param) list) Hashtbl.t =
  Hashtbl.create 8

let register_custom_view id name params =
  Hashtbl.replace custom_view_names id name;
  Hashtbl.replace custom_view_params id params

let note_varying vid axes = Hashtbl.replace varying_axes vid axes

let py_tuple items =
  match items with
  | [] -> "()"
  | [ x ] -> "(" ^ x ^ ",)"
  | _ -> "(" ^ String.concat ", " items ^ ")"

let py_ints a =
  py_tuple
    (List.mapi
       (fun i x ->
         if Np_ints.is_marked a i then Printf.sprintf "np.int64(%d)" x
         else string_of_int x)
       (Array.to_list a))

let py_bool b = if b then "True" else "False"

let py_list a =
  "[" ^ String.concat ", " (Array.to_list (Array.map string_of_int a)) ^ "]"

let py_pairs a =
  py_tuple (Array.to_list (Array.map (fun (x, y) -> py_ints [| x; y |]) a))

let py_platforms platforms =
  let one = function
    | None -> "None"
    | Some names -> (
        let items = Array.to_list names |> List.map (fun n -> "'" ^ n ^ "'") in
        match items with
        | [ only ] -> "(" ^ only ^ ",)"
        | _ -> "(" ^ String.concat ", " items ^ ")")
  in
  let items = Array.to_list platforms |> List.map one in
  match items with
  | [ only ] -> "(" ^ only ^ ",)"
  | _ -> "(" ^ String.concat ", " items ^ ")"

let py_triples a =
  let one i j x =
    if Np_ints.triple_marked a i j then Printf.sprintf "np.int64(%d)" x
    else string_of_int x
  in
  py_tuple
    (Array.to_list
       (Array.mapi
          (fun i (x, y, z) -> py_tuple [ one i 0 x; one i 1 y; one i 2 z ])
          a))

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

let scatter_flags p =
  Option.value (Np_ints.scatter_flags_of p) ~default:(false, false)

let scatter_mode p =
  if Np_ints.prim_marked p then "PROMISE_IN_BOUNDS"
  else if Np_ints.scatter_one_hot p then "ONE_HOT"
  else "FILL_OR_DROP"

let scatter_params ?(flags = (false, false)) ?(sorted = false) ?update
    ?(mode = "FILL_OR_DROP") d unique =
  let sorted = sorted || fst flags and unique = unique || snd flags in
  [
    ("dimension_numbers", Plain (scatter_numbers d));
    ("indices_are_sorted", Plain (py_bool (sorted || d.s_sorted_unique)));
    ("mode", Plain ("GatherScatterMode." ^ if d.s_clip then "CLIP" else mode));
    ("unique_indices", Plain (py_bool (unique || d.s_sorted_unique)));
    ("update_consts", Plain "()");
    ( "update_jaxpr",
      match update with
      | Some (prim, dt) -> Sub (update_jaxpr prim dt)
      | None -> Plain "None" );
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

let closed_to_string : (closed_jaxpr -> string) ref = ref (fun _ -> "")

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
      let weak =
        match e.outs with (v : var) :: _ -> v.vaval.weak_type | [] -> false
      in
      [ ("new_dtype", Plain (dn dt)); ("weak_type", Plain (py_bool weak)) ]
  | Bitcast_convert_type dt -> [ ("new_dtype", Plain (dn dt)) ]
  | Reshape ns ->
      [
        ("dimensions", Plain "None");
        ("new_sizes", Plain (py_ints ns));
        ("sharding", Plain "None");
      ]
  | Decomposable { dparams; _ } -> List.map (fun (k, v) -> (k, Plain v)) dparams
  | Dce_sink { prevent_mlir_dce } ->
      [
        ( "prevent_mlir_dce",
          Plain (if prevent_mlir_dce then "True" else "False") );
      ]
  | Reshape_perm { new_sizes; dimensions } ->
      [
        ("dimensions", Plain (py_ints dimensions));
        ("new_sizes", Plain (py_ints new_sizes));
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
      ]
      @ (match d.dot_precision with
        | Some (l, r) ->
            [
              ( "precision",
                Plain (Printf.sprintf "(Precision.%s, Precision.%s)" l r) );
            ]
        | None -> [])
      @
      if d.untyped_dot then []
      else [ ("preferred_element_type", Plain (dn (out_dtype ()))) ]
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
  | Empty2 dt ->
      [ ("dtype", Plain (dn dt)); ("memory_space", Plain "MemorySpace.Device") ]
  | Pad cfg -> [ ("padding_config", Plain (py_triples cfg)) ]
  | Platform_index platforms ->
      [ ("platforms", Plain (py_platforms platforms)) ]
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
  | Collective { kind = Psum; axis_name } ->
      [ ("axes", Plain ("('" ^ axis_name ^ "',)")) ]
  | Collective { kind = Pmax | Pmin; axis_name } ->
      [
        ("axes", Plain ("('" ^ axis_name ^ "',)"));
        ("axis_index_groups", Plain "None");
      ]
  | Collective { kind = Axis_index; axis_name } ->
      [ ("axis_name", Plain axis_name) ]
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
  | (Argmax { axis; index_dtype } | Argmin { axis; index_dtype }) as p
    when Np_ints.prim_marked p ->
      [
        ("axes", Plain (Printf.sprintf "(np.int64(%d),)" axis));
        ("index_dtype", Plain (dn index_dtype));
      ]
  | Argmax { axis; index_dtype } | Argmin { axis; index_dtype } ->
      [
        ("axes", Plain (py_ints [| axis |]));
        ("index_dtype", Plain (dn index_dtype));
      ]
  | Top_k { k; axis } ->
      [ ("axis", Plain (string_of_int axis)); ("k", Plain (string_of_int k)) ]
  | Gather { dimension_numbers; slice_sizes } as p ->
      let fill = Np_ints.gather_fill_of p in
      let clip = Np_ints.gather_clip_of p in
      let sorted = clip = Some true in
      let flag_sorted, flag_unique =
        Option.value (Np_ints.gather_flags_of p) ~default:(false, false)
      in
      [
        ("dimension_numbers", Plain (gather_numbers dimension_numbers));
        ( "fill_value",
          Plain (match fill with Some (r, _) -> r | None -> "None") );
        ( "indices_are_sorted",
          Plain
            (py_bool
               (sorted || flag_sorted || dimension_numbers.g_sorted_unique)) );
        ( "mode",
          Plain
            (if fill <> None then "GatherScatterMode.FILL_OR_DROP"
             else if clip <> None then "GatherScatterMode.CLIP"
             else "GatherScatterMode.PROMISE_IN_BOUNDS") );
        ("slice_sizes", Plain (py_ints slice_sizes));
        ( "unique_indices",
          Plain
            (py_bool
               (sorted || flag_unique || dimension_numbers.g_sorted_unique)) );
      ]
  | Scatter { dimension_numbers; unique_indices } as p ->
      scatter_params ~flags:(scatter_flags p) ~sorted:(Np_ints.scatter_marked p)
        ~mode:(scatter_mode p) dimension_numbers unique_indices
  | Scatter_mul { dimension_numbers; unique_indices } as p ->
      scatter_params ~flags:(scatter_flags p) ~mode:(scatter_mode p)
        ~update:(Mul, in_dtype ())
        dimension_numbers unique_indices
  | Scatter_add { dimension_numbers } as p when Np_ints.scatter_marked p ->
      scatter_params ~sorted:true
        ~update:(Add, in_dtype ())
        dimension_numbers true
  | Scatter_add { dimension_numbers } as p ->
      scatter_params ~flags:(scatter_flags p) ~mode:(scatter_mode p)
        ~update:(Add, in_dtype ())
        dimension_numbers false
  | Scatter_sub { dimension_numbers } as p ->
      scatter_params ~flags:(scatter_flags p) ~mode:(scatter_mode p)
        ~update:(Sub, in_dtype ())
        dimension_numbers false
  | Scatter_min { dimension_numbers } as p ->
      scatter_params ~flags:(scatter_flags p) ~mode:(scatter_mode p)
        ~update:(Min, in_dtype ())
        dimension_numbers false
  | Scatter_max { dimension_numbers } as p ->
      scatter_params ~flags:(scatter_flags p) ~mode:(scatter_mode p)
        ~update:(Max, in_dtype ())
        dimension_numbers false
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
        ( "precision",
          Plain
            (match c.conv_precision with
            | Some (l, r) -> Printf.sprintf "(Precision.%s, Precision.%s)" l r
            | None -> "None") );
        ( "preferred_element_type",
          Plain
            (match c.conv_preferred with
            | Some d -> Dtype.short_name d
            | None -> "None") );
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
        ( "computation",
          Plain
            (match jaxpr.jaxpr.eqns with
            | [ { prim = (Add | Mul | And | Or | Xor | Max | Min) as p; _ } ] ->
                let fname =
                  match p with
                  | And -> "bitwise_and"
                  | Or -> "bitwise_or"
                  | Xor -> "bitwise_xor"
                  | p -> primitive_name p
                in
                Printf.sprintf "<function %s at 0xXXXXXXXXXXXX>" fname
            | _ -> "<function>") );
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
  | Qr_pivoted full ->
      [
        ("full_matrices", Plain (py_bool full));
        ("pivoting", Plain "True");
        ("use_magma", Plain "None");
      ]
  | Geqp3 -> [ ("use_magma", Plain "None") ]
  | Ormqr { ormqr_left; ormqr_transpose } ->
      [
        ("left", Plain (py_bool ormqr_left));
        ("transpose", Plain (py_bool ormqr_transpose));
      ]
  | Rng_bit_generator { rbg_shape; rbg_dtype; rbg_algorithm } ->
      [
        ( "algorithm",
          Plain
            (if rbg_algorithm = "DEFAULT" then "RNG_DEFAULT"
             else "RNG_" ^ rbg_algorithm) );
        ("dtype", Plain (Dtype.numpy_name rbg_dtype));
        ("out_sharding", Plain "None");
        ("shape", Plain (py_ints rbg_shape));
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
  | Symmetric_product { alpha; beta } ->
      [
        ("alpha", Plain (python_repr_float alpha));
        ("beta", Plain (python_repr_float beta));
      ]
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
  | Random_gamma { log_space } -> [ ("log_space", Plain (py_bool log_space)) ]
  | Random_seed tag -> [ ("impl", Plain tag) ]
  | Random_wrap -> [ ("impl", Plain "fry") ]
  | While { cond_nconsts; body_nconsts; cond; body } ->
      [
        ("body_jaxpr", Sub body);
        ("body_nconsts", Plain (string_of_int body_nconsts));
        ("cond_jaxpr", Sub cond);
        ("cond_nconsts", Plain (string_of_int cond_nconsts));
      ]
  | Cond { branches; platforms = None } -> [ ("branches", Subs branches) ]
  | Cond { branches; platforms = Some platforms } ->
      [
        ("branches", Subs branches);
        ("branches_platforms", Plain (py_platforms platforms));
      ]
  | Scan { length; reverse; num_consts; num_carry; jaxpr; unroll } ->
      [
        ("jaxpr", Sub jaxpr);
        ("length", Plain (string_of_int length));
        ("num_carry", Plain (string_of_int num_carry));
        ("num_consts", Plain (string_of_int num_consts));
        ("reverse", Plain (py_bool reverse));
        ("unroll", Plain (string_of_int unroll));
      ]
  | Xla_call c -> [ ("jaxpr", Sub c) ]
  | Custom_linear_solve
      { matvec; vecmat; solve; transpose_solve; const_lengths } ->
      let opt = function Some c -> !closed_to_string c | None -> "None" in
      [
        ( "const_lengths",
          Plain
            (Printf.sprintf
               "_LinearSolveTuple(matvec=%d, vecmat=%d, solve=%d, \
                transpose_solve=%d)"
               const_lengths.(0) const_lengths.(1) const_lengths.(2)
               const_lengths.(3)) );
        ( "jaxprs",
          Plain
            (Printf.sprintf
               "_LinearSolveTuple(matvec=%s, vecmat=%s, solve=%s, \
                transpose_solve=%s)"
               (!closed_to_string matvec) (opt vecmat) (!closed_to_string solve)
               (opt transpose_solve)) );
      ]
  | Select_and_scatter { select_jaxpr; scatter_jaxpr; window } ->
      List.sort compare
        (("scatter_jaxpr", Sub scatter_jaxpr)
        :: ("select_jaxpr", Sub select_jaxpr)
        :: window_params window)
  | _ -> []

let insertion_order (e : eqn) =
  match e.prim with
  | Custom_linear_solve { matvec; vecmat; solve; transpose_solve; _ } ->
      (matvec :: Option.to_list vecmat)
      @ (solve :: Option.to_list transpose_solve)
  | While { cond; body; _ } -> [ cond; body ]
  | Cond { branches; _ } -> branches
  | Select_and_scatter { select_jaxpr; scatter_jaxpr; _ } ->
      [ select_jaxpr; scatter_jaxpr ]
  | Custom_jvp_call id when Hashtbl.mem custom_view_params id ->
      List.concat_map
        (fun (_, p) ->
          match p with Sub c -> [ c ] | Subs cs -> cs | Plain _ -> [])
        (Hashtbl.find custom_view_params id)
  | Custom_jvp_call _ -> (
      match custom_call_of e with Some (_, _, c) -> [ c ] | None -> [])
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
  keys : (int, unit) Hashtbl.t;
  mutable next_name : int;
  var_names : (int, string) Hashtbl.t;
  shared : (int, string) Hashtbl.t;
  shared_names : (string, unit) Hashtbl.t;
}

let rec fresh_var_name ctx =
  let name = alpha_name ctx.next_name in
  ctx.next_name <- ctx.next_name + 1;
  if Hashtbl.mem ctx.shared_names name then fresh_var_name ctx else name

let dropped_vars : (int, unit) Hashtbl.t = Hashtbl.create 16
let mark_dropped (v : var) = Hashtbl.replace dropped_vars v.vid ()

let name_of_var ctx (v : var) =
  match Hashtbl.find_opt ctx.var_names v.vid with
  | Some n -> n
  | None when Hashtbl.mem dropped_vars v.vid -> "_"
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

let python_float_argument nd =
  let x = Ndarray.get_f nd [||] in
  let bits = Int32.bits_of_float x in
  let rec shortest p =
    let d = float_of_string (Printf.sprintf "%.*g" p x) in
    if p >= 9 || Int32.bits_of_float d = bits then d else shortest (p + 1)
  in
  if Float.is_finite x then
    let shown =
      match Ndarray.py_value nd with Some d -> d | None -> shortest 1
    in
    lit_value (Ndarray.of_floats Dtype.F64 [||] [| shown |])
    ^ ":" ^ lit_short nd
  else jax_literal nd

let pp_atoms ctx ?(separator = "") atoms =
  let doc = function
    | A_var v -> Layout.Text (name_of_var ctx v)
    | A_lit nd
      when Ndarray.weak nd
           && Ndarray.dtype nd = Dtype.F32
           && Array.length (Ndarray.shape nd) = 0 ->
        Layout.Text (python_float_argument nd)
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

let key_short (a : aval) =
  let rank = Array.length a.shape in
  let dims = Array.to_list (Array.sub a.shape 0 (max 0 (rank - 1))) in
  let tag =
    if rank = 0 || a.shape.(rank - 1) <> 4 then "fry"
    else if a.weak_type then "urbg"
    else "rbg"
  in
  "key<" ^ tag ^ ">[" ^ String.concat "," (List.map string_of_int dims) ^ "]"

let token_vars : (int, unit) Hashtbl.t = Hashtbl.create 8

let pp_binders ctx (vars : var list) =
  let doc (v : var) =
    let name = name_of_var ctx v in
    let short =
      if Hashtbl.mem token_vars v.vid then "Tok"
      else if Hashtbl.mem ctx.keys v.vid then key_short v.vaval
      else aval_short v.vaval
    in
    let short =
      match Hashtbl.find_opt varying_axes v.vid with
      | Some (_ :: _ as axes) -> short ^ "{V:" ^ String.concat "," axes ^ "}"
      | _ -> short
    in
    Layout.Text (name ^ ":" ^ short)
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
  (match e.prim with
  | Create_token | After_all | Decomposable { dname = "psend"; _ } ->
      List.iter (fun (v : var) -> Hashtbl.replace token_vars v.vid ()) e.outs
  | _ -> ());
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
    | Custom_vjp_call id when Hashtbl.mem callback_views id ->
        List.map
          (fun (k, v) -> (k, Plain v))
          (snd (Hashtbl.find callback_views id))
    | Custom_jvp_call id when Hashtbl.mem custom_view_params id ->
        Hashtbl.find custom_view_params id
    | Checkpoint_name name -> [ ("name", Plain name) ]
    | Custom_lin { rule; n_res; in_zeros; _ } -> (
        match !custom_lin_vjp rule with
        | true ->
            [
              ("bwd", Plain "bwd");
              ( "in_zeros",
                Plain
                  ("["
                  ^ String.concat ", "
                      (List.map
                         (fun z -> if z then "True" else "False")
                         in_zeros)
                  ^ "]") );
              ("num_res", Plain (string_of_int n_res));
              ("symbolic_zeros", Plain "False");
            ]
        | false -> [])
    | Custom_jvp_call _ -> (
        match custom_call_of e with
        | Some (name, jvp_name, c) ->
            [
              ("name", Plain name);
              ("call_jaxpr", Sub c);
              ("jvp", Plain jvp_name);
            ]
            @ (match !custom_num_consts e with
              | Some n -> [ ("num_consts", Plain (string_of_int n)) ]
              | None -> [])
            @ [ ("symbolic_zeros", Plain "False") ]
        | None -> [])
    | _ -> []
  in
  let binders =
    match (e.prim, e.outs) with
    | Svd _, [ u; s; v ] -> [ s; u; v ]
    | _ -> e.outs
  in
  let lhs = pp_binders ctx binders in
  let params =
    let strip a = Array.sub a 0 (Array.length a - 1) in
    let keyed =
      match e.inputs with
      | A_var v :: _ -> Hashtbl.mem ctx.keys v.vid
      | _ -> false
    in
    let shown =
      match e.prim with
      | Slice { start_indices; limit_indices; strides } when keyed ->
          {
            e with
            prim =
              Slice
                {
                  start_indices = strip start_indices;
                  limit_indices = strip limit_indices;
                  strides = Option.map strip strides;
                };
          }
      | _ -> e
    in
    pp_params ctx (name_param @ jax_params shown)
  in
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

let rec mark_keys keys (jx : jaxpr) =
  let mark (v : var) = Hashtbl.replace keys v.vid () in
  let is_key = function
    | A_var (v : var) -> Hashtbl.mem keys v.vid
    | A_lit _ | DropVar _ -> false
  in
  List.iter
    (fun (e : eqn) ->
      match e.prim with
      | Random_seed _ | Random_wrap | Random_split _ | Random_fold_in ->
          List.iter mark e.outs
      | Slice _ | Squeeze _ | Reshape _ | Copy | Stop_gradient ->
          if List.exists is_key e.inputs then List.iter mark e.outs
      | Xla_call c ->
          let args =
            List.filteri
              (fun i _ -> i >= List.length c.consts)
              c.jaxpr.in_binders
          in
          List.iter2
            (fun (b : var) input -> if is_key input then mark b)
            args e.inputs;
          mark_keys keys c.jaxpr;
          List.iter2
            (fun (out : var) inner -> if is_key inner then mark out)
            e.outs c.jaxpr.outs
      | While { cond_nconsts; body_nconsts; cond; body } ->
          let n = cond_nconsts + body_nconsts in
          let mark_carry (c : closed_jaxpr) nconsts =
            let binders =
              List.filteri
                (fun i _ -> i >= List.length c.consts + nconsts)
                c.jaxpr.in_binders
            in
            let carry = List.filteri (fun i _ -> i >= n) e.inputs in
            List.iter2
              (fun (b : var) input -> if is_key input then mark b)
              binders carry;
            mark_keys keys c.jaxpr
          in
          mark_carry cond cond_nconsts;
          mark_carry body body_nconsts;
          let carry = List.filteri (fun i _ -> i >= n) e.inputs in
          List.iter2
            (fun (out : var) input -> if is_key input then mark out)
            e.outs carry
      | Scan { jaxpr = c; _ } ->
          let binders =
            List.filteri
              (fun i _ -> i >= List.length c.consts)
              c.jaxpr.in_binders
          in
          if List.length binders = List.length e.inputs then begin
            List.iter2
              (fun (b : var) input -> if is_key input then mark b)
              binders e.inputs;
            mark_keys keys c.jaxpr
          end
      | _ -> ())
    jx.eqns

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
      keys = Hashtbl.create 8;
    }
  in
  mark_keys ctx.keys cj.jaxpr;
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
            if not (Hashtbl.mem names inner.jid) then begin
              let raw = xla_call_name inner in
              let n = String.length raw in
              let name =
                if n >= 2 && raw.[0] = '<' && raw.[n - 1] = '>' then
                  String.sub raw 1 (n - 2)
                else raw
              in
              Hashtbl.replace names inner.jid name
            end
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

let () = closed_to_string := closed_jaxpr_to_jax_string

let eqn_to_jax_string (e : eqn) =
  let ctx =
    {
      next_name = 0;
      var_names = Hashtbl.create 8;
      shared = Hashtbl.create 1;
      shared_names = Hashtbl.create 1;
      keys = Hashtbl.create 1;
    }
  in
  Layout.render (pp_eqn ctx e)
