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
  | Round -> "round"
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
  | Cond _ -> "cond"

let prim_params = function
  | Convert_element_type dt -> "[new_dtype=" ^ Dtype.short_name dt ^ "]"
  | Broadcast_in_dim { shape; dims } ->
      "[broadcast_dimensions=" ^ int_tuple dims ^ " shape=" ^ int_tuple shape
      ^ "]"
  | Reshape ns -> "[new_sizes=" ^ int_tuple ns ^ "]"
  | Reduce_sum axes -> "[axes=" ^ int_tuple axes ^ "]"
  | Dot_general dd -> "[dimension_numbers=" ^ dot_dims_str dd ^ "]"
  | Integer_pow y -> "[y=" ^ string_of_int y ^ "]"
  | _ -> ""

let lit_short nd =
  Dtype.short_name (Ndarray.dtype nd) ^ shape_str (Ndarray.shape nd)

let lit_value nd =
  let scalar = [||] in
  match Ndarray.dtype nd with
  | Dtype.Bool -> if Ndarray.get_f nd scalar = 0.0 then "False" else "True"
  | Dtype.I32 | Dtype.I64 -> Int64.to_string (Ndarray.get_i64 nd scalar)
  | Dtype.F32 | Dtype.F64 -> python_repr_float (Ndarray.get_f nd scalar)

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
