open Types

module Doc = struct
  type t = Text of string | Concat of t list | Group of t

  let text s = Text s
  let concat ts = Concat ts
  let group d = Group d

  let rec flat = function
    | Text s -> s
    | Concat ts -> String.concat "" (List.map flat ts)
    | Group d -> flat d

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
  else begin
    let rec find p =
      if p > 17 then Printf.sprintf "%.17g" x
      else
        let s = Printf.sprintf "%.*g" p x in
        if float_of_string s = x then s else find (p + 1)
    in
    let s = find 1 in
    if
      String.contains s '.' || String.contains s 'e' || String.contains s 'E'
      || String.contains s 'n'
    then s
    else s ^ ".0"
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
    match Hashtbl.find_opt names v.vid with Some n -> n | None -> "?"
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
    Doc.group
      (Doc.concat
         (match jx.eqns with
         | [] -> []
         | first :: rest ->
             eqn_doc first
             :: List.concat_map (fun e -> [ Doc.text " ; "; eqn_doc e ]) rest))
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
