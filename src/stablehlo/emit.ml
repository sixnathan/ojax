open Types

let aval_of_ndarray nd =
  { shape = Ndarray.shape nd; dtype = Ndarray.dtype nd; weak_type = false }

let atom_aval = function
  | A_var v -> v.vaval
  | A_lit nd -> aval_of_ndarray nd
  | DropVar a -> a

let element_literal nd idx =
  match Ndarray.dtype nd with
  | Dtype.F32 | Dtype.F64 ->
      Ir.float_literal (Ndarray.dtype nd) (Ndarray.get_f nd idx)
  | Dtype.I32 | Dtype.I64 | Dtype.Uint32 ->
      Ir.int_literal (Ndarray.get_i64 nd idx)
  | Dtype.Bool -> Ir.bool_literal (Ndarray.get_i64 nd idx <> 0L)

let all_literals nd =
  let shape = Ndarray.shape nd in
  let rec loop dims = function
    | [] -> [ element_literal nd (Array.of_list (List.rev dims)) ]
    | d :: rest -> List.concat (List.init d (fun i -> loop (i :: dims) rest))
  in
  loop [] (Array.to_list shape)

let nested_literals nd =
  let shape = Ndarray.shape nd in
  let rec build dims = function
    | [] -> element_literal nd (Array.of_list (List.rev dims))
    | d :: rest ->
        "["
        ^ String.concat ", " (List.init d (fun i -> build (i :: dims) rest))
        ^ "]"
  in
  build [] (Array.to_list shape)

let dense_body nd =
  match all_literals nd with
  | [] -> invalid_arg "Stablehlo.Emit: empty constant"
  | first :: rest when List.for_all (String.equal first) rest -> first
  | _ -> nested_literals nd

type ctx = { buf : Buffer.t; ids : (int, int) Hashtbl.t; mutable next : int }

let fresh ctx =
  let n = ctx.next in
  ctx.next <- n + 1;
  n

let bind_var ctx (v : var) =
  let n = fresh ctx in
  Hashtbl.replace ctx.ids v.vid n;
  n

let ssa_of_var ctx (v : var) =
  match Hashtbl.find_opt ctx.ids v.vid with
  | Some n -> "%" ^ string_of_int n
  | None -> invalid_arg "Stablehlo.Emit: unbound variable"

let emit_constant_at ctx n nd =
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %%%d = stablehlo.constant %s : %s\n" n
       (Ir.dense (dense_body nd))
       (Ir.tensor_type (Ndarray.dtype nd) (Ndarray.shape nd)))

let name n = "%" ^ string_of_int n

let ssa_of_atom ctx = function
  | A_var v -> ssa_of_var ctx v
  | A_lit nd ->
      let n = fresh ctx in
      emit_constant_at ctx n nd;
      name n
  | DropVar _ -> invalid_arg "Stablehlo.Emit: DropVar in value position"

let id_of_atom ctx = function
  | A_var v -> (
      match Hashtbl.find_opt ctx.ids v.vid with
      | Some n -> n
      | None -> invalid_arg "Stablehlo.Emit: unbound variable")
  | A_lit nd ->
      let n = fresh ctx in
      emit_constant_at ctx n nd;
      n
  | DropVar _ -> invalid_arg "Stablehlo.Emit: DropVar in value position"

let sole = function [ x ] -> x | _ -> invalid_arg "Stablehlo.Emit: arity"

let emit_stablehlo_unary ctx (eqn : eqn) in_ids op =
  let x = sole in_ids in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.%s %s : %s\n" (name n) op (name x)
       (Ir.tensor_type_of_aval out.vaval))

let emit_chlo_unary ctx (eqn : eqn) in_ids op =
  let x = sole in_ids in
  let inty = Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)) in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = chlo.%s %s : %s -> %s\n" (name n) op (name x) inty
       (Ir.tensor_type_of_aval out.vaval))

let pair = function
  | [ a; b ] -> (a, b)
  | _ -> invalid_arg "Stablehlo.Emit: arity"

let emit_stablehlo_binary ctx (eqn : eqn) in_ids op =
  let a, b = pair in_ids in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.%s %s, %s : %s\n" (name n) op (name a)
       (name b)
       (Ir.tensor_type_of_aval out.vaval))

let emit_chlo_binary ctx (eqn : eqn) in_ids op =
  let a, b = pair in_ids in
  let lhs, rhs =
    match eqn.inputs with
    | [ l; r ] -> (l, r)
    | _ -> invalid_arg "Stablehlo.Emit: arity"
  in
  let lty = Ir.tensor_type_of_aval (atom_aval lhs) in
  let rty = Ir.tensor_type_of_aval (atom_aval rhs) in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = chlo.%s %s, %s : %s, %s -> %s\n" (name n) op
       (name a) (name b) lty rty
       (Ir.tensor_type_of_aval out.vaval))

let emit_exp2 ctx (eqn : eqn) in_ids =
  let x = sole in_ids in
  let out = sole eqn.outs in
  let dt = out.vaval.dtype in
  let xty = Ir.tensor_type_of_aval out.vaval in
  let sty = Ir.tensor_type dt [||] in
  let c = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c)
       (Ir.dense (Ir.float_literal dt (Float.log 2.0)))
       sty);
  let b = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.broadcast_in_dim %s, dims = [] : (%s) -> %s\n"
       (name b) (name c) sty xty);
  let m = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.multiply %s, %s : %s\n" (name m)
       (name b) (name x) xty);
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.exponential %s : %s\n" (name n) (name m)
       xty)

let emit_is_finite ctx (eqn : eqn) in_ids =
  let x = sole in_ids in
  let inty = Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)) in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.is_finite %s : (%s) -> %s\n" (name n)
       (name x) inty
       (Ir.tensor_type_of_aval out.vaval))

let emit_logistic ctx (eqn : eqn) in_ids =
  let x = sole in_ids in
  let out = sole eqn.outs in
  let dt = out.vaval.dtype in
  let xty = Ir.tensor_type_of_aval out.vaval in
  let sty = Ir.tensor_type dt [||] in
  let one = Ir.dense (Ir.float_literal dt 1.0) in
  let ng = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.negate %s : %s\n" (name ng) (name x) xty);
  let e = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.exponential %s : %s\n" (name e)
       (name ng) xty);
  let c1 = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c1) one sty);
  let b1 = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.broadcast_in_dim %s, dims = [] : (%s) -> %s\n"
       (name b1) (name c1) sty xty);
  let a = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.add %s, %s : %s\n" (name a) (name b1)
       (name e) xty);
  let c2 = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c2) one sty);
  let b2 = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.broadcast_in_dim %s, dims = [] : (%s) -> %s\n"
       (name b2) (name c2) sty xty);
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.divide %s, %s : %s\n" (name n) (name b2)
       (name a) xty)

let emit_integer_pow ctx (eqn : eqn) in_ids y =
  let x = sole in_ids in
  let out = sole eqn.outs in
  let ty = Ir.tensor_type_of_aval out.vaval in
  let mul a b =
    let n = fresh ctx in
    Buffer.add_string ctx.buf
      (Printf.sprintf "    %s = stablehlo.multiply %s, %s : %s\n" (name n)
         (name a) (name b) ty);
    n
  in
  let result =
    if y = 1 then x
    else if y = 2 then mul x x
    else if y = 3 then
      let m = mul x x in
      mul m x
    else if y >= 4 then begin
      let acc = ref None in
      let base = ref x in
      let yy = ref y in
      while !yy > 0 do
        if !yy land 1 = 1 then
          acc := Some (match !acc with None -> !base | Some a -> mul a !base);
        yy := !yy asr 1;
        if !yy > 0 then base := mul !base !base
      done;
      match !acc with Some a -> a | None -> assert false
    end
    else failwith "Stablehlo.Emit: Integer_pow non-positive exponent (M5)"
  in
  Hashtbl.replace ctx.ids out.vid result

let compare_type_of dtype total_order =
  match dtype with
  | Dtype.F32 | Dtype.F64 -> if total_order then "TOTALORDER" else "FLOAT"
  | Dtype.I32 | Dtype.I64 -> "SIGNED"
  | Dtype.Bool | Dtype.Uint32 -> "UNSIGNED"

let emit_compare ctx (eqn : eqn) in_ids dir total_order =
  let a, b = pair in_ids in
  let inty = atom_aval (List.hd eqn.inputs) in
  let ctype = compare_type_of inty.dtype total_order in
  let tystr = Ir.tensor_type_of_aval inty in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.compare %s, %s, %s, %s : (%s, %s) -> %s\n" (name n)
       dir (name a) (name b) ctype tystr tystr
       (Ir.tensor_type_of_aval out.vaval))

let emit_clamp ctx (eqn : eqn) in_ids =
  let mn, op, mx =
    match in_ids with
    | [ a; b; c ] -> (a, b, c)
    | _ -> invalid_arg "Stablehlo.Emit: clamp arity"
  in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.clamp %s, %s, %s : %s\n" (name n)
       (name mn) (name op) (name mx)
       (Ir.tensor_type_of_aval out.vaval))

let emit_select_n ctx (eqn : eqn) in_ids =
  let which, cases =
    match in_ids with
    | w :: cs -> (w, cs)
    | [] -> invalid_arg "Stablehlo.Emit: select_n arity"
  in
  let out = sole eqn.outs in
  let vty = Ir.tensor_type_of_aval out.vaval in
  let which_aval = atom_aval (List.hd eqn.inputs) in
  let wty = Ir.tensor_type_of_aval which_aval in
  let bty = Ir.tensor_type Dtype.Bool which_aval.shape in
  let emit_select pred a b =
    let n = fresh ctx in
    Buffer.add_string ctx.buf
      (Printf.sprintf "    %s = stablehlo.select %s, %s, %s : %s, %s\n" (name n)
         (name pred) (name a) (name b) bty vty);
    n
  in
  let result =
    match which_aval.dtype with
    | Dtype.Bool -> (
        match cases with
        | [ c ] -> c
        | [ c0; c1 ] -> emit_select which c1 c0
        | _ -> invalid_arg "Stablehlo.Emit: bool select_n arity")
    | dt ->
        let ctype =
          match dt with Dtype.I32 | Dtype.I64 -> "SIGNED" | _ -> "UNSIGNED"
        in
        let sty = Ir.tensor_type dt [||] in
        let rec sel offset cs =
          match cs with
          | [ c ] -> c
          | _ ->
              let mid = List.length cs / 2 in
              let cid = fresh ctx in
              Buffer.add_string ctx.buf
                (Printf.sprintf "    %s = stablehlo.constant %s : %s\n"
                   (name cid)
                   (Ir.dense (Ir.int_literal (Int64.of_int (offset + mid))))
                   sty);
              let bid = fresh ctx in
              Buffer.add_string ctx.buf
                (Printf.sprintf
                   "    %s = stablehlo.broadcast_in_dim %s, dims = [] : (%s) \
                    -> %s\n"
                   (name bid) (name cid) sty wty);
              let pid = fresh ctx in
              Buffer.add_string ctx.buf
                (Printf.sprintf
                   "    %s = stablehlo.compare LT, %s, %s, %s : (%s, %s) -> %s\n"
                   (name pid) (name which) (name bid) ctype wty wty bty);
              let left, right =
                match Util.split_list cs [ mid ] with
                | [ l; r ] -> (l, r)
                | _ -> invalid_arg "Stablehlo.Emit: select split"
              in
              let lv = sel offset left in
              let rv = sel (offset + mid) right in
              emit_select pid lv rv
        in
        sel 0 cases
  in
  Hashtbl.replace ctx.ids out.vid result

let zero_dense dt =
  match dt with
  | Dtype.F32 | Dtype.F64 -> Ir.float_literal dt 0.0
  | Dtype.I32 | Dtype.I64 | Dtype.Uint32 -> Ir.int_literal 0L
  | Dtype.Bool -> Ir.bool_literal false

let emit_convert ctx (eqn : eqn) in_ids =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let out = sole eqn.outs in
  let in_dt = in_aval.dtype in
  let out_dt = out.vaval.dtype in
  let oty = Ir.tensor_type_of_aval out.vaval in
  if out_dt = Dtype.Bool && in_dt <> Dtype.Bool then begin
    let inty = Ir.tensor_type_of_aval in_aval in
    let sty = Ir.tensor_type in_dt [||] in
    let c = fresh ctx in
    Buffer.add_string ctx.buf
      (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c)
         (Ir.dense (zero_dense in_dt))
         sty);
    let b = fresh ctx in
    Buffer.add_string ctx.buf
      (Printf.sprintf
         "    %s = stablehlo.broadcast_in_dim %s, dims = [] : (%s) -> %s\n"
         (name b) (name c) sty inty);
    let cmp = fresh ctx in
    Buffer.add_string ctx.buf
      (Printf.sprintf
         "    %s = stablehlo.compare NE, %s, %s, %s : (%s, %s) -> %s\n"
         (name cmp) (name x) (name b)
         (compare_type_of in_dt false)
         inty inty oty);
    let n = bind_var ctx out in
    Buffer.add_string ctx.buf
      (Printf.sprintf "    %s = stablehlo.convert %s : %s\n" (name n) (name cmp)
         oty)
  end
  else if in_dt = out_dt then Hashtbl.replace ctx.ids out.vid x
  else begin
    let inty = Ir.tensor_type_of_aval in_aval in
    let n = bind_var ctx out in
    Buffer.add_string ctx.buf
      (Printf.sprintf "    %s = stablehlo.convert %s : (%s) -> %s\n" (name n)
         (name x) inty oty)
  end

let emit_bitcast ctx (eqn : eqn) in_ids =
  let x = sole in_ids in
  let inty = Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)) in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.bitcast_convert %s : (%s) -> %s\n"
       (name n) (name x) inty
       (Ir.tensor_type_of_aval out.vaval))

let emit_opt_barrier ctx (eqn : eqn) in_ids =
  match (eqn.outs, in_ids) with
  | [ out ], [ x ] ->
      let n = bind_var ctx out in
      Buffer.add_string ctx.buf
        (Printf.sprintf "    %s = stablehlo.optimization_barrier %s : %s\n"
           (name n) (name x)
           (Ir.tensor_type_of_aval out.vaval))
  | _ ->
      failwith
        "Stablehlo.Emit: optimization_barrier multi-result grouped SSA deferred"

let emit_reduce_precision ctx (eqn : eqn) in_ids exponent_bits mantissa_bits =
  let x = sole in_ids in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.reduce_precision %s, format = e%dm%d : %s\n" (name n)
       (name x) exponent_bits mantissa_bits
       (Ir.tensor_type_of_aval out.vaval))

let emit_empty ctx (eqn : eqn) =
  let out = sole eqn.outs in
  let dt = out.vaval.dtype in
  let oty = Ir.tensor_type_of_aval out.vaval in
  let sty = Ir.tensor_type dt [||] in
  let c = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c)
       (Ir.dense (zero_dense dt))
       sty);
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.broadcast_in_dim %s, dims = [] : (%s) -> %s\n"
       (name n) (name c) sty oty)

let platform_cpu_index platforms =
  let n = Array.length platforms in
  let rec find_cpu i =
    if i >= n then find_default 0
    else
      match platforms.(i) with
      | Some names when Array.exists (fun p -> p = "cpu") names -> i
      | _ -> find_cpu (i + 1)
  and find_default i =
    if i >= n then
      failwith "Stablehlo.Emit: platform_index has no cpu or default branch"
    else match platforms.(i) with None -> i | Some _ -> find_default (i + 1)
  in
  find_cpu 0

let emit_platform_index ctx (eqn : eqn) platforms =
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name n)
       (Ir.dense (Ir.int_literal (Int64.of_int (platform_cpu_index platforms))))
       (Ir.tensor_type_of_aval out.vaval))

let emit_eqn ctx (eqn : eqn) (in_ids : int list) : unit =
  match eqn.prim with
  | Abs -> emit_stablehlo_unary ctx eqn in_ids "abs"
  | Cbrt -> emit_stablehlo_unary ctx eqn in_ids "cbrt"
  | Ceil -> emit_stablehlo_unary ctx eqn in_ids "ceil"
  | Clz -> emit_stablehlo_unary ctx eqn in_ids "count_leading_zeros"
  | Cos -> emit_stablehlo_unary ctx eqn in_ids "cosine"
  | Exp -> emit_stablehlo_unary ctx eqn in_ids "exponential"
  | Expm1 -> emit_stablehlo_unary ctx eqn in_ids "exponential_minus_one"
  | Floor -> emit_stablehlo_unary ctx eqn in_ids "floor"
  | Acos -> emit_chlo_unary ctx eqn in_ids "acos"
  | Acosh -> emit_chlo_unary ctx eqn in_ids "acosh"
  | Asin -> emit_chlo_unary ctx eqn in_ids "asin"
  | Asinh -> emit_chlo_unary ctx eqn in_ids "asinh"
  | Atan -> emit_chlo_unary ctx eqn in_ids "atan"
  | Atanh -> emit_chlo_unary ctx eqn in_ids "atanh"
  | Cosh -> emit_chlo_unary ctx eqn in_ids "cosh"
  | Copy -> Hashtbl.replace ctx.ids (sole eqn.outs).vid (sole in_ids)
  | Exp2 -> emit_exp2 ctx eqn in_ids
  | Conj -> failwith "Stablehlo.Emit: Conj requires complex dtype (M5)"
  | Log -> emit_stablehlo_unary ctx eqn in_ids "log"
  | Log1p -> emit_stablehlo_unary ctx eqn in_ids "log_plus_one"
  | Neg -> emit_stablehlo_unary ctx eqn in_ids "negate"
  | Not -> emit_stablehlo_unary ctx eqn in_ids "not"
  | Population_count -> emit_stablehlo_unary ctx eqn in_ids "popcnt"
  | Round -> emit_stablehlo_unary ctx eqn in_ids "round_nearest_afz"
  | Rsqrt -> emit_stablehlo_unary ctx eqn in_ids "rsqrt"
  | Sign -> emit_stablehlo_unary ctx eqn in_ids "sign"
  | Sin -> emit_stablehlo_unary ctx eqn in_ids "sine"
  | Sqrt -> emit_stablehlo_unary ctx eqn in_ids "sqrt"
  | Tan -> emit_stablehlo_unary ctx eqn in_ids "tan"
  | Tanh -> emit_stablehlo_unary ctx eqn in_ids "tanh"
  | Sinh -> emit_chlo_unary ctx eqn in_ids "sinh"
  | Square -> emit_chlo_unary ctx eqn in_ids "square"
  | Is_finite -> emit_is_finite ctx eqn in_ids
  | Logistic -> emit_logistic ctx eqn in_ids
  | Integer_pow y -> emit_integer_pow ctx eqn in_ids y
  | Imag -> failwith "Stablehlo.Emit: Imag requires complex dtype (M5)"
  | Real -> failwith "Stablehlo.Emit: Real requires complex dtype (M5)"
  | Add -> emit_stablehlo_binary ctx eqn in_ids "add"
  | And -> emit_stablehlo_binary ctx eqn in_ids "and"
  | Atan2 -> emit_stablehlo_binary ctx eqn in_ids "atan2"
  | Div -> emit_stablehlo_binary ctx eqn in_ids "divide"
  | Max -> emit_stablehlo_binary ctx eqn in_ids "maximum"
  | Min -> emit_stablehlo_binary ctx eqn in_ids "minimum"
  | Mul -> emit_stablehlo_binary ctx eqn in_ids "multiply"
  | Or -> emit_stablehlo_binary ctx eqn in_ids "or"
  | Pow -> emit_stablehlo_binary ctx eqn in_ids "power"
  | Rem -> emit_stablehlo_binary ctx eqn in_ids "remainder"
  | Shift_left -> emit_stablehlo_binary ctx eqn in_ids "shift_left"
  | Shift_right_arithmetic ->
      emit_stablehlo_binary ctx eqn in_ids "shift_right_arithmetic"
  | Shift_right_logical ->
      emit_stablehlo_binary ctx eqn in_ids "shift_right_logical"
  | Sub -> emit_stablehlo_binary ctx eqn in_ids "subtract"
  | Xor -> emit_stablehlo_binary ctx eqn in_ids "xor"
  | Mulhi -> emit_chlo_binary ctx eqn in_ids "mulhi"
  | Nextafter -> emit_chlo_binary ctx eqn in_ids "next_after"
  | Complex -> failwith "Stablehlo.Emit: Complex requires complex dtype (M5)"
  | Eq -> emit_compare ctx eqn in_ids "EQ" false
  | Ne -> emit_compare ctx eqn in_ids "NE" false
  | Ge -> emit_compare ctx eqn in_ids "GE" false
  | Gt -> emit_compare ctx eqn in_ids "GT" false
  | Le -> emit_compare ctx eqn in_ids "LE" false
  | Lt -> emit_compare ctx eqn in_ids "LT" false
  | Eq_to -> emit_compare ctx eqn in_ids "EQ" true
  | Le_to -> emit_compare ctx eqn in_ids "LE" true
  | Lt_to -> emit_compare ctx eqn in_ids "LT" true
  | Clamp -> emit_clamp ctx eqn in_ids
  | Select_n -> emit_select_n ctx eqn in_ids
  | Convert_element_type _ -> emit_convert ctx eqn in_ids
  | Bitcast_convert_type _ -> emit_bitcast ctx eqn in_ids
  | Optimization_barrier -> emit_opt_barrier ctx eqn in_ids
  | Reduce_precision { exponent_bits; mantissa_bits } ->
      emit_reduce_precision ctx eqn in_ids exponent_bits mantissa_bits
  | Tie -> (
      match in_ids with
      | [ _; b ] -> Hashtbl.replace ctx.ids (sole eqn.outs).vid b
      | _ -> invalid_arg "Stablehlo.Emit: tie arity")
  | Empty _ -> emit_empty ctx eqn
  | Platform_index platforms -> emit_platform_index ctx eqn platforms
  | After_all ->
      failwith
        "Stablehlo.Emit: After_all produces a token (no ShapedArray aval), \
         non-representable (M5)"
  | Create_token ->
      failwith
        "Stablehlo.Emit: Create_token produces a token (no ShapedArray aval), \
         non-representable (M5)"
  | Dce_sink ->
      failwith
        "Stablehlo.Emit: Dce_sink is a DCE-effect marker, non-representable"
  | Empty2 _ ->
      failwith "Stablehlo.Emit: Empty2 (extended dtype) deferred to M5"
  | From_edtype _ ->
      failwith "Stablehlo.Emit: From_edtype (extended dtypes) deferred to M5"
  | To_edtype _ ->
      failwith "Stablehlo.Emit: To_edtype (extended dtypes) deferred to M5"
  | _ ->
      failwith
        "Stablehlo.Emit: no lowering rule for this primitive (rows 81-87)"

let emit_closed_jaxpr (cj : closed_jaxpr) : string =
  let n_consts = List.length cj.consts in
  let jx = cj.jaxpr in
  let const_binders, arg_binders =
    match Util.split_list jx.in_binders [ n_consts ] with
    | [ cb; ob ] -> (cb, ob)
    | _ -> invalid_arg "Stablehlo.Emit: binder split"
  in
  let ctx = { buf = Buffer.create 256; ids = Hashtbl.create 16; next = 0 } in
  let params =
    List.map
      (fun (v : var) ->
        Printf.sprintf "%s: %s"
          ("%" ^ string_of_int (bind_var ctx v))
          (Ir.tensor_type_of_aval v.vaval))
      arg_binders
  in
  List.iter2
    (fun (v : var) nd ->
      let n = bind_var ctx v in
      emit_constant_at ctx n nd)
    const_binders cj.consts;
  List.iter
    (fun (e : eqn) ->
      let in_ids = List.map (id_of_atom ctx) e.inputs in
      emit_eqn ctx e in_ids)
    jx.eqns;
  let out_names = List.map (ssa_of_atom ctx) jx.outs in
  let out_types =
    List.map (fun a -> Ir.tensor_type_of_aval (atom_aval a)) jx.outs
  in
  if out_names = [] then Buffer.add_string ctx.buf "    return\n"
  else
    Buffer.add_string ctx.buf
      (Printf.sprintf "    return %s : %s\n"
         (String.concat ", " out_names)
         (String.concat ", " out_types));
  Printf.sprintf "module {\n  func.func public @main(%s) -> (%s) {\n%s  }\n}\n"
    (String.concat ", " params)
    (String.concat ", " out_types)
    (Buffer.contents ctx.buf)
