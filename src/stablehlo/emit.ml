open Types

let aval_of_ndarray nd =
  {
    shape = Ndarray.shape nd;
    dtype = Ndarray.dtype nd;
    weak_type = Ndarray.weak nd;
  }

let atom_aval = function
  | A_var v -> v.vaval
  | A_lit nd -> aval_of_ndarray nd
  | DropVar a -> a

let element_literal nd idx =
  match Ndarray.dtype nd with
  | Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn
  | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
  | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
  | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
      Ir.float_literal (Ndarray.dtype nd) (Ndarray.get_f nd idx)
  | Dtype.I32 | Dtype.I64 | Dtype.Uint32 | Dtype.I8 | Dtype.I16 | Dtype.I2
  | Dtype.I4 | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4
  | Dtype.Uint64 ->
      Ir.int_literal (Ndarray.get_i64 nd idx)
  | Dtype.Bool -> Ir.bool_literal (Ndarray.get_i64 nd idx <> 0L)
  | Dtype.Complex64 | Dtype.Complex128 ->
      Ir.complex_literal (Ndarray.dtype nd) (Ndarray.get_c nd idx)
  | Dtype.Float0 -> invalid_arg "Stablehlo.Emit: float0 has no stablehlo type"

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

type ctx = {
  mutable buf : Buffer.t;
  extras : Buffer.t;
  ids : (int, int) Hashtbl.t;
  names : (int, string) Hashtbl.t;
  mutable next : int;
  mutable literals : (Ndarray.t * int) list;
}

let fresh ctx =
  let n = ctx.next in
  ctx.next <- n + 1;
  n

let bind_var ctx (v : var) =
  let n = fresh ctx in
  Hashtbl.replace ctx.ids v.vid n;
  n

let used_symbols : (string, int) Hashtbl.t = Hashtbl.create 8
let symbol_counter = ref 0

let unique_symbol base =
  let rec probe candidate =
    if Hashtbl.mem used_symbols candidate then begin
      let n = !symbol_counter in
      incr symbol_counter;
      probe (Printf.sprintf "%s_%d" base n)
    end
    else begin
      Hashtbl.replace used_symbols candidate 1;
      candidate
    end
  in
  probe base

let call_symbols : (int, string) Hashtbl.t = Hashtbl.create 8
let result_aliases : (int, string) Hashtbl.t = Hashtbl.create 16
let next_alias = ref (-1)

let alias_result text =
  let id = !next_alias in
  decr next_alias;
  Hashtbl.replace result_aliases id text;
  id

let name n =
  if n < 0 then Hashtbl.find result_aliases n else "%" ^ string_of_int n

let ssa_of_var ctx (v : var) =
  match Hashtbl.find_opt ctx.names v.vid with
  | Some s -> s
  | None -> (
      match Hashtbl.find_opt ctx.ids v.vid with
      | Some n -> name n
      | None -> invalid_arg "Stablehlo.Emit: unbound variable")

let emit_constant_at ctx n nd =
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %%%d = stablehlo.constant %s : %s\n" n
       (Ir.dense (dense_body nd))
       (Ir.tensor_type (Ndarray.dtype nd) (Ndarray.shape nd)))

let capture ctx f =
  let saved = ctx.buf in
  let saved_literals = ctx.literals in
  ctx.buf <- Buffer.create 128;
  ctx.literals <- [];
  f ();
  let s = Buffer.contents ctx.buf in
  ctx.buf <- saved;
  ctx.literals <- saved_literals;
  s

let reindent extra s =
  let pad = String.make extra ' ' in
  String.split_on_char '\n' s
  |> List.map (fun ln -> if ln = "" then "" else pad ^ ln)
  |> String.concat "\n"

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

let is_complex_atom atom =
  match (atom_aval atom).dtype with
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let emit_stablehlo_unary ctx (eqn : eqn) in_ids op =
  let x = sole in_ids in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  let out_ty = Ir.tensor_type_of_aval out.vaval in
  let in_ty = Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)) in
  let signature =
    if in_ty = out_ty then out_ty else Printf.sprintf "(%s) -> %s" in_ty out_ty
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.%s %s : %s\n" (name n) op (name x)
       signature)

let scalar_literal dt v =
  match dt with
  | Dtype.Complex64 | Dtype.Complex128 ->
      Ir.complex_literal dt { Complex.re = v; im = 0.0 }
  | _ -> Ir.float_literal dt v

let is_complex_dtype = function
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let is_unsigned_dtype = function
  | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4 | Dtype.Uint32
  | Dtype.Uint64 ->
      true
  | _ -> false

let is_integer_like = function
  | Dtype.Bool | Dtype.I8 | Dtype.I16 | Dtype.I2 | Dtype.I4 | Dtype.I32
  | Dtype.I64 | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4
  | Dtype.Uint32 | Dtype.Uint64 ->
      true
  | _ -> false

let emit_self_multiply ctx (eqn : eqn) in_ids =
  let x = sole in_ids in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.multiply %s, %s : %s\n" (name n)
       (name x) (name x)
       (Ir.tensor_type_of_aval out.vaval))

let emit_unsigned_sign ctx (eqn : eqn) in_ids =
  let x = sole in_ids in
  let out = sole eqn.outs in
  let ty = Ir.tensor_type_of_aval out.vaval in
  let flag_ty = Ir.tensor_type Dtype.Bool out.vaval.shape in
  let scalar_ty = Ir.tensor_type out.vaval.dtype [||] in
  let splat value =
    let c = fresh ctx in
    Buffer.add_string ctx.buf
      (Printf.sprintf "    %s = stablehlo.constant dense<%d> : %s\n" (name c)
         value scalar_ty);
    let n = fresh ctx in
    Buffer.add_string ctx.buf
      (Printf.sprintf
         "    %s = stablehlo.broadcast_in_dim %s, dims = [] : (%s) -> %s\n"
         (name n) (name c) scalar_ty ty);
    n
  in
  let zero = splat 0 in
  let flag = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.compare EQ, %s, %s, UNSIGNED : (%s, %s) -> %s\n"
       (name flag) (name x) (name zero) ty ty flag_ty);
  let zeros = splat 0 in
  let ones = splat 1 in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.select %s, %s, %s : %s, %s\n" (name n)
       (name flag) (name zeros) (name ones) flag_ty ty)

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

let emit_complex_pick ctx (eqn : eqn) a b ~direction =
  let out = sole eqn.outs in
  let cty = Ir.tensor_type_of_aval out.vaval in
  let rdt =
    match out.vaval.dtype with Dtype.Complex128 -> Dtype.F64 | _ -> Dtype.F32
  in
  let rty = Ir.tensor_type rdt out.vaval.shape in
  let bty = Ir.tensor_type Dtype.Bool out.vaval.shape in
  let unary op v =
    let n = fresh ctx in
    Stdlib.Buffer.add_string ctx.buf
      (Printf.sprintf "    %s = stablehlo.%s %s : (%s) -> %s\n" (name n) op
         (name v) cty rty);
    n
  in
  let compare dir x y =
    let n = fresh ctx in
    Stdlib.Buffer.add_string ctx.buf
      (Printf.sprintf
         "    %s = stablehlo.compare %s, %s, %s, FLOAT : (%s, %s) -> %s\n"
         (name n) dir (name x) (name y) rty rty bty);
    n
  in
  let ra = unary "real" a in
  let rb = unary "real" b in
  let same = compare "EQ" ra rb in
  let by_real = compare direction ra rb in
  let ia = unary "imag" a in
  let ib = unary "imag" b in
  let by_imag = compare direction ia ib in
  let pick = fresh ctx in
  Stdlib.Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.select %s, %s, %s : %s, %s\n"
       (name pick) (name same) (name by_imag) (name by_real) bty bty);
  let n = bind_var ctx out in
  Stdlib.Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.select %s, %s, %s : %s, %s\n" (name n)
       (name pick) (name a) (name b) bty cty)

let emit_stablehlo_binary ctx (eqn : eqn) in_ids op =
  let a, b = pair in_ids in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.%s %s, %s : %s\n" (name n) op (name a)
       (name b)
       (Ir.tensor_type_of_aval out.vaval))

let is_complex_atom atom =
  match (atom_aval atom).dtype with
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

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
       (Ir.dense (scalar_literal dt (Float.log 2.0)))
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
  let one = Ir.dense (scalar_literal dt 1.0) in
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
    else failwith "Stablehlo.Emit: negative Integer_pow"
  in
  Hashtbl.replace ctx.ids out.vid result

let compare_type_of dtype total_order =
  match dtype with
  | Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn
  | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
  | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
  | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
      if total_order then "TOTALORDER" else "FLOAT"
  | Dtype.I32 | Dtype.I64 | Dtype.I8 | Dtype.I16 | Dtype.I2 | Dtype.I4 ->
      "SIGNED"
  | Dtype.Bool | Dtype.Uint32 | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2
  | Dtype.Uint4 | Dtype.Uint64 ->
      "UNSIGNED"
  | Dtype.Complex64 | Dtype.Complex128 -> "FLOAT"
  | Dtype.Float0 -> invalid_arg "Stablehlo.Emit: float0 has no stablehlo type"

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
  | Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn
  | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
  | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
  | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
      Ir.float_literal dt 0.0
  | Dtype.I32 | Dtype.I64 | Dtype.Uint32 | Dtype.I8 | Dtype.I16 | Dtype.I2
  | Dtype.I4 | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4
  | Dtype.Uint64 ->
      Ir.int_literal 0L
  | Dtype.Bool -> Ir.bool_literal false
  | Dtype.Complex64 | Dtype.Complex128 -> Ir.complex_literal dt Complex.zero
  | Dtype.Float0 -> invalid_arg "Stablehlo.Emit: float0 has no stablehlo type"

let emit_convert ctx (eqn : eqn) in_ids =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let out = sole eqn.outs in
  let in_dt = in_aval.dtype in
  let out_dt = out.vaval.dtype in
  let oty = Ir.tensor_type_of_aval out.vaval in
  let x, in_dt, in_aval =
    match (in_dt, out_dt) with
    | (Dtype.Complex64 | Dtype.Complex128), Dtype.Bool ->
        let rdt = if in_dt = Dtype.Complex128 then Dtype.F64 else Dtype.F32 in
        let re = fresh ctx in
        Buffer.add_string ctx.buf
          (Printf.sprintf "    %s = stablehlo.real %s : (%s) -> %s\n" (name re)
             (name x)
             (Ir.tensor_type_of_aval in_aval)
             (Ir.tensor_type rdt in_aval.shape));
        (re, rdt, { in_aval with dtype = rdt })
    | _ -> (x, in_dt, in_aval)
  in
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
  else if in_dt = out_dt then begin
    let n = bind_var ctx out in
    Buffer.add_string ctx.buf
      (Printf.sprintf "    %s = stablehlo.convert %s : %s\n" (name n) (name x)
         oty)
  end
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
  | outs, _ ->
      let group = fresh ctx in
      let types =
        String.concat ", "
          (List.map (fun (v : var) -> Ir.tensor_type_of_aval v.vaval) outs)
      in
      Buffer.add_string ctx.buf
        (Printf.sprintf "    %%%d:%d = stablehlo.optimization_barrier %s : %s\n"
           group (List.length outs)
           (String.concat ", " (List.map name in_ids))
           types);
      List.iteri
        (fun i (v : var) ->
          Hashtbl.replace ctx.ids v.vid
            (alias_result (Printf.sprintf "%%%d#%d" group i)))
        outs

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

let platform_branch_index platforms =
  let current = Config.current_platform () in
  let n = Array.length platforms in
  let rec find_named i =
    if i >= n then find_default 0
    else
      match platforms.(i) with
      | Some names when Array.exists (fun p -> p = current) names -> i
      | _ -> find_named (i + 1)
  and find_default i =
    if i >= n then
      failwith
        (Printf.sprintf
           "Stablehlo.Emit: platform_index has no %s or default branch" current)
    else match platforms.(i) with None -> i | Some _ -> find_default (i + 1)
  in
  find_named 0

let emit_platform_index ctx (eqn : eqn) platforms =
  let out = sole eqn.outs in
  let ty = Ir.tensor_type_of_aval out.vaval in
  let constant n v =
    Printf.sprintf "%s = stablehlo.constant %s : %s\n" (name n)
      (Ir.dense (Ir.int_literal (Int64.of_int v)))
      ty
  in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    ("    " ^ constant n (platform_branch_index platforms))

let emit_broadcast_in_dim ctx (eqn : eqn) in_ids dims =
  let x = sole in_ids in
  let inty = Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)) in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.broadcast_in_dim %s, dims = %s : (%s) -> %s\n"
       (name n) (name x) (Ir.int_array_attr dims) inty
       (Ir.tensor_type_of_aval out.vaval))

let emit_concatenate ctx (eqn : eqn) in_ids dim =
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  let operands = String.concat ", " (List.map name in_ids) in
  let intys =
    String.concat ", "
      (List.map (fun a -> Ir.tensor_type_of_aval (atom_aval a)) eqn.inputs)
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.concatenate %s, dim = %d : (%s) -> %s\n"
       (name n) operands dim intys
       (Ir.tensor_type_of_aval out.vaval))

let emit_iota ctx (eqn : eqn) dimension =
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.iota dim = %d : %s\n" (name n) dimension
       (Ir.tensor_type_of_aval out.vaval))

let emit_pad ctx (eqn : eqn) in_ids config =
  let op, pv = pair in_ids in
  let opty, pvty =
    match eqn.inputs with
    | [ o; p ] ->
        ( Ir.tensor_type_of_aval (atom_aval o),
          Ir.tensor_type_of_aval (atom_aval p) )
    | _ -> invalid_arg "Stablehlo.Emit: pad arity"
  in
  let out = sole eqn.outs in
  let low = Array.map (fun (l, _, _) -> l) config in
  let high = Array.map (fun (_, h, _) -> h) config in
  let interior = Array.map (fun (_, _, i) -> i) config in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.pad %s, %s, low = %s, high = %s, interior = %s : \
        (%s, %s) -> %s\n"
       (name n) (name op) (name pv) (Ir.int_array_attr low)
       (Ir.int_array_attr high)
       (Ir.int_array_attr interior)
       opty pvty
       (Ir.tensor_type_of_aval out.vaval))

let emit_reshape ctx (eqn : eqn) in_ids =
  let x = sole in_ids in
  let inty = Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)) in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.reshape %s : (%s) -> %s\n" (name n)
       (name x) inty
       (Ir.tensor_type_of_aval out.vaval))

let emit_rev ctx (eqn : eqn) in_ids dims =
  let x = sole in_ids in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.reverse %s, dims = %s : %s\n" (name n)
       (name x) (Ir.int_array_attr dims)
       (Ir.tensor_type_of_aval out.vaval))

let emit_transpose ctx (eqn : eqn) in_ids perm =
  let x = sole in_ids in
  let inty = Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)) in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.transpose %s, dims = %s : (%s) -> %s\n"
       (name n) (name x) (Ir.int_array_attr perm) inty
       (Ir.tensor_type_of_aval out.vaval))

let emit_slice ctx x_id inty ranges outty =
  let idx =
    String.concat ", "
      (List.map (fun (lo, hi) -> Printf.sprintf "%d:%d" lo hi) ranges)
  in
  let n = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.slice %s [%s] : (%s) -> %s\n" (name n)
       (name x_id) idx inty outty);
  n

let emit_split ctx (eqn : eqn) in_ids axis sizes =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let inty = Ir.tensor_type_of_aval in_aval in
  let ndim = Array.length in_aval.shape in
  let start = ref 0 in
  List.iteri
    (fun i (out : var) ->
      let sz = sizes.(i) in
      let ranges =
        List.init ndim (fun d ->
            if d = axis then (!start, !start + sz) else (0, in_aval.shape.(d)))
      in
      let sid =
        emit_slice ctx x inty ranges (Ir.tensor_type_of_aval out.vaval)
      in
      Hashtbl.replace ctx.ids out.vid sid;
      start := !start + sz)
    eqn.outs

let emit_unstack ctx (eqn : eqn) in_ids axis =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let inty = Ir.tensor_type_of_aval in_aval in
  let ndim = Array.length in_aval.shape in
  let dt = in_aval.dtype in
  List.iteri
    (fun i (out : var) ->
      let ranges =
        List.init ndim (fun d ->
            if d = axis then (i, i + 1) else (0, in_aval.shape.(d)))
      in
      let mid_shape =
        Array.mapi (fun d s -> if d = axis then 1 else s) in_aval.shape
      in
      let midty = Ir.tensor_type dt mid_shape in
      let sid = emit_slice ctx x inty ranges midty in
      let n = bind_var ctx out in
      Buffer.add_string ctx.buf
        (Printf.sprintf "    %s = stablehlo.reshape %s : (%s) -> %s\n" (name n)
           (name sid) midty
           (Ir.tensor_type_of_aval out.vaval)))
    eqn.outs

let insert_dim shape axis v =
  Array.init
    (Array.length shape + 1)
    (fun i ->
      if i < axis then shape.(i) else if i = axis then v else shape.(i - 1))

let emit_stack ctx (eqn : eqn) in_ids axis =
  let out = sole eqn.outs in
  let outty = Ir.tensor_type_of_aval out.vaval in
  let expanded =
    List.map2
      (fun xid a ->
        let av = atom_aval a in
        let ndim = Array.length av.shape in
        let new_shape = insert_dim av.shape axis 1 in
        let dims =
          Array.of_list
            (List.filter (fun i -> i <> axis) (List.init (ndim + 1) Fun.id))
        in
        let ty_in = Ir.tensor_type_of_aval av in
        let ty_exp = Ir.tensor_type av.dtype new_shape in
        let n = fresh ctx in
        Buffer.add_string ctx.buf
          (Printf.sprintf
             "    %s = stablehlo.broadcast_in_dim %s, dims = %s : (%s) -> %s\n"
             (name n) (name xid) (Ir.int_array_attr dims) ty_in ty_exp);
        (n, ty_exp))
      in_ids eqn.inputs
  in
  let n = bind_var ctx out in
  let operands =
    String.concat ", " (List.map (fun (i, _) -> name i) expanded)
  in
  let intys = String.concat ", " (List.map snd expanded) in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.concatenate %s, dim = %d : (%s) -> %s\n"
       (name n) operands axis intys outty)

let emit_tile ctx (eqn : eqn) in_ids reps =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let inty = Ir.tensor_type_of_aval in_aval in
  let dt = in_aval.dtype in
  let shape = in_aval.shape in
  let ndim = Array.length shape in
  let out = sole eqn.outs in
  let outty = Ir.tensor_type_of_aval out.vaval in
  let expand_shape =
    Array.init (2 * ndim) (fun i -> if i land 1 = 0 then 1 else shape.(i / 2))
  in
  let expty = Ir.tensor_type dt expand_shape in
  let r1 = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.reshape %s : (%s) -> %s\n" (name r1)
       (name x) inty expty);
  let bshape =
    Array.init (2 * ndim) (fun i ->
        if i land 1 = 0 then reps.(i / 2) else shape.(i / 2))
  in
  let bty = Ir.tensor_type dt bshape in
  let dims = Array.init (2 * ndim) Fun.id in
  let b = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.broadcast_in_dim %s, dims = %s : (%s) -> %s\n"
       (name b) (name r1) (Ir.int_array_attr dims) expty bty);
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.reshape %s : (%s) -> %s\n" (name n)
       (name b) bty outty)

let reduce_zero dt = Ir.dense (zero_dense dt)

let reduce_one dt =
  Ir.dense
    (match dt with
    | Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn
    | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
    | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
    | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
        Ir.float_literal dt 1.0
    | Dtype.I32 | Dtype.I64 | Dtype.Uint32 | Dtype.I8 | Dtype.I16 | Dtype.I2
    | Dtype.I4 | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4
    | Dtype.Uint64 ->
        Ir.int_literal 1L
    | Dtype.Bool -> Ir.bool_literal true
    | Dtype.Complex64 | Dtype.Complex128 -> Ir.complex_literal dt Complex.one
    | Dtype.Float0 -> invalid_arg "Stablehlo.Emit: float0 has no stablehlo type")

let reduce_allones dt =
  Ir.dense
    (match dt with
    | Dtype.I32 | Dtype.I64 | Dtype.I8 | Dtype.I16 | Dtype.I2 | Dtype.I4 ->
        Ir.int_literal (-1L)
    | Dtype.Uint2 -> Ir.int_literal 3L
    | Dtype.Uint4 -> Ir.int_literal 15L
    | Dtype.Uint8 -> Ir.int_literal 255L
    | Dtype.Uint16 -> Ir.int_literal 65535L
    | Dtype.Uint32 -> Ir.int_literal 4294967295L
    | Dtype.Uint64 -> "18446744073709551615"
    | Dtype.Bool -> Ir.bool_literal true
    | Dtype.Float0 -> invalid_arg "Stablehlo.Emit: float0 has no stablehlo type"
    | Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn
    | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
    | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
    | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
        invalid_arg "Stablehlo.Emit: bitwise reduce on floating-point dtype"
    | Dtype.Complex64 | Dtype.Complex128 ->
        invalid_arg "Stablehlo.Emit: complex unsupported (row 96)")

let reduce_maxid dt =
  Ir.dense
    (match dt with
    | Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn
    | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
    | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
    | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
        Ir.float_literal dt (Dtypes.max_identity dt)
    | Dtype.I2 -> Ir.int_literal (-2L)
    | Dtype.I4 -> Ir.int_literal (-8L)
    | Dtype.I8 -> Ir.int_literal (-128L)
    | Dtype.I16 -> Ir.int_literal (-32768L)
    | Dtype.I32 -> Ir.int_literal (-2147483648L)
    | Dtype.I64 -> Ir.int_literal Int64.min_int
    | Dtype.Uint32 | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4
    | Dtype.Uint64 ->
        Ir.int_literal 0L
    | Dtype.Bool -> Ir.bool_literal false
    | Dtype.Float0 -> invalid_arg "Stablehlo.Emit: float0 has no stablehlo type"
    | Dtype.Complex64 | Dtype.Complex128 ->
        Ir.complex_literal dt { Complex.re = neg_infinity; im = 0.0 })

let reduce_minid dt =
  Ir.dense
    (match dt with
    | Dtype.F32 | Dtype.F64 | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn
    | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3
    | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
    | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
        Ir.float_literal dt (Dtypes.min_identity dt)
    | Dtype.I2 -> Ir.int_literal 1L
    | Dtype.I4 -> Ir.int_literal 7L
    | Dtype.I8 -> Ir.int_literal 127L
    | Dtype.I16 -> Ir.int_literal 32767L
    | Dtype.I32 -> Ir.int_literal 2147483647L
    | Dtype.I64 -> Ir.int_literal Int64.max_int
    | Dtype.Uint2 -> Ir.int_literal 3L
    | Dtype.Uint4 -> Ir.int_literal 15L
    | Dtype.Uint64 -> "18446744073709551615"
    | Dtype.Uint8 -> Ir.int_literal 255L
    | Dtype.Uint16 -> Ir.int_literal 65535L
    | Dtype.Uint32 -> Ir.int_literal 4294967295L
    | Dtype.Bool -> Ir.bool_literal true
    | Dtype.Float0 -> invalid_arg "Stablehlo.Emit: float0 has no stablehlo type"
    | Dtype.Complex64 | Dtype.Complex128 ->
        Ir.complex_literal dt { Complex.re = infinity; im = 0.0 })

let emit_min_max ctx (eqn : eqn) in_ids ~op ~direction =
  match (sole eqn.outs).vaval.dtype with
  | Dtype.Complex64 | Dtype.Complex128 ->
      let a, b = pair in_ids in
      emit_complex_pick ctx eqn a b ~direction
  | _ -> emit_stablehlo_binary ctx eqn in_ids op

let emit_reduce ctx (eqn : eqn) in_ids axes op init =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let dt = in_aval.dtype in
  let sty = Ir.tensor_type dt [||] in
  let out = sole eqn.outs in
  let c = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c) (init dt)
       sty);
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.reduce(%s init: %s) applies stablehlo.%s across \
        dimensions = %s : (%s, %s) -> %s\n"
       (name n) (name x) (name c) op (Ir.int_array_attr axes)
       (Ir.tensor_type_of_aval in_aval)
       sty
       (Ir.tensor_type_of_aval out.vaval))

let emit_argminmax ctx (eqn : eqn) in_ids ~is_max ~axis ~index_dtype =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let dt = in_aval.dtype in
  let inty = Ir.tensor_type_of_aval in_aval in
  let out = sole eqn.outs in
  let outty = Ir.tensor_type_of_aval out.vaval in
  let fname = unique_symbol (if is_max then "argmax" else "argmin") in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = call @%s(%s) : (%s) -> %s\n" (name n) fname
       (name x) inty outty);
  let vscalar = Ir.tensor_type dt [||] in
  let iscalar = Ir.tensor_type index_dtype [||] in
  let iota_ty = Ir.tensor_type index_dtype in_aval.shape in
  let vout = Ir.tensor_type dt out.vaval.shape in
  let bty = Ir.tensor_type Dtype.Bool [||] in
  let dir = if is_max then "GT" else "LT" in
  let ct = compare_type_of dt false in
  let ict = compare_type_of index_dtype false in
  let init_v = if is_max then reduce_maxid dt else reduce_minid dt in
  let init_i = Ir.dense (zero_dense index_dtype) in
  let b = ctx.extras in
  Buffer.add_string b
    (Printf.sprintf "  func.func private @%s(%%0: %s) -> %s {\n" fname inty
       outty);
  Buffer.add_string b
    (Printf.sprintf "    %%1 = stablehlo.iota dim = %d : %s\n" axis iota_ty);
  Buffer.add_string b
    (Printf.sprintf "    %%2 = stablehlo.constant %s : %s\n" init_v vscalar);
  Buffer.add_string b
    (Printf.sprintf "    %%3 = stablehlo.constant %s : %s\n" init_i iscalar);
  Buffer.add_string b
    (Printf.sprintf
       "    %%4:2 = stablehlo.reduce(%%0 init: %%2), (%%1 init: %%3) across \
        dimensions = [%d] : (%s, %s, %s, %s) -> (%s, %s)\n"
       axis inty iota_ty vscalar iscalar vout outty);
  Buffer.add_string b
    (Printf.sprintf "     reducer(%%5: %s, %%6: %s) (%%7: %s, %%8: %s)  {\n"
       vscalar vscalar iscalar iscalar);
  Buffer.add_string b
    (Printf.sprintf
       "      %%9 = stablehlo.compare %s, %%5, %%6, %s : (%s, %s) -> %s\n" dir
       ct vscalar vscalar bty);
  Buffer.add_string b
    (Printf.sprintf
       "      %%10 = stablehlo.compare NE, %%5, %%5, %s : (%s, %s) -> %s\n" ct
       vscalar vscalar bty);
  Buffer.add_string b
    (Printf.sprintf "      %%11 = stablehlo.or %%9, %%10 : %s\n" bty);
  Buffer.add_string b
    (Printf.sprintf
       "      %%12 = stablehlo.compare EQ, %%5, %%6, %s : (%s, %s) -> %s\n" ct
       vscalar vscalar bty);
  Buffer.add_string b
    (Printf.sprintf
       "      %%13 = stablehlo.compare LT, %%7, %%8, %s : (%s, %s) -> %s\n" ict
       iscalar iscalar bty);
  Buffer.add_string b
    (Printf.sprintf "      %%14 = stablehlo.and %%12, %%13 : %s\n" bty);
  Buffer.add_string b
    (Printf.sprintf "      %%15 = stablehlo.or %%11, %%14 : %s\n" bty);
  Buffer.add_string b
    (Printf.sprintf "      %%16 = stablehlo.select %%11, %%5, %%6 : %s, %s\n"
       bty vscalar);
  Buffer.add_string b
    (Printf.sprintf "      %%17 = stablehlo.select %%15, %%7, %%8 : %s, %s\n"
       bty iscalar);
  Buffer.add_string b
    (Printf.sprintf "      stablehlo.return %%16, %%17 : %s, %s\n" vscalar
       iscalar);
  Buffer.add_string b "    }\n";
  Buffer.add_string b (Printf.sprintf "    return %%4#1 : %s\n" outty);
  Buffer.add_string b "  }\n"

let cum_window_attrs shape axis reverse =
  let ndim = Array.length shape in
  let ones = Array.make ndim 1 in
  let arr a =
    "array<i64: "
    ^ (Array.to_list a |> List.map string_of_int |> String.concat ", ")
    ^ ">"
  in
  let window_dims = Array.mapi (fun d s -> if d = axis then s else 1) shape in
  let pad =
    Array.to_list
      (Array.mapi
         (fun d _ ->
           if d = axis then
             if reverse then (0, shape.(axis) - 1) else (shape.(axis) - 1, 0)
           else (0, 0))
         shape)
  in
  let pad_str =
    "dense<["
    ^ (List.map (fun (l, h) -> Printf.sprintf "[%d, %d]" l h) pad
      |> String.concat ", ")
    ^ Printf.sprintf "]> : tensor<%dx2xi64>" ndim
  in
  Printf.sprintf
    "base_dilations = %s, padding = %s, window_dilations = %s, \
     window_dimensions = %s, window_strides = %s"
    (arr ones) pad_str (arr ones) (arr window_dims) (arr ones)

let cum_functions : (string, string) Hashtbl.t = Hashtbl.create 8

let emit_cum_single ctx (eqn : eqn) in_ids ~fname:base_name ~axis ~reverse ~init
    ~broadcast ~op =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let dt = in_aval.dtype in
  let ty = Ir.tensor_type_of_aval in_aval in
  let sty = Ir.tensor_type dt [||] in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  let attrs = cum_window_attrs in_aval.shape axis reverse in
  let key = String.concat "|" [ base_name; ty; attrs ] in
  let known = Hashtbl.find_opt cum_functions key in
  let fname =
    match known with Some f -> f | None -> unique_symbol base_name
  in
  Hashtbl.replace cum_functions key fname;
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = call @%s(%s) : (%s) -> %s\n" (name n) fname
       (name x) ty ty);
  if known = None then begin
    let b = ctx.extras in
    Buffer.add_string b
      (Printf.sprintf "  func.func private @%s(%%0: %s) -> %s {\n" fname ty ty);
    Buffer.add_string b
      (Printf.sprintf "    %%2 = stablehlo.constant %s : %s\n" (init dt) sty);
    if broadcast then (
      Buffer.add_string b
        (Printf.sprintf
           "    %%1 = stablehlo.broadcast_in_dim %%2, dims = [] : (%s) -> %s\n"
           sty sty);
      Buffer.add_string b
        (Printf.sprintf
           "    %%3 = \"stablehlo.reduce_window\"(%%0, %%1) <{%s}> ({\n" attrs);
      Buffer.add_string b
        (Printf.sprintf "    ^bb0(%%4: %s, %%5: %s):\n" sty sty);
      Buffer.add_string b
        (Printf.sprintf "      %%6 = stablehlo.%s %%4, %%5 : %s\n" op sty);
      Buffer.add_string b
        (Printf.sprintf "      stablehlo.return %%6 : %s\n" sty);
      Buffer.add_string b (Printf.sprintf "    }) : (%s, %s) -> %s\n" ty sty ty);
      Buffer.add_string b (Printf.sprintf "    return %%3 : %s\n" ty))
    else begin
      Buffer.add_string b
        (Printf.sprintf
           "    %%1 = \"stablehlo.reduce_window\"(%%0, %%2) <{%s}> ({\n" attrs);
      Buffer.add_string b
        (Printf.sprintf "    ^bb0(%%3: %s, %%4: %s):\n" sty sty);
      Buffer.add_string b
        (Printf.sprintf "      %%5 = stablehlo.%s %%3, %%4 : %s\n" op sty);
      Buffer.add_string b
        (Printf.sprintf "      stablehlo.return %%5 : %s\n" sty);
      Buffer.add_string b (Printf.sprintf "    }) : (%s, %s) -> %s\n" ty sty ty);
      Buffer.add_string b (Printf.sprintf "    return %%1 : %s\n" ty)
    end;
    Buffer.add_string b "  }\n"
  end

let emit_cum_logsumexp ctx (eqn : eqn) in_ids ~axis ~reverse =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let dt = in_aval.dtype in
  let ty = Ir.tensor_type_of_aval in_aval in
  let sty = Ir.tensor_type dt [||] in
  let bty = Ir.tensor_type Dtype.Bool [||] in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  let fname = unique_symbol "cumlogsumexp" in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = call @%s(%s) : (%s) -> %s\n" (name n) fname
       (name x) ty ty);
  let attrs = cum_window_attrs in_aval.shape axis reverse in
  let b = ctx.extras in
  Buffer.add_string b
    (Printf.sprintf "  func.func private @%s(%%0: %s) -> %s {\n" fname ty ty);
  Buffer.add_string b
    (Printf.sprintf "    %%2 = stablehlo.constant %s : %s\n" (reduce_maxid dt)
       sty);
  Buffer.add_string b
    (Printf.sprintf
       "    %%1 = \"stablehlo.reduce_window\"(%%0, %%2) <{%s}> ({\n" attrs);
  Buffer.add_string b (Printf.sprintf "    ^bb0(%%3: %s, %%4: %s):\n" sty sty);
  Buffer.add_string b
    (Printf.sprintf "      %%5 = stablehlo.maximum %%3, %%4 : %s\n" sty);
  Buffer.add_string b
    (Printf.sprintf "      %%6 = stablehlo.subtract %%3, %%4 : %s\n" sty);
  Buffer.add_string b
    (Printf.sprintf
       "      %%7 = stablehlo.compare NE, %%6, %%6, FLOAT : (%s, %s) -> %s\n"
       sty sty bty);
  Buffer.add_string b
    (Printf.sprintf "      %%8 = stablehlo.add %%3, %%4 : %s\n" sty);
  Buffer.add_string b
    (Printf.sprintf "      %%9 = stablehlo.abs %%6 : %s\n" sty);
  Buffer.add_string b
    (Printf.sprintf "      %%10 = stablehlo.negate %%9 : %s\n" sty);
  Buffer.add_string b
    (Printf.sprintf "      %%11 = stablehlo.exponential %%10 : %s\n" sty);
  Buffer.add_string b
    (Printf.sprintf "      %%12 = stablehlo.log_plus_one %%11 : %s\n" sty);
  Buffer.add_string b
    (Printf.sprintf "      %%13 = stablehlo.add %%5, %%12 : %s\n" sty);
  Buffer.add_string b
    (Printf.sprintf "      %%14 = stablehlo.select %%7, %%8, %%13 : %s, %s\n"
       bty sty);
  Buffer.add_string b (Printf.sprintf "      stablehlo.return %%14 : %s\n" sty);
  Buffer.add_string b (Printf.sprintf "    }) : (%s, %s) -> %s\n" ty sty ty);
  Buffer.add_string b (Printf.sprintf "    return %%1 : %s\n" ty);
  Buffer.add_string b "  }\n"

let array_i64 a =
  "array<i64: "
  ^ (Array.to_list a |> List.map string_of_int |> String.concat ", ")
  ^ ">"

let emit_slice_op ctx (eqn : eqn) in_ids start_indices limit_indices strides =
  let x = sole in_ids in
  let inty = Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)) in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  let ndim = Array.length start_indices in
  let idx =
    String.concat ", "
      (List.init ndim (fun d ->
           let lo = start_indices.(d) and hi = limit_indices.(d) in
           let st = match strides with None -> 1 | Some s -> s.(d) in
           if st = 1 then Printf.sprintf "%d:%d" lo hi
           else Printf.sprintf "%d:%d:%d" lo hi st))
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.slice %s [%s] : (%s) -> %s\n" (name n)
       (name x) idx inty
       (Ir.tensor_type_of_aval out.vaval))

let emit_dynamic_slice ctx (eqn : eqn) in_ids slice_sizes =
  let op_id, idx_ids =
    match in_ids with
    | o :: r -> (o, r)
    | _ -> invalid_arg "Stablehlo.Emit: arity"
  in
  let op_ty, idx_tys =
    match eqn.inputs with
    | o :: r ->
        ( Ir.tensor_type_of_aval (atom_aval o),
          List.map (fun a -> Ir.tensor_type_of_aval (atom_aval a)) r )
    | _ -> invalid_arg "Stablehlo.Emit: arity"
  in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  let operands = String.concat ", " (List.map name (op_id :: idx_ids)) in
  let intys = String.concat ", " (op_ty :: idx_tys) in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.dynamic_slice %s, sizes = %s : (%s) -> %s\n" (name n)
       operands
       (Ir.int_array_attr slice_sizes)
       intys
       (Ir.tensor_type_of_aval out.vaval))

let emit_dynamic_update_slice ctx (eqn : eqn) in_ids =
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  let operands = String.concat ", " (List.map name in_ids) in
  let intys =
    String.concat ", "
      (List.map (fun a -> Ir.tensor_type_of_aval (atom_aval a)) eqn.inputs)
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.dynamic_update_slice %s : (%s) -> %s\n"
       (name n) operands intys
       (Ir.tensor_type_of_aval out.vaval))

let dim_field name a =
  if Array.length a = 0 then None
  else Some (Printf.sprintf "%s = %s" name (Ir.int_array_attr a))

let gather_dnums_attr (d : gather_dims) index_vector_dim =
  let parts =
    List.filter_map Fun.id
      [
        dim_field "offset_dims" d.offset_dims;
        dim_field "collapsed_slice_dims" d.collapsed_slice_dims;
        dim_field "operand_batching_dims" d.g_operand_batching_dims;
        dim_field "start_indices_batching_dims" d.g_start_indices_batching_dims;
        dim_field "start_index_map" d.start_index_map;
        Some (Printf.sprintf "index_vector_dim = %d" index_vector_dim);
      ]
  in
  "#stablehlo.gather<" ^ String.concat ", " parts ^ ">"

let scatter_dnums_attr (d : scatter_dims) index_vector_dim =
  let parts =
    List.filter_map Fun.id
      [
        dim_field "update_window_dims" d.update_window_dims;
        dim_field "inserted_window_dims" d.inserted_window_dims;
        dim_field "input_batching_dims" d.s_operand_batching_dims;
        dim_field "scatter_indices_batching_dims"
          d.s_scatter_indices_batching_dims;
        dim_field "scatter_dims_to_operand_dims" d.scatter_dims_to_operand_dims;
        Some (Printf.sprintf "index_vector_dim = %d" index_vector_dim);
      ]
  in
  "#stablehlo.scatter<" ^ String.concat ", " parts ^ ">"

let emit_gather ctx (eqn : eqn) in_ids dnums slice_sizes =
  let op_id, idx_id = pair in_ids in
  let op_av, idx_av =
    match eqn.inputs with
    | [ o; i ] -> (atom_aval o, atom_aval i)
    | _ -> invalid_arg "Stablehlo.Emit: gather arity"
  in
  let ivd = Array.length idx_av.shape - 1 in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = \"stablehlo.gather\"(%s, %s) <{dimension_numbers = %s, \
        indices_are_sorted = false, slice_sizes = %s}> : (%s, %s) -> %s\n"
       (name n) (name op_id) (name idx_id)
       (gather_dnums_attr dnums ivd)
       (array_i64 slice_sizes)
       (Ir.tensor_type_of_aval op_av)
       (Ir.tensor_type_of_aval idx_av)
       (Ir.tensor_type_of_aval out.vaval))

let emit_scatter ?(unique = false) ctx (eqn : eqn) in_ids dnums combiner =
  let op_id, idx_id, upd_id =
    match in_ids with
    | [ a; b; c ] -> (a, b, c)
    | _ -> invalid_arg "Stablehlo.Emit: scatter arity"
  in
  let op_av, idx_av, upd_av =
    match eqn.inputs with
    | [ a; b; c ] -> (atom_aval a, atom_aval b, atom_aval c)
    | _ -> invalid_arg "Stablehlo.Emit: scatter arity"
  in
  let dt = op_av.dtype in
  let sty = Ir.tensor_type dt [||] in
  let ivd = Array.length idx_av.shape - 1 in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = \"stablehlo.scatter\"(%s, %s, %s) <{indices_are_sorted = %b, \
        scatter_dimension_numbers = %s, unique_indices = %b}> ({\n"
       (name n) (name op_id) (name idx_id) (name upd_id) unique
       (scatter_dnums_attr dnums ivd)
       unique);
  let a = fresh ctx in
  let b = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    ^bb0(%s: %s, %s: %s):\n" (name a) sty (name b) sty);
  let ret =
    match combiner with
    | None -> b
    | Some op ->
        let c = fresh ctx in
        Buffer.add_string ctx.buf
          (Printf.sprintf "      %s = stablehlo.%s %s, %s : %s\n" (name c) op
             (name a) (name b) sty);
        c
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "      stablehlo.return %s : %s\n" (name ret) sty);
  Buffer.add_string ctx.buf
    (Printf.sprintf "    }) : (%s, %s, %s) -> %s\n"
       (Ir.tensor_type_of_aval op_av)
       (Ir.tensor_type_of_aval idx_av)
       (Ir.tensor_type_of_aval upd_av)
       (Ir.tensor_type_of_aval out.vaval))

let dot_dims_str (dd : dot_dims) =
  let batch =
    if Array.length dd.lhs_batch > 0 then
      Printf.sprintf "batching_dims = %s x %s, "
        (Ir.int_array_attr dd.lhs_batch)
        (Ir.int_array_attr dd.rhs_batch)
    else ""
  in
  let level = Option.value (Config.matmul_level ()) ~default:"DEFAULT" in
  Printf.sprintf "%scontracting_dims = %s x %s, precision = [%s, %s]" batch
    (Ir.int_array_attr dd.lhs_contract)
    (Ir.int_array_attr dd.rhs_contract)
    level level

let dot_preset_info = function
  | "F16_F16_F16" -> ("f16", "f16", "f16", 1, [ Dtype.F16 ], Dtype.F16)
  | "F32_F32_F32" -> ("f32", "f32", "f32", 1, [ Dtype.F32 ], Dtype.F32)
  | "F64_F64_F64" -> ("f64", "f64", "f64", 1, [ Dtype.F64 ], Dtype.F64)
  | "BF16_BF16_F32" ->
      ("bf16", "bf16", "f32", 1, [ Dtype.Bf16; Dtype.F32 ], Dtype.F32)
  | "BF16_BF16_F32_X3" -> ("bf16", "bf16", "f32", 3, [ Dtype.F32 ], Dtype.F32)
  | "BF16_BF16_F32_X6" -> ("bf16", "bf16", "f32", 6, [ Dtype.F32 ], Dtype.F32)
  | other -> invalid_arg ("Stablehlo.Emit: dot preset " ^ other)

let emit_dot_general ctx (eqn : eqn) in_ids dd =
  let a, b = pair in_ids in
  let la, lb =
    match eqn.inputs with
    | [ l; r ] ->
        ( Ir.tensor_type_of_aval (atom_aval l),
          Ir.tensor_type_of_aval (atom_aval r) )
    | _ -> invalid_arg "Stablehlo.Emit: dot_general arity"
  in
  let out = sole eqn.outs in
  Config.check_matmul_preset_on_cpu ();
  match Config.matmul_preset () with
  | None ->
      let n = bind_var ctx out in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %s = stablehlo.dot_general %s, %s, %s : (%s, %s) -> %s\n"
           (name n) (name a) (name b) (dot_dims_str dd) la lb
           (Ir.tensor_type_of_aval out.vaval))
  | Some preset ->
      let attr_l, attr_r, attr_acc, ops, supported, acc =
        dot_preset_info preset
      in
      let retyped (v : aval) dtype = Ir.tensor_type_of_aval { v with dtype } in
      let convert_input id (v : aval) =
        if List.mem v.dtype supported then (id, v.dtype)
        else
          let target = List.hd supported in
          let c = fresh ctx in
          Buffer.add_string ctx.buf
            (Printf.sprintf "    %s = stablehlo.convert %s : (%s) -> %s\n"
               (name c) (name id) (Ir.tensor_type_of_aval v) (retyped v target));
          (c, target)
      in
      let aval_l, aval_r =
        match eqn.inputs with
        | [ l; r ] -> (atom_aval l, atom_aval r)
        | _ -> invalid_arg "Stablehlo.Emit: dot_general arity"
      in
      let a', dt_l = convert_input a aval_l in
      let b', dt_r = convert_input b aval_r in
      let acc_dtype = acc in
      let dot_id =
        if acc_dtype = out.vaval.dtype then bind_var ctx out else fresh ctx
      in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %s = stablehlo.dot_general %s, %s, %s, algorithm = \
            <lhs_precision_type = %s, rhs_precision_type = %s, \
            accumulation_type = %s, lhs_component_count = 1, \
            rhs_component_count = 1, num_primitive_operations = %d, \
            allow_imprecise_accumulation = false> : (%s, %s) -> %s\n"
           (name dot_id) (name a') (name b') (dot_dims_str dd) attr_l attr_r
           attr_acc ops (retyped aval_l dt_l) (retyped aval_r dt_r)
           (retyped out.vaval acc_dtype));
      if acc_dtype <> out.vaval.dtype then
        let n = bind_var ctx out in
        Buffer.add_string ctx.buf
          (Printf.sprintf "    %s = stablehlo.convert %s : (%s) -> %s\n"
             (name n) (name dot_id)
             (retyped out.vaval acc_dtype)
             (Ir.tensor_type_of_aval out.vaval))

let conv_spec_str spec l0 l1 =
  let ndim = Array.length spec in
  let role = Array.make ndim "" in
  role.(spec.(0)) <- l0;
  role.(spec.(1)) <- l1;
  for k = 2 to ndim - 1 do
    role.(spec.(k)) <- string_of_int (k - 2)
  done;
  "[" ^ String.concat ", " (Array.to_list role) ^ "]"

let pad_pairs_str padding =
  "["
  ^ (Array.to_list padding
    |> List.map (fun (l, h) -> Printf.sprintf "[%d, %d]" l h)
    |> String.concat ", ")
  ^ "]"

let emit_conv ctx (eqn : eqn) in_ids ~window_strides ~padding ~lhs_dilation
    ~rhs_dilation ~(dn : conv_dims) ~feature_group_count ~batch_group_count =
  let a, b = pair in_ids in
  let la, lb =
    match eqn.inputs with
    | [ l; r ] ->
        ( Ir.tensor_type_of_aval (atom_aval l),
          Ir.tensor_type_of_aval (atom_aval r) )
    | _ -> invalid_arg "Stablehlo.Emit: convolution arity"
  in
  let out = sole eqn.outs in
  let n_spatial = Array.length window_strides in
  let reverse =
    "[" ^ String.concat ", " (List.init n_spatial (fun _ -> "false")) ^ "]"
  in
  let dim_numbers =
    Printf.sprintf "%sx%s->%s"
      (conv_spec_str dn.lhs_spec "b" "f")
      (conv_spec_str dn.rhs_spec "o" "i")
      (conv_spec_str dn.out_spec "b" "f")
  in
  let n = bind_var ctx out in
  let level = Option.value (Config.matmul_level ()) ~default:"DEFAULT" in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.convolution(%s, %s) dim_numbers = %s, window = \
        {stride = %s, pad = %s, lhs_dilate = %s, rhs_dilate = %s, reverse = \
        %s} {batch_group_count = %d : i64, feature_group_count = %d : i64, \
        precision_config = [#stablehlo<precision %s>, #stablehlo<precision \
        %s>]} : (%s, %s) -> %s\n"
       (name n) (name a) (name b) dim_numbers
       (Ir.int_array_attr window_strides)
       (pad_pairs_str padding)
       (Ir.int_array_attr lhs_dilation)
       (Ir.int_array_attr rhs_dilation)
       reverse batch_group_count feature_group_count level level la lb
       (Ir.tensor_type_of_aval out.vaval))

let pad_dense_str padding =
  let n = Array.length padding in
  let flat =
    Array.to_list padding |> List.concat_map (fun (l, h) -> [ l; h ])
  in
  match flat with
  | first :: rest when List.for_all (Int.equal first) rest ->
      Printf.sprintf "dense<%d> : tensor<%dx2xi64>" first n
  | _ -> Printf.sprintf "dense<%s> : tensor<%dx2xi64>" (pad_pairs_str padding) n

let reduce_window_attrs (w : window_dims) =
  Printf.sprintf
    "base_dilations = %s, padding = %s, window_dilations = %s, \
     window_dimensions = %s, window_strides = %s"
    (array_i64 w.base_dilation)
    (pad_dense_str w.w_padding)
    (array_i64 w.window_dilation)
    (array_i64 w.window_dimensions)
    (array_i64 w.window_strides)

let emit_reduce_window_op ctx (eqn : eqn) in_ids ~window ~op ~init =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let dt = in_aval.dtype in
  let sty = Ir.tensor_type dt [||] in
  let out = sole eqn.outs in
  let c = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c) (init dt)
       sty);
  let b = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.broadcast_in_dim %s, dims = [] : (%s) -> %s\n"
       (name b) (name c) sty sty);
  let n = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = \"stablehlo.reduce_window\"(%s, %s) <{%s}> ({\n"
       (name n) (name x) (name b)
       (reduce_window_attrs window));
  let a1 = fresh ctx in
  let a2 = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    ^bb0(%s: %s, %s: %s):\n" (name a1) sty (name a2) sty);
  let r = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "      %s = stablehlo.%s %s, %s : %s\n" (name r) op
       (name a1) (name a2) sty);
  Buffer.add_string ctx.buf
    (Printf.sprintf "      stablehlo.return %s : %s\n" (name r) sty);
  Buffer.add_string ctx.buf
    (Printf.sprintf "    }) : (%s, %s) -> %s\n"
       (Ir.tensor_type_of_aval in_aval)
       sty
       (Ir.tensor_type_of_aval out.vaval));
  Hashtbl.replace ctx.ids out.vid n

let window_select_dir = function Wge -> "GE" | Wle -> "LE"
let window_select_init = function Wge -> reduce_maxid | Wle -> reduce_minid

let emit_select_and_gather_add ctx (eqn : eqn) in_ids ~select ~window =
  let t, o = pair in_ids in
  let t_av, o_av =
    match eqn.inputs with
    | [ ta; oa ] -> (atom_aval ta, atom_aval oa)
    | _ -> invalid_arg "Stablehlo.Emit: select_and_gather_add arity"
  in
  let dt = o_av.dtype in
  let sty = Ir.tensor_type dt [||] in
  let bty = Ir.tensor_type Dtype.Bool [||] in
  let out = sole eqn.outs in
  let outty = Ir.tensor_type_of_aval out.vaval in
  let dir = window_select_dir select in
  let c_op = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c_op)
       (window_select_init select dt)
       sty);
  let c_tan = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c_tan)
       (reduce_zero dt) sty);
  let grp = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s:2 = \"stablehlo.reduce_window\"(%s, %s, %s, %s) <{%s}> ({\n"
       (name grp) (name o) (name t) (name c_op) (name c_tan)
       (reduce_window_attrs window));
  let kx = fresh ctx in
  let vx = fresh ctx in
  let ky = fresh ctx in
  let vy = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    ^bb0(%s: %s, %s: %s, %s: %s, %s: %s):\n" (name kx) sty
       (name vx) sty (name ky) sty (name vy) sty);
  let cmp = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "      %s = stablehlo.compare %s, %s, %s, FLOAT : (%s, %s) -> %s\n"
       (name cmp) dir (name kx) (name ky) sty sty bty);
  let s1 = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "      %s = stablehlo.select %s, %s, %s : %s, %s\n"
       (name s1) (name cmp) (name kx) (name ky) bty sty);
  let s2 = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "      %s = stablehlo.select %s, %s, %s : %s, %s\n"
       (name s2) (name cmp) (name vx) (name vy) bty sty);
  Buffer.add_string ctx.buf
    (Printf.sprintf "      stablehlo.return %s, %s : %s, %s\n" (name s1)
       (name s2) sty sty);
  Buffer.add_string ctx.buf
    (Printf.sprintf "    }) : (%s, %s, %s, %s) -> (%s, %s)\n"
       (Ir.tensor_type_of_aval o_av)
       (Ir.tensor_type_of_aval t_av)
       sty sty outty outty);
  Hashtbl.replace ctx.names out.vid (Printf.sprintf "%%%d#1" grp)

let emit_select_and_scatter_add ctx (eqn : eqn) in_ids ~select ~window =
  let s, o = pair in_ids in
  let s_av, o_av =
    match eqn.inputs with
    | [ sa; oa ] -> (atom_aval sa, atom_aval oa)
    | _ -> invalid_arg "Stablehlo.Emit: select_and_scatter_add arity"
  in
  let dt = o_av.dtype in
  let sty = Ir.tensor_type dt [||] in
  let bty = Ir.tensor_type Dtype.Bool [||] in
  let out = sole eqn.outs in
  let dir = window_select_dir select in
  let low = Array.map fst window.w_padding in
  let high = Array.map snd window.w_padding in
  let interior = Array.map (fun _ -> 0) window.w_padding in
  let padded_shape =
    Array.mapi (fun d s -> s + low.(d) + high.(d)) o_av.shape
  in
  let padty = Ir.tensor_type dt padded_shape in
  let c_pad = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c_pad)
       (window_select_init select dt)
       sty);
  let padded = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.pad %s, %s, low = %s, high = %s, interior = %s : \
        (%s, %s) -> %s\n"
       (name padded) (name o) (name c_pad) (Ir.int_array_attr low)
       (Ir.int_array_attr high)
       (Ir.int_array_attr interior)
       (Ir.tensor_type_of_aval o_av)
       sty padty);
  let c_init = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c_init)
       (reduce_zero dt) sty);
  let zero_pad = Array.map (fun _ -> (0, 0)) window.w_padding in
  let sas = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = \"stablehlo.select_and_scatter\"(%s, %s, %s) <{padding = %s, \
        window_dimensions = %s, window_strides = %s}> ({\n"
       (name sas) (name padded) (name s) (name c_init) (pad_dense_str zero_pad)
       (array_i64 window.window_dimensions)
       (array_i64 window.window_strides));
  let a1 = fresh ctx in
  let a2 = fresh ctx in
  let r = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    ^bb0(%s: %s, %s: %s):\n" (name a1) sty (name a2) sty);
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "      %s = stablehlo.compare %s, %s, %s, FLOAT : (%s, %s) -> %s\n"
       (name r) dir (name a1) (name a2) sty sty bty);
  Buffer.add_string ctx.buf
    (Printf.sprintf "      stablehlo.return %s : %s\n" (name r) bty);
  Buffer.add_string ctx.buf
    (Printf.sprintf "    }, {\n    ^bb0(%s: %s, %s: %s):\n" (name a1) sty
       (name a2) sty);
  Buffer.add_string ctx.buf
    (Printf.sprintf "      %s = stablehlo.add %s, %s : %s\n" (name r) (name a1)
       (name a2) sty);
  Buffer.add_string ctx.buf
    (Printf.sprintf "      stablehlo.return %s : %s\n" (name r) sty);
  Buffer.add_string ctx.buf
    (Printf.sprintf "    }) : (%s, %s, %s) -> %s\n" padty
       (Ir.tensor_type_of_aval s_av)
       sty padty);
  let ranges =
    List.init (Array.length o_av.shape) (fun d ->
        (low.(d), low.(d) + o_av.shape.(d)))
  in
  let sliced =
    emit_slice ctx sas padty ranges (Ir.tensor_type_of_aval out.vaval)
  in
  Hashtbl.replace ctx.ids out.vid sliced

let emit_sort_canonical_key ctx dt operand =
  let sty = Ir.tensor_type dt [||] in
  let bty = Ir.tensor_type Dtype.Bool [||] in
  let c0 = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "      %s = stablehlo.constant %s : %s\n" (name c0)
       (Ir.dense (Ir.float_literal dt 0.0))
       sty);
  let iszero = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "      %s = stablehlo.compare EQ, %s, %s, FLOAT : (%s, %s) -> %s\n"
       (name iszero) (name operand) (name c0) sty sty bty);
  let c0b = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "      %s = stablehlo.constant %s : %s\n" (name c0b)
       (Ir.dense (Ir.float_literal dt 0.0))
       sty);
  let selz = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "      %s = stablehlo.select %s, %s, %s : %s, %s\n"
       (name selz) (name iszero) (name c0b) (name operand) bty sty);
  let isnan = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "      %s = stablehlo.compare NE, %s, %s, FLOAT : (%s, %s) -> %s\n"
       (name isnan) (name operand) (name operand) sty sty bty);
  let cnan = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "      %s = stablehlo.constant %s : %s\n" (name cnan)
       (Ir.dense (Ir.float_literal dt Float.nan))
       sty);
  let cres = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "      %s = stablehlo.select %s, %s, %s : %s, %s\n"
       (name cres) (name isnan) (name cnan) (name selz) bty sty);
  cres

let emit_sort_multi ctx (eqn : eqn) in_ids ~dimension ~is_stable ~num_keys =
  let inputs = List.map atom_aval eqn.inputs in
  let bty = Ir.tensor_type Dtype.Bool [||] in
  let group = fresh ctx in
  let single = List.length inputs = 1 in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = \"stablehlo.sort\"(%s) <{dimension = %d : i64, is_stable = \
        %s}> ({\n"
       (if single then name group
        else Printf.sprintf "%%%d:%d" group (List.length inputs))
       (String.concat ", " (List.map name in_ids))
       dimension
       (if is_stable then "true" else "false"));
  let args =
    List.concat_map
      (fun (a : aval) ->
        let sty = Ir.tensor_type a.dtype [||] in
        let x = fresh ctx in
        let y = fresh ctx in
        [ (x, sty); (y, sty) ])
      inputs
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    ^bb0(%s):\n"
       (String.concat ", "
          (List.map (fun (id, ty) -> name id ^ ": " ^ ty) args)));
  let compare_op kind x y sty cmp =
    let p = fresh ctx in
    Buffer.add_string ctx.buf
      (Printf.sprintf
         "      %s = stablehlo.compare %s, %s, %s, %s : (%s, %s) -> %s\n"
         (name p) kind (name x) (name y) cmp sty sty bty);
    p
  in
  let binary op l r =
    let p = fresh ctx in
    Buffer.add_string ctx.buf
      (Printf.sprintf "      %s = stablehlo.%s %s, %s : %s\n" (name p) op
         (name l) (name r) bty);
    p
  in
  let part op dt v =
    let part_dt =
      match dt with Dtype.Complex128 -> Dtype.F64 | _ -> Dtype.F32
    in
    let n = fresh ctx in
    Buffer.add_string ctx.buf
      (Printf.sprintf "      %s = stablehlo.%s %s : (%s) -> %s\n" (name n) op
         (name v) (Ir.tensor_type dt [||])
         (Ir.tensor_type part_dt [||]));
    (part_dt, n)
  in
  let keys_of dt v =
    match dt with
    | Dtype.Complex64 | Dtype.Complex128 ->
        let rdt, re = part "real" dt v in
        let re = emit_sort_canonical_key ctx rdt re in
        let idt, im = part "imag" dt v in
        let im = emit_sort_canonical_key ctx idt im in
        [ (rdt, re); (idt, im) ]
    | Dtype.F16 | Dtype.Bf16 | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4
    | Dtype.F8_e4m3 | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz
    | Dtype.F8_e5m2 | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu | Dtype.F32
    | Dtype.F64 ->
        [ (dt, emit_sort_canonical_key ctx dt v) ]
    | _ -> [ (dt, v) ]
  in
  let key_operands = List.filteri (fun i _ -> i < num_keys) inputs in
  let pairs =
    List.concat
      (List.mapi
         (fun k (key : aval) ->
           let x, _ = List.nth args (2 * k)
           and y, _ = List.nth args ((2 * k) + 1) in
           let xk = keys_of key.dtype x in
           let yk = keys_of key.dtype y in
           List.map2 (fun (dt, xv) (_, yv) -> (dt, xv, yv)) xk yk)
         key_operands)
  in
  let comparison dt =
    match dt with
    | Dtype.F4_e2m1fn -> "FLOAT"
    | Dtype.F16 | Dtype.Bf16 | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn | Dtype.F8_e3m4
    | Dtype.F8_e4m3 | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz
    | Dtype.F8_e5m2 | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu | Dtype.F32
    | Dtype.F64 ->
        "TOTALORDER"
    | dt -> compare_type_of dt false
  in
  let result =
    List.fold_left
      (fun acc (dt, x, y) ->
        let sty = Ir.tensor_type dt [||] in
        let cmp = comparison dt in
        let lt = compare_op "LT" x y sty cmp in
        match acc with
        | None -> Some lt
        | Some p ->
            let eq = compare_op "EQ" x y sty cmp in
            Some (binary "or" lt (binary "and" eq p)))
      None (List.rev pairs)
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "      stablehlo.return %s : %s\n"
       (name (Option.get result))
       bty);
  let in_types = String.concat ", " (List.map Ir.tensor_type_of_aval inputs) in
  let out_types =
    String.concat ", "
      (List.map (fun (v : var) -> Ir.tensor_type_of_aval v.vaval) eqn.outs)
  in
  Buffer.add_string ctx.buf
    (if single then Printf.sprintf "    }) : (%s) -> %s\n" in_types out_types
     else Printf.sprintf "    }) : (%s) -> (%s)\n" in_types out_types);
  if single then Hashtbl.replace ctx.ids (sole eqn.outs).vid group
  else
    List.iteri
      (fun i (v : var) ->
        Hashtbl.replace ctx.ids v.vid
          (alias_result (Printf.sprintf "%%%d#%d" group i)))
      eqn.outs

let emit_sort ctx (eqn : eqn) in_ids ~dimension ~is_stable =
  emit_sort_multi ctx eqn in_ids ~dimension ~is_stable ~num_keys:1

let emit_top_k ctx (eqn : eqn) in_ids k =
  let x = sole in_ids in
  let inty = Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)) in
  let vout, iout =
    match eqn.outs with
    | [ v; i ] -> (v, i)
    | _ -> invalid_arg "Stablehlo.Emit: top_k expects two outputs"
  in
  let r1 = bind_var ctx vout in
  let r2 = bind_var ctx iout in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s, %s = chlo.top_k(%s, k = %d) : %s -> (%s, %s)\n"
       (name r1) (name r2) (name x) k inty
       (Ir.tensor_type_of_aval vout.vaval)
       (Ir.tensor_type_of_aval iout.vaval))

let shape_i64_body dims =
  let n = Array.length dims in
  if n > 0 && Array.for_all (fun d -> d = dims.(0)) dims then
    Ir.int_literal (Int64.of_int dims.(0))
  else
    "["
    ^ String.concat ", "
        (Array.to_list
           (Array.map (fun d -> Ir.int_literal (Int64.of_int d)) dims))
    ^ "]"

let emit_rng_uniform ctx (eqn : eqn) in_ids =
  let a, b = pair in_ids in
  let la, ra =
    match eqn.inputs with
    | [ l; r ] -> (l, r)
    | _ -> invalid_arg "Stablehlo.Emit: rng_uniform expects two operands"
  in
  let aty = Ir.tensor_type_of_aval (atom_aval la) in
  let bty = Ir.tensor_type_of_aval (atom_aval ra) in
  let out = sole eqn.outs in
  let dims = out.vaval.shape in
  let shty = Ir.tensor_type Dtype.I64 [| Array.length dims |] in
  let c = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.constant %s : %s\n" (name c)
       (Ir.dense (shape_i64_body dims))
       shty);
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.rng %s, %s, %s, distribution = UNIFORM : (%s, %s, \
        %s) -> %s\n"
       (name n) (name a) (name b) (name c) aty bty shty
       (Ir.tensor_type_of_aval out.vaval))

let emit_rng_bit_generator ctx (eqn : eqn) in_ids algorithm =
  let key = atom_aval (sole eqn.inputs) in
  let key_id = sole in_ids in
  let out_key, out_vals =
    match eqn.outs with
    | [ a; b ] -> (a, b)
    | _ -> invalid_arg "Stablehlo.Emit: rng_bit_generator arity"
  in
  let wide = Ir.tensor_type Dtype.Uint64 [| 2 |] in
  let pair = Ir.tensor_type Dtype.Uint32 [| 2; 2 |] in
  let key_ty = Ir.tensor_type_of_aval key in
  let narrow = key.dtype = Dtype.Uint32 in
  let state =
    if narrow then begin
      let r = fresh ctx in
      Buffer.add_string ctx.buf
        (Printf.sprintf "    %s = stablehlo.reshape %s : (%s) -> %s\n" (name r)
           (name key_id) key_ty pair);
      let k = fresh ctx in
      Buffer.add_string ctx.buf
        (Printf.sprintf "    %s = stablehlo.bitcast_convert %s : (%s) -> %s\n"
           (name k) (name r) pair wide);
      k
    end
    else key_id
  in
  let next = fresh ctx in
  let vals = bind_var ctx out_vals in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s, %s = stablehlo.rng_bit_generator %s, algorithm = %s : (%s) -> \
        (%s, %s)\n"
       (name next) (name vals) (name state) algorithm wide wide
       (Ir.tensor_type_of_aval out_vals.vaval));
  if narrow then begin
    let b = fresh ctx in
    Buffer.add_string ctx.buf
      (Printf.sprintf "    %s = stablehlo.bitcast_convert %s : (%s) -> %s\n"
         (name b) (name next) wide pair);
    let ok = bind_var ctx out_key in
    Buffer.add_string ctx.buf
      (Printf.sprintf "    %s = stablehlo.reshape %s : (%s) -> %s\n" (name ok)
         (name b) pair key_ty)
  end
  else Hashtbl.replace ctx.ids out_key.vid next

let mlir_string s =
  let b = Buffer.create (String.length s + 2) in
  Buffer.add_char b '"';
  String.iter
    (fun c ->
      match c with
      | '"' -> Buffer.add_string b "\\\""
      | '\\' -> Buffer.add_string b "\\\\"
      | c when Char.code c < 32 || Char.code c >= 127 ->
          Buffer.add_string b (Printf.sprintf "\\%02x" (Char.code c))
      | c -> Buffer.add_char b c)
    s;
  Buffer.add_char b '"';
  Buffer.contents b

let ffi_attr_key k =
  let ident =
    k <> ""
    && String.for_all
         (fun c ->
           (c >= 'a' && c <= 'z')
           || (c >= 'A' && c <= 'Z')
           || (c >= '0' && c <= '9')
           || c = '_')
         k
    && not (k.[0] >= '0' && k.[0] <= '9')
  in
  if ident then k else mlir_string k

let rec ffi_attr_text = function
  | Ffi_bool b -> Ir.bool_literal b
  | Ffi_int n -> Printf.sprintf "%Ld : i64" n
  | Ffi_float f -> Ir.float_literal Dtype.F64 f ^ " : f64"
  | Ffi_string s -> mlir_string s
  | Ffi_dtype d -> Ir.element_type d
  | Ffi_seq l -> "[" ^ String.concat ", " (List.map ffi_attr_text l) ^ "]"
  | Ffi_dict kvs -> ffi_attr_dict kvs
  | Ffi_array nd ->
      let dt = Ndarray.dtype nd and shape = Ndarray.shape nd in
      if Array.length shape = 0 then
        let lit = element_literal nd [||] in
        if dt = Dtype.Bool then lit
        else Printf.sprintf "%s : %s" lit (Ir.element_type dt)
      else
        Printf.sprintf "%s : %s"
          (Ir.dense (dense_body nd))
          (Ir.tensor_type dt shape)

and ffi_attr_dict kvs =
  let sorted = List.sort (fun (a, _) (b, _) -> String.compare a b) kvs in
  "{"
  ^ String.concat ", "
      (List.map
         (fun (k, v) ->
           Printf.sprintf "%s = %s" (ffi_attr_key k) (ffi_attr_text v))
         sorted)
  ^ "}"

let ffi_layout_text layout =
  let n = Array.length layout in
  let body =
    if n = 0 then ""
    else
      "["
      ^ String.concat ", " (Array.to_list (Array.map string_of_int layout))
      ^ "]"
  in
  Printf.sprintf "dense<%s> : tensor<%dxindex>" body n

let lapack_ffi_ready : (unit -> bool) ref = ref (fun () -> false)
let lapack_ffi () = Config.current_platform () = "cpu" && !lapack_ffi_ready ()

let lapack_prec dt =
  match dt with
  | Dtype.F32 -> 's'
  | Dtype.F64 -> 'd'
  | Dtype.Complex64 -> 'c'
  | Dtype.Complex128 -> 'z'
  | _ -> failwith "Stablehlo.Emit: LAPACK has no routine for this dtype"

let lapack_layout rank ~batch =
  if rank = batch + 2 then
    Array.append
      [| rank - 2; rank - 1 |]
      (Array.init (rank - 2) (fun i -> rank - 3 - i))
  else Array.init rank (fun i -> rank - 1 - i)

let sdy_factor_name i =
  if i < 17 then String.make 1 (Char.chr (Char.code 'i' + i))
  else Printf.sprintf "z_%d" (i - 17)

let lapack_aval shape dtype = { shape; dtype; weak_type = false }

let emit_lapack_call ?target ctx ~base ~config ~aliases in_ids in_avals
    out_avals =
  let first = List.hd in_avals in
  let target =
    match target with
    | Some name -> name
    | None -> Printf.sprintf "lapack_%c%s_ffi" (lapack_prec first.dtype) base
  in
  let max_rank =
    List.fold_left
      (fun acc (a : aval) -> max acc (Array.length a.shape))
      0 in_avals
  in
  let batch = max_rank - 2 in
  let group = fresh ctx in
  let n = List.length out_avals in
  let layouts avals =
    String.concat ", "
      (List.map
         (fun (a : aval) ->
           match lapack_layout (Array.length a.shape) ~batch with
           | [| only |] -> Printf.sprintf "dense<%d> : tensor<1xindex>" only
           | layout -> ffi_layout_text layout)
         avals)
  in
  let alias_text =
    if aliases = [] then ""
    else
      Printf.sprintf ", output_operand_aliases = [%s]"
        (String.concat ", "
           (List.map
              (fun (input, output) ->
                Printf.sprintf
                  "#stablehlo.output_operand_alias<output_tuple_indices = \
                   [%s], operand_index = %d, operand_tuple_indices = []>"
                  (if n > 1 then string_of_int output else "")
                  input)
              aliases))
  in
  let types avals =
    String.concat ", " (List.map Ir.tensor_type_of_aval avals)
  in
  let sizes = ref [] in
  let next_factor = ref 0 in
  let fresh_factor size =
    let letter = sdy_factor_name !next_factor in
    incr next_factor;
    sizes := (letter, size) :: !sizes;
    letter
  in
  let batch_factors =
    Array.init batch (fun d -> fresh_factor first.shape.(d))
  in
  let factors_of (a : aval) =
    let inner =
      Array.to_list a.shape
      |> List.mapi (fun d size ->
          if d < batch then batch_factors.(d) else fresh_factor size)
    in
    "[" ^ String.concat ", " inner ^ "]"
  in
  let operand_rule = List.map factors_of in_avals in
  let result_rule = List.map factors_of out_avals in
  let sharding_rule =
    Printf.sprintf "#sdy.op_sharding_rule<(%s)->(%s) {%s}, custom>"
      (String.concat ", " operand_rule)
      (String.concat ", " result_rule)
      (String.concat ", "
         (List.rev_map
            (fun (letter, size) -> Printf.sprintf "%s=%d" letter size)
            !sizes))
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %%%d%s = stablehlo.custom_call @%s(%s) {backend_config = \"\", \
        mhlo.backend_config = {%s}, mhlo.frontend_attributes = {num_batch_dims \
        = \"%d\"}, operand_layouts = [%s]%s, result_layouts = [%s], \
        sdy.sharding_rule = %s} : (%s) -> %s\n"
       group
       (if n > 1 then Printf.sprintf ":%d" n else "")
       target
       (String.concat ", " (List.map name in_ids))
       config batch (layouts in_avals) alias_text (layouts out_avals)
       sharding_rule (types in_avals)
       (if n > 1 then "(" ^ types out_avals ^ ")" else types out_avals));
  if n = 1 then [ group ]
  else List.init n (fun i -> alias_result (Printf.sprintf "%%%d#%d" group i))

let emit_ffi_call ctx (eqn : eqn) in_ids (c : ffi_call) =
  let operand_types =
    List.map (fun a -> Ir.tensor_type_of_aval (atom_aval a)) eqn.inputs
  in
  let result_types =
    List.map (fun (v : var) -> Ir.tensor_type_of_aval v.vaval) eqn.outs
  in
  let attrs =
    [
      (if c.ffi_api_version < 4 then
         Some (Printf.sprintf "api_version = %d : i32" c.ffi_api_version)
       else None);
      Some
        (Printf.sprintf "backend_config = %s"
           (mlir_string (Option.value c.ffi_legacy_backend_config ~default:"")));
      (if c.ffi_has_side_effect then Some "has_side_effect = true" else None);
      (if c.ffi_api_version >= 4 then
         Some ("mhlo.backend_config = " ^ ffi_attr_dict c.ffi_attributes)
       else None);
      Some
        (Printf.sprintf "operand_layouts = [%s]"
           (String.concat ", " (List.map ffi_layout_text c.ffi_input_layouts)));
      (if c.ffi_aliases = [] then None
       else
         Some
           (Printf.sprintf "output_operand_aliases = [%s]"
              (String.concat ", "
                 (List.map
                    (fun (input, output) ->
                      Printf.sprintf
                        "#stablehlo.output_operand_alias<output_tuple_indices \
                         = [%s], operand_index = %d, operand_tuple_indices = \
                         []>"
                        (if List.length result_types > 1 then
                           string_of_int output
                         else "")
                        input)
                    c.ffi_aliases))));
      Some
        (Printf.sprintf "result_layouts = [%s]"
           (String.concat ", " (List.map ffi_layout_text c.ffi_output_layouts)));
    ]
  in
  let lhs =
    match List.map (fun v -> name (bind_var ctx v)) eqn.outs with
    | [] -> ""
    | names -> String.concat ", " names ^ " = "
  in
  let results =
    match result_types with [ t ] -> t | l -> "(" ^ String.concat ", " l ^ ")"
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %sstablehlo.custom_call @%s(%s) {%s} : (%s) -> %s\n"
       lhs c.ffi_target
       (String.concat ", " (List.map name in_ids))
       (String.concat ", " (List.filter_map Fun.id attrs))
       (String.concat ", " operand_types)
       results)

type partition = {
  pt_mesh : (string * int) list;
  pt_devices : int array;
  pt_in : string list array list;
  pt_out : string list array option list;
}

let active_mesh : (string * int) list option ref = ref None
let manual_axes : string list ref = ref []
let seen_mesh : ((string * int) list * int array) option ref = ref None
let channel = ref 0

let note_mesh mesh devices =
  match !seen_mesh with
  | None -> seen_mesh := Some (mesh, devices)
  | Some (m, d) when m = mesh && d = devices -> ()
  | Some _ -> failwith "Stablehlo.Emit: a program can only use one device mesh"

let shard_mesh_seen () = !seen_mesh

let sdy_sharding_body spec =
  let dim axes =
    "{" ^ String.concat ", " (List.map (Printf.sprintf "\"%s\"") axes) ^ "}"
  in
  "<@mesh, [" ^ String.concat ", " (Array.to_list (Array.map dim spec)) ^ "]>"

let sdy_sharding spec = "#sdy.sharding" ^ sdy_sharding_body spec

let sdy_mesh mesh =
  Printf.sprintf "  sdy.mesh @mesh = <[%s]> {stablehlo.mesh = {axes = [%s]}}\n"
    (String.concat ", "
       (List.map (fun (n, s) -> Printf.sprintf "\"%s\"=%d" n s) mesh))
    (String.concat ", "
       (List.map
          (fun (n, s) -> Printf.sprintf "{name = \"%s\", size = %d : i64}" n s)
          mesh))

let list_index x l =
  let rec go i = function
    | [] -> invalid_arg "Stablehlo.Emit: unknown mesh axis"
    | y :: _ when y = x -> i
    | _ :: rest -> go (i + 1) rest
  in
  go 0 l

let collective_groups mesh axis =
  let sizes = List.map snd mesh in
  let total = List.fold_left ( * ) 1 sizes in
  let position = list_index axis (List.map fst mesh) in
  let stride =
    List.fold_left ( * ) 1 (List.filteri (fun i _ -> i > position) sizes)
  in
  let size = List.nth sizes position in
  List.filter_map
    (fun p ->
      if p / stride mod size = 0 then
        Some (List.init size (fun k -> p + (k * stride)))
      else None)
    (List.init total Fun.id)

let dense_i64_matrix rows =
  let n = List.length rows in
  let m = match rows with r :: _ -> List.length r | [] -> 0 in
  Printf.sprintf "dense<[%s]> : tensor<%dx%dxi64>"
    (String.concat ", "
       (List.map
          (fun r -> "[" ^ String.concat ", " (List.map string_of_int r) ^ "]")
          rows))
    n m

let callback_seen = ref false
let default_layout rank = Array.init rank (fun i -> rank - 1 - i)

let emit_host_callback ctx (eqn : eqn) in_ids id =
  if (not (lapack_ffi ())) || !active_mesh <> None then
    failwith "Stablehlo.Emit: callbacks run on the host interpreter";
  callback_seen := true;
  let in_avals = List.map atom_aval eqn.inputs in
  let out_avals = List.map (fun (v : var) -> v.vaval) eqn.outs in
  let layouts avals =
    String.concat ", "
      (List.map
         (fun (a : aval) ->
           ffi_layout_text (default_layout (Array.length a.shape)))
         avals)
  in
  let types avals =
    String.concat ", " (List.map Ir.tensor_type_of_aval avals)
  in
  let per_value =
    let n = max 1 (List.length out_avals) in
    String.concat ", " (List.init n (fun _ -> "<@maximal_mesh_0, []>"))
  in
  let results =
    match out_avals with
    | [] -> "()"
    | [ a ] -> Ir.tensor_type_of_aval a
    | avals -> "(" ^ types avals ^ ")"
  in
  let attrs =
    Printf.sprintf
      "{backend_config = \"\", %smhlo.backend_config = {index = %d : ui64}, \
       operand_layouts = [%s], result_layouts = [%s], sdy.sharding = \
       #sdy.sharding_per_value<[%s]>} : (%s) -> %s\n"
      (if out_avals = [] then "has_side_effect = true, " else "")
      id (layouts in_avals) (layouts out_avals) per_value (types in_avals)
      results
  in
  let args = String.concat ", " (List.map name in_ids) in
  match eqn.outs with
  | [] ->
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    stablehlo.custom_call @xla_ffi_python_cpu_callback(%s) %s" args
           attrs)
  | [ only ] ->
      let n = bind_var ctx only in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %s = stablehlo.custom_call @xla_ffi_python_cpu_callback(%s) %s"
           (name n) args attrs)
  | outs ->
      let group = fresh ctx in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %%%d:%d = stablehlo.custom_call \
            @xla_ffi_python_cpu_callback(%s) %s"
           group (List.length outs) args attrs);
      List.iteri
        (fun i (v : var) ->
          Hashtbl.replace ctx.ids v.vid
            (alias_result (Printf.sprintf "%%%d#%d" group i)))
        outs

let next_channel () =
  incr channel;
  Printf.sprintf "#stablehlo.channel_handle<handle = %d, type = 1>" !channel

let collective_mesh axis_name =
  match !active_mesh with
  | Some mesh
    when List.mem axis_name !manual_axes && List.mem_assoc axis_name mesh ->
      mesh
  | _ ->
      failwith "Stablehlo.Emit: collectives must be bound by an enclosing vmap"

let emit_all_reduce ctx (eqn : eqn) in_ids op mesh axis_name =
  let in_aval = atom_aval (List.hd eqn.inputs) in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  let scalar = Ir.tensor_type in_aval.dtype [||] in
  let a = fresh ctx and b = fresh ctx and c = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = \"stablehlo.all_reduce\"(%s) <{channel_handle = %s, \
        replica_groups = %s, use_global_device_ids}> ({\n\
       \    ^bb0(%s: %s, %s: %s):\n\
       \      %s = stablehlo.%s %s, %s : %s\n\
       \      stablehlo.return %s : %s\n\
       \    }) : (%s) -> %s\n"
       (name n)
       (name (sole in_ids))
       (next_channel ())
       (dense_i64_matrix (collective_groups mesh axis_name))
       (name a) scalar (name b) scalar (name c) op (name a) (name b) scalar
       (name c) scalar
       (Ir.tensor_type_of_aval in_aval)
       (Ir.tensor_type_of_aval in_aval))

let emit_all_gather ctx in_id (in_aval : aval) axis size mesh axis_name =
  let rank = Array.length in_aval.shape in
  let expanded_shape =
    Array.init (rank + 1) (fun i ->
        if i < axis then in_aval.shape.(i)
        else if i = axis then 1
        else in_aval.shape.(i - 1))
  in
  let gathered_shape =
    Array.mapi (fun i d -> if i = axis then size else d) expanded_shape
  in
  let expanded = fresh ctx in
  let gathered = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.reshape %s : (%s) -> %s\n\
       \    %s = \"stablehlo.all_gather\"(%s) <{all_gather_dim = %d : i64, \
        channel_handle = %s, replica_groups = %s, use_global_device_ids}> : \
        (%s) -> %s\n"
       (name expanded) (name in_id)
       (Ir.tensor_type_of_aval in_aval)
       (Ir.tensor_type in_aval.dtype expanded_shape)
       (name gathered) (name expanded) axis (next_channel ())
       (dense_i64_matrix (collective_groups mesh axis_name))
       (Ir.tensor_type in_aval.dtype expanded_shape)
       (Ir.tensor_type in_aval.dtype gathered_shape));
  (gathered, gathered_shape)

let emit_pbroadcast ctx (eqn : eqn) in_ids source size mesh axis_name =
  let in_aval = atom_aval (List.hd eqn.inputs) in
  let gathered, shape =
    emit_all_gather ctx (sole in_ids) in_aval 0 size mesh axis_name
  in
  let ranges =
    String.concat ", "
      (Array.to_list
         (Array.mapi
            (fun i d ->
              if i = 0 then Printf.sprintf "%d:%d" source (source + 1)
              else Printf.sprintf "0:%d" d)
            shape))
  in
  let sliced = fresh ctx in
  let piece = Array.mapi (fun i d -> if i = 0 then 1 else d) shape in
  let n = bind_var ctx (sole eqn.outs) in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.slice %s [%s] : (%s) -> %s\n\
       \    %s = stablehlo.reshape %s : (%s) -> %s\n"
       (name sliced) (name gathered) ranges
       (Ir.tensor_type in_aval.dtype shape)
       (Ir.tensor_type in_aval.dtype piece)
       (name n) (name sliced)
       (Ir.tensor_type in_aval.dtype piece)
       (Ir.tensor_type_of_aval in_aval))

let emit_ppermute ctx (eqn : eqn) in_ids pairs mesh axis_name =
  let in_aval = atom_aval (List.hd eqn.inputs) in
  let n = bind_var ctx (sole eqn.outs) in
  let rows =
    List.concat_map
      (fun group ->
        let group = Array.of_list group in
        List.map
          (fun (src, dst) -> [ group.(src); group.(dst) ])
          (Array.to_list pairs))
      (collective_groups mesh axis_name)
  in
  let ty = Ir.tensor_type_of_aval in_aval in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = \"stablehlo.collective_permute\"(%s) <{channel_handle = %s, \
        source_target_pairs = %s}> : (%s) -> %s\n"
       (name n)
       (name (sole in_ids))
       (next_channel ()) (dense_i64_matrix rows) ty ty)

let emit_axis_index ctx (eqn : eqn) mesh axis_name =
  let sizes = List.map snd mesh in
  let position = list_index axis_name (List.map fst mesh) in
  let stride =
    List.fold_left ( * ) 1 (List.filteri (fun i _ -> i > position) sizes)
  in
  let size = List.nth sizes position in
  let n = bind_var ctx (sole eqn.outs) in
  let pid = fresh ctx
  and as_int = fresh ctx
  and stride_c = fresh ctx
  and size_c = fresh ctx
  and div = fresh ctx in
  let i32 = Ir.tensor_type Dtype.I32 [||] in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.partition_id : tensor<ui32>\n\
       \    %s = stablehlo.convert %s : (tensor<ui32>) -> %s\n\
       \    %s = stablehlo.constant dense<%d> : %s\n\
       \    %s = stablehlo.constant dense<%d> : %s\n\
       \    %s = stablehlo.divide %s, %s : %s\n\
       \    %s = stablehlo.remainder %s, %s : %s\n"
       (name pid) (name as_int) (name pid) i32 (name stride_c) stride i32
       (name size_c) size i32 (name div) (name as_int) (name stride_c) i32
       (name n) (name div) (name size_c) i32)

let rec emit_region_returning ctx (jx : jaxpr) =
  let s =
    capture ctx (fun () ->
        List.iter
          (fun (e : eqn) ->
            let ids = List.map (id_of_atom ctx) e.inputs in
            emit_eqn ctx e ids)
          jx.eqns;
        let outs = List.map (ssa_of_atom ctx) jx.outs in
        let otys =
          List.map (fun a -> Ir.tensor_type_of_aval (atom_aval a)) jx.outs
        in
        Buffer.add_string ctx.buf
          (Printf.sprintf "    stablehlo.return %s : %s\n"
             (String.concat ", " outs) (String.concat ", " otys)))
  in
  Buffer.add_string ctx.buf (reindent 2 s)

and map_binders ctx (jx : jaxpr) target_ids =
  List.iter2
    (fun (v : var) tid -> Hashtbl.replace ctx.ids v.vid tid)
    jx.in_binders target_ids

and bind_results ctx (outs : var list) =
  match outs with
  | [ out ] ->
      let n = bind_var ctx out in
      name n
  | _ ->
      let group = fresh ctx in
      List.iteri
        (fun i (v : var) ->
          Hashtbl.replace ctx.ids v.vid
            (alias_result (Printf.sprintf "%%%d#%d" group i)))
        outs;
      Printf.sprintf "%%%d:%d" group (List.length outs)

and enter_region ctx (cj : closed_jaxpr) target_ids =
  let n_consts = List.length cj.consts in
  let const_binders, arg_binders =
    match Util.split_list cj.jaxpr.in_binders [ n_consts ] with
    | [ cb; ab ] -> (cb, ab)
    | _ -> invalid_arg "Stablehlo.Emit: region binder split"
  in
  List.iter2
    (fun (v : var) tid -> Hashtbl.replace ctx.ids v.vid tid)
    arg_binders target_ids;
  List.iter2
    (fun (v : var) nd ->
      let n = bind_var ctx v in
      let text = capture ctx (fun () -> emit_constant_at ctx n nd) in
      Buffer.add_string ctx.buf (reindent 2 text))
    const_binders cj.consts

and emit_cond ctx (eqn : eqn) in_ids ~platforms branches =
  let pred_id, operand_ids =
    match in_ids with
    | i :: r -> (i, r)
    | [] -> invalid_arg "Stablehlo.Emit: cond arity"
  in
  let pred_aval = atom_aval (List.hd eqn.inputs) in
  let index_id =
    match pred_aval.dtype with
    | Dtype.Bool ->
        let c = fresh ctx in
        Buffer.add_string ctx.buf
          (Printf.sprintf "    %s = stablehlo.convert %s : (%s) -> %s\n"
             (name c) (name pred_id)
             (Ir.tensor_type_of_aval pred_aval)
             (Ir.tensor_type Dtype.I32 [||]));
        c
    | _ -> pred_id
  in
  let index_id =
    match platforms with
    | None -> index_id
    | Some _ ->
        let i32 = Ir.tensor_type Dtype.I32 [||] in
        let n = fresh ctx in
        let start = ctx.next in
        let k = fresh ctx in
        let region =
          Printf.sprintf
            "      %s = stablehlo.constant %s : %s\n\
            \      stablehlo.return %s : %s\n"
            (name k)
            (Ir.dense (Ir.int_literal 0L))
            i32 (name k) i32
        in
        Buffer.add_string ctx.buf
          (Printf.sprintf "    %s = \"stablehlo.case\"(%s) ({\n%s" (name n)
             (name index_id) region);
        List.iteri
          (fun i _ ->
            if i > 0 then
              Buffer.add_string ctx.buf (Printf.sprintf "    }, {\n%s" region))
          branches;
        Buffer.add_string ctx.buf
          (Printf.sprintf "    }) : (%s) -> %s\n" i32 i32);
        ctx.next <- start;
        n
  in
  let result = bind_results ctx eqn.outs in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = \"stablehlo.case\"(%s) ({\n" result
       (name index_id));
  let start = ctx.next in
  let emit_branch (cj : closed_jaxpr) =
    ctx.next <- start;
    enter_region ctx cj operand_ids;
    emit_region_returning ctx cj.jaxpr
  in
  (match platforms with
  | None ->
      List.iteri
        (fun i b ->
          if i > 0 then Buffer.add_string ctx.buf "    }, {\n";
          emit_branch b)
        branches
  | Some _ -> emit_branch (List.nth branches (List.length branches - 1)));
  let out_types =
    List.map (fun (v : var) -> Ir.tensor_type_of_aval v.vaval) eqn.outs
  in
  let out_type =
    match out_types with [ t ] -> t | ts -> "(" ^ String.concat ", " ts ^ ")"
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    }) : (%s) -> %s\n"
       (Ir.tensor_type Dtype.I32 [||])
       out_type)

and emit_reduce_general ctx (eqn : eqn) in_ids (reducer : closed_jaxpr) dims =
  let op_id, init_id = pair in_ids in
  let in_aval = atom_aval (List.hd eqn.inputs) in
  let dt = in_aval.dtype in
  let sty = Ir.tensor_type dt [||] in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  let single_op =
    match
      (reducer.jaxpr.in_binders, reducer.jaxpr.eqns, reducer.jaxpr.outs)
    with
    | ( [ a; b ],
        [ { prim; inputs = [ A_var x; A_var y ]; outs = [ o ]; _ } ],
        [ A_var r ] )
      when x.vid = a.vid && y.vid = b.vid && r.vid = o.vid -> (
        match prim with
        | Add -> Some "add"
        | Mul -> Some "multiply"
        | And -> Some "and"
        | Or -> Some "or"
        | Xor -> Some "xor"
        | _ -> None)
    | _ -> None
  in
  match single_op with
  | Some op ->
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %s = stablehlo.reduce(%s init: %s) applies stablehlo.%s across \
            dimensions = %s : (%s, %s) -> %s\n"
           (name n) (name op_id) (name init_id) op (Ir.int_array_attr dims)
           (Ir.tensor_type_of_aval in_aval)
           sty
           (Ir.tensor_type_of_aval out.vaval))
  | None ->
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %s = stablehlo.reduce(%s init: %s) across dimensions = %s : \
            (%s, %s) -> %s\n"
           (name n) (name op_id) (name init_id) (Ir.int_array_attr dims)
           (Ir.tensor_type_of_aval in_aval)
           sty
           (Ir.tensor_type_of_aval out.vaval));
      let ba, bb =
        match reducer.jaxpr.in_binders with
        | [ a; b ] ->
            let ia = bind_var ctx a in
            let ib = bind_var ctx b in
            (ia, ib)
        | _ -> invalid_arg "Stablehlo.Emit: reduce reducer arity"
      in
      Buffer.add_string ctx.buf
        (Printf.sprintf "     reducer(%s: %s, %s: %s)  {\n" (name ba) sty
           (name bb) sty);
      emit_region_returning ctx reducer.jaxpr;
      Buffer.add_string ctx.buf "    }\n"

and emit_reduce_window_general ctx (eqn : eqn) in_ids (reducer : closed_jaxpr)
    window =
  let op_id, init_id = pair in_ids in
  let in_aval = atom_aval (List.hd eqn.inputs) in
  let dt = in_aval.dtype in
  let sty = Ir.tensor_type dt [||] in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = \"stablehlo.reduce_window\"(%s, %s) <{%s}> ({\n"
       (name n) (name op_id) (name init_id)
       (reduce_window_attrs window));
  let ba, bb =
    match reducer.jaxpr.in_binders with
    | [ a; b ] ->
        let ia = bind_var ctx a in
        let ib = bind_var ctx b in
        (ia, ib)
    | _ -> invalid_arg "Stablehlo.Emit: reduce_window reducer arity"
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    ^bb0(%s: %s, %s: %s):\n" (name ba) sty (name bb) sty);
  emit_region_returning ctx reducer.jaxpr;
  Buffer.add_string ctx.buf
    (Printf.sprintf "    }) : (%s, %s) -> %s\n"
       (Ir.tensor_type_of_aval in_aval)
       sty
       (Ir.tensor_type_of_aval out.vaval))

and emit_while ctx (eqn : eqn) in_ids ~cond_nconsts ~body_nconsts
    (cond : closed_jaxpr) (body : closed_jaxpr) =
  let pred_rank =
    match cond.jaxpr.outs with
    | [ A_var v ] -> Array.length v.vaval.shape
    | _ -> 0
  in
  if pred_rank > 0 then begin
    let n_consts = cond_nconsts + body_nconsts in
    let in_avals = List.map atom_aval eqn.inputs in
    let cond', body' =
      Jaxpr.lift_while ~cond_nconsts ~body_nconsts cond body in_avals
    in
    let const_avals = List.filteri (fun i _ -> i < n_consts) in_avals in
    let const_outs =
      List.map (fun a -> { vid = Core.fresh_id (); vaval = a }) const_avals
    in
    emit_while_plain ctx
      { eqn with outs = const_outs @ eqn.outs }
      in_ids ~cond_nconsts:0 ~body_nconsts:0 cond' body'
  end
  else emit_while_plain ctx eqn in_ids ~cond_nconsts ~body_nconsts cond body

and emit_while_plain ctx (eqn : eqn) in_ids ~cond_nconsts ~body_nconsts
    (cond : closed_jaxpr) (body : closed_jaxpr) =
  let n_consts = cond_nconsts + body_nconsts in
  let const_avals =
    List.filteri (fun i _ -> i < n_consts) (List.map atom_aval eqn.inputs)
  in
  let const_outs =
    List.map (fun a -> { vid = Core.fresh_id (); vaval = a }) const_avals
  in
  let all_outs = const_outs @ eqn.outs in
  let result = bind_results ctx all_outs in
  let carry_ids = List.map (fun _ -> fresh ctx) in_ids in
  let carry_tys =
    List.map (fun (v : var) -> Ir.tensor_type_of_aval v.vaval) all_outs
  in
  let header_pairs =
    List.map2
      (fun ba init -> Printf.sprintf "%s = %s" (name ba) (name init))
      carry_ids in_ids
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.while(%s) : %s\n" result
       (String.concat ", " header_pairs)
       (String.concat ", " carry_tys));
  let start = ctx.next in
  let cond_ids =
    List.filteri (fun i _ -> i < cond_nconsts || i >= n_consts) carry_ids
  in
  enter_region_later ctx cond cond_ids "    cond {\n";
  Buffer.add_string ctx.buf "    } do {\n";
  ctx.next <- start;
  let cond_fake =
    List.filteri (fun i _ -> i < cond_nconsts) const_avals
    |> List.map (fun a -> { vid = Core.fresh_id (); vaval = a })
  in
  List.iteri
    (fun i (v : var) -> Hashtbl.replace ctx.ids v.vid (List.nth carry_ids i))
    cond_fake;
  let body_ids = List.filteri (fun i _ -> i >= cond_nconsts) carry_ids in
  enter_region ctx body body_ids;
  let body_consts =
    List.filteri (fun i _ -> i < body_nconsts) body.jaxpr.in_binders
  in
  emit_region_returning ctx
    {
      body.jaxpr with
      outs =
        List.map (fun v -> A_var v) (cond_fake @ body_consts) @ body.jaxpr.outs;
    };
  Buffer.add_string ctx.buf "    }\n"

and enter_region_later ctx cj target_ids header =
  Buffer.add_string ctx.buf header;
  enter_region ctx cj target_ids;
  emit_region_returning ctx cj.jaxpr

and emit_inlined ctx (eqn : eqn) in_ids f =
  emit_traced ctx (List.map atom_aval eqn.inputs) in_ids eqn.outs f

and emit_traced ctx avals in_ids outs f =
  List.iter2
    (fun (v : var) id -> Hashtbl.replace ctx.ids v.vid id)
    outs
    (emit_traced_ids ctx avals in_ids f)

and emit_traced_ids ctx avals in_ids f =
  emit_closed_inline ctx (Jaxpr.make_jaxpr avals f) in_ids

and hoist_custom_consts ctx (jx : jaxpr) =
  List.iter
    (fun (e : eqn) ->
      match e.prim with
      | Custom_jvp_call id | Custom_vjp_call id -> (
          match Jaxpr.custom_call_jaxpr id (List.map atom_aval e.inputs) with
          | Some c -> begin
              let binders =
                match
                  Util.split_list c.jaxpr.in_binders [ List.length c.consts ]
                with
                | [ cb; _ ] -> cb
                | _ -> invalid_arg "Stablehlo.Emit: binder split"
              in
              List.iter2
                (fun (v : var) nd ->
                  if not (Hashtbl.mem ctx.ids v.vid) then
                    emit_constant_at ctx (bind_var ctx v) nd)
                binders c.consts;
              hoist_custom_consts ctx c.jaxpr
            end
          | None -> ())
      | _ -> ())
    jx.eqns

and emit_closed_inline ctx (traced : closed_jaxpr) in_ids =
  let jx = traced.jaxpr in
  let const_binders, arg_binders =
    match Util.split_list jx.in_binders [ List.length traced.consts ] with
    | [ cb; ab ] -> (cb, ab)
    | _ -> invalid_arg "Stablehlo.Emit: binder split"
  in
  List.iter2
    (fun (v : var) nd ->
      if not (Hashtbl.mem ctx.ids v.vid) then
        emit_constant_at ctx (bind_var ctx v) nd)
    const_binders traced.consts;
  List.iter2
    (fun (v : var) id -> Hashtbl.replace ctx.ids v.vid id)
    arg_binders in_ids;
  List.iter
    (fun (e : eqn) -> emit_eqn ctx e (List.map (id_of_atom ctx) e.inputs))
    jx.eqns;
  List.map (id_of_atom ctx) jx.outs

and emit_eigh_call ctx input ~lower ~vectors ~values =
  let group = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %%%d:2 = stablehlo.custom_call @Eigh(%s) {api_version = 1 : i32, \
        backend_config = \"%d,1,100,1e-30\"} : (%s) -> (%s, %s)\n"
       group (name input)
       (if lower then 1 else 0)
       vectors vectors values);
  ( alias_result (Printf.sprintf "%%%d#0" group),
    alias_result (Printf.sprintf "%%%d#1" group) )

and placed_call ctx (eqn : eqn) in_ids id =
  let usable mesh devices =
    note_mesh mesh devices;
    !active_mesh = Some mesh
  in
  match (Core.shard_rule id, Core.constraint_rule id) with
  | Some rule, _ when !manual_axes = [] && usable rule.sr_mesh rule.sr_devices
    ->
      emit_manual_computation ctx eqn in_ids rule;
      true
  | _, Some rule when !manual_axes = [] && usable rule.cr_mesh rule.cr_devices
    ->
      let in_aval = atom_aval (List.hd eqn.inputs) in
      let n = bind_var ctx (sole eqn.outs) in
      Buffer.add_string ctx.buf
        (Printf.sprintf "    %s = sdy.sharding_constraint %s %s : %s\n" (name n)
           (name (sole in_ids))
           (sdy_sharding_body rule.cr_spec)
           (Ir.tensor_type_of_aval in_aval));
      true
  | _ -> false

and emit_manual_computation ctx (eqn : eqn) in_ids (rule : Core.shard_rule) =
  let body = rule.sr_body in
  let n_consts = List.length body.consts in
  let arg_binders =
    match Util.split_list body.jaxpr.in_binders [ n_consts ] with
    | [ _; ab ] -> ab
    | _ -> invalid_arg "Stablehlo.Emit: shard_map binder split"
  in
  let targets = List.map (fun _ -> fresh ctx) arg_binders in
  let header = bind_results ctx eqn.outs in
  let shardings specs =
    "[" ^ String.concat ", " (List.map sdy_sharding_body specs) ^ "]"
  in
  let params =
    List.map2
      (fun tid (v : var) ->
        Printf.sprintf "%s: %s" (name tid) (Ir.tensor_type_of_aval v.vaval))
      targets arg_binders
  in
  let out_types =
    List.map (fun (v : var) -> Ir.tensor_type_of_aval v.vaval) eqn.outs
  in
  let in_types =
    List.map (fun a -> Ir.tensor_type_of_aval (atom_aval a)) eqn.inputs
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = sdy.manual_computation(%s) in_shardings=%s out_shardings=%s \
        manual_axes={%s} (%s) {\n"
       header
       (String.concat ", " (List.map name in_ids))
       (shardings rule.sr_in) (shardings rule.sr_out)
       (String.concat ", " (List.map (Printf.sprintf "\"%s\"") rule.sr_manual))
       (String.concat ", " params));
  let saved = !manual_axes in
  manual_axes := rule.sr_manual;
  Fun.protect
    ~finally:(fun () -> manual_axes := saved)
    (fun () ->
      enter_region ctx body targets;
      let text =
        capture ctx (fun () ->
            List.iter
              (fun (e : eqn) ->
                emit_eqn ctx e (List.map (id_of_atom ctx) e.inputs))
              body.jaxpr.eqns;
            let outs = List.map (ssa_of_atom ctx) body.jaxpr.outs in
            let otys =
              List.map
                (fun a -> Ir.tensor_type_of_aval (atom_aval a))
                body.jaxpr.outs
            in
            Buffer.add_string ctx.buf
              (Printf.sprintf "    sdy.return %s : %s\n"
                 (String.concat ", " outs) (String.concat ", " otys)))
      in
      Buffer.add_string ctx.buf (reindent 2 text));
  Buffer.add_string ctx.buf
    (Printf.sprintf "    } : (%s) -> %s\n"
       (String.concat ", " in_types)
       (match out_types with
       | [ t ] -> t
       | ts -> "(" ^ String.concat ", " ts ^ ")"))

and emit_call ctx (eqn : eqn) in_ids (c : closed_jaxpr) =
  let fname =
    match Hashtbl.find_opt call_symbols c.jid with
    | Some s -> s
    | None ->
        let s =
          Mlir_names.symbol_name
            (unique_symbol (Pretty_printer.call_name c.jid))
        in
        Hashtbl.replace call_symbols c.jid s;
        emit_private_function ctx s c;
        s
  in
  let arg_types =
    List.map (fun a -> Ir.tensor_type_of_aval (atom_aval a)) eqn.inputs
  in
  let out_types =
    List.map (fun (v : var) -> Ir.tensor_type_of_aval v.vaval) eqn.outs
  in
  let results =
    match out_types with [ t ] -> t | ts -> "(" ^ String.concat ", " ts ^ ")"
  in
  let args = String.concat ", " (List.map name in_ids) in
  match eqn.outs with
  | [] ->
      Buffer.add_string ctx.buf
        (Printf.sprintf "    call @%s(%s) : (%s) -> ()\n" fname args
           (String.concat ", " arg_types))
  | [ out ] ->
      let n = bind_var ctx out in
      Buffer.add_string ctx.buf
        (Printf.sprintf "    %s = call @%s(%s) : (%s) -> %s\n" (name n) fname
           args
           (String.concat ", " arg_types)
           results)
  | outs ->
      let n = fresh ctx in
      List.iteri
        (fun i (v : var) ->
          Hashtbl.replace ctx.ids v.vid
            (alias_result (Printf.sprintf "%%%d#%d" n i)))
        outs;
      Buffer.add_string ctx.buf
        (Printf.sprintf "    %%%d:%d = call @%s(%s) : (%s) -> %s\n" n
           (List.length outs) fname args
           (String.concat ", " arg_types)
           results)

and emit_private_function ctx fname (c : closed_jaxpr) =
  let inner =
    {
      buf = Buffer.create 256;
      extras = Buffer.create 256;
      ids = Hashtbl.create 16;
      names = Hashtbl.create 4;
      next = 0;
      literals = [];
    }
  in
  let const_binders, arg_binders =
    match Util.split_list c.jaxpr.in_binders [ List.length c.consts ] with
    | [ cb; ab ] -> (cb, ab)
    | _ -> invalid_arg "Stablehlo.Emit: binder split"
  in
  let params =
    List.map
      (fun (v : var) ->
        Printf.sprintf "%%%d: %s" (bind_var inner v)
          (Ir.tensor_type_of_aval v.vaval))
      arg_binders
  in
  List.iter2
    (fun (v : var) nd -> emit_constant_at inner (bind_var inner v) nd)
    const_binders c.consts;
  hoist_custom_consts inner c.jaxpr;
  List.iter
    (fun (e : eqn) -> emit_eqn inner e (List.map (id_of_atom inner) e.inputs))
    c.jaxpr.eqns;
  let out_names = List.map (ssa_of_atom inner) c.jaxpr.outs in
  let out_types =
    List.map (fun a -> Ir.tensor_type_of_aval (atom_aval a)) c.jaxpr.outs
  in
  let results =
    match out_types with [ t ] -> t | ts -> "(" ^ String.concat ", " ts ^ ")"
  in
  Buffer.add_string ctx.extras
    (Printf.sprintf
       "  func.func private @%s(%s) -> %s {\n%s    return %s : %s\n  }\n" fname
       (String.concat ", " params)
       results
       (Buffer.contents inner.buf)
       (String.concat ", " out_names)
       (String.concat ", " out_types));
  Buffer.add_buffer ctx.extras inner.extras

and widen_operands ctx (eqn : eqn) (in_ids : int list) =
  let needs_widening =
    match eqn.prim with Clamp -> true | p -> Core.is_elementwise p
  in
  if not needs_widening then (eqn, in_ids)
  else
    let shapes = List.map (fun a -> (atom_aval a).shape) eqn.inputs in
    let out = Core.elementwise_shape shapes in
    if List.for_all (fun s -> s = out) shapes then (eqn, in_ids)
    else
      let widened =
        List.map2
          (fun atom id ->
            let aval = atom_aval atom in
            if aval.shape = out then (atom, id)
            else begin
              let k = Array.length out - Array.length aval.shape in
              let dims =
                Array.init (Array.length aval.shape) (fun i -> i + k)
              in
              let var =
                { vid = Core.fresh_id (); vaval = { aval with shape = out } }
              in
              emit_broadcast_in_dim ctx
                {
                  prim = Broadcast_in_dim { shape = out; dims };
                  inputs = [ atom ];
                  outs = [ var ];
                  multiple_results = false;
                }
                [ id ] dims;
              (A_var var, Hashtbl.find ctx.ids var.vid)
            end)
          eqn.inputs in_ids
      in
      ({ eqn with inputs = List.map fst widened }, List.map snd widened)

and emit_eqn ctx (eqn : eqn) (in_ids : int list) : unit =
  let eqn, in_ids = widen_operands ctx eqn in_ids in
  try emit_eqn_unchecked ctx eqn in_ids
  with Invalid_argument msg when msg = "Stablehlo.Emit: arity" ->
    failwith
      ("Stablehlo.Emit: arity in " ^ Pretty_printer.primitive_name eqn.prim)

and emit_eqn_unchecked ctx (eqn : eqn) (in_ids : int list) : unit =
  match eqn.prim with
  | Abs when is_unsigned_dtype (atom_aval (sole eqn.inputs)).dtype ->
      emit_stablehlo_unary ctx eqn in_ids "convert"
  | Abs -> emit_stablehlo_unary ctx eqn in_ids "abs"
  | Cbrt -> emit_stablehlo_unary ctx eqn in_ids "cbrt"
  | Ceil -> emit_stablehlo_unary ctx eqn in_ids "ceil"
  | Clz -> emit_stablehlo_unary ctx eqn in_ids "count_leading_zeros"
  | Cos when Dtypes.issubdtype (atom_aval (List.hd eqn.inputs)).dtype `Complex
    ->
      emit_inlined ctx eqn in_ids Native_lowering.complex_cos
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
  | Copy | Checkpoint_name _ ->
      Hashtbl.replace ctx.ids (sole eqn.outs).vid (sole in_ids)
  | Exp2 -> emit_exp2 ctx eqn in_ids
  | Conj
    when match (atom_aval (sole eqn.inputs)).dtype with
         | Dtype.F32 | Dtype.F64 -> true
         | _ -> false ->
      let x = sole in_ids in
      let aval = atom_aval (sole eqn.inputs) in
      let partty = Ir.tensor_type aval.dtype aval.shape in
      let im = fresh ctx in
      let out = sole eqn.outs in
      let n = bind_var ctx out in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %s = stablehlo.constant %s : %s\n\
           \    %s = stablehlo.complex %s, %s : %s\n"
           (name im)
           (Ir.dense (Ir.float_literal aval.dtype 0.0))
           partty (name n) (name x) (name im)
           (Ir.tensor_type_of_aval out.vaval))
  | Conj ->
      let x = sole in_ids in
      let inty = Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)) in
      let aval = atom_aval (sole eqn.inputs) in
      let part_dt =
        match aval.dtype with Dtype.Complex128 -> Dtype.F64 | _ -> Dtype.F32
      in
      let partty = Ir.tensor_type part_dt aval.shape in
      let re = fresh ctx and im = fresh ctx and negated = fresh ctx in
      let out = sole eqn.outs in
      let n = bind_var ctx out in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %s = stablehlo.real %s : (%s) -> %s\n\
           \    %s = stablehlo.imag %s : (%s) -> %s\n\
           \    %s = stablehlo.negate %s : %s\n\
           \    %s = stablehlo.complex %s, %s : %s\n"
           (name re) (name x) inty partty (name im) (name x) inty partty
           (name negated) (name im) partty (name n) (name re) (name negated)
           inty)
  | Log -> emit_stablehlo_unary ctx eqn in_ids "log"
  | Log1p -> emit_stablehlo_unary ctx eqn in_ids "log_plus_one"
  | Neg -> emit_stablehlo_unary ctx eqn in_ids "negate"
  | Not -> emit_stablehlo_unary ctx eqn in_ids "not"
  | Population_count -> emit_stablehlo_unary ctx eqn in_ids "popcnt"
  | Round -> emit_stablehlo_unary ctx eqn in_ids "round_nearest_afz"
  | Round_nearest_even ->
      emit_stablehlo_unary ctx eqn in_ids "round_nearest_even"
  | Rsqrt -> emit_stablehlo_unary ctx eqn in_ids "rsqrt"
  | Sign when is_unsigned_dtype (atom_aval (sole eqn.inputs)).dtype ->
      emit_unsigned_sign ctx eqn in_ids
  | Sign -> emit_stablehlo_unary ctx eqn in_ids "sign"
  | Sin when Dtypes.issubdtype (atom_aval (List.hd eqn.inputs)).dtype `Complex
    ->
      emit_inlined ctx eqn in_ids Native_lowering.complex_sin
  | Sin -> emit_stablehlo_unary ctx eqn in_ids "sine"
  | Sqrt -> emit_stablehlo_unary ctx eqn in_ids "sqrt"
  | Tan -> emit_stablehlo_unary ctx eqn in_ids "tan"
  | Tanh -> emit_stablehlo_unary ctx eqn in_ids "tanh"
  | Sinh -> emit_chlo_unary ctx eqn in_ids "sinh"
  | Square when is_integer_like (atom_aval (sole eqn.inputs)).dtype ->
      emit_self_multiply ctx eqn in_ids
  | Square -> emit_chlo_unary ctx eqn in_ids "square"
  | Is_finite -> emit_is_finite ctx eqn in_ids
  | Logistic -> emit_logistic ctx eqn in_ids
  | Integer_pow y when y < 0 ->
      emit_inlined ctx eqn in_ids (function
        | [ x ] ->
            let positive = Core.bind1 (Integer_pow (-y)) [ x ] in
            let a = Core.get_aval x in
            let one =
              Core.bind1
                (Broadcast_in_dim { shape = a.shape; dims = [||] })
                [ Core.splat a.dtype [||] 1.0 ]
            in
            [ Core.bind1 Div [ one; positive ] ]
        | _ -> invalid_arg "Stablehlo.Emit: integer_pow arity")
  | Integer_pow y -> emit_integer_pow ctx eqn in_ids y
  | Imag ->
      let x = sole in_ids in
      let out = sole eqn.outs in
      let n = bind_var ctx out in
      Buffer.add_string ctx.buf
        (Printf.sprintf "    %s = stablehlo.imag %s : (%s) -> %s\n" (name n)
           (name x)
           (Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)))
           (Ir.tensor_type_of_aval out.vaval))
  | Real ->
      let x = sole in_ids in
      let out = sole eqn.outs in
      let n = bind_var ctx out in
      Buffer.add_string ctx.buf
        (Printf.sprintf "    %s = stablehlo.real %s : (%s) -> %s\n" (name n)
           (name x)
           (Ir.tensor_type_of_aval (atom_aval (sole eqn.inputs)))
           (Ir.tensor_type_of_aval out.vaval))
  | Add | Add_any -> emit_stablehlo_binary ctx eqn in_ids "add"
  | And -> emit_stablehlo_binary ctx eqn in_ids "and"
  | Atan2 -> emit_stablehlo_binary ctx eqn in_ids "atan2"
  | Div -> emit_stablehlo_binary ctx eqn in_ids "divide"
  | Max -> emit_min_max ctx eqn in_ids ~op:"maximum" ~direction:"GT"
  | Min -> emit_min_max ctx eqn in_ids ~op:"minimum" ~direction:"LT"
  | Mul -> emit_stablehlo_binary ctx eqn in_ids "multiply"
  | Or -> emit_stablehlo_binary ctx eqn in_ids "or"
  | Pow
    when match eqn.inputs with
         | [ l; r ] -> (atom_aval l).dtype <> (atom_aval r).dtype
         | _ -> false ->
      emit_inlined ctx eqn in_ids (function
        | [ x; y ] ->
            [
              Core.bind1 Pow
                [
                  x;
                  Core.bind1 (Convert_element_type (Core.get_aval x).dtype)
                    [ y ];
                ];
            ]
        | _ -> invalid_arg "Stablehlo.Emit: pow arity")
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
  | Complex ->
      let x, y = pair in_ids in
      let out = sole eqn.outs in
      let n = bind_var ctx out in
      Buffer.add_string ctx.buf
        (Printf.sprintf "    %s = stablehlo.complex %s, %s : %s\n" (name n)
           (name x) (name y)
           (Ir.tensor_type_of_aval out.vaval))
  | Eq -> emit_compare ctx eqn in_ids "EQ" false
  | Ne -> emit_compare ctx eqn in_ids "NE" false
  | Ge -> emit_compare ctx eqn in_ids "GE" false
  | Gt -> emit_compare ctx eqn in_ids "GT" false
  | Le -> emit_compare ctx eqn in_ids "LE" false
  | Lt -> emit_compare ctx eqn in_ids "LT" false
  | Eq_to -> emit_compare ctx eqn in_ids "EQ" true
  | Le_to -> emit_compare ctx eqn in_ids "LE" true
  | Lt_to -> emit_compare ctx eqn in_ids "LT" true
  | Clamp
    when List.exists
           (fun a -> (atom_aval a).shape <> (sole eqn.outs).vaval.shape)
           eqn.inputs ->
      emit_inlined ctx eqn in_ids (fun args ->
          let shape = (sole eqn.outs).vaval.shape in
          let widen v =
            if (Core.get_aval v).shape = shape then v
            else Core.bind1 (Broadcast_in_dim { shape; dims = [||] }) [ v ]
          in
          [ Core.bind1 Clamp (List.map widen args) ])
  | Clamp
    when List.exists
           (fun a -> (atom_aval a).shape <> (sole eqn.outs).vaval.shape)
           eqn.inputs ->
      let target = (sole eqn.outs).vaval.shape in
      emit_inlined ctx eqn in_ids (fun args ->
          Core.bind Clamp
            (List.map
               (fun a ->
                 if (Core.get_aval a).shape = target then a
                 else
                   Core.bind1
                     (Broadcast_in_dim { shape = target; dims = [||] })
                     [ a ])
               args))
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
  | Ffi_call c -> emit_ffi_call ctx eqn in_ids c
  | Platform_index platforms -> emit_platform_index ctx eqn platforms
  | Broadcast_in_dim { dims; _ } -> emit_broadcast_in_dim ctx eqn in_ids dims
  | Concatenate dim -> emit_concatenate ctx eqn in_ids dim
  | Iota { dimension; _ } -> emit_iota ctx eqn dimension
  | Pad config -> emit_pad ctx eqn in_ids config
  | Reshape _ -> emit_reshape ctx eqn in_ids
  | Squeeze _ -> emit_reshape ctx eqn in_ids
  | Rev dims -> emit_rev ctx eqn in_ids dims
  | Split { sizes; axis } -> emit_split ctx eqn in_ids axis sizes
  | Stack axis -> emit_stack ctx eqn in_ids axis
  | Tile reps -> emit_tile ctx eqn in_ids reps
  | Transpose perm -> emit_transpose ctx eqn in_ids perm
  | Unstack axis -> emit_unstack ctx eqn in_ids axis
  | After_all | Create_token ->
      emit_inlined ctx eqn in_ids (fun _ ->
          [ Concrete (Ndarray.of_floats Dtype.F32 [| 0 |] [||]) ])
  | Dce_sink ->
      failwith
        "Stablehlo.Emit: Dce_sink is a DCE-effect marker, non-representable"
  | Empty2 _ ->
      failwith "Stablehlo.Emit: Empty2 (extended dtype) deferred to M5"
  | From_edtype _ ->
      failwith "Stablehlo.Emit: From_edtype (extended dtypes) deferred to M5"
  | To_edtype _ ->
      failwith "Stablehlo.Emit: To_edtype (extended dtypes) deferred to M5"
  | Reduce_sum axes -> emit_reduce ctx eqn in_ids axes "add" reduce_zero
  | Reduce_max axes when is_complex_dtype (atom_aval (sole eqn.inputs)).dtype ->
      emit_inlined ctx eqn in_ids
        (Native_lowering.complex_reduce ~is_max:true ~axes)
  | Reduce_min axes when is_complex_dtype (atom_aval (sole eqn.inputs)).dtype ->
      emit_inlined ctx eqn in_ids
        (Native_lowering.complex_reduce ~is_max:false ~axes)
  | Reduce_max axes -> emit_reduce ctx eqn in_ids axes "maximum" reduce_maxid
  | Reduce_min axes -> emit_reduce ctx eqn in_ids axes "minimum" reduce_minid
  | Reduce_prod axes -> emit_reduce ctx eqn in_ids axes "multiply" reduce_one
  | Reduce_and axes -> emit_reduce ctx eqn in_ids axes "and" reduce_allones
  | Reduce_or axes -> emit_reduce ctx eqn in_ids axes "or" reduce_zero
  | Reduce_xor axes -> emit_reduce ctx eqn in_ids axes "xor" reduce_zero
  | Argmax { axis; index_dtype } ->
      emit_argminmax ctx eqn in_ids ~is_max:true ~axis ~index_dtype
  | Argmin { axis; index_dtype } ->
      emit_argminmax ctx eqn in_ids ~is_max:false ~axis ~index_dtype
  | Cumsum { axis; reverse } ->
      emit_cum_single ctx eqn in_ids ~fname:"cumsum" ~axis ~reverse
        ~init:reduce_zero ~broadcast:true ~op:"add"
  | Cumprod { axis; reverse } ->
      emit_cum_single ctx eqn in_ids ~fname:"cumprod" ~axis ~reverse
        ~init:reduce_one ~broadcast:false ~op:"multiply"
  | Cummax { axis; reverse } ->
      emit_cum_single ctx eqn in_ids ~fname:"cummax" ~axis ~reverse
        ~init:reduce_maxid ~broadcast:true ~op:"maximum"
  | Cummin { axis; reverse } ->
      emit_cum_single ctx eqn in_ids ~fname:"cummin" ~axis ~reverse
        ~init:reduce_minid ~broadcast:true ~op:"minimum"
  | Cumlogsumexp { axis; reverse } ->
      emit_cum_logsumexp ctx eqn in_ids ~axis ~reverse
  | Slice { start_indices; limit_indices; strides } ->
      emit_slice_op ctx eqn in_ids start_indices limit_indices strides
  | Dynamic_slice { slice_sizes } ->
      emit_dynamic_slice ctx eqn in_ids slice_sizes
  | Dynamic_update_slice -> emit_dynamic_update_slice ctx eqn in_ids
  | Gather { dimension_numbers; slice_sizes } ->
      emit_gather ctx eqn in_ids dimension_numbers slice_sizes
  | Scatter { dimension_numbers; unique_indices } ->
      emit_scatter ~unique:unique_indices ctx eqn in_ids dimension_numbers None
  | Scatter_add { dimension_numbers } ->
      emit_scatter ctx eqn in_ids dimension_numbers (Some "add")
  | Scatter_sub { dimension_numbers } ->
      emit_scatter ctx eqn in_ids dimension_numbers (Some "subtract")
  | Scatter_mul { dimension_numbers; _ } ->
      emit_scatter ctx eqn in_ids dimension_numbers (Some "multiply")
  | Scatter_min { dimension_numbers } ->
      emit_scatter ctx eqn in_ids dimension_numbers (Some "minimum")
  | Scatter_max { dimension_numbers } ->
      emit_scatter ctx eqn in_ids dimension_numbers (Some "maximum")
  | Dot_general dd
    when (atom_aval (List.hd eqn.inputs)).dtype = Dtype.Complex64
         && Config.current_platform () = "cuda" ->
      emit_inlined ctx eqn in_ids (function
        | [ a; b ] ->
            let dot x y = Core.bind1 (Dot_general dd) [ x; y ] in
            let real v = Core.bind1 Real [ v ]
            and imag v = Core.bind1 Imag [ v ] in
            let ar = real a and ai = imag a and br = real b and bi = imag b in
            [
              Core.bind1 Complex
                [
                  Core.bind1 Sub [ dot ar br; dot ai bi ];
                  Core.bind1 Add [ dot ar bi; dot ai br ];
                ];
            ]
        | _ -> invalid_arg "Stablehlo.Emit: dot_general arity")
  | Dot_general dd -> emit_dot_general ctx eqn in_ids dd
  | Conv_general_dilated _
    when match eqn.inputs with lhs :: _ -> is_complex_atom lhs | [] -> false ->
      emit_inlined ctx eqn in_ids (function
        | [ x; y ] ->
            let b1 = Core.bind1 in
            let conv a b = b1 eqn.prim [ a; b ] in
            let x_re = b1 Real [ x ] in
            let x_im = b1 Imag [ x ] in
            let y_re = b1 Real [ y ] in
            let y_im = b1 Imag [ y ] in
            let k1 = conv (b1 Add [ x_re; x_im ]) y_re in
            let k2 = conv x_re (b1 Sub [ y_im; y_re ]) in
            let k3 = conv x_im (b1 Add [ y_re; y_im ]) in
            let re = b1 Sub [ k1; k3 ] in
            let im = b1 Add [ k1; k2 ] in
            [ b1 Complex [ re; im ] ]
        | _ -> invalid_arg "Stablehlo.Emit: conv arity")
  | Conv_general_dilated
      {
        window_strides;
        padding;
        lhs_dilation;
        rhs_dilation;
        dimension_numbers;
        feature_group_count;
        batch_group_count;
      } ->
      emit_conv ctx eqn in_ids ~window_strides ~padding ~lhs_dilation
        ~rhs_dilation ~dn:dimension_numbers ~feature_group_count
        ~batch_group_count
  | Ragged_dot_general ->
      failwith
        "Stablehlo.Emit: Ragged_dot_general non-representable (needs \
         group_sizes machinery, matches host row 27)"
  | Reduce_window_sum window ->
      emit_reduce_window_op ctx eqn in_ids ~window ~op:"add" ~init:reduce_zero
  | Reduce_window_max window ->
      emit_reduce_window_op ctx eqn in_ids ~window ~op:"maximum"
        ~init:reduce_maxid
  | Reduce_window_min window ->
      emit_reduce_window_op ctx eqn in_ids ~window ~op:"minimum"
        ~init:reduce_minid
  | Select_and_gather_add { select; window } ->
      emit_select_and_gather_add ctx eqn in_ids ~select ~window
  | Select_and_scatter_add { select; window } ->
      emit_select_and_scatter_add ctx eqn in_ids ~select ~window
  | Sort { dimension; is_stable; num_keys } ->
      if List.length in_ids = 1 then
        emit_sort ctx eqn in_ids ~dimension ~is_stable
      else emit_sort_multi ctx eqn in_ids ~dimension ~is_stable ~num_keys
  | Top_k { k; _ } -> emit_top_k ctx eqn in_ids k
  | Bessel_i1e -> emit_chlo_unary ctx eqn in_ids "bessel_i1e"
  | Digamma -> emit_chlo_unary ctx eqn in_ids "digamma"
  | Erf -> emit_chlo_unary ctx eqn in_ids "erf"
  | Erf_inv -> emit_chlo_unary ctx eqn in_ids "erf_inv"
  | Erfc -> emit_chlo_unary ctx eqn in_ids "erfc"
  | Lgamma -> emit_chlo_unary ctx eqn in_ids "lgamma"
  | Polygamma -> emit_chlo_binary ctx eqn in_ids "polygamma"
  | Zeta -> emit_chlo_binary ctx eqn in_ids "zeta"
  | Bessel_i0e -> emit_inlined ctx eqn in_ids Native_lowering.bessel_i0e
  | Igamma -> emit_inlined ctx eqn in_ids Native_lowering.igamma
  | Igamma_grad_a -> emit_inlined ctx eqn in_ids Native_lowering.igamma_grad_a
  | Igammac -> emit_inlined ctx eqn in_ids Native_lowering.igammac
  | Regularized_incomplete_beta ->
      emit_inlined ctx eqn in_ids Native_lowering.betainc
  | Rng_uniform -> emit_rng_uniform ctx eqn in_ids
  | Rng_bit_generator { rbg_algorithm; _ } ->
      emit_rng_bit_generator ctx eqn in_ids rbg_algorithm
  | Threefry2x32 ->
      emit_inlined ctx eqn in_ids (function
        | [ k1; k2; c1; c2 ] -> Prng_lowering.threefry2x32 k1 k2 c1 c2
        | _ -> invalid_arg "Stablehlo.Emit: threefry arity")
  | Iota_2x32_shape shape ->
      emit_inlined ctx eqn in_ids (fun _ ->
          let hi, lo = Prng_lowering.iota_counters shape in
          [ hi; lo ])
  | Random_bits { bit_width = 32; shape } ->
      emit_inlined ctx eqn in_ids (fun args ->
          [ Prng_lowering.random_bits ~shape (List.hd args) ])
  | Random_bits { shape; _ } ->
      emit_inlined ctx eqn in_ids (fun args ->
          [ Prng_lowering.random_bits64 ~shape (List.hd args) ])
  | Random_seed ->
      emit_inlined ctx eqn in_ids (fun args ->
          [ Prng_lowering.random_seed (List.hd args) ])
  | Random_split shape ->
      emit_inlined ctx eqn in_ids (fun args ->
          [ Prng_lowering.random_split ~shape (List.hd args) ])
  | Random_fold_in ->
      emit_inlined ctx eqn in_ids (function
        | [ key; data ] -> [ Prng_lowering.fold_in key data ]
        | _ -> invalid_arg "Stablehlo.Emit: fold_in arity")
  | Random_wrap | Random_unwrap -> emit_inlined ctx eqn in_ids Fun.id
  | Cond { branches; platforms } -> emit_cond ctx eqn in_ids ~platforms branches
  | Reduce { jaxpr; dimensions } ->
      emit_reduce_general ctx eqn in_ids jaxpr dimensions
  | Reduce_window { reducer; window } ->
      emit_reduce_window_general ctx eqn in_ids reducer window
  | While { cond_nconsts; body_nconsts; cond; body } ->
      emit_while ctx eqn in_ids ~cond_nconsts ~body_nconsts cond body
  | Scan { length; reverse; num_consts; num_carry; jaxpr; unroll } ->
      emit_inlined ctx eqn in_ids
        (Lax.lower_scan ~length ~reverse ~num_consts ~num_carry ~unroll jaxpr)
  | Select_and_scatter _ ->
      failwith
        "Stablehlo.Emit: general Select_and_scatter lowers on cpu to a pad + \
         stablehlo.select_and_scatter (two arbitrary regions) + slice \
         decomposition; the common add-variant is goldened at row 84, the \
         general dual-region form is deferred to the PJRT execution rows"
  | Composite decomposition ->
      emit_inlined ctx eqn in_ids (Jaxpr.eval_closed_jaxpr decomposition)
  | Custom_linear_solve { solve; _ } ->
      emit_inlined ctx eqn in_ids (Jaxpr.eval_closed_jaxpr solve)
  | Cholesky when lapack_ffi () ->
      let a = atom_aval (sole eqn.inputs) in
      let info =
        lapack_aval (Array.sub a.shape 0 (Array.length a.shape - 2)) Dtype.I32
      in
      let ids =
        emit_lapack_call ctx ~base:"potrf" ~config:"uplo = 76 : ui8"
          ~aliases:[ (0, 0) ]
          in_ids [ a ] [ a; info ]
      in
      emit_traced ctx [ a; info ] ids eqn.outs Native_lowering.lapack_cholesky
  | Triangular_solve ({ transpose_a = false; conjugate_a = true; _ } as o)
    when lapack_ffi () ->
      emit_inlined ctx eqn in_ids (function
        | [ a; b ] ->
            let a =
              if is_complex_dtype (Core.get_aval a).dtype then
                Core.bind1 Conj [ a ]
              else a
            in
            Core.bind (Triangular_solve { o with conjugate_a = false }) [ a; b ]
        | _ -> invalid_arg "Stablehlo.Emit: triangular_solve arity")
  | Triangular_solve o when lapack_ffi () ->
      let inputs = List.map atom_aval eqn.inputs in
      let b = List.nth inputs 1 in
      let char_attr key c = Printf.sprintf "%s = %d : ui8" key (Char.code c) in
      let trans =
        if not o.transpose_a then 'N'
        else if o.conjugate_a && is_complex_dtype b.dtype then 'C'
        else 'T'
      in
      let config =
        String.concat ", "
          [
            char_attr "diag" (if o.unit_diagonal then 'U' else 'N');
            char_attr "side" (if o.left_side then 'L' else 'R');
            char_attr "trans_x" trans;
            char_attr "uplo" (if o.lower then 'L' else 'U');
          ]
      in
      let ids =
        emit_lapack_call ctx ~base:"trsm" ~config
          ~aliases:[ (1, 0) ]
          in_ids inputs [ b ]
      in
      Hashtbl.replace ctx.ids (sole eqn.outs).vid (List.hd ids)
  | Householder_product when lapack_ffi () ->
      let inputs = List.map atom_aval eqn.inputs in
      let a = List.hd inputs in
      let base = if is_complex_dtype a.dtype then "ungqr" else "orgqr" in
      let ids =
        emit_lapack_call ctx ~base ~config:""
          ~aliases:[ (0, 0) ]
          in_ids inputs [ a ]
      in
      Hashtbl.replace ctx.ids (sole eqn.outs).vid (List.hd ids)
  | Lu when lapack_ffi () ->
      let a = atom_aval (sole eqn.inputs) in
      let rank = Array.length a.shape in
      if Array.exists (fun d -> d = 0) a.shape then
        failwith "Stablehlo.Emit: an empty lu runs on the host";
      let batch = Array.sub a.shape 0 (rank - 2) in
      let m = a.shape.(rank - 2) and n = a.shape.(rank - 1) in
      let pivots = lapack_aval (Array.append batch [| min m n |]) Dtype.I32 in
      let info = lapack_aval batch Dtype.I32 in
      let ids =
        emit_lapack_call ctx ~base:"getrf" ~config:""
          ~aliases:[ (0, 0) ]
          in_ids [ a ] [ a; pivots; info ]
      in
      emit_traced ctx [ a; pivots; info ] ids eqn.outs
        (Native_lowering.lapack_lu ~permutation_size:m)
  | Qr full_matrices when lapack_ffi () ->
      let a = atom_aval (sole eqn.inputs) in
      let rank = Array.length a.shape in
      if Array.exists (fun d -> d = 0) a.shape then
        failwith "Stablehlo.Emit: an empty qr runs on the host";
      let taus =
        lapack_aval
          (Array.append
             (Array.sub a.shape 0 (rank - 2))
             [| min a.shape.(rank - 2) a.shape.(rank - 1) |])
          a.dtype
      in
      let ids =
        emit_lapack_call ctx ~base:"geqrf" ~config:""
          ~aliases:[ (0, 0) ]
          in_ids [ a ] [ a; taus ]
      in
      emit_traced ctx [ a; taus ] ids eqn.outs
        (Native_lowering.qr_from_factors ~full_matrices)
  | Qr_pivoted full_matrices ->
      if Array.exists (fun d -> d = 0) (atom_aval (sole eqn.inputs)).shape then
        failwith "Stablehlo.Emit: an empty qr runs on the host";
      emit_inlined ctx eqn in_ids (function
        | [ a ] -> Native_lowering.qr_pivoted ~full_matrices a
        | _ -> invalid_arg "Stablehlo.Emit: qr arity")
  | Geqrf when lapack_ffi () ->
      let a = atom_aval (sole eqn.inputs) in
      let rank = Array.length a.shape in
      let taus =
        lapack_aval
          (Array.append
             (Array.sub a.shape 0 (rank - 2))
             [| min a.shape.(rank - 2) a.shape.(rank - 1) |])
          a.dtype
      in
      let ids =
        emit_lapack_call ctx ~base:"geqrf" ~config:""
          ~aliases:[ (0, 0) ]
          in_ids [ a ] [ a; taus ]
      in
      List.iter2
        (fun (v : var) id -> Hashtbl.replace ctx.ids v.vid id)
        eqn.outs ids
  | Geqp3 when lapack_ffi () ->
      let inputs = List.map atom_aval eqn.inputs in
      let a = List.hd inputs and jpvt = List.nth inputs 1 in
      let rank = Array.length a.shape in
      let taus =
        lapack_aval
          (Array.append
             (Array.sub a.shape 0 (rank - 2))
             [| min a.shape.(rank - 2) a.shape.(rank - 1) |])
          a.dtype
      in
      let ids =
        emit_lapack_call ctx ~base:"geqp3" ~config:""
          ~aliases:[ (0, 0); (1, 1) ]
          in_ids inputs [ a; jpvt; taus ]
      in
      List.iter2
        (fun (v : var) id -> Hashtbl.replace ctx.ids v.vid id)
        eqn.outs ids
  | Ormqr { ormqr_left; ormqr_transpose } when lapack_ffi () ->
      let inputs = List.map atom_aval eqn.inputs in
      let a = List.hd inputs and c = List.nth inputs 2 in
      let base = if is_complex_dtype a.dtype then "unmqr" else "ormqr" in
      let ids =
        emit_lapack_call ctx ~base
          ~config:
            (Printf.sprintf "left = %b, transpose = %b" ormqr_left
               ormqr_transpose)
          ~aliases:[ (2, 0) ]
          in_ids inputs [ c ]
      in
      Hashtbl.replace ctx.ids (sole eqn.outs).vid (List.hd ids)
  | Tridiagonal_solve { perturb_singular = true } when lapack_ffi () ->
      let inputs = List.map atom_aval eqn.inputs in
      let b = List.nth inputs 3 in
      let ids =
        emit_lapack_call ctx ~target:"tridiagonal_solve_perturbed_ffi"
          ~base:"tridiagonal_solve_perturbed" ~config:"" ~aliases:[] in_ids
          inputs [ b ]
      in
      emit_traced ctx [ b ] ids eqn.outs Fun.id
  | Tridiagonal_solve { perturb_singular = false } when lapack_ffi () ->
      let inputs = List.map atom_aval eqn.inputs in
      let d = List.nth inputs 1 and b = List.nth inputs 3 in
      if Array.length b.shape <> Array.length d.shape + 1 then
        failwith "Stablehlo.Emit: gtsv expects a matrix right-hand side";
      let info =
        lapack_aval (Array.sub b.shape 0 (Array.length b.shape - 2)) Dtype.I32
      in
      let out_avals = inputs @ [ info ] in
      let ids =
        emit_lapack_call ctx ~base:"gtsv" ~config:""
          ~aliases:[ (0, 0); (1, 1); (2, 2); (3, 3) ]
          in_ids inputs out_avals
      in
      emit_traced ctx out_avals ids eqn.outs (fun outs ->
          match Native_lowering.lapack_guarded outs with
          | [ _; _; _; x ] -> [ x ]
          | _ -> invalid_arg "Stablehlo.Emit: gtsv arity")
  | Eigh { lower } when lapack_ffi () ->
      let a = atom_aval (sole eqn.inputs) in
      let rank = Array.length a.shape in
      let batch = Array.sub a.shape 0 (rank - 2) in
      let c = is_complex_dtype a.dtype in
      let real_dt =
        match a.dtype with
        | Dtype.Complex64 -> Dtype.F32
        | Dtype.Complex128 -> Dtype.F64
        | d -> d
      in
      if Array.exists (fun d -> d = 0) a.shape then
        failwith "Stablehlo.Emit: an empty eigh runs on the host";
      let w =
        lapack_aval (Array.append batch [| a.shape.(rank - 1) |]) real_dt
      in
      let info = lapack_aval batch Dtype.I32 in
      let config =
        Printf.sprintf "mode = 86 : ui8, uplo = %d : ui8"
          (Char.code (if lower then 'L' else 'U'))
      in
      let ids =
        emit_lapack_call ctx
          ~base:(if c then "heevd" else "syevd")
          ~config
          ~aliases:[ (0, 0) ]
          in_ids [ a ] [ a; w; info ]
      in
      emit_traced ctx [ a; w; info ] ids eqn.outs Native_lowering.lapack_guarded
  | Svd { full_matrices; compute_uv; gesvd } when lapack_ffi () ->
      let a = atom_aval (sole eqn.inputs) in
      let rank = Array.length a.shape in
      if Array.exists (fun d -> d = 0) a.shape then
        failwith "Stablehlo.Emit: an empty svd runs on the host";
      let batch = Array.sub a.shape 0 (rank - 2) in
      let m = a.shape.(rank - 2) and n = a.shape.(rank - 1) in
      let mn = min m n in
      let real_dt =
        match a.dtype with
        | Dtype.Complex64 -> Dtype.F32
        | Dtype.Complex128 -> Dtype.F64
        | d -> d
      in
      let s = lapack_aval (Array.append batch [| mn |]) real_dt in
      let u =
        lapack_aval
          (Array.append batch [| m; (if full_matrices then m else mn) |])
          a.dtype
      in
      let vt =
        lapack_aval
          (Array.append batch [| (if full_matrices then n else mn); n |])
          a.dtype
      in
      let info = lapack_aval batch Dtype.I32 in
      let mode =
        if not compute_uv then 'N' else if full_matrices then 'A' else 'S'
      in
      let out_avals = [ a; s; u; vt; info ] in
      let ids =
        emit_lapack_call ctx
          ~base:(if gesvd then "gesvd" else "gesdd")
          ~config:(Printf.sprintf "mode = %d : ui8" (Char.code mode))
          ~aliases:[ (0, 0) ]
          in_ids [ a ] out_avals
      in
      emit_traced ctx out_avals ids eqn.outs
        (Native_lowering.lapack_svd ~compute_uv)
  | Eig { compute_left; compute_right; _ } when lapack_ffi () ->
      let a = atom_aval (sole eqn.inputs) in
      let rank = Array.length a.shape in
      if Array.exists (fun d -> d = 0) a.shape then
        failwith "Stablehlo.Emit: an empty eig runs on the host";
      let batch = Array.sub a.shape 0 (rank - 2) in
      let n = a.shape.(rank - 1) in
      let real = not (is_complex_dtype a.dtype) in
      let cdt =
        match a.dtype with
        | Dtype.F32 | Dtype.Complex64 -> Dtype.Complex64
        | _ -> Dtype.Complex128
      in
      let w = lapack_aval (Array.append batch [| n |]) a.dtype in
      let vec = lapack_aval a.shape cdt in
      let info = lapack_aval batch Dtype.I32 in
      let out_avals = (if real then [ w; w ] else [ w ]) @ [ vec; vec; info ] in
      let flag b = Char.code (if b then 'V' else 'N') in
      let ids =
        emit_lapack_call ctx ~base:"geev"
          ~config:
            (Printf.sprintf "compute_left = %d : ui8, compute_right = %d : ui8"
               (flag compute_left) (flag compute_right))
          ~aliases:[] in_ids [ a ] out_avals
      in
      emit_traced ctx out_avals ids eqn.outs
        (Native_lowering.lapack_eig ~real ~compute_left ~compute_right)
  | Hessenberg when lapack_ffi () ->
      let a = atom_aval (sole eqn.inputs) in
      let rank = Array.length a.shape in
      let n = a.shape.(rank - 1) in
      if n < 1 then
        failwith "Stablehlo.Emit: an empty hessenberg runs on the host";
      let batch = Array.sub a.shape 0 (rank - 2) in
      let taus = lapack_aval (Array.append batch [| n - 1 |]) a.dtype in
      let info = lapack_aval batch Dtype.I32 in
      let out_avals = [ a; taus; info ] in
      let ids =
        emit_lapack_call ctx ~base:"gehrd"
          ~config:(Printf.sprintf "high = %d : i32, low = 1 : i32" n)
          ~aliases:[ (0, 0) ]
          in_ids [ a ] out_avals
      in
      emit_traced ctx out_avals ids eqn.outs Native_lowering.lapack_guarded
  | Schur { compute_vectors } when lapack_ffi () ->
      let a = atom_aval (sole eqn.inputs) in
      let rank = Array.length a.shape in
      if Array.exists (fun d -> d = 0) a.shape then
        failwith "Stablehlo.Emit: an empty schur runs on the host";
      let batch = Array.sub a.shape 0 (rank - 2) in
      let w =
        lapack_aval (Array.append batch [| a.shape.(rank - 1) |]) a.dtype
      in
      let scalar = lapack_aval batch Dtype.I32 in
      let values = if is_complex_dtype a.dtype then [ w ] else [ w; w ] in
      let out_avals = [ a; a ] @ values @ [ scalar; scalar ] in
      let mode = Char.code (if compute_vectors then 'V' else 'N') in
      let ids =
        emit_lapack_call ctx ~base:"gees"
          ~config:(Printf.sprintf "mode = %d : ui8, sort = 78 : ui8" mode)
          ~aliases:[ (0, 0) ]
          in_ids [ a ] out_avals
      in
      emit_traced ctx out_avals ids eqn.outs
        (Native_lowering.lapack_schur ~compute_vectors)
  | Tridiagonal { lower } when lapack_ffi () ->
      let a = atom_aval (sole eqn.inputs) in
      let rank = Array.length a.shape in
      let n = a.shape.(rank - 1) in
      if n < 2 then
        failwith "Stablehlo.Emit: a small tridiagonal runs on the host";
      let batch = Array.sub a.shape 0 (rank - 2) in
      let real_dt =
        match a.dtype with
        | Dtype.Complex64 -> Dtype.F32
        | Dtype.Complex128 -> Dtype.F64
        | d -> d
      in
      let d = lapack_aval (Array.append batch [| n |]) real_dt in
      let e = lapack_aval (Array.append batch [| n - 1 |]) real_dt in
      let taus = lapack_aval (Array.append batch [| n - 1 |]) a.dtype in
      let info = lapack_aval batch Dtype.I32 in
      let out_avals = [ a; d; e; taus; info ] in
      let ids =
        emit_lapack_call ctx
          ~base:(if is_complex_dtype a.dtype then "hetrd" else "sytrd")
          ~config:
            (Printf.sprintf "uplo = %d : ui8"
               (Char.code (if lower then 'L' else 'U')))
          ~aliases:[ (0, 0) ]
          in_ids [ a ] out_avals
      in
      emit_traced ctx out_avals ids eqn.outs Native_lowering.lapack_guarded
  | Cholesky ->
      let aval = atom_aval (sole eqn.inputs) in
      let ty = Ir.tensor_type_of_aval aval in
      let raw = fresh ctx in
      let mask_id = fresh ctx in
      let rank = Array.length aval.shape in
      let rows = aval.shape.(rank - 2) and cols = aval.shape.(rank - 1) in
      let total = Array.fold_left ( * ) 1 aval.shape in
      let mask =
        Ndarray.of_floats aval.dtype aval.shape
          (Array.init total (fun i ->
               let local = i mod (rows * cols) in
               if local mod cols <= local / cols then 1.0 else 0.0))
      in
      emit_constant_at ctx mask_id mask;
      let out = sole eqn.outs in
      let n = bind_var ctx out in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %s = stablehlo.cholesky %s, lower = true : %s\n\
           \    %s = stablehlo.multiply %s, %s : %s\n"
           (name raw)
           (name (sole in_ids))
           ty (name n) (name raw) (name mask_id) ty)
  | Triangular_solve ({ transpose_a = false; conjugate_a = true; _ } as o) ->
      emit_inlined ctx eqn in_ids (function
        | [ a; b ] ->
            let a =
              if is_complex_dtype (Core.get_aval a).dtype then
                Core.bind1 Conj [ a ]
              else a
            in
            Core.bind (Triangular_solve { o with conjugate_a = false }) [ a; b ]
        | _ -> invalid_arg "Stablehlo.Emit: triangular_solve arity")
  | Triangular_solve o ->
      let a, b = pair in_ids in
      let inputs = List.map atom_aval eqn.inputs in
      let aty = Ir.tensor_type_of_aval (List.hd inputs) in
      let bty = Ir.tensor_type_of_aval (List.nth inputs 1) in
      let out = sole eqn.outs in
      let n = bind_var ctx out in
      let transpose =
        match (o.transpose_a, o.conjugate_a) with
        | false, false -> "NO_TRANSPOSE"
        | true, false -> "TRANSPOSE"
        | true, true -> "ADJOINT"
        | false, true -> invalid_arg "Stablehlo.Emit: conjugate lowered above"
      in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %s = \"stablehlo.triangular_solve\"(%s, %s) <{left_side = %b, \
            lower = %b, transpose_a = #stablehlo<transpose %s>, unit_diagonal \
            = %b}> : (%s, %s) -> %s\n"
           (name n) (name a) (name b) o.left_side o.lower transpose
           o.unit_diagonal aty bty bty)
  | Householder_product ->
      let inputs = List.map atom_aval eqn.inputs in
      let n = bind_var ctx (sole eqn.outs) in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %s = stablehlo.custom_call \
            @ProductOfElementaryHouseholderReflectors(%s) {api_version = 1 : \
            i32} : (%s) -> %s\n"
           (name n)
           (String.concat ", " (List.map name in_ids))
           (String.concat ", " (List.map Ir.tensor_type_of_aval inputs))
           (Ir.tensor_type_of_aval (sole eqn.outs).vaval))
  | Lu ->
      if Config.current_platform () = "cpu" then
        failwith
          "Stablehlo.Emit: lu on cpu uses LAPACK getrf, whose rounding the \
           unblocked jax lowering does not reproduce";
      emit_inlined ctx eqn in_ids Native_lowering.lu
  | Lu_pivots_to_permutation permutation_size ->
      emit_inlined ctx eqn in_ids
        (Native_lowering.lu_pivots_to_permutation ~permutation_size)
  | Geqrf | Geqp3 | Ormqr _ ->
      failwith "Stablehlo.Emit: this LAPACK target needs the scipy library"
  | Qr full_matrices ->
      let a = atom_aval (sole eqn.inputs) in
      let rank = Array.length a.shape in
      if Array.exists (fun d -> d = 0) a.shape then
        failwith "Stablehlo.Emit: an empty qr runs on the host";
      let taus =
        {
          a with
          shape =
            Array.append
              (Array.sub a.shape 0 (rank - 2))
              [| min a.shape.(rank - 2) a.shape.(rank - 1) |];
        }
      in
      let group = fresh ctx in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %%%d:2 = stablehlo.custom_call @Qr(%s) {api_version = 1 : i32} \
            : (%s) -> (%s, %s)\n"
           group
           (name (sole in_ids))
           (Ir.tensor_type_of_aval a) (Ir.tensor_type_of_aval a)
           (Ir.tensor_type_of_aval taus));
      emit_traced ctx [ a; taus ]
        [
          alias_result (Printf.sprintf "%%%d#0" group);
          alias_result (Printf.sprintf "%%%d#1" group);
        ]
        eqn.outs
        (Native_lowering.qr_from_factors ~full_matrices)
  | Tridiagonal_solve { perturb_singular } ->
      if perturb_singular || Config.current_platform () = "cpu" then
        failwith
          "Stablehlo.Emit: tridiagonal_solve on cpu uses LAPACK gtsv with \
           pivoting, which the unpivoted jax lowering cannot match";
      emit_inlined ctx eqn in_ids Native_lowering.tridiagonal_solve
  | Eigh { lower } ->
      if Config.current_platform () = "cpu" then
        failwith
          "Stablehlo.Emit: eigh on cpu uses LAPACK syevd, whose eigenvector \
           signs the Jacobi expansion does not reproduce";
      let a = atom_aval (sole eqn.inputs) in
      let v, w =
        match eqn.outs with
        | [ v; w ] -> (v, w)
        | _ -> invalid_arg "Stablehlo.Emit: eigh arity"
      in
      let vectors = Ir.tensor_type_of_aval v.vaval in
      let values = Ir.tensor_type_of_aval w.vaval in
      let input = sole in_ids in
      let v1, _ = emit_eigh_call ctx input ~lower ~vectors ~values in
      emit_traced ctx [ a; v.vaval ] [ input; v1 ] [ v; w ]
        (Native_lowering.eigh_refined ~lower)
  | Svd { full_matrices; compute_uv; _ } ->
      if Config.current_platform () = "cpu" then
        failwith
          "Stablehlo.Emit: svd on cpu uses LAPACK gesdd, whose singular vector \
           signs the qdwh expansion does not reproduce";
      emit_inlined ctx eqn in_ids
        (Native_lowering.svd ~full_matrices ~compute_uv)
  | Eig _ | Hessenberg | Schur _ | Tridiagonal _ ->
      failwith
        "Stablehlo.Emit: eig, hessenberg, schur and tridiagonal have no \
         StableHLO lowering in jax; they run on the host through LAPACK"
  | Collective { kind; axis_name } -> (
      let mesh = collective_mesh axis_name in
      match kind with
      | Psum -> emit_all_reduce ctx eqn in_ids "add" mesh axis_name
      | Pmax -> emit_all_reduce ctx eqn in_ids "maximum" mesh axis_name
      | Pmin -> emit_all_reduce ctx eqn in_ids "minimum" mesh axis_name
      | All_gather { axis; size } ->
          let in_aval = atom_aval (List.hd eqn.inputs) in
          let gathered, _ =
            emit_all_gather ctx (sole in_ids) in_aval axis size mesh axis_name
          in
          Hashtbl.replace ctx.ids (sole eqn.outs).vid gathered
      | Pbroadcast source ->
          emit_pbroadcast ctx eqn in_ids source
            (List.assoc axis_name mesh)
            mesh axis_name
      | Ppermute pairs -> emit_ppermute ctx eqn in_ids pairs mesh axis_name
      | Axis_index -> emit_axis_index ctx eqn mesh axis_name)
  | Fft { fft_type; fft_lengths } ->
      let out = sole eqn.outs in
      let n = bind_var ctx out in
      let kind =
        match fft_type with
        | Fft_forward -> "FFT"
        | Fft_inverse -> "IFFT"
        | Fft_real -> "RFFT"
        | Fft_inverse_real -> "IRFFT"
      in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "    %s = stablehlo.fft %s, type = %s, length = %s : (%s) -> %s\n"
           (name n)
           (name (sole in_ids))
           kind
           (Ir.int_array_attr fft_lengths)
           (Ir.tensor_type_of_aval (atom_aval (List.hd eqn.inputs)))
           (Ir.tensor_type_of_aval out.vaval))
  | Custom_jvp_call id when placed_call ctx eqn in_ids id -> ()
  | Custom_jvp_call id | Custom_vjp_call id -> (
      match Jaxpr.custom_call_jaxpr id (List.map atom_aval eqn.inputs) with
      | Some c ->
          List.iter2
            (fun (v : var) out -> Hashtbl.replace ctx.ids v.vid out)
            eqn.outs
            (emit_closed_inline ctx c in_ids)
      | None -> (
          match Core.custom_rule id with
          | Core.Callback_rule _ -> emit_host_callback ctx eqn in_ids id
          | _ -> invalid_arg "Stablehlo.Emit: linear rule has no primal call"))
  | Custom_lin { rule; n_res; _ } -> (
      match Core.custom_rule rule with
      | Core.Linear_rule { forward; _ } ->
          let inline args =
            match Util.split_list args [ n_res ] with
            | [ res; linear ] -> forward res linear
            | _ -> invalid_arg "Stablehlo.Emit: custom_lin arity"
          in
          emit_inlined ctx eqn in_ids inline
      | Core.Jvp_rule _ | Core.Vjp_rule _ | Core.Callback_rule _ ->
          failwith
            "can't apply forward-mode autodiff (jvp) to a custom_vjp function.")
  | Xla_call c -> emit_call ctx eqn in_ids c
  | Stop_gradient -> Hashtbl.replace ctx.ids (sole eqn.outs).vid (sole in_ids)

let qualify_nested_calls ~depth text =
  let level = ref depth in
  String.split_on_char '\n' text
  |> List.map (fun line ->
      let qualified =
        match String.index_opt line '=' with
        | Some i
          when !level > 1
               && i + 7 <= String.length line
               && String.sub line (i + 1) 6 = " call " ->
            String.sub line 0 (i + 1)
            ^ " func.call "
            ^ String.sub line (i + 7) (String.length line - i - 7)
        | _ -> line
      in
      String.iter
        (function '{' -> incr level | '}' -> decr level | _ -> ())
        line;
      qualified)
  |> String.concat "\n"

let emission_depth = ref 0
let emitting () = !emission_depth > 0

let emit_closed_jaxpr_body ?partition (cj : closed_jaxpr) : string =
  Hashtbl.reset used_symbols;
  Hashtbl.reset cum_functions;
  Hashtbl.reset call_symbols;
  symbol_counter := 0;
  seen_mesh := None;
  callback_seen := false;
  channel := 0;
  manual_axes := [];
  active_mesh := Option.map (fun p -> p.pt_mesh) partition;
  let n_consts = List.length cj.consts in
  let jx = cj.jaxpr in
  let const_binders, arg_binders =
    match Util.split_list jx.in_binders [ n_consts ] with
    | [ cb; ob ] -> (cb, ob)
    | _ -> invalid_arg "Stablehlo.Emit: binder split"
  in
  let ctx =
    {
      buf = Buffer.create 256;
      extras = Buffer.create 256;
      ids = Hashtbl.create 16;
      names = Hashtbl.create 4;
      next = 0;
      literals = [];
    }
  in
  let in_specs =
    match partition with
    | Some p -> List.map Option.some p.pt_in
    | None -> List.map (fun _ -> None) arg_binders
  in
  let attribute = function
    | Some spec -> Printf.sprintf " {sdy.sharding = %s}" (sdy_sharding spec)
    | None -> ""
  in
  let params =
    List.map2
      (fun (v : var) spec ->
        Printf.sprintf "%s: %s%s"
          ("%" ^ string_of_int (bind_var ctx v))
          (Ir.tensor_type_of_aval v.vaval)
          (attribute spec))
      arg_binders in_specs
  in
  List.iter2
    (fun (v : var) nd ->
      let n = bind_var ctx v in
      emit_constant_at ctx n nd)
    const_binders cj.consts;
  hoist_custom_consts ctx jx;
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
  let out_attributes =
    match partition with
    | Some { pt_out; _ } when pt_out <> [] ->
        List.map2 (fun ty spec -> ty ^ attribute spec) out_types pt_out
    | _ -> out_types
  in
  Printf.sprintf
    "module {\n%s  func.func public @main(%s) -> (%s) {\n%s  }\n%s}\n"
    (match partition with
    | Some p -> sdy_mesh p.pt_mesh
    | None ->
        if !callback_seen then
          "  sdy.mesh @maximal_mesh_0 = <[], device_ids=[0]> {stablehlo.mesh = \
           {axes = [], device_ids = dense<0> : tensor<1xi64>}}\n"
        else "")
    (String.concat ", " params)
    (String.concat ", " out_attributes)
    (qualify_nested_calls ~depth:1 (Buffer.contents ctx.buf))
    (qualify_nested_calls ~depth:0 (Buffer.contents ctx.extras))

let emit_closed_jaxpr ?partition cj =
  incr emission_depth;
  Fun.protect
    ~finally:(fun () -> decr emission_depth)
    (fun () -> emit_closed_jaxpr_body ?partition cj)

type lowered = { text : string; kept : int list }

let lower ~name cj =
  let pruned, kept = Dce.prune_unused_inputs cj in
  {
    text = Mlir_names.render ~module_name:name (emit_closed_jaxpr pruned);
    kept;
  }

let emit_jax_text ~name cj = Mlir_names.renumber_callbacks (lower ~name cj).text

let emit_export_text ~name ~uses_shape_polymorphism cj =
  let pruned, kept = Dce.prune_unused_inputs cj in
  {
    text =
      Mlir_names.render ~module_name:name
        ~module_attributes:
          [
            Printf.sprintf "jax.uses_shape_polymorphism = %b"
              uses_shape_polymorphism;
          ]
        (Mlir_names.hoist_constants (emit_closed_jaxpr pruned));
    kept;
  }

let lower_partitioned ~name partition cj =
  let pruned, kept = Dce.prune_unused_inputs cj in
  let pt_in = List.map (List.nth partition.pt_in) kept in
  {
    text =
      Mlir_names.render ~module_name:name
        ~num_partitions:(Array.length partition.pt_devices)
        (emit_closed_jaxpr ~partition:{ partition with pt_in } pruned);
    kept;
  }
