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
  | _ ->
      failwith
        "Stablehlo.Emit: no lowering rule for this primitive (rows 77-87)"

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
