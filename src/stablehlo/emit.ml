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

let ssa_of_atom ctx = function
  | A_var v -> ssa_of_var ctx v
  | A_lit nd ->
      let n = fresh ctx in
      emit_constant_at ctx n nd;
      "%" ^ string_of_int n
  | DropVar _ -> invalid_arg "Stablehlo.Emit: DropVar in value position"

let emit_eqn ctx (eqn : eqn) (in_names : string list) : unit =
  ignore in_names;
  ignore ctx;
  match eqn.prim with
  | _ ->
      failwith
        "Stablehlo.Emit: no lowering rule for this primitive (rows 76-87)"

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
      let in_names = List.map (ssa_of_atom ctx) e.inputs in
      emit_eqn ctx e in_names)
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
