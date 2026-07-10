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

type ctx = {
  mutable buf : Buffer.t;
  extras : Buffer.t;
  ids : (int, int) Hashtbl.t;
  names : (int, string) Hashtbl.t;
  mutable next : int;
}

let fresh ctx =
  let n = ctx.next in
  ctx.next <- n + 1;
  n

let bind_var ctx (v : var) =
  let n = fresh ctx in
  Hashtbl.replace ctx.ids v.vid n;
  n

let ssa_of_var ctx (v : var) =
  match Hashtbl.find_opt ctx.names v.vid with
  | Some s -> s
  | None -> (
      match Hashtbl.find_opt ctx.ids v.vid with
      | Some n -> "%" ^ string_of_int n
      | None -> invalid_arg "Stablehlo.Emit: unbound variable")

let emit_constant_at ctx n nd =
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %%%d = stablehlo.constant %s : %s\n" n
       (Ir.dense (dense_body nd))
       (Ir.tensor_type (Ndarray.dtype nd) (Ndarray.shape nd)))

let name n = "%" ^ string_of_int n

let capture ctx f =
  let saved = ctx.buf in
  ctx.buf <- Buffer.create 128;
  f ();
  let s = Buffer.contents ctx.buf in
  ctx.buf <- saved;
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
    | Dtype.F32 | Dtype.F64 -> Ir.float_literal dt 1.0
    | Dtype.I32 | Dtype.I64 | Dtype.Uint32 -> Ir.int_literal 1L
    | Dtype.Bool -> Ir.bool_literal true)

let reduce_allones dt =
  Ir.dense
    (match dt with
    | Dtype.I32 | Dtype.I64 -> Ir.int_literal (-1L)
    | Dtype.Uint32 -> Ir.int_literal 4294967295L
    | Dtype.Bool -> Ir.bool_literal true
    | Dtype.F32 | Dtype.F64 ->
        invalid_arg "Stablehlo.Emit: bitwise reduce on floating-point dtype")

let reduce_maxid dt =
  Ir.dense
    (match dt with
    | Dtype.F32 | Dtype.F64 -> Ir.float_literal dt neg_infinity
    | Dtype.I32 -> Ir.int_literal (-2147483648L)
    | Dtype.I64 -> Ir.int_literal Int64.min_int
    | Dtype.Uint32 -> Ir.int_literal 0L
    | Dtype.Bool -> Ir.bool_literal false)

let reduce_minid dt =
  Ir.dense
    (match dt with
    | Dtype.F32 | Dtype.F64 -> Ir.float_literal dt infinity
    | Dtype.I32 -> Ir.int_literal 2147483647L
    | Dtype.I64 -> Ir.int_literal Int64.max_int
    | Dtype.Uint32 -> Ir.int_literal 4294967295L
    | Dtype.Bool -> Ir.bool_literal true)

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
  let fname = if is_max then "argmax" else "argmin" in
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

let emit_cum_single ctx (eqn : eqn) in_ids ~fname ~axis ~reverse ~init
    ~broadcast ~op =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let dt = in_aval.dtype in
  let ty = Ir.tensor_type_of_aval in_aval in
  let sty = Ir.tensor_type dt [||] in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = call @%s(%s) : (%s) -> %s\n" (name n) fname
       (name x) ty ty);
  let attrs = cum_window_attrs in_aval.shape axis reverse in
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
    Buffer.add_string b (Printf.sprintf "    ^bb0(%%4: %s, %%5: %s):\n" sty sty);
    Buffer.add_string b
      (Printf.sprintf "      %%6 = stablehlo.%s %%4, %%5 : %s\n" op sty);
    Buffer.add_string b (Printf.sprintf "      stablehlo.return %%6 : %s\n" sty);
    Buffer.add_string b (Printf.sprintf "    }) : (%s, %s) -> %s\n" ty sty ty);
    Buffer.add_string b (Printf.sprintf "    return %%3 : %s\n" ty))
  else begin
    Buffer.add_string b
      (Printf.sprintf
         "    %%1 = \"stablehlo.reduce_window\"(%%0, %%2) <{%s}> ({\n" attrs);
    Buffer.add_string b (Printf.sprintf "    ^bb0(%%3: %s, %%4: %s):\n" sty sty);
    Buffer.add_string b
      (Printf.sprintf "      %%5 = stablehlo.%s %%3, %%4 : %s\n" op sty);
    Buffer.add_string b (Printf.sprintf "      stablehlo.return %%5 : %s\n" sty);
    Buffer.add_string b (Printf.sprintf "    }) : (%s, %s) -> %s\n" ty sty ty);
    Buffer.add_string b (Printf.sprintf "    return %%1 : %s\n" ty)
  end;
  Buffer.add_string b "  }\n"

let emit_cum_logsumexp ctx (eqn : eqn) in_ids ~axis ~reverse =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let dt = in_aval.dtype in
  let ty = Ir.tensor_type_of_aval in_aval in
  let sty = Ir.tensor_type dt [||] in
  let bty = Ir.tensor_type Dtype.Bool [||] in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = call @cumlogsumexp(%s) : (%s) -> %s\n" (name n)
       (name x) ty ty);
  let attrs = cum_window_attrs in_aval.shape axis reverse in
  let b = ctx.extras in
  Buffer.add_string b
    (Printf.sprintf "  func.func private @cumlogsumexp(%%0: %s) -> %s {\n" ty ty);
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

let emit_scatter ctx (eqn : eqn) in_ids dnums combiner =
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
       "    %s = \"stablehlo.scatter\"(%s, %s, %s) <{indices_are_sorted = \
        false, scatter_dimension_numbers = %s, unique_indices = false}> ({\n"
       (name n) (name op_id) (name idx_id) (name upd_id)
       (scatter_dnums_attr dnums ivd));
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
  Printf.sprintf "%scontracting_dims = %s x %s, precision = [DEFAULT, DEFAULT]"
    batch
    (Ir.int_array_attr dd.lhs_contract)
    (Ir.int_array_attr dd.rhs_contract)

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
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.dot_general %s, %s, %s : (%s, %s) -> %s\n" (name n)
       (name a) (name b) (dot_dims_str dd) la lb
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
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.convolution(%s, %s) dim_numbers = %s, window = \
        {stride = %s, pad = %s, lhs_dilate = %s, rhs_dilate = %s, reverse = \
        %s} {batch_group_count = %d : i64, feature_group_count = %d : i64, \
        precision_config = [#stablehlo<precision DEFAULT>, \
        #stablehlo<precision DEFAULT>]} : (%s, %s) -> %s\n"
       (name n) (name a) (name b) dim_numbers
       (Ir.int_array_attr window_strides)
       (pad_pairs_str padding)
       (Ir.int_array_attr lhs_dilation)
       (Ir.int_array_attr rhs_dilation)
       reverse batch_group_count feature_group_count la lb
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

let emit_sort_comparator ctx dt =
  let sty = Ir.tensor_type dt [||] in
  let bty = Ir.tensor_type Dtype.Bool [||] in
  let x = fresh ctx in
  let y = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    ^bb0(%s: %s, %s: %s):\n" (name x) sty (name y) sty);
  match dt with
  | Dtype.F32 | Dtype.F64 ->
      let canon operand =
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
      in
      let cx = canon x in
      let cy = canon y in
      let p = fresh ctx in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "      %s = stablehlo.compare LT, %s, %s, TOTALORDER : (%s, %s) -> %s\n"
           (name p) (name cx) (name cy) sty sty bty);
      Buffer.add_string ctx.buf
        (Printf.sprintf "      stablehlo.return %s : %s\n" (name p) bty)
  | _ ->
      let ct = compare_type_of dt false in
      let p = fresh ctx in
      Buffer.add_string ctx.buf
        (Printf.sprintf
           "      %s = stablehlo.compare LT, %s, %s, %s : (%s, %s) -> %s\n"
           (name p) (name x) (name y) ct sty sty bty);
      Buffer.add_string ctx.buf
        (Printf.sprintf "      stablehlo.return %s : %s\n" (name p) bty)

let emit_sort ctx (eqn : eqn) in_ids ~dimension ~is_stable =
  let x = sole in_ids in
  let in_aval = atom_aval (sole eqn.inputs) in
  let dt = in_aval.dtype in
  let out = sole eqn.outs in
  let n = fresh ctx in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = \"stablehlo.sort\"(%s) <{dimension = %d : i64, is_stable = \
        %s}> ({\n"
       (name n) (name x) dimension
       (if is_stable then "true" else "false"));
  emit_sort_comparator ctx dt;
  Buffer.add_string ctx.buf
    (Printf.sprintf "    }) : (%s) -> %s\n"
       (Ir.tensor_type_of_aval in_aval)
       (Ir.tensor_type_of_aval out.vaval));
  Hashtbl.replace ctx.ids out.vid n

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

and emit_cond ctx (eqn : eqn) in_ids branch_f branch_t =
  let index_id, operand_ids =
    match in_ids with
    | i :: r -> (i, r)
    | [] -> invalid_arg "Stablehlo.Emit: cond arity"
  in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = \"stablehlo.case\"(%s) ({\n" (name n)
       (name index_id));
  let start = ctx.next in
  let emit_branch (cj : closed_jaxpr) =
    ctx.next <- start;
    map_binders ctx cj.jaxpr operand_ids;
    emit_region_returning ctx cj.jaxpr
  in
  emit_branch branch_f;
  Buffer.add_string ctx.buf "    }, {\n";
  emit_branch branch_t;
  Buffer.add_string ctx.buf
    (Printf.sprintf "    }) : (%s) -> %s\n"
       (Ir.tensor_type_of_aval (atom_aval (List.hd eqn.inputs)))
       (Ir.tensor_type_of_aval out.vaval))

and emit_reduce_general ctx (eqn : eqn) in_ids (reducer : closed_jaxpr) dims =
  let op_id, init_id = pair in_ids in
  let in_aval = atom_aval (List.hd eqn.inputs) in
  let dt = in_aval.dtype in
  let sty = Ir.tensor_type dt [||] in
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  Buffer.add_string ctx.buf
    (Printf.sprintf
       "    %s = stablehlo.reduce(%s init: %s) across dimensions = %s : (%s, \
        %s) -> %s\n"
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
    (Printf.sprintf "     reducer(%s: %s, %s: %s)  {\n" (name ba) sty (name bb)
       sty);
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

and emit_while ctx (eqn : eqn) in_ids (cond : closed_jaxpr)
    (body : closed_jaxpr) =
  let out = sole eqn.outs in
  let n = bind_var ctx out in
  let carry_ids = List.map (fun _ -> fresh ctx) in_ids in
  let carry_tys =
    List.map (fun (v : var) -> Ir.tensor_type_of_aval v.vaval) eqn.outs
  in
  let header_pairs =
    List.map2
      (fun ba init -> Printf.sprintf "%s = %s" (name ba) (name init))
      carry_ids in_ids
  in
  Buffer.add_string ctx.buf
    (Printf.sprintf "    %s = stablehlo.while(%s) : %s\n" (name n)
       (String.concat ", " header_pairs)
       (String.concat ", " carry_tys));
  let start = ctx.next in
  map_binders ctx cond.jaxpr carry_ids;
  Buffer.add_string ctx.buf "    cond {\n";
  emit_region_returning ctx cond.jaxpr;
  Buffer.add_string ctx.buf "    } do {\n";
  ctx.next <- start;
  map_binders ctx body.jaxpr carry_ids;
  emit_region_returning ctx body.jaxpr;
  Buffer.add_string ctx.buf "    }\n"

and emit_eqn ctx (eqn : eqn) (in_ids : int list) : unit =
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
  | Reduce_sum axes -> emit_reduce ctx eqn in_ids axes "add" reduce_zero
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
  | Scatter { dimension_numbers; _ } ->
      emit_scatter ctx eqn in_ids dimension_numbers None
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
  | Dot_general dd -> emit_dot_general ctx eqn in_ids dd
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
  | Sort { dimension; is_stable; _ } ->
      emit_sort ctx eqn in_ids ~dimension ~is_stable
  | Top_k { k; _ } -> emit_top_k ctx eqn in_ids k
  | Bessel_i1e -> emit_chlo_unary ctx eqn in_ids "bessel_i1e"
  | Digamma -> emit_chlo_unary ctx eqn in_ids "digamma"
  | Erf -> emit_chlo_unary ctx eqn in_ids "erf"
  | Erf_inv -> emit_chlo_unary ctx eqn in_ids "erf_inv"
  | Erfc -> emit_chlo_unary ctx eqn in_ids "erfc"
  | Lgamma -> emit_chlo_unary ctx eqn in_ids "lgamma"
  | Polygamma -> emit_chlo_binary ctx eqn in_ids "polygamma"
  | Zeta -> emit_chlo_binary ctx eqn in_ids "zeta"
  | Bessel_i0e ->
      failwith
        "Stablehlo.Emit: Bessel_i0e has no native chlo op; jax lowers it via a \
         lower_fun Cephes decomposition into many stablehlo ops \
         (non-single-op), end-to-end decomposition deferred"
  | Igamma ->
      failwith
        "Stablehlo.Emit: Igamma has no native chlo op; jax lowers it via a \
         lower_fun decomposition into many stablehlo ops (non-single-op), \
         end-to-end decomposition deferred"
  | Igamma_grad_a ->
      failwith
        "Stablehlo.Emit: Igamma_grad_a has no native chlo op; jax lowers it \
         via a lower_fun decomposition into many stablehlo ops \
         (non-single-op), end-to-end decomposition deferred"
  | Igammac ->
      failwith
        "Stablehlo.Emit: Igammac has no native chlo op; jax lowers it via a \
         lower_fun decomposition into many stablehlo ops (non-single-op), \
         end-to-end decomposition deferred"
  | Regularized_incomplete_beta ->
      failwith
        "Stablehlo.Emit: Regularized_incomplete_beta has no native chlo op; \
         jax lowers it via a lower_fun decomposition into many stablehlo ops \
         (non-single-op), end-to-end decomposition deferred"
  | Rng_uniform -> emit_rng_uniform ctx eqn in_ids
  | Rng_bit_generator ->
      failwith
        "Stablehlo.Emit: Rng_bit_generator lowers via a u32[4] -> u64[2] \
         bitcast_convert into stablehlo.rng_bit_generator; uint64 is not in \
         the M4 dtype set (F32/F64/I32/I64/Bool/Uint32) and Ojax has no \
         abstract_eval for it, so it is non-representable (M5)"
  | Threefry2x32 ->
      failwith
        "Stablehlo.Emit: Threefry2x32 lowers on cpu to a stablehlo.while \
         rolled loop (unrolled elsewhere) with private subfunctions \
         (non-single-op decomposition); end-to-end deferred to the PJRT \
         execution rows"
  | Iota_2x32_shape _ ->
      failwith
        "Stablehlo.Emit: Iota_2x32_shape lowers via a uint64 iota + \
         shift_right_logical + convert decomposition; uint64 is not in the M4 \
         dtype set, non-representable (M5)"
  | Random_bits _ ->
      failwith
        "Stablehlo.Emit: Random_bits lowers via mlir.lower_fun into the full \
         threefry bit-generation decomposition (many stablehlo ops); \
         non-single-op, end-to-end deferred to the PJRT execution rows"
  | Random_seed ->
      failwith
        "Stablehlo.Emit: Random_seed lowers via mlir.lower_fun into a \
         shift/convert/and/concatenate decomposition plus a \
         sdy.sharding_constraint; sharding machinery is out of scope \
         (STRIP-ON-PORT), non-representable"
  | Random_split _ ->
      failwith
        "Stablehlo.Emit: Random_split lowers via mlir.lower_fun into the \
         threefry fold/split decomposition (many stablehlo ops); \
         non-single-op, end-to-end deferred to the PJRT execution rows"
  | Random_fold_in ->
      failwith
        "Stablehlo.Emit: Random_fold_in lowers via mlir.lower_fun into the \
         threefry fold-in decomposition (many stablehlo ops); non-single-op, \
         end-to-end deferred to the PJRT execution rows"
  | Random_wrap ->
      failwith
        "Stablehlo.Emit: Random_wrap lowers to a sdy.sharding_constraint over \
         the physical key array; sharding machinery is out of scope \
         (STRIP-ON-PORT), non-representable"
  | Random_unwrap ->
      failwith
        "Stablehlo.Emit: Random_unwrap lowers to a sdy.sharding_constraint \
         over the physical key array; sharding machinery is out of scope \
         (STRIP-ON-PORT), non-representable"
  | Cond { t; f } -> emit_cond ctx eqn in_ids f t
  | Reduce { jaxpr; dimensions } ->
      emit_reduce_general ctx eqn in_ids jaxpr dimensions
  | Reduce_window { reducer; window } ->
      emit_reduce_window_general ctx eqn in_ids reducer window
  | While { cond; body } -> emit_while ctx eqn in_ids cond body
  | Scan _ ->
      failwith
        "Stablehlo.Emit: Scan lowers to a counted stablehlo.while over sliced \
         xs plus three private helper functions (@dynamic_index_in_dim, \
         @closed_call, @dynamic_update_index_in_dim); non-single-op \
         decomposition, end-to-end deferred to the PJRT execution rows"
  | Select_and_scatter _ ->
      failwith
        "Stablehlo.Emit: general Select_and_scatter lowers on cpu to a pad + \
         stablehlo.select_and_scatter (two arbitrary regions) + slice \
         decomposition; the common add-variant is goldened at row 84, the \
         general dual-region form is deferred to the PJRT execution rows"
  | Composite _ ->
      failwith
        "Stablehlo.Emit: Composite lowers to a stablehlo.composite op that \
         references a separate decomposition func by symbol; the \
         decomposition-symbol machinery is out of scope for this row, deferred"
  | Custom_linear_solve _ ->
      failwith
        "Stablehlo.Emit: Custom_linear_solve lowers to an iterative \
         while/solve decomposition (non-single-op); end-to-end deferred to the \
         PJRT execution rows"
  | Xla_call _ ->
      failwith
        "Stablehlo.Emit: a nested Xla_call must be inlined by the caller; the \
         top-level compiled jaxpr body is @main (emitter core row 75), an \
         eqn-level Xla_call is non-representable here"

let emit_closed_jaxpr (cj : closed_jaxpr) : string =
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
    }
  in
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
  Printf.sprintf
    "module {\n  func.func public @main(%s) -> (%s) {\n%s  }\n%s}\n"
    (String.concat ", " params)
    (String.concat ", " out_types)
    (Buffer.contents ctx.buf)
    (Buffer.contents ctx.extras)
