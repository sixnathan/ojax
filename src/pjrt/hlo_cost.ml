type shape = {
  ty : int;
  dims : int array;
  dynamic : bool;
  tuple : shape array;
  minor_to_major : int array;
  has_layout : bool;
  element_bits : int;
}

type window_dim = {
  size : int;
  stride : int;
  pad_low : int;
  pad_high : int;
  window_dilation : int;
  base_dilation : int;
}

type conv_dnums = {
  input_batch : int;
  input_feature : int;
  input_spatial : int array;
  kernel_spatial : int array;
  output_feature : int;
  output_spatial : int array;
}

type instr = {
  id : int;
  opcode : string;
  shape : shape;
  operands : int array;
  called : int array;
  control_preds : int array;
  dimensions : int array;
  tuple_index : int;
  param_no : int;
  fusion_kind : string;
  window : window_dim array;
  lhs_contracting : int array;
  rhs_group : int array;
  conv : conv_dnums option;
  feature_group_count : int;
  batch_group_count : int;
  fft_length : int array;
}

type comp = {
  cid : int;
  instrs : instr array;
  by_id : (int, instr) Hashtbl.t;
  users : (int, instr list) Hashtbl.t;
  root : instr;
}

type hlo_module = { comps : (int, comp) Hashtbl.t; entry : comp }

let read_varint text pos =
  let rec go shift acc =
    let byte = Char.code text.[!pos] in
    incr pos;
    let acc = acc lor ((byte land 0x7f) lsl shift) in
    if byte land 0x80 = 0 then acc else go (shift + 7) acc
  in
  go 0 0

let fields text =
  let pos = ref 0 in
  let out = ref [] in
  while !pos < String.length text do
    let tag = read_varint text pos in
    let field = tag lsr 3 in
    match tag land 7 with
    | 0 -> out := (field, `Int (read_varint text pos)) :: !out
    | 1 -> pos := !pos + 8
    | 2 ->
        let n = read_varint text pos in
        out := (field, `Bytes (String.sub text !pos n)) :: !out;
        pos := !pos + n
    | 5 -> pos := !pos + 4
    | wire -> failwith (Printf.sprintf "Hlo_cost: proto wire type %d" wire)
  done;
  List.rev !out

let ints field fs =
  List.concat_map
    (fun (f, v) ->
      if f <> field then []
      else
        match v with
        | `Int i -> [ i ]
        | `Bytes b ->
            let pos = ref 0 in
            let acc = ref [] in
            while !pos < String.length b do
              acc := read_varint b pos :: !acc
            done;
            List.rev !acc)
    fs
  |> Array.of_list

let int_field field default fs =
  match List.assoc_opt field fs with Some (`Int i) -> i | _ -> default

let bytes_field field fs =
  match List.assoc_opt field fs with Some (`Bytes b) -> Some b | _ -> None

let string_field field fs = Option.value ~default:"" (bytes_field field fs)

let subs field fs =
  List.filter_map
    (fun (f, v) ->
      match v with `Bytes b when f = field -> Some (fields b) | _ -> None)
    fs

let rec decode_shape fs =
  let layout = bytes_field 5 fs |> Option.map fields in
  {
    ty = int_field 2 0 fs;
    dims = ints 3 fs;
    dynamic = Array.exists (fun b -> b <> 0) (ints 6 fs);
    tuple = Array.of_list (List.map decode_shape (subs 4 fs));
    minor_to_major = (match layout with Some l -> ints 1 l | None -> [||]);
    has_layout = Option.is_some layout;
    element_bits = (match layout with Some l -> int_field 7 0 l | None -> 0);
  }

let decode_window fs =
  Array.of_list
    (List.map
       (fun d ->
         {
           size = int_field 1 0 d;
           stride = int_field 2 0 d;
           pad_low = int_field 3 0 d;
           pad_high = int_field 4 0 d;
           window_dilation = int_field 5 0 d;
           base_dilation = int_field 6 0 d;
         })
       (subs 1 fs))

let decode_instr fs =
  let sub field = Option.map fields (bytes_field field fs) in
  let dot = sub 30 in
  let ragged = sub 90 in
  {
    id = int_field 35 0 fs;
    opcode = string_field 2 fs;
    shape = decode_shape (Option.value ~default:[] (sub 3));
    operands = ints 36 fs;
    called = ints 38 fs;
    control_preds = ints 37 fs;
    dimensions = ints 14 fs;
    tuple_index = int_field 13 0 fs;
    param_no = int_field 9 0 fs;
    fusion_kind = string_field 11 fs;
    window = (match sub 15 with Some w -> decode_window w | None -> [||]);
    lhs_contracting =
      (match (dot, ragged) with
      | Some d, _ -> ints 1 d
      | None, Some r -> (
          match bytes_field 1 r with Some d -> ints 1 (fields d) | None -> [||])
      | None, None -> [||]);
    rhs_group = (match ragged with Some r -> ints 3 r | None -> [||]);
    conv =
      Option.map
        (fun c ->
          {
            input_batch = int_field 7 0 c;
            input_feature = int_field 8 0 c;
            input_spatial = ints 11 c;
            kernel_spatial = ints 6 c;
            output_feature = int_field 10 0 c;
            output_spatial = ints 12 c;
          })
        (sub 16);
    feature_group_count = int_field 50 1 fs;
    batch_group_count = int_field 58 1 fs;
    fft_length = ints 32 fs;
  }

let decode_comp fs =
  let instrs = Array.of_list (List.map decode_instr (subs 2 fs)) in
  let by_id = Hashtbl.create 16 in
  Array.iter (fun i -> Hashtbl.replace by_id i.id i) instrs;
  let users = Hashtbl.create 16 in
  Array.iter
    (fun i ->
      let seen = Hashtbl.create 4 in
      Array.iter
        (fun o ->
          if not (Hashtbl.mem seen o) then (
            Hashtbl.replace seen o ();
            let prev = Option.value ~default:[] (Hashtbl.find_opt users o) in
            Hashtbl.replace users o (i :: prev)))
        i.operands)
    instrs;
  Hashtbl.filter_map_inplace (fun _ l -> Some (List.rev l)) users;
  let root_id = int_field 6 0 fs in
  {
    cid = int_field 5 0 fs;
    instrs;
    by_id;
    users;
    root = Hashtbl.find by_id root_id;
  }

let decode_module text =
  let fs = fields text in
  let comps = Hashtbl.create 8 in
  List.iter
    (fun c ->
      let c = decode_comp c in
      Hashtbl.replace comps c.cid c)
    (subs 3 fs);
  { comps; entry = Hashtbl.find comps (int_field 6 0 fs) }

let f32 x = Int32.float_of_bits (Int32.bits_of_float x)
let ( +. ) a b = f32 (Stdlib.( +. ) a b)
let ( *. ) a b = f32 (Stdlib.( *. ) a b)
let fi n = f32 (float_of_int n)
let is_array s = s.ty <> 13 && s.ty <> 14 && s.ty <> 17
let elements (s : shape) = Array.fold_left ( * ) 1 s.dims

let byte_width = function
  | 1 | 2 | 6 | 19 | 20 | 21 | 22 | 23 | 24 | 25 | 26 | 27 | 28 | 29 | 30 | 31
  | 32 | 33 ->
      1
  | 3 | 7 | 10 | 16 -> 2
  | 4 | 8 | 11 -> 4
  | 5 | 9 | 12 | 15 -> 8
  | 18 -> 16
  | _ -> 0

let rec has_layout s =
  if s.ty = 13 then Array.for_all has_layout s.tuple
  else if is_array s then s.has_layout
  else true

let shape_size s =
  if not (has_layout s) then 0
  else if s.ty = 14 then 8
  else if s.ty = 13 then 8 * Array.length s.tuple
  else if s.ty = 17 then 0
  else
    let body =
      if s.element_bits <> 0 then ((elements s * s.element_bits) + 7) / 8
      else elements s * byte_width s.ty
    in
    if s.dynamic then body + (4 * Array.length s.dims) else body

let index_string idx =
  "{" ^ String.concat "," (List.map string_of_int idx) ^ "}"

let bytes_key = "bytes accessed"
let util_key = "utilization"
let operand_bytes_key i idx = bytes_key ^ string_of_int i ^ index_string idx
let operand_util_key i idx = util_key ^ string_of_int i ^ index_string idx
let output_key idx = bytes_key ^ "out" ^ index_string idx

type props = (string, float) Hashtbl.t

let get (p : props) k = Option.value ~default:0.0 (Hashtbl.find_opt p k)
let set (p : props) k v = Hashtbl.replace p k (f32 v)
let add (p : props) k v = set p k (get p k +. v)

let rec for_each_leaf s idx f =
  if s.ty = 13 then
    Array.iteri (fun i t -> for_each_leaf t (idx @ [ i ]) f) s.tuple
  else f s idx

let rec for_each_subshape s idx f =
  f s idx;
  if s.ty = 13 then
    Array.iteri (fun i t -> for_each_subshape t (idx @ [ i ]) f) s.tuple

let transcendental = function
  | "acos" | "acosh" | "asin" | "asinh" | "atan2" | "atanh" | "cbrt" | "cosine"
  | "cosh" | "erf" | "exponential" | "exponential-minus-one" | "log"
  | "log-plus-one" | "logistic" | "power" | "rsqrt" | "sine" | "sinh" | "sqrt"
  | "tan" | "tanh" ->
      true
  | _ -> false

let elementwise = function
  | "abs" | "round-nearest-afz" | "round-nearest-even" | "ceil"
  | "count-leading-zeros" | "floor" | "imag" | "is-finite" | "not" | "negate"
  | "popcnt" | "real" | "sign" | "bitcast-convert" | "convert"
  | "stochastic-convert" | "add" | "complex" | "divide" | "maximum" | "minimum"
  | "multiply" | "remainder" | "subtract" | "and" | "or" | "xor" | "shift-left"
  | "shift-right-arithmetic" | "shift-right-logical" | "compare" | "clamp"
  | "select" | "reduce-precision" ->
      true
  | op -> transcendental op

let no_op = function
  | "iota" | "reverse" | "concatenate" | "copy" | "pad" | "reshape"
  | "dynamic-reshape" | "batch-norm-training" | "batch-norm-inference"
  | "batch-norm-grad" | "async-update" | "async-done" | "copy-start"
  | "copy-done" | "send" | "send-done" | "recv" | "recv-done"
  | "optimization-barrier" | "all-gather" | "all-gather-start"
  | "all-gather-done" | "all-reduce-done" | "reduce-scatter" | "all-to-all"
  | "ragged-all-to-all" | "collective-broadcast" | "collective-permute"
  | "collective-permute-start" | "collective-permute-done" | "partition-id"
  | "replica-id" | "rng-get-and-update-state" | "topk" | "get-dimension-size"
  | "set-dimension-size" ->
      true
  | _ -> false

let key_to_copy k =
  not
    (String.starts_with ~prefix:bytes_key k
    || String.starts_with ~prefix:util_key k)

let log2_floor n =
  let rec go n acc = if n <= 1 then acc else go (n lsr 1) (acc + 1) in
  if n <= 0 then -1 else go n 0

let log2_ceiling n =
  let f = log2_floor n in
  if n > 0 && 1 lsl f = n then f else f + 1

let conv_flops ins lhs rhs result =
  let d = Option.get ins.conv in
  let input_feature = lhs.dims.(d.input_feature) in
  let output_feature = result.dims.(d.output_feature) in
  let batch = lhs.dims.(d.input_batch) in
  let window, kernel_limits, output_limits, input_limits =
    if Array.length ins.window = 0 then
      ( [|
          {
            size = 1;
            stride = 1;
            pad_low = 0;
            pad_high = 0;
            window_dilation = 1;
            base_dilation = 1;
          };
        |],
        [| 1 |],
        [| 1 |],
        [| 1 |] )
    else
      ( ins.window,
        Array.mapi (fun i _ -> rhs.dims.(d.kernel_spatial.(i))) ins.window,
        Array.mapi (fun i _ -> result.dims.(d.output_spatial.(i))) ins.window,
        Array.mapi (fun i _ -> lhs.dims.(d.input_spatial.(i))) ins.window )
  in
  let counts =
    Array.mapi
      (fun s w ->
        let il = input_limits.(s)
        and kl = kernel_limits.(s)
        and ol = output_limits.(s) in
        if
          il = ol && kl = ol && il = w.base_dilation && w.window_dilation = 1
          && max 1 (il - 1) = w.stride
          && w.pad_low = 0 && w.pad_high = 0
        then il
        else if
          il = 1 && kl = ol && w.window_dilation = 1 && w.base_dilation = 1
          && w.stride = 1
          && w.pad_high = ol - 1
          && w.pad_low = ol - 1
        then ol
        else
          let count = ref 0 in
          for k = 0 to kl - 1 do
            if w.stride = 1 && w.base_dilation = 1 then
              let base = w.pad_low - (k * w.window_dilation) in
              count := !count + max (min (il + base) ol - max base 0) 0
            else
              for o = 0 to ol - 1 do
                let undilated =
                  (o * w.stride) - w.pad_low + (k * w.window_dilation)
                in
                let idx =
                  if w.base_dilation > 1 then undilated / w.base_dilation
                  else undilated
                in
                if undilated = idx * w.base_dilation && idx >= 0 && idx < il
                then incr count
              done
          done;
          !count)
      window
  in
  let product = Array.fold_left ( * ) 1 counts in
  2
  * (input_feature / ins.feature_group_count)
  * output_feature
  * (batch / ins.batch_group_count)
  * product

let transpose_is_bitcast input output mapping =
  input.ty = output.ty
  && Array.length output.minor_to_major = Array.length mapping
  && Array.map (fun i -> mapping.(i)) output.minor_to_major
     = input.minor_to_major

type state = { m : hlo_module; hlo_props : (int * int, props) Hashtbl.t }

let hlo_prop st comp ins =
  let key = (comp.cid, ins.id) in
  match Hashtbl.find_opt st.hlo_props key with
  | Some p -> p
  | None ->
      let p = Hashtbl.create 4 in
      Hashtbl.replace st.hlo_props key p;
      p

let nonzero_iter (p : props) f =
  Hashtbl.iter (fun k v -> if v <> 0.0 then f k v) p

let rec analyze st comp =
  let sum : props = Hashtbl.create 16 in
  let visited = Hashtbl.create 16 in
  let operand ins i = Hashtbl.find comp.by_id ins.operands.(i) in
  let users_of ins =
    Option.value ~default:[] (Hashtbl.find_opt comp.users ins.id)
  in
  let rec visit ins =
    if not (Hashtbl.mem visited ins.id) then (
      Hashtbl.replace visited ins.id ();
      Array.iter (fun o -> visit (Hashtbl.find comp.by_id o)) ins.operands;
      Array.iter (fun o -> visit (Hashtbl.find comp.by_id o)) ins.control_preds;
      let cur = visit_one ins in
      nonzero_iter cur (fun k v -> add sum k v);
      Hashtbl.replace st.hlo_props (comp.cid, ins.id) cur)
  and visit_one ins =
    let cur : props = Hashtbl.create 8 in
    let ba = ref 0.0 in
    for_each_leaf ins.shape [] (fun s _ -> ba := !ba +. fi (shape_size s));
    set cur (output_key []) !ba;
    Array.iteri
      (fun i _ ->
        let sz = shape_size (operand ins i).shape in
        ba := !ba +. fi sz;
        set cur (operand_bytes_key i []) (fi sz);
        set cur (operand_util_key i []) 1.0)
      ins.operands;
    set cur bytes_key !ba;
    let zero_all () =
      set cur bytes_key 0.0;
      set cur (output_key []) 0.0
    in
    let sub_copy c scale_ =
      let sp = analyze st (Hashtbl.find st.m.comps c) in
      nonzero_iter sp (fun k v -> if key_to_copy k then set cur k (v *. scale_))
    in
    let op = ins.opcode in
    (if elementwise op then
       let n = fi (elements ins.shape) in
       if transcendental op then set cur "transcendentals" n
       else set cur "flops" n
     else if no_op op then ()
     else
       match op with
       | "parameter" | "constant" -> zero_all ()
       | "map" -> sub_copy ins.called.(0) (fi (elements ins.shape))
       | "scan" -> sub_copy ins.called.(0) (fi (elements (operand ins 1).shape))
       | "domain" ->
           zero_all ();
           Array.iteri
             (fun i _ -> set cur (operand_bytes_key i []) 0.0)
             ins.operands
       | "get-tuple-element" ->
           zero_all ();
           set cur (operand_bytes_key 0 []) 0.0
       | "slice" ->
           let out = shape_size ins.shape in
           set cur bytes_key (fi (out * 2));
           set cur (output_key []) (fi out);
           set cur (operand_bytes_key 0 []) (fi out);
           set cur (operand_util_key 0 [])
             (float_of_int (elements ins.shape)
             /. float_of_int (elements (operand ins 0).shape))
       | "dynamic-slice" ->
           let out = shape_size ins.shape in
           let idx = shape_size (operand ins 1).shape in
           set cur bytes_key (fi ((out * 2) + idx));
           set cur (output_key []) (fi out);
           set cur (operand_bytes_key 0 []) (fi out);
           set cur (operand_bytes_key 1 []) (fi idx);
           set cur (operand_util_key 0 [])
             (float_of_int (elements ins.shape)
             /. float_of_int (elements (operand ins 0).shape))
       | "dynamic-update-slice" ->
           let upd = shape_size (operand ins 1).shape in
           let idx = shape_size (operand ins 2).shape in
           let nu = elements (operand ins 1).shape in
           let no = elements ins.shape in
           set cur bytes_key (fi ((upd * 2) + idx));
           set cur (output_key []) (fi upd);
           set cur (operand_bytes_key 0 []) 0.0;
           set cur (operand_bytes_key 1 []) (fi upd);
           set cur (operand_bytes_key 2 []) (fi idx);
           set cur (operand_util_key 0 [])
             (float_of_int (no - nu) /. float_of_int no)
       | "tuple" ->
           let sz = fi (shape_size ins.shape) in
           set cur bytes_key sz;
           set cur (output_key []) sz;
           Array.iteri
             (fun i _ -> set cur (operand_bytes_key i []) 0.0)
             ins.operands
       | "after-all" | "add-dependency" ->
           zero_all ();
           Array.iteri
             (fun i _ -> set cur (operand_bytes_key i []) 0.0)
             ins.operands
       | "dot" | "scaled-dot" ->
           let lhs = (operand ins 0).shape in
           let width =
             Array.fold_left (fun a d -> a * lhs.dims.(d)) 1 ins.lhs_contracting
           in
           set cur "flops" (fi (2 * elements ins.shape * width))
       | "ragged-dot" ->
           let lhs = (operand ins 0).shape in
           let width =
             Array.fold_left (fun a d -> a * lhs.dims.(d)) 1 ins.lhs_contracting
           in
           let dims = ref (Array.to_list ins.shape.dims) in
           Array.iteri
             (fun i _ -> dims := List.filteri (fun j _ -> j <> i) !dims)
             ins.rhs_group;
           let n = List.fold_left ( * ) 1 !dims in
           set cur "flops" (fi (2 * n * width))
       | "convolution" ->
           set cur "flops"
             (fi
                (conv_flops ins (operand ins 0).shape (operand ins 1).shape
                   ins.shape))
       | "fft" ->
           let real =
             let s = (operand ins 0).shape in
             if s.ty = 13 then s.tuple.(0) else s
           in
           let lf =
             Array.fold_left (fun a d -> a * log2_floor d) 1 ins.fft_length
           in
           set cur "flops" (fi (2 * 4 * lf * elements real))
       | "triangular-solve" ->
           let out = fi (shape_size ins.shape) in
           let a = (operand ins 0).shape and b = (operand ins 1).shape in
           let half = f32 (float_of_int (shape_size a) /. 2.0) in
           set cur (output_key []) out;
           set cur (operand_bytes_key 0 []) (fi (shape_size b));
           set cur bytes_key (out +. half +. fi (shape_size b));
           let n = a.dims.(Array.length a.dims - 1) * elements b in
           set cur "flops" (fi (2 * n))
       | "cholesky" ->
           let a = (operand ins 0).shape in
           let half = f32 (float_of_int (shape_size a) /. 2.0) in
           set cur (output_key []) half;
           set cur (operand_bytes_key 0 []) half;
           set cur bytes_key (half +. half);
           let n = a.dims.(Array.length a.dims - 1) * elements a in
           set cur "flops" (fi (n / 3))
       | "all-reduce" | "all-reduce-start" ->
           let flops = ref 0 and out = ref 0 in
           for_each_subshape ins.shape [] (fun s _ ->
               if is_array s then (
                 flops := !flops + elements s;
                 out := !out + shape_size s));
           let total =
             Array.fold_left
               (fun a o -> a + shape_size (Hashtbl.find comp.by_id o).shape)
               !out ins.operands
           in
           set cur "flops" (fi !flops);
           set cur (output_key []) (fi !out);
           set cur bytes_key (fi total)
       | "rng" -> set cur "transcendentals" (fi (elements ins.shape))
       | "rng-bit-generator" ->
           let n = ref 0 in
           for_each_leaf ins.shape [] (fun s _ ->
               if is_array s then n := !n + elements s);
           set cur "transcendentals" (fi !n)
       | "bitcast" ->
           zero_all ();
           set cur (operand_bytes_key 0 []) 0.0
       | "transpose" ->
           if
             transpose_is_bitcast (operand ins 0).shape ins.shape ins.dimensions
           then (
             zero_all ();
             set cur (operand_bytes_key 0 []) 0.0)
       | "broadcast" -> ()
       | "reduce" ->
           let arg = (operand ins 0).shape in
           let out =
             if is_array ins.shape then ins.shape else ins.shape.tuple.(0)
           in
           sub_copy ins.called.(0) (fi (elements arg - elements out))
       | "reduce-window" ->
           let w = ins.window in
           let wcount = ref (Array.fold_left (fun a d -> a * d.size) 1 w) in
           let out =
             elements
               (if is_array ins.shape then ins.shape else ins.shape.tuple.(0))
           in
           let count = ref ((!wcount - 1) * out) in
           let nred =
             Array.fold_left (fun a d -> if d.size <> 1 then a + 1 else a) 0 w
           in
           let npad =
             Array.fold_left
               (fun a d ->
                 if d.pad_low <> 0 || d.pad_high <> 0 then a + 1 else a)
               0 w
           in
           if nred = 1 && npad = 1 && is_array ins.shape then (
             let found = ref None in
             Array.iteri
               (fun i d ->
                 if
                   !found = None && d.size <> 1 && d.pad_low <> 0
                   && d.pad_high <> 0 && d.pad_low = d.pad_high
                   && d.size = (2 * d.pad_low) + 1
                 then found := Some (i, d))
               w;
             match !found with
             | Some (i, d) when d.pad_low = ins.shape.dims.(i) - 1 ->
                 wcount := ins.shape.dims.(i);
                 count := (out / !wcount) + (!wcount - 1)
             | _ -> ());
           sub_copy ins.called.(0) (fi !count)
       | "select-and-scatter" ->
           let src = elements (operand ins 1).shape in
           let wcount = Array.fold_left (fun a d -> a * d.size) 1 ins.window in
           let sel = analyze st (Hashtbl.find st.m.comps ins.called.(0)) in
           let sca = analyze st (Hashtbl.find st.m.comps ins.called.(1)) in
           nonzero_iter sel (fun k v ->
               if key_to_copy k then add cur k (v *. fi (src * (wcount - 1))));
           nonzero_iter sca (fun k v ->
               if key_to_copy k then add cur k (v *. fi src))
       | "fusion" -> fusion cur ins
       | "call" ->
           let sp = analyze st (Hashtbl.find st.m.comps ins.called.(0)) in
           Hashtbl.reset cur;
           Hashtbl.iter (fun k v -> Hashtbl.replace cur k v) sp
       | "async-start" ->
           let sp = analyze st (Hashtbl.find st.m.comps ins.called.(0)) in
           Hashtbl.reset cur;
           Hashtbl.iter (fun k v -> Hashtbl.replace cur k v) sp
       | "custom-call" ->
           set cur "optimal_seconds" (-1.0);
           set cur bytes_key (-1.0);
           set cur (output_key []) (-1.0);
           Array.iteri
             (fun i _ -> set cur (operand_bytes_key i []) (-1.0))
             ins.operands;
           set cur "flops" (-1.0)
       | "sort" ->
           let n = elements (operand ins 0).shape in
           set cur "flops" (fi (n * log2_ceiling n))
       | "while" ->
           let props =
             Array.to_list ins.called |> List.rev
             |> List.map (fun c -> analyze st (Hashtbl.find st.m.comps c))
           in
           Hashtbl.reset cur;
           List.iter (fun p -> nonzero_iter p (fun k v -> add cur k v)) props
       | "conditional" ->
           let first = analyze st (Hashtbl.find st.m.comps ins.called.(0)) in
           Hashtbl.reset cur;
           Hashtbl.iter (fun k v -> Hashtbl.replace cur k v) first;
           for j = 1 to Array.length ins.called - 1 do
             let p = analyze st (Hashtbl.find st.m.comps ins.called.(j)) in
             nonzero_iter p (fun k v -> set cur k (Float.max (get cur k) v))
           done
       | "gather" -> gather cur ins
       | "scatter" -> scatter cur ins
       | "infeed" ->
           let size = ref 0 in
           for_each_leaf ins.shape [] (fun s idx ->
               size := !size + shape_size s;
               set cur (output_key idx) (fi (shape_size s)));
           set cur (output_key []) (fi !size);
           set cur bytes_key (fi !size)
       | "outfeed" ->
           set cur bytes_key 0.0;
           Array.iteri
             (fun i _ ->
               let size = ref 0 in
               for_each_leaf (operand ins i).shape [] (fun s idx ->
                   size := !size + shape_size s;
                   set cur (operand_bytes_key i idx) (fi (shape_size s)));
               set cur (operand_bytes_key i []) (fi !size);
               add cur bytes_key (fi !size))
             ins.operands
       | other -> failwith ("Hlo_cost: unsupported opcode " ^ other));
    cur
  and gather cur ins =
    let out = shape_size ins.shape in
    let idx = shape_size (operand ins 1).shape in
    set cur bytes_key (fi ((out * 2) + idx));
    set cur (operand_bytes_key 0 []) (fi out);
    set cur (operand_bytes_key 1 []) (fi idx);
    set cur (operand_util_key 0 [])
      (float_of_int (elements ins.shape)
      /. float_of_int (elements (operand ins 0).shape));
    set cur (output_key []) (fi out)
  and scatter cur ins =
    let n = Array.length ins.operands / 2 in
    let total = ref 0 in
    for i = 0 to n - 1 do
      let upd = shape_size (operand ins (n + 1 + i)).shape in
      set cur (operand_bytes_key i []) (fi upd);
      set cur (operand_bytes_key (n + 1 + i) []) (fi upd);
      total := !total + upd
    done;
    let idx = shape_size (operand ins n).shape in
    set cur (operand_bytes_key n []) (fi idx);
    set cur bytes_key (fi ((!total * 3) + idx));
    set cur (output_key []) (fi !total);
    let count = elements (operand ins (n + 1)).shape in
    let sp = analyze st (Hashtbl.find st.m.comps ins.called.(0)) in
    nonzero_iter sp (fun k v -> if key_to_copy k then set cur k (v *. fi count))
  and fusion cur ins =
    let fc = Hashtbl.find st.m.comps ins.called.(0) in
    let custom_hit =
      if ins.fusion_kind = "kCustom" then
        Array.to_list fc.instrs
        |> List.find_opt (fun i -> i.opcode = "gather" || i.opcode = "scatter")
      else None
    in
    match custom_hit with
    | Some _ -> failwith "Hlo_cost: custom fusion with gather or scatter"
    | None ->
        let sp = analyze st fc in
        Hashtbl.reset cur;
        Hashtbl.iter (fun k v -> Hashtbl.replace cur k v) sp;
        set cur bytes_key 0.0;
        let root = fc.root in
        for_each_subshape ins.shape [] (fun sub idx ->
            if is_array sub then begin
              let r =
                if List.length idx = 1 && root.opcode = "tuple" then
                  Hashtbl.find fc.by_id root.operands.(List.hd idx)
                else root
              in
              if r.opcode = "dynamic-update-slice" then (
                let size =
                  shape_size (Hashtbl.find fc.by_id r.operands.(1)).shape
                in
                add cur bytes_key (fi size);
                set cur (output_key idx) (fi size);
                set (hlo_prop st fc r) (operand_util_key 0 []) 0.0)
              else (
                add cur bytes_key (fi (shape_size sub));
                set cur (output_key idx) (fi (shape_size sub)))
            end);
        if ins.shape.ty = 13 then begin
          let rec prop s idx =
            let b = get cur (output_key idx) in
            if b <> 0.0 then b
            else if s.ty <> 13 then b
            else
              let acc = ref b in
              Array.iteri
                (fun i t ->
                  acc := !acc +. prop t (idx @ [ i ]);
                  set cur (output_key idx) !acc)
                s.tuple;
              !acc
          in
          set cur (output_key []) 0.0;
          ignore (prop ins.shape [])
        end;
        Array.iter (fun i -> set (hlo_prop st fc i) util_key 1.0) fc.instrs;
        Array.iter
          (fun i ->
            if i.opcode = "constant" && elements i.shape > 1 then
              let u = Float.min (get (hlo_prop st fc i) util_key) 1.0 in
              add cur bytes_key (fi (shape_size i.shape) *. u))
          fc.instrs;
        let params =
          Array.to_list fc.instrs
          |> List.filter (fun i -> i.opcode = "parameter")
          |> List.sort (fun a b -> compare (param_number a) (param_number b))
        in
        List.iteri
          (fun i p ->
            let size =
              if p.shape.ty <> 13 then param_read_bytes fc p
              else
                let total = ref 0 in
                for_each_leaf p.shape [] (fun _ idx ->
                    let gte =
                      List.fold_left
                        (fun g sub ->
                          match
                            List.find_opt
                              (fun u ->
                                u.opcode = "get-tuple-element"
                                && u.tuple_index = sub)
                              (fusion_users fc g)
                          with
                          | Some u -> u
                          | None -> g)
                        p idx
                    in
                    let s = param_read_bytes fc gte in
                    total := !total + s;
                    set cur (operand_bytes_key i idx) (fi s));
                !total
            in
            add cur bytes_key (fi size);
            set cur (operand_bytes_key i []) (fi size);
            set cur (operand_util_key i []) (get (hlo_prop st fc p) util_key))
          params
  and param_number p = p.param_no
  and fusion_users fc i =
    Option.value ~default:[] (Hashtbl.find_opt fc.users i.id)
  and param_read_bytes fc hlo =
    let size = ref 0 in
    let seen = ref false in
    let dyn_slice hlo user seen =
      if user.operands.(0) = hlo.id then shape_size user.shape
      else if not !seen then (
        seen := true;
        shape_size hlo.shape)
      else 0
    in
    let dus hlo user seen =
      if user.operands.(0) <> hlo.id && not !seen then (
        seen := true;
        shape_size hlo.shape)
      else 0
    in
    List.iter
      (fun user ->
        match user.opcode with
        | "fusion" ->
            let inner = Hashtbl.find st.m.comps user.called.(0) in
            let inner_params =
              Array.to_list inner.instrs
              |> List.filter (fun i -> i.opcode = "parameter")
              |> List.sort (fun a b ->
                  compare (param_number a) (param_number b))
              |> Array.of_list
            in
            Array.iteri
              (fun idx o ->
                if o = hlo.id then begin
                  let nested_seen = ref false in
                  let root = inner.root in
                  let simple =
                    Array.length root.operands = 0
                    || inner_params.(idx).id = root.operands.(0)
                  in
                  List.iter
                    (fun fu ->
                      if simple && fu.opcode = "slice" then
                        size := !size + shape_size fu.shape
                      else if simple && fu.opcode = "dynamic-slice" then
                        size := !size + dyn_slice user fu nested_seen
                      else if simple && fu.opcode = "dynamic-update-slice" then
                        size := !size + dus user fu nested_seen
                      else if not !nested_seen then (
                        nested_seen := true;
                        size :=
                          !size + param_read_bytes inner inner_params.(idx)))
                    (fusion_users fc user)
                end)
              user.operands
        | "slice" -> size := !size + shape_size user.shape
        | "dynamic-slice" -> size := !size + dyn_slice hlo user seen
        | "dynamic-update-slice" -> size := !size + dus hlo user seen
        | "broadcast" | "reshape" -> size := !size + shape_size hlo.shape
        | _ ->
            if not !seen then (
              seen := true;
              size := !size + shape_size hlo.shape))
      (fusion_users fc hlo);
    !size
  in
  let dead =
    Array.to_list comp.instrs
    |> List.filter (fun i -> users_of i = [] && i.id <> comp.root.id)
  in
  let has_succ i =
    Array.exists (fun j -> Array.mem i.id j.control_preds) comp.instrs
  in
  List.iter (fun i -> if not (has_succ i) then visit i) dead;
  visit comp.root;
  sum

let of_module text =
  let m = decode_module text in
  let st = { m; hlo_props = Hashtbl.create 64 } in
  let sum = analyze st m.entry in
  let order =
    [
      "flops";
      "transcendentals";
      bytes_key;
      "optimal_seconds";
      util_key;
      operand_util_key 0 [];
      operand_util_key 1 [];
      operand_bytes_key 0 [];
      operand_bytes_key 1 [];
      output_key [];
    ]
  in
  let fixed =
    List.filter_map
      (fun k ->
        let v = get sum k in
        if v <> 0.0 then Some (k, v) else None)
      order
  in
  let rest =
    Hashtbl.fold
      (fun k v acc ->
        if v <> 0.0 && not (List.mem k order) then (k, v) :: acc else acc)
      sum []
    |> List.sort compare
  in
  fixed @ rest
