open Types

let axis_size_of (trace : trace) =
  match trace.global_data with
  | GAxisSize n -> n
  | _ -> failwith "batching: expected axis size in global_data"

let mapped_aval (bdim : int) (a : aval) : aval =
  let n = Array.length a.shape in
  let shape = Array.make (n - 1) 0 in
  let k = ref 0 in
  for i = 0 to n - 1 do
    if i <> bdim then begin
      shape.(!k) <- a.shape.(i);
      incr k
    end
  done;
  { a with shape }

let unmapped_aval (axis_size : int) (bdim : int) (a : aval) : aval =
  let n = Array.length a.shape in
  let shape = Array.make (n + 1) 0 in
  for i = 0 to n do
    if i < bdim then shape.(i) <- a.shape.(i)
    else if i = bdim then shape.(i) <- axis_size
    else shape.(i) <- a.shape.(i - 1)
  done;
  { a with shape }

let b1 = Core.bind1

let insert_size (shape : int array) (dst : int) (size : int) : int array =
  let n = Array.length shape in
  Array.init (n + 1) (fun i ->
      if i < dst then shape.(i) else if i = dst then size else shape.(i - 1))

let broadcast_insert (axis_size : int) (dst : int) (v : value) : value =
  let a = Core.get_aval v in
  let n = Array.length a.shape in
  let shape = insert_size a.shape dst axis_size in
  let dims = Array.init n (fun i -> if i < dst then i else i + 1) in
  b1 (Broadcast_in_dim { shape; dims }) [ v ]

let moveaxis (src : int) (dst : int) (v : value) : value =
  let a = Core.get_aval v in
  let n = Array.length a.shape in
  let order = Array.make n 0 in
  let k = ref 0 in
  for i = 0 to n - 1 do
    if i <> src then begin
      order.(!k) <- i;
      incr k
    end
  done;
  for i = n - 1 downto dst + 1 do
    order.(i) <- order.(i - 1)
  done;
  order.(dst) <- src;
  let shape = Array.map (fun ax -> a.shape.(ax)) order in
  let dims = Array.make n 0 in
  Array.iteri (fun out_pos in_ax -> dims.(in_ax) <- out_pos) order;
  b1 (Broadcast_in_dim { shape; dims }) [ v ]

let move_batch_axis (axis_size : int) (src : int option) (dst : int) (v : value)
    : value =
  match src with
  | None -> broadcast_insert axis_size dst v
  | Some s -> if s = dst then v else moveaxis s dst v

let new_batch_tracer (trace : trace) (v : value) (bdim : int option) : tracer =
  let base = Core.get_aval v in
  let aval = match bdim with None -> base | Some d -> mapped_aval d base in
  { id = Core.fresh_id (); trace; aval; payload = Batch { v; bdim } }

let batch_pure (trace : trace) (v : value) : value =
  Tracer (new_batch_tracer trace v None)

let to_batch_info (trace : trace) (v : value) : value * int option =
  match v with
  | Tracer t when t.trace.level = trace.level -> (
      match t.payload with
      | Batch { v; bdim } -> (v, bdim)
      | _ -> failwith "batching: expected a Batch tracer")
  | _ -> (v, None)

let unop_rule (prim : primitive) (x : value) (bdim : int option) :
    value * int option =
  (b1 prim [ x ], bdim)

let binop_rule (prim : primitive) (axis_size : int) (x : value)
    (x_bdim : int option) (y : value) (y_bdim : int option) : value * int option
    =
  if x_bdim = y_bdim then (b1 prim [ x; y ], x_bdim)
  else
    match x_bdim with
    | None ->
        let dst = match y_bdim with Some d -> d | None -> 0 in
        let x = move_batch_axis axis_size x_bdim dst x in
        (b1 prim [ x; y ], y_bdim)
    | Some d ->
        let y = move_batch_axis axis_size y_bdim d y in
        (b1 prim [ x; y ], x_bdim)

let align_to (axis_size : int) (dst : int) (v : value) (bdim : int option) :
    value =
  move_batch_axis axis_size bdim dst v

let select_rule (axis_size : int) (vals : value list) (bdims : int option list)
    : value * int option =
  let dst =
    match List.find_opt (fun b -> b <> None) bdims with
    | Some (Some d) -> d
    | _ -> 0
  in
  let aligned = List.map2 (align_to axis_size dst) vals bdims in
  (b1 Select_n aligned, Some dst)

let reduce_sum_rule (axes : int array) (x : value) (bdim : int option) :
    value * int option =
  match bdim with
  | None -> (b1 (Reduce_sum axes) [ x ], None)
  | Some d ->
      let new_axes =
        Array.map (fun ax -> if d <= ax then ax + 1 else ax) axes
      in
      let shift =
        Array.fold_left (fun acc ax -> if ax < d then acc + 1 else acc) 0 axes
      in
      let out_bdim = d - shift in
      (b1 (Reduce_sum new_axes) [ x ], Some out_bdim)

let reshape_rule (axis_size : int) (ns : int array) (x : value)
    (bdim : int option) : value * int option =
  match bdim with
  | None -> (b1 (Reshape ns) [ x ], None)
  | Some d ->
      let x = move_batch_axis axis_size bdim 0 x in
      let new_sizes = insert_size ns 0 axis_size in
      (b1 (Reshape new_sizes) [ x ], Some 0)

let broadcast_rule (axis_size : int) (shape : int array) (dims : int array)
    (x : value) (bdim : int option) : value * int option =
  match bdim with
  | None -> (b1 (Broadcast_in_dim { shape; dims }) [ x ], None)
  | Some d ->
      let x = move_batch_axis axis_size bdim 0 x in
      let new_shape = insert_size shape 0 axis_size in
      let new_dims = Array.append [| 0 |] (Array.map (fun i -> i + 1) dims) in
      ( b1 (Broadcast_in_dim { shape = new_shape; dims = new_dims }) [ x ],
        Some 0 )

let convert_rule (dt : Dtype.t) (x : value) (bdim : int option) :
    value * int option =
  (b1 (Convert_element_type dt) [ x ], bdim)

let insert_pad_cfg (cfg : (int * int * int) array) (idx : int)
    (v : int * int * int) : (int * int * int) array =
  let n = Array.length cfg in
  Array.init (n + 1) (fun i ->
      if i < idx then cfg.(i) else if i = idx then v else cfg.(i - 1))

let concatenate_batch (axis_size : int) (dim : int) (vals : value list)
    (bdims : int option list) : value * int option =
  let aligned =
    List.map2 (fun v b -> move_batch_axis axis_size b 0 v) vals bdims
  in
  (b1 (Concatenate (dim + 1)) aligned, Some 0)

let stack_batch (axis_size : int) (axis : int) (vals : value list)
    (bdims : int option list) : value * int option =
  let aligned =
    List.map2 (fun v b -> move_batch_axis axis_size b 0 v) vals bdims
  in
  (b1 (Stack (axis + 1)) aligned, Some 0)

let pad_batch (cfg : (int * int * int) array) (vals : value list)
    (bdims : int option list) : value * int option =
  match (vals, bdims) with
  | [ operand; pv ], [ ob; pvb ] -> (
      match pvb with
      | Some _ ->
          failwith
            "batching: pad with batched padding_value not supported in M1"
      | None -> (
          match ob with
          | Some d ->
              let cfg2 = insert_pad_cfg cfg d (0, 0, 0) in
              (b1 (Pad cfg2) [ operand; pv ], Some d)
          | None ->
              failwith
                "batching: pad with unbatched operand not supported in M1"))
  | _ -> failwith "batching: pad expects 2 operands"

let rev_batch (dims : int array) (x : value) (bdim : int option) :
    value * int option =
  match bdim with
  | None -> (b1 (Rev dims) [ x ], None)
  | Some d ->
      let nd = Array.map (fun i -> if i >= d then i + 1 else i) dims in
      (b1 (Rev nd) [ x ], Some d)

let squeeze_batch (axis_size : int) (dims : int array) (x : value)
    (bdim : int option) : value * int option =
  match bdim with
  | None -> (b1 (Squeeze dims) [ x ], None)
  | Some d ->
      let x = move_batch_axis axis_size (Some d) 0 x in
      let nd = Array.map (fun i -> i + 1) dims in
      (b1 (Squeeze nd) [ x ], Some 0)

let tile_batch (reps : int array) (x : value) (bdim : int option) :
    value * int option =
  match bdim with
  | None -> (b1 (Tile reps) [ x ], None)
  | Some d ->
      let nr = insert_size reps d 1 in
      (b1 (Tile nr) [ x ], Some d)

let transpose_batch (perm : int array) (x : value) (bdim : int option) :
    value * int option =
  match bdim with
  | None -> (b1 (Transpose perm) [ x ], None)
  | Some d ->
      let np = Array.map (fun i -> if i < d then i else i + 1) perm in
      (b1 (Transpose (Array.append [| d |] np)) [ x ], Some 0)

let split_batch (sizes : int array) (axis : int) (x : value) (bdim : int option)
    : value list * int option list =
  match bdim with
  | None ->
      let outs = Core.bind (Split { sizes; axis }) [ x ] in
      (outs, List.map (fun _ -> None) outs)
  | Some d ->
      let ax = if axis >= d then axis + 1 else axis in
      let outs = Core.bind (Split { sizes; axis = ax }) [ x ] in
      (outs, List.map (fun _ -> Some d) outs)

let unstack_batch (axis : int) (x : value) (bdim : int option) :
    value list * int option list =
  match bdim with
  | None ->
      let outs = Core.bind (Unstack axis) [ x ] in
      (outs, List.map (fun _ -> None) outs)
  | Some d ->
      let ax, out_bdim = if axis < d then (axis, d - 1) else (axis + 1, d) in
      let outs = Core.bind (Unstack ax) [ x ] in
      (outs, List.map (fun _ -> Some out_bdim) outs)

let vmap_rule (axis_size : int) (prim : primitive) (vals : value list)
    (bdims : int option list) : value * int option =
  let un () =
    match (vals, bdims) with
    | [ x ], [ b ] -> unop_rule prim x b
    | _ -> failwith "batching: expected 1 operand"
  in
  let bin () =
    match (vals, bdims) with
    | [ x; y ], [ bx; by ] -> binop_rule prim axis_size x bx y by
    | _ -> failwith "batching: expected 2 operands"
  in
  match prim with
  | Neg | Sin | Cos | Exp | Log | Tanh | Abs | Sign | Acos | Acosh | Asin
  | Asinh | Atan | Atanh | Cbrt | Ceil | Clz | Conj | Copy | Cosh | Exp2 | Expm1
  | Floor | Imag | Integer_pow _ | Is_finite | Log1p | Logistic | Not
  | Population_count | Real | Round | Rsqrt | Sinh | Sqrt | Square | Tan ->
      un ()
  | Add | Sub | Mul | Div | Max | Min | Pow | Eq | Lt | Gt | Ge | Le | Eq_to
  | Le_to | Lt_to | And | Atan2 | Complex | Mulhi | Ne | Nextafter | Or | Rem
  | Shift_left | Shift_right_arithmetic | Shift_right_logical | Xor ->
      bin ()
  | Select_n -> select_rule axis_size vals bdims
  | Convert_element_type dt -> (
      match (vals, bdims) with
      | [ x ], [ b ] -> convert_rule dt x b
      | _ -> failwith "batching: expected 1 operand")
  | Reduce_sum axes -> (
      match (vals, bdims) with
      | [ x ], [ b ] -> reduce_sum_rule axes x b
      | _ -> failwith "batching: expected 1 operand")
  | Reshape ns -> (
      match (vals, bdims) with
      | [ x ], [ b ] -> reshape_rule axis_size ns x b
      | _ -> failwith "batching: expected 1 operand")
  | Broadcast_in_dim { shape; dims } -> (
      match (vals, bdims) with
      | [ x ], [ b ] -> broadcast_rule axis_size shape dims x b
      | _ -> failwith "batching: expected 1 operand")
  | Concatenate dim -> concatenate_batch axis_size dim vals bdims
  | Stack axis -> stack_batch axis_size axis vals bdims
  | Pad cfg -> pad_batch cfg vals bdims
  | Rev dims -> (
      match (vals, bdims) with
      | [ x ], [ b ] -> rev_batch dims x b
      | _ -> failwith "batching: expected 1 operand")
  | Squeeze dims -> (
      match (vals, bdims) with
      | [ x ], [ b ] -> squeeze_batch axis_size dims x b
      | _ -> failwith "batching: expected 1 operand")
  | Tile reps -> (
      match (vals, bdims) with
      | [ x ], [ b ] -> tile_batch reps x b
      | _ -> failwith "batching: expected 1 operand")
  | Transpose perm -> (
      match (vals, bdims) with
      | [ x ], [ b ] -> transpose_batch perm x b
      | _ -> failwith "batching: expected 1 operand")
  | Split _ | Unstack _ ->
      failwith "batching: multi-output handled by batch_process_primitive"
  | Dot_general _ ->
      failwith "batching: vmap of dot_general not supported in M1"
  | Xla_call _ | Cond _ ->
      failwith "batching: vmap of control primitive not supported in M1"

let vmap_rule_multi (prim : primitive) (vals : value list)
    (bdims : int option list) : value list * int option list =
  match (prim, vals, bdims) with
  | Split { sizes; axis }, [ x ], [ b ] -> split_batch sizes axis x b
  | Unstack axis, [ x ], [ b ] -> unstack_batch axis x b
  | _ -> failwith "batching: expected a multi-output primitive with 1 operand"

let batch_process_primitive (trace : trace) (prim : primitive)
    (args : value list) : value list =
  let axis_size = axis_size_of trace in
  let pairs = List.map (to_batch_info trace) args in
  let vals = List.map fst pairs and bdims = List.map snd pairs in
  match prim with
  | Split _ | Unstack _ ->
      let outs, out_bdims = vmap_rule_multi prim vals bdims in
      List.map2 (fun o b -> Tracer (new_batch_tracer trace o b)) outs out_bdims
  | _ ->
      let out, out_bdim = vmap_rule axis_size prim vals bdims in
      [ Tracer (new_batch_tracer trace out out_bdim) ]

let full_lower_batch (v : value) : value =
  match v with
  | Tracer t -> (
      match t.payload with
      | Batch { v; bdim = None } -> Core.full_lower v
      | Batch _ -> v
      | _ -> v)
  | Concrete _ -> v

let interpreter : Core.interpreter =
  {
    i_pure = batch_pure;
    i_lift = batch_pure;
    i_full_lower = full_lower_batch;
    i_process_primitive = batch_process_primitive;
    i_process_custom_jvp =
      (fun _ ~primal:_ ~jvp:_ _ ->
        failwith "batching: custom_jvp not supported in M1");
    i_process_custom_vjp =
      (fun _ ~primal:_ ~fwd:_ ~bwd:_ _ ->
        failwith "batching: custom_vjp not supported in M1");
  }

let install () = Core.register_interpreter KBatch interpreter
let () = install ()

let vmap_flat (f : value list -> value list) (in_axes : int option list)
    (args : value list) : value list =
  let axis_size =
    let rec find axes vals =
      match (axes, vals) with
      | Some d :: _, x :: _ -> (Core.get_aval x).shape.(d)
      | None :: axes, _ :: vals -> find axes vals
      | _ -> failwith "batching: vmap needs at least one mapped input"
    in
    find in_axes args
  in
  Core.with_new_main KBatch (GAxisSize axis_size) (fun main ->
      let tracers_in =
        List.map2
          (fun x ax ->
            match ax with
            | Some _ -> Tracer (new_batch_tracer main x ax)
            | None -> x)
          args in_axes
      in
      let outs = f tracers_in in
      let vals_bdims =
        List.map (fun o -> to_batch_info main (Core.full_raise main o)) outs
      in
      List.map (fun (v, bdim) -> move_batch_axis axis_size bdim 0 v) vals_bdims)

let vmap (f : value list -> value list) (in_axes : int option list) :
    value list -> value list =
 fun args -> vmap_flat f in_axes args
