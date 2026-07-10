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

let clamp_rule (axis_size : int) (vals : value list) (bdims : int option list) :
    value * int option =
  let dst =
    match List.find_opt (fun b -> b <> None) bdims with
    | Some (Some d) -> d
    | _ -> 0
  in
  let aligned = List.map2 (align_to axis_size dst) vals bdims in
  (b1 Clamp aligned, Some dst)

let naryop_rule (axis_size : int) (prim : primitive) (vals : value list)
    (bdims : int option list) : value * int option =
  let dst =
    match List.find_opt (fun b -> b <> None) bdims with
    | Some (Some d) -> d
    | _ -> 0
  in
  let aligned = List.map2 (align_to axis_size dst) vals bdims in
  (b1 prim aligned, Some dst)

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

let cumred_rule (make : int -> primitive) (axis : int) (x : value)
    (bdim : int option) : value * int option =
  match bdim with
  | None -> (b1 (make axis) [ x ], None)
  | Some d ->
      let new_axis = if axis < d then axis else axis + 1 in
      (b1 (make new_axis) [ x ], Some d)

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

let reduce_axes_batch (make : int array -> primitive) (axes : int array)
    (x : value) (bdim : int option) : value * int option =
  match bdim with
  | None -> (b1 (make axes) [ x ], None)
  | Some d ->
      let new_axes =
        Array.map (fun ax -> if d <= ax then ax + 1 else ax) axes
      in
      let shift =
        Array.fold_left (fun acc ax -> if ax < d then acc + 1 else acc) 0 axes
      in
      (b1 (make new_axes) [ x ], Some (d - shift))

let argminmax_batch (make : int -> primitive) (axis : int) (x : value)
    (bdim : int option) : value * int option =
  match bdim with
  | None -> (b1 (make axis) [ x ], None)
  | Some d ->
      let new_axis = if d <= axis then axis + 1 else axis in
      let out_bdim = if axis < d then d - 1 else d in
      (b1 (make new_axis) [ x ], Some out_bdim)

let reduce_general_batch (axis_size : int) (jaxpr : closed_jaxpr)
    (dimensions : int array) (vals : value list) (bdims : int option list) :
    value * int option =
  let num = List.length vals / 2 in
  match (Util.split_list vals [ num ], Util.split_list bdims [ num ]) with
  | [ operands; inits ], [ op_bdims; init_bdims ] ->
      if List.for_all (fun b -> b = None) init_bdims then begin
        let moved =
          List.map2
            (fun v b -> move_batch_axis axis_size b 0 v)
            operands op_bdims
        in
        let new_dims = Array.map (fun d -> d + 1) dimensions in
        let outs =
          Core.bind (Reduce { jaxpr; dimensions = new_dims }) (moved @ inits)
        in
        (List.hd outs, Some 0)
      end
      else failwith "batching: reduce with batched init not supported in M1"
  | _ -> failwith "batching: reduce arity"

let slice_batch (start_indices : int array) (limit_indices : int array)
    (strides : int array option) (x : value) (bdim : int option) :
    value * int option =
  match bdim with
  | None -> (b1 (Slice { start_indices; limit_indices; strides }) [ x ], None)
  | Some d ->
      let dim_size = (Core.get_aval x).shape.(d) in
      let ns = insert_size start_indices d 0 in
      let nl = insert_size limit_indices d dim_size in
      let nst =
        match strides with None -> None | Some s -> Some (insert_size s d 1)
      in
      ( b1
          (Slice { start_indices = ns; limit_indices = nl; strides = nst })
          [ x ],
        Some d )

let index_zero (idx_vals : value list) : value =
  let dt =
    match idx_vals with v :: _ -> (Core.get_aval v).dtype | [] -> Dtype.I32
  in
  Concrete (Ndarray.of_floats dt [||] [| 0.0 |])

let dynamic_slice_batch (axis_size : int) (slice_sizes : int array)
    (vals : value list) (bdims : int option list) : value * int option =
  match (vals, bdims) with
  | operand :: idx_vals, ob :: idx_bdims -> (
      if List.exists (fun b -> b <> None) idx_bdims then
        failwith
          "batching: dynamic_slice with batched indices needs gather (row 29)"
      else
        match ob with
        | None -> (b1 (Dynamic_slice { slice_sizes }) vals, None)
        | Some d ->
            let x = move_batch_axis axis_size (Some d) 0 operand in
            let zero = index_zero idx_vals in
            let new_sizes = insert_size slice_sizes 0 axis_size in
            ( Core.bind1
                (Dynamic_slice { slice_sizes = new_sizes })
                (x :: zero :: idx_vals),
              Some 0 ))
  | _ -> failwith "batching: dynamic_slice expects an operand"

let dynamic_update_slice_batch (axis_size : int) (vals : value list)
    (bdims : int option list) : value * int option =
  match (vals, bdims) with
  | operand :: update :: idx_vals, ob :: ub :: idx_bdims ->
      if List.exists (fun b -> b <> None) idx_bdims then
        failwith
          "batching: dynamic_update_slice with batched indices needs scatter \
           (row 29)"
      else
        let x = move_batch_axis axis_size ob 0 operand in
        let u = move_batch_axis axis_size ub 0 update in
        let zero = index_zero idx_vals in
        (Core.bind1 Dynamic_update_slice (x :: u :: zero :: idx_vals), Some 0)
  | _ -> failwith "batching: dynamic_update_slice expects operand and update"

let bump arr = Array.map (fun i -> i + 1) arr

let gather_batch (axis_size : int) (gd : gather_dims) (slice_sizes : int array)
    (vals : value list) (bdims : int option list) : value * int option =
  match (vals, bdims) with
  | [ operand; indices ], [ ob; ib ] -> (
      match (ob, ib) with
      | None, None ->
          ( b1
              (Gather { dimension_numbers = gd; slice_sizes })
              [ operand; indices ],
            None )
      | Some od, None ->
          let operand = move_batch_axis axis_size (Some od) 0 operand in
          let op0 = (Core.get_aval operand).shape.(0) in
          let slice_sizes = Array.append [| op0 |] slice_sizes in
          let gd' =
            {
              offset_dims = Array.append [| 0 |] (bump gd.offset_dims);
              collapsed_slice_dims = bump gd.collapsed_slice_dims;
              start_index_map = bump gd.start_index_map;
              g_operand_batching_dims = bump gd.g_operand_batching_dims;
              g_start_indices_batching_dims = gd.g_start_indices_batching_dims;
            }
          in
          ( b1
              (Gather { dimension_numbers = gd'; slice_sizes })
              [ operand; indices ],
            Some 0 )
      | None, Some idd ->
          let indices = move_batch_axis axis_size (Some idd) 0 indices in
          let gd' =
            {
              offset_dims = bump gd.offset_dims;
              collapsed_slice_dims = gd.collapsed_slice_dims;
              start_index_map = gd.start_index_map;
              g_operand_batching_dims = gd.g_operand_batching_dims;
              g_start_indices_batching_dims =
                bump gd.g_start_indices_batching_dims;
            }
          in
          ( b1
              (Gather { dimension_numbers = gd'; slice_sizes })
              [ operand; indices ],
            Some 0 )
      | Some od, Some idd ->
          let operand = move_batch_axis axis_size (Some od) 0 operand in
          let indices = move_batch_axis axis_size (Some idd) 0 indices in
          let op0 = (Core.get_aval operand).shape.(0) in
          let slice_sizes =
            Array.append [| (if op0 = 0 then 0 else 1) |] slice_sizes
          in
          let gd' =
            {
              offset_dims = bump gd.offset_dims;
              collapsed_slice_dims = bump gd.collapsed_slice_dims;
              start_index_map = bump gd.start_index_map;
              g_operand_batching_dims =
                Array.append [| 0 |] (bump gd.g_operand_batching_dims);
              g_start_indices_batching_dims =
                Array.append [| 0 |] (bump gd.g_start_indices_batching_dims);
            }
          in
          ( b1
              (Gather { dimension_numbers = gd'; slice_sizes })
              [ operand; indices ],
            Some 0 ))
  | _ -> failwith "batching: gather expects operand and indices"

let scatter_dnums_of = function
  | Scatter { dimension_numbers; _ }
  | Scatter_add { dimension_numbers }
  | Scatter_sub { dimension_numbers }
  | Scatter_mul { dimension_numbers; _ }
  | Scatter_min { dimension_numbers }
  | Scatter_max { dimension_numbers } ->
      dimension_numbers
  | _ -> failwith "batching: not a scatter primitive"

let rebuild_scatter prim (sd : scatter_dims) =
  match prim with
  | Scatter { unique_indices; _ } ->
      Scatter { dimension_numbers = sd; unique_indices }
  | Scatter_add _ -> Scatter_add { dimension_numbers = sd }
  | Scatter_sub _ -> Scatter_sub { dimension_numbers = sd }
  | Scatter_mul { unique_indices; _ } ->
      Scatter_mul { dimension_numbers = sd; unique_indices }
  | Scatter_min _ -> Scatter_min { dimension_numbers = sd }
  | Scatter_max _ -> Scatter_max { dimension_numbers = sd }
  | _ -> failwith "batching: not a scatter primitive"

let scatter_batch (axis_size : int) (prim : primitive) (vals : value list)
    (bdims : int option list) : value * int option =
  match (vals, bdims) with
  | [ operand; indices; updates ], [ ob; ib; ub ] ->
      let sd = scatter_dnums_of prim in
      if ob = None && ib = None && ub = None then (b1 prim vals, None)
      else begin
        let operand = move_batch_axis axis_size ob 0 operand in
        let updates = move_batch_axis axis_size ub 0 updates in
        match ib with
        | None ->
            let sd' =
              {
                update_window_dims =
                  Array.append [| 0 |] (bump sd.update_window_dims);
                inserted_window_dims = bump sd.inserted_window_dims;
                scatter_dims_to_operand_dims =
                  bump sd.scatter_dims_to_operand_dims;
                s_operand_batching_dims = bump sd.s_operand_batching_dims;
                s_scatter_indices_batching_dims =
                  sd.s_scatter_indices_batching_dims;
              }
            in
            ( Core.bind1 (rebuild_scatter prim sd') [ operand; indices; updates ],
              Some 0 )
        | Some idd ->
            let indices = move_batch_axis axis_size (Some idd) 0 indices in
            let sd' =
              {
                update_window_dims = bump sd.update_window_dims;
                inserted_window_dims = bump sd.inserted_window_dims;
                scatter_dims_to_operand_dims =
                  bump sd.scatter_dims_to_operand_dims;
                s_operand_batching_dims =
                  Array.append [| 0 |] (bump sd.s_operand_batching_dims);
                s_scatter_indices_batching_dims =
                  Array.append [| 0 |] (bump sd.s_scatter_indices_batching_dims);
              }
            in
            ( Core.bind1 (rebuild_scatter prim sd') [ operand; indices; updates ],
              Some 0 )
      end
  | _ -> failwith "batching: scatter expects operand, indices and updates"

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
  | Population_count | Real | Round | Rsqrt | Sinh | Sqrt | Square | Tan
  | Bitcast_convert_type _ | Reduce_precision _ | Bessel_i0e | Bessel_i1e
  | Digamma | Erf | Erf_inv | Erfc | Lgamma ->
      un ()
  | Igamma | Igamma_grad_a | Igammac | Polygamma | Zeta
  | Regularized_incomplete_beta ->
      naryop_rule axis_size prim vals bdims
  | Add | Sub | Mul | Div | Max | Min | Pow | Eq | Lt | Gt | Ge | Le | Eq_to
  | Le_to | Lt_to | And | Atan2 | Complex | Mulhi | Ne | Nextafter | Or | Rem
  | Shift_left | Shift_right_arithmetic | Shift_right_logical | Xor | Tie ->
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
  | Reduce_max axes
  | Reduce_min axes
  | Reduce_prod axes
  | Reduce_and axes
  | Reduce_or axes
  | Reduce_xor axes -> (
      match (vals, bdims) with
      | [ x ], [ b ] ->
          let make a =
            match prim with
            | Reduce_max _ -> Reduce_max a
            | Reduce_min _ -> Reduce_min a
            | Reduce_prod _ -> Reduce_prod a
            | Reduce_and _ -> Reduce_and a
            | Reduce_or _ -> Reduce_or a
            | _ -> Reduce_xor a
          in
          reduce_axes_batch make axes x b
      | _ -> failwith "batching: expected 1 operand")
  | Argmax { axis; index_dtype } -> (
      match (vals, bdims) with
      | [ x ], [ b ] ->
          argminmax_batch (fun ax -> Argmax { axis = ax; index_dtype }) axis x b
      | _ -> failwith "batching: expected 1 operand")
  | Argmin { axis; index_dtype } -> (
      match (vals, bdims) with
      | [ x ], [ b ] ->
          argminmax_batch (fun ax -> Argmin { axis = ax; index_dtype }) axis x b
      | _ -> failwith "batching: expected 1 operand")
  | Reduce { jaxpr; dimensions } ->
      reduce_general_batch axis_size jaxpr dimensions vals bdims
  | Cumsum { axis; reverse }
  | Cumprod { axis; reverse }
  | Cummax { axis; reverse }
  | Cummin { axis; reverse }
  | Cumlogsumexp { axis; reverse } -> (
      match (vals, bdims) with
      | [ x ], [ b ] ->
          let make a =
            match prim with
            | Cumsum _ -> Cumsum { axis = a; reverse }
            | Cumprod _ -> Cumprod { axis = a; reverse }
            | Cummax _ -> Cummax { axis = a; reverse }
            | Cummin _ -> Cummin { axis = a; reverse }
            | _ -> Cumlogsumexp { axis = a; reverse }
          in
          cumred_rule make axis x b
      | _ -> failwith "batching: expected 1 operand")
  | Clamp -> clamp_rule axis_size vals bdims
  | Slice { start_indices; limit_indices; strides } -> (
      match (vals, bdims) with
      | [ x ], [ b ] -> slice_batch start_indices limit_indices strides x b
      | _ -> failwith "batching: expected 1 operand")
  | Dynamic_slice { slice_sizes } ->
      dynamic_slice_batch axis_size slice_sizes vals bdims
  | Dynamic_update_slice -> dynamic_update_slice_batch axis_size vals bdims
  | Gather { dimension_numbers; slice_sizes } ->
      gather_batch axis_size dimension_numbers slice_sizes vals bdims
  | Scatter _ | Scatter_add _ | Scatter_sub _ | Scatter_mul _ | Scatter_min _
  | Scatter_max _ ->
      scatter_batch axis_size prim vals bdims
  | Split _ | Unstack _ | Optimization_barrier | Sort _ | Top_k _ | Scan _
  | Custom_linear_solve _ ->
      failwith "batching: multi-output handled by batch_process_primitive"
  | Iota _ | Empty _ | Empty2 _ | Create_token | After_all | Composite _
  | Dce_sink | From_edtype _ | Ragged_dot_general | Rng_bit_generator
  | Rng_uniform | To_edtype _ ->
      failwith "batching: vmap of this primitive not supported in M1"
  | Dot_general _ ->
      failwith "batching: vmap of dot_general not supported in M1"
  | Conv_general_dilated _ ->
      failwith "batching: vmap of conv_general_dilated not supported in M2"
  | Reduce_window _ | Reduce_window_max _ | Reduce_window_min _
  | Reduce_window_sum _ | Select_and_gather_add _ | Select_and_scatter _
  | Select_and_scatter_add _ ->
      failwith "batching: vmap of windowed reductions not supported in M2"
  | Platform_index _ ->
      failwith "batching: vmap of platform_index not supported"
  | Cond _ -> failwith "batching: cond handled by batch_process_primitive"
  | Threefry2x32 | Iota_2x32_shape _ | Random_seed | Random_split _
  | Random_fold_in | Random_bits _ | Random_wrap | Random_unwrap ->
      failwith "batching: vmap of prng primitive not supported"
  | While _ ->
      failwith
        "batching: vmap of while_loop needs the batched-predicate fixpoint (M2 \
         gap)"
  | Cholesky | Householder_product | Lu | Lu_pivots_to_permutation _ | Qr _
  | Triangular_solve _ | Tridiagonal_solve | Eig _ | Eigh _ | Hessenberg
  | Schur _ | Svd _ | Tridiagonal _ ->
      failwith "batching: vmap of linalg primitive deferred (M5 gap)"
  | Xla_call _ -> failwith "batching: vmap of xla_call not supported in M1"

let opt_barrier_batch (vals : value list) (bdims : int option list) :
    value list * int option list =
  (Core.bind Optimization_barrier vals, bdims)

let sort_batch (axis_size : int) (dimension : int) (is_stable : bool)
    (num_keys : int) (vals : value list) (bdims : int option list) :
    value list * int option list =
  let dst =
    match List.find_opt (fun b -> b <> None) bdims with
    | Some (Some d) -> d
    | _ -> 0
  in
  let aligned = List.map2 (align_to axis_size dst) vals bdims in
  let new_dimension = if dst <= dimension then dimension + 1 else dimension in
  let outs =
    Core.bind (Sort { dimension = new_dimension; is_stable; num_keys }) aligned
  in
  (outs, List.map (fun _ -> Some dst) outs)

let top_k_batch (k : int) (axis : int) (x : value) (bdim : int option) :
    value list * int option list =
  match bdim with
  | None ->
      let outs = Core.bind (Top_k { k; axis }) [ x ] in
      (outs, List.map (fun _ -> None) outs)
  | Some d ->
      let ax = if d <= axis then axis + 1 else axis in
      let outs = Core.bind (Top_k { k; axis = ax }) [ x ] in
      (outs, List.map (fun _ -> Some d) outs)

let vmap_rule_multi (axis_size : int) (prim : primitive) (vals : value list)
    (bdims : int option list) : value list * int option list =
  match (prim, vals, bdims) with
  | Split { sizes; axis }, [ x ], [ b ] -> split_batch sizes axis x b
  | Unstack axis, [ x ], [ b ] -> unstack_batch axis x b
  | Optimization_barrier, _, _ -> opt_barrier_batch vals bdims
  | Sort { dimension; is_stable; num_keys }, _, _ ->
      sort_batch axis_size dimension is_stable num_keys vals bdims
  | Top_k { k; axis }, [ x ], [ b ] -> top_k_batch k axis x b
  | _ -> failwith "batching: expected a multi-output primitive"

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

let cond_batch (axis_size : int) (t : closed_jaxpr) (f : closed_jaxpr)
    (vals : value list) (bdims : int option list) : value list * int option list
    =
  match (vals, bdims) with
  | pred :: ops, pred_bdim :: op_bdims ->
      if pred_bdim <> None then
        failwith
          "batching: vmap of cond with batched predicate needs select_n (M2 \
           gap)";
      let moved =
        List.map2 (fun v b -> move_batch_axis axis_size b 0 v) ops op_bdims
      in
      let moved_avals = List.map Core.get_aval moved in
      let batch_branch (branch : closed_jaxpr) : closed_jaxpr =
        Jaxpr.make_jaxpr moved_avals (fun bops ->
            vmap
              (fun a -> Jaxpr.eval_closed_jaxpr branch a)
              (List.map (fun _ -> Some 0) bops)
              bops)
      in
      let t' = batch_branch t and f' = batch_branch f in
      let outs = Core.bind (Cond { t = t'; f = f' }) (pred :: moved) in
      (outs, List.map (fun _ -> Some 0) outs)
  | _ -> failwith "batching: cond expects a predicate"

let scan_batch (axis_size : int) ~(length : int) ~(reverse : bool)
    ~(num_carry : int) (jaxpr : closed_jaxpr) (vals : value list)
    (bdims : int option list) : value list * int option list =
  let split2 l n =
    match Util.split_list l [ n ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "batching: scan split"
  in
  let carry, xs = split2 vals num_carry in
  let carry_b, xs_b = split2 bdims num_carry in
  let new_carry =
    List.map2 (fun v b -> move_batch_axis axis_size b 0 v) carry carry_b
  in
  let new_xs = List.map2 (fun v b -> move_batch_axis axis_size b 1 v) xs xs_b in
  let carry_slice_avals = List.map Core.get_aval new_carry in
  let x_slice_avals =
    List.map (fun v -> mapped_aval 0 (Core.get_aval v)) new_xs
  in
  let new_body =
    Jaxpr.make_jaxpr (carry_slice_avals @ x_slice_avals) (fun args ->
        vmap
          (fun a -> Jaxpr.eval_closed_jaxpr jaxpr a)
          (List.map (fun _ -> Some 0) args)
          args)
  in
  let outs =
    Core.bind
      (Scan { length; reverse; num_carry; jaxpr = new_body })
      (new_carry @ new_xs)
  in
  let num_ys = List.length outs - num_carry in
  let out_bdims =
    List.init num_carry (fun _ -> Some 0) @ List.init num_ys (fun _ -> Some 1)
  in
  (outs, out_bdims)

let batch_process_primitive (trace : trace) (prim : primitive)
    (args : value list) : value list =
  let axis_size = axis_size_of trace in
  let pairs = List.map (to_batch_info trace) args in
  let vals = List.map fst pairs and bdims = List.map snd pairs in
  match prim with
  | Split _ | Unstack _ | Optimization_barrier | Sort _ | Top_k _ ->
      let outs, out_bdims = vmap_rule_multi axis_size prim vals bdims in
      List.map2 (fun o b -> Tracer (new_batch_tracer trace o b)) outs out_bdims
  | Cond { t; f } ->
      let outs, out_bdims = cond_batch axis_size t f vals bdims in
      List.map2 (fun o b -> Tracer (new_batch_tracer trace o b)) outs out_bdims
  | Scan { length; reverse; num_carry; jaxpr } ->
      let outs, out_bdims =
        scan_batch axis_size ~length ~reverse ~num_carry jaxpr vals bdims
      in
      List.map2 (fun o b -> Tracer (new_batch_tracer trace o b)) outs out_bdims
  | Custom_linear_solve _ ->
      failwith
        "batching: vmap of custom_linear_solve needs the batched fixpoint (M2 \
         gap)"
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
  | Device _ -> v

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
