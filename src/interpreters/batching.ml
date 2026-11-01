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
  let dims =
    Np_ints.mark (Array.init n (fun i -> if i < dst then i else i + 1))
  in
  b1 (Broadcast_in_dim { shape; dims }) [ v ]

let moveaxis (src : int) (dst : int) (v : value) : value =
  if src = dst then v
  else
    let n = Array.length (Core.get_aval v).shape in
    let rest = List.filter (fun i -> i <> src) (List.init n Fun.id) in
    let rec insert k l =
      if k = 0 then src :: l
      else match l with x :: tl -> x :: insert (k - 1) tl | [] -> [ src ]
    in
    b1 (Transpose (Array.of_list (insert dst rest))) [ v ]

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

let bdim_at_front (size : int) (v : value) (bdim : int option) : value =
  match bdim with None -> broadcast_insert size 0 v | Some d -> moveaxis d 0 v

let broadcast_batcher (prim : primitive) (vals : value list)
    (bdims : int option list) : value * int option =
  if List.for_all (fun b -> b = None) bdims then (b1 prim vals, None)
  else
    let shape_of v = (Core.get_aval v).shape in
    let shape, dim =
      List.find_map
        (fun (v, b) ->
          match b with Some d -> Some (shape_of v, d) | None -> None)
        (List.combine vals bdims)
      |> Option.get
    in
    if
      List.for_all2
        (fun v b ->
          Array.length (shape_of v) = 0 || (shape_of v = shape && b = Some dim))
        vals bdims
    then (b1 prim vals, Some dim)
    else
      let args =
        List.map2
          (fun v b ->
            if Array.length (shape_of v) = 0 then v else bdim_at_front 1 v b)
          vals bdims
      in
      let nd =
        List.fold_left (fun acc v -> max acc (Array.length (shape_of v))) 0 args
      in
      let args =
        List.map2
          (fun v b ->
            let r = Array.length (shape_of v) in
            if b = None || nd = r then v
            else
              let sh = shape_of v in
              b1
                (Broadcast_in_dim
                   {
                     shape =
                       Array.init nd (fun i -> if i < r then sh.(i) else 1);
                     dims = Array.init r Fun.id;
                   })
                [ v ])
          args bdims
      in
      (b1 prim args, Some 0)

let binop_rule (prim : primitive) (x : value) (x_bdim : int option) (y : value)
    (y_bdim : int option) : value * int option =
  broadcast_batcher prim [ x; y ] [ x_bdim; y_bdim ]

let align_to (axis_size : int) (dst : int) (v : value) (bdim : int option) :
    value =
  move_batch_axis axis_size bdim dst v

let select_rule (axis_size : int) (vals : value list) (bdims : int option list)
    : value * int option =
  match (vals, bdims) with
  | which :: cases, which_bdim :: case_bdims ->
      let shape_of v = (Core.get_aval v).shape in
      let ndim v = Array.length (shape_of v) in
      if List.for_all (fun b -> b = None) bdims then (b1 Select_n vals, None)
      else if List.for_all (fun b -> b = which_bdim) case_bdims then
        if shape_of which = shape_of (List.hd cases) then
          (b1 Select_n vals, which_bdim)
        else
          let wb = Option.get which_bdim in
          let which =
            b1
              (Broadcast_in_dim
                 { shape = shape_of (List.hd cases); dims = [| wb |] })
              [ which ]
          in
          (b1 Select_n (which :: cases), which_bdim)
      else if ndim which = 0 && List.for_all (fun b -> b <> None) case_bdims
      then
        let first = List.hd case_bdims in
        if List.for_all (fun b -> b = first) case_bdims then
          (b1 Select_n vals, first)
        else
          let dst = Option.get first in
          let moved =
            List.map2
              (fun c b -> moveaxis (Option.get b) dst c)
              (List.tl cases) (List.tl case_bdims)
          in
          (b1 Select_n (which :: List.hd cases :: moved), first)
      else
        let which =
          if ndim which > 0 then bdim_at_front axis_size which which_bdim
          else which
        in
        let cases =
          if List.for_all (fun c -> ndim c = 0) cases then cases
          else List.map2 (bdim_at_front axis_size) cases case_bdims
        in
        let first = List.hd cases in
        let which =
          if ndim which > 0 && ndim which < ndim first then
            b1
              (Broadcast_in_dim { shape = shape_of first; dims = [| 0 |] })
              [ which ]
          else which
        in
        let cases =
          if ndim which > ndim first then
            List.map
              (fun c ->
                b1
                  (Broadcast_in_dim { shape = shape_of which; dims = [||] })
                  [ c ])
              cases
          else cases
        in
        (b1 Select_n (which :: cases), Some 0)
  | _ -> failwith "batching: select arity"

let clamp_rule (axis_size : int) (vals : value list) (bdims : int option list) :
    value * int option =
  let dst =
    match List.find_opt (fun b -> b <> None) bdims with
    | Some (Some d) -> d
    | _ -> 0
  in
  let aligned =
    List.map2
      (fun v b ->
        if b = None && Array.length (Core.get_aval v).shape = 0 then
          match vals with
          | [ _; x; _ ] when x == v -> align_to axis_size dst v b
          | _ -> v
        else align_to axis_size dst v b)
      vals bdims
  in
  (b1 Clamp aligned, Some dst)

let naryop_rule (prim : primitive) (vals : value list) (bdims : int option list)
    : value * int option =
  broadcast_batcher prim vals bdims

let reduce_sum_rule (axes : int array) (x : value) (bdim : int option) :
    value * int option =
  match bdim with
  | None -> (b1 (Reduce_sum axes) [ x ], None)
  | Some d ->
      let new_axes =
        Np_ints.mark (Array.map (fun ax -> if d <= ax then ax + 1 else ax) axes)
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
      let new_dims =
        Np_ints.mark_where
          (fun i -> i > 0)
          (Array.append [| 0 |] (Array.map (fun i -> i + 1) dims))
      in
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
    List.map2
      (fun v b ->
        match b with
        | Some _ -> move_batch_axis axis_size b 0 v
        | None ->
            let a = Core.get_aval v in
            let n = Array.length a.shape in
            b1
              (Broadcast_in_dim
                 {
                   shape = insert_size a.shape 0 axis_size;
                   dims = Array.init n (fun i -> i + 1);
                 })
              [ v ])
      vals bdims
  in
  (b1 (Concatenate (dim + 1)) aligned, Some 0)

let stack_batch (axis_size : int) (axis : int) (vals : value list)
    (bdims : int option list) : value * int option =
  match bdims with
  | Some bdim :: rest when List.for_all (fun b -> b = Some bdim) rest ->
      if axis <= bdim then (b1 (Stack axis) vals, Some (bdim + 1))
      else (b1 (Stack (axis + 1)) vals, Some bdim)
  | _ ->
      let aligned =
        List.map2
          (fun v b ->
            match b with
            | Some d -> moveaxis d 0 v
            | None ->
                let a = Core.get_aval v in
                let n = Array.length a.shape in
                b1
                  (Broadcast_in_dim
                     {
                       shape = Array.append [| axis_size |] a.shape;
                       dims = Array.init n (fun i -> i + 1);
                     })
                  [ v ])
          vals bdims
      in
      (b1 (Stack (axis + 1)) aligned, Some 0)

let pad_batch (cfg : (int * int * int) array) (vals : value list)
    (bdims : int option list) : value * int option =
  match (vals, bdims) with
  | [ operand; pv ], [ Some d; None ] ->
      let cfg2 = insert_pad_cfg cfg d (0, 0, 0) in
      (b1 (Pad cfg2) [ operand; pv ], Some d)
  | [ operand; pv ], [ ob; pvb ] ->
      let size =
        match (ob, pvb) with
        | Some d, _ -> (Core.get_aval operand).shape.(d)
        | None, Some d -> (Core.get_aval pv).shape.(d)
        | None, None -> failwith "batching: pad with nothing batched"
      in
      let leading v bdim =
        let a = Core.get_aval v in
        match bdim with
        | Some d ->
            let n = Array.length a.shape in
            let perm =
              Array.of_list
                (d :: List.filter (fun i -> i <> d) (List.init n Fun.id))
            in
            b1 (Transpose perm) [ v ]
        | None ->
            b1
              (Broadcast_in_dim
                 {
                   shape = Array.append [| size |] a.shape;
                   dims = Array.init (Array.length a.shape) (fun i -> i + 1);
                 })
              [ v ]
      in
      let x = leading operand ob in
      let fill = leading pv pvb in
      let cfg2 = insert_pad_cfg cfg 0 (0, 0, 0) in
      let dt = (Core.get_aval x).dtype in
      let zero = Concrete (Ndarray.of_floats dt [||] [| 0.0 |]) in
      let padded = b1 (Pad cfg2) [ x; zero ] in
      let keep =
        Concrete
          (Ndarray.of_floats Dtype.Bool (Core.get_aval x).shape
             (Array.make (Array.fold_left ( * ) 1 (Core.get_aval x).shape) 1.0))
      in
      let off = Concrete (Ndarray.of_floats Dtype.Bool [||] [| 0.0 |]) in
      let mask = b1 (Pad cfg2) [ keep; off ] in
      let out_shape = (Core.get_aval padded).shape in
      let wide_fill =
        b1 (Broadcast_in_dim { shape = out_shape; dims = [| 0 |] }) [ fill ]
      in
      (b1 Select_n [ mask; wide_fill; padded ], Some 0)
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

let dot_general_batch (dims : dot_dims) (vals : value list)
    (bdims : int option list) : value * int option =
  let bump a b = Array.map (fun d -> if d >= b then d + 1 else d) a in
  match (vals, bdims) with
  | [ lhs; rhs ], [ None; None ] -> (b1 (Dot_general dims) [ lhs; rhs ], None)
  | [ lhs; rhs ], [ Some l; Some r ] ->
      let dims' =
        {
          lhs_batch = Array.append [| l |] (bump dims.lhs_batch l);
          rhs_batch = Array.append [| r |] (bump dims.rhs_batch r);
          lhs_contract = bump dims.lhs_contract l;
          rhs_contract = bump dims.rhs_contract r;
        }
      in
      (b1 (Dot_general dims') [ lhs; rhs ], Some 0)
  | [ lhs; rhs ], [ Some l; None ] ->
      let ndim = Array.length (Core.get_aval lhs).shape in
      let dims' =
        {
          dims with
          lhs_batch = bump dims.lhs_batch l;
          lhs_contract = bump dims.lhs_contract l;
        }
      in
      let tensors_before =
        List.length
          (List.filter
             (fun d ->
               d < l
               && (not (Array.mem d dims'.lhs_batch))
               && not (Array.mem d dims'.lhs_contract))
             (List.init ndim Fun.id))
      in
      ( b1 (Dot_general dims') [ lhs; rhs ],
        Some (Array.length dims.lhs_batch + tensors_before) )
  | [ lhs; rhs ], [ None; Some r ] ->
      let lhs_ndim = Array.length (Core.get_aval lhs).shape in
      let rhs_ndim = Array.length (Core.get_aval rhs).shape in
      let dims' =
        {
          dims with
          rhs_batch = bump dims.rhs_batch r;
          rhs_contract = bump dims.rhs_contract r;
        }
      in
      let tensors_before =
        List.length
          (List.filter
             (fun d ->
               d < r
               && (not (Array.mem d dims'.rhs_batch))
               && not (Array.mem d dims'.rhs_contract))
             (List.init rhs_ndim Fun.id))
      in
      ( b1 (Dot_general dims') [ lhs; rhs ],
        Some (lhs_ndim - Array.length dims.lhs_contract + tensors_before) )
  | _ -> failwith "batching: dot_general arity"

let vmap_rule (axis_size : int) (prim : primitive) (vals : value list)
    (bdims : int option list) : value * int option =
  let un () =
    match (vals, bdims) with
    | [ x ], [ b ] -> unop_rule prim x b
    | _ -> failwith "batching: expected 1 operand"
  in
  let bin () =
    match (vals, bdims) with
    | [ x; y ], [ bx; by ] -> binop_rule prim x bx y by
    | _ -> failwith "batching: expected 2 operands"
  in
  match prim with
  | Neg | Sin | Cos | Exp | Log | Tanh | Abs | Sign | Acos | Acosh | Asin
  | Asinh | Atan | Atanh | Cbrt | Ceil | Clz | Conj | Copy | Cosh | Exp2 | Expm1
  | Floor | Imag | Integer_pow _ | Is_finite | Log1p | Logistic | Not
  | Population_count | Real | Round | Round_nearest_even | Rsqrt | Sinh | Sqrt
  | Square | Tan | Bitcast_convert_type _ | Reduce_precision _ | Bessel_i0e
  | Bessel_i1e | Digamma | Erf | Erf_inv | Erfc | Lgamma | Stop_gradient ->
      un ()
  | Igamma | Igamma_grad_a | Igammac | Polygamma | Zeta
  | Regularized_incomplete_beta ->
      naryop_rule prim vals bdims
  | Add | Add_any | Sub | Mul | Div | Max | Min | Pow | Eq | Lt | Gt | Ge | Le
  | Eq_to | Le_to | Lt_to | And | Atan2 | Complex | Mulhi | Ne | Nextafter | Or
  | Rem | Shift_left | Shift_right_arithmetic | Shift_right_logical | Xor | Tie
    ->
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
          argminmax_batch
            (fun ax -> Np_ints.mark_prim (Argmax { axis = ax; index_dtype }))
            axis x b
      | _ -> failwith "batching: expected 1 operand")
  | Argmin { axis; index_dtype } -> (
      match (vals, bdims) with
      | [ x ], [ b ] ->
          argminmax_batch
            (fun ax -> Np_ints.mark_prim (Argmin { axis = ax; index_dtype }))
            axis x b
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
  | Dce_sink | From_edtype _ | Ragged_dot_general | Rng_bit_generator _
  | Rng_uniform | To_edtype _ ->
      failwith "batching: vmap of this primitive not supported in M1"
  | Dot_general dims -> dot_general_batch dims vals bdims
  | Conv_general_dilated _ | Reduce_window _ | Reduce_window_max _
  | Reduce_window_min _ | Reduce_window_sum _ | Select_and_gather_add _
  | Select_and_scatter _ | Select_and_scatter_add _ ->
      failwith "batching: handled by unstacking in batch_process_primitive"
  | Platform_index _ ->
      failwith "batching: vmap of platform_index not supported"
  | Cond _ -> failwith "batching: cond handled by batch_process_primitive"
  | Threefry2x32 | Iota_2x32_shape _ | Random_seed | Random_split _
  | Random_fold_in | Random_bits _ | Random_wrap | Random_unwrap | While _
  | Cholesky | Householder_product | Lu | Lu_pivots_to_permutation _ | Qr _
  | Qr_pivoted _ | Geqrf | Geqp3 | Ormqr _ | Triangular_solve _
  | Tridiagonal_solve _ | Eig _ | Eigh _ | Hessenberg | Schur _ | Svd _
  | Tridiagonal _ ->
      failwith "batching: handled by unstacking in batch_process_primitive"
  | Collective _ ->
      failwith "batching: collectives handled by batch_process_primitive"
  | Fft _ -> (
      match (vals, bdims) with
      | [ x ], [ Some d ] -> (b1 prim [ moveaxis d 0 x ], Some 0)
      | [ x ], [ None ] -> (b1 prim [ x ], None)
      | _ -> failwith "batching: fft arity")
  | Custom_jvp_call _ | Custom_vjp_call _ | Custom_lin _ ->
      failwith "batching: custom rules handled by batch_process_primitive"
  | Checkpoint_name _ -> (
      match (vals, bdims) with
      | [ x ], [ d ] -> (b1 prim [ x ], d)
      | _ -> failwith "batching: checkpoint_name arity")
  | Xla_call _ ->
      failwith "batching: xla_call handled by batch_process_primitive"
  | Ffi_call _ -> failwith "batching: ffi_call handled by ffi_batch"

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

let named_axes : (string * trace * int) list ref = ref []

let find_axis name =
  match List.find_opt (fun (n, _, _) -> n = name) !named_axes with
  | Some entry -> entry
  | None -> invalid_arg ("unbound axis name: " ^ name)

let () =
  Core.named_axis_trace :=
    fun name ->
      List.find_map
        (fun (n, t, _) -> if n = name then Some t else None)
        !named_axes

let vmap_flat ?axis_name ?(out_axis = 0) (f : value list -> value list)
    (in_axes : int option list) (args : value list) : value list =
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
      let saved = !named_axes in
      (match axis_name with
      | Some name -> named_axes := (name, main, axis_size) :: saved
      | None -> ());
      let outs =
        Fun.protect
          ~finally:(fun () -> named_axes := saved)
          (fun () -> f tracers_in)
      in
      let vals_bdims =
        List.map (fun o -> to_batch_info main (Core.full_raise main o)) outs
      in
      List.map
        (fun (v, bdim) -> move_batch_axis axis_size bdim out_axis v)
        vals_bdims)

let vmap_out f in_axes args out_axis = vmap_flat ~out_axis f in_axes args

let vmap ?axis_name (f : value list -> value list) (in_axes : int option list) :
    value list -> value list =
 fun args -> vmap_flat ?axis_name f in_axes args

let collective_batch trace axis_size kind axis_name vals bdims =
  let bound =
    List.exists
      (fun (n, t, _) -> n = axis_name && t.level = trace.level)
      !named_axes
  in
  match (kind, vals, bdims) with
  | Axis_index, [], [] ->
      if bound then
        ( b1
            (Iota { dtype = Dtype.I32; shape = [| axis_size |]; dimension = 0 })
            [],
          Some 0 )
      else (b1 (Collective { kind; axis_name }) [], None)
  | _, [ v ], [ b ] when not bound ->
      (b1 (Collective { kind; axis_name }) [ v ], b)
  | Psum, [ v ], [ Some d ] -> (b1 (Reduce_sum [| d |]) [ v ], None)
  | Psum, [ v ], [ None ] ->
      let a = Core.get_aval v in
      let n = Array.fold_left ( * ) 1 a.shape in
      let scale =
        Concrete
          (Ndarray.of_floats a.dtype a.shape
             (Array.make n (float_of_int axis_size)))
      in
      (b1 Mul [ v; scale ], None)
  | Pmax, [ v ], [ Some d ] -> (b1 (Reduce_max [| d |]) [ v ], None)
  | Pmin, [ v ], [ Some d ] -> (b1 (Reduce_min [| d |]) [ v ], None)
  | (Pmax | Pmin), [ v ], [ None ] -> (v, None)
  | All_gather { axis; _ }, [ v ], [ b ] ->
      (move_batch_axis axis_size b axis v, None)
  | Pbroadcast source, [ v ], [ Some d ] ->
      (List.nth (Core.bind (Unstack d) [ v ]) source, None)
  | Pbroadcast _, [ v ], [ None ] -> (v, None)
  | Ppermute pairs, [ v ], [ b ] ->
      let moved = move_batch_axis axis_size b 0 v in
      let rows = Array.of_list (Core.bind (Unstack 0) [ moved ]) in
      let zero = Ad_util.zeros_like_aval (Core.get_aval rows.(0)) in
      let out = Array.make axis_size zero in
      Array.iter (fun (src, dst) -> out.(dst) <- rows.(src)) pairs;
      (b1 (Stack 0) (Array.to_list out), Some 0)
  | _ -> failwith "batching: collective arity"

let stop_gradient_rule =
  lazy
    (Core.register_custom_rule
       (Core.Jvp_rule
          {
            primal = Fun.id;
            jvp =
              (fun primals _ ->
                (primals, List.map Ad_util.symbolic_zero_like_value primals));
          }))

let stop_gradient v = b1 (Custom_jvp_call (Lazy.force stop_gradient_rule)) [ v ]

let cond_batch ~platforms (axis_size : int) (branches : closed_jaxpr list)
    (vals : value list) (bdims : int option list) : value list * int option list
    =
  let bcast shape v =
    let a = Core.get_aval v in
    if Array.length a.shape = Array.length shape then v
    else
      b1
        (Broadcast_in_dim
           { shape; dims = Array.init (Array.length a.shape) Fun.id })
        [ v ]
  in
  match (vals, bdims, platforms) with
  | _ :: ops, Some _ :: op_bdims, Some p ->
      let moved =
        List.map2 (fun v b -> move_batch_axis axis_size b 0 v) ops op_bdims
      in
      let index = b1 (Platform_index p) [] in
      let moved_avals = List.map Core.get_aval moved in
      let batch_branch (branch : closed_jaxpr) : closed_jaxpr =
        Jaxpr.make_jaxpr moved_avals (fun bops ->
            vmap
              (fun a -> Jaxpr.eval_closed_jaxpr branch a)
              (List.map (fun _ -> Some 0) bops)
              bops)
      in
      let outs =
        Core.bind
          (Cond { branches = List.map batch_branch branches; platforms })
          (index :: moved)
      in
      (outs, List.map (fun _ -> Some 0) outs)
  | index :: ops, Some pred_dim :: op_bdims, None ->
      let index = if pred_dim = 0 then index else moveaxis pred_dim 0 index in
      let moved =
        List.map2 (fun v b -> move_batch_axis axis_size b 0 v) ops op_bdims
      in
      let branch_outs =
        List.mapi
          (fun i (branch : closed_jaxpr) ->
            let predicate =
              b1 Eq
                [
                  index;
                  Concrete
                    (Ndarray.of_floats (Core.get_aval index).dtype [||]
                       [| float_of_int i |]);
                ]
            in
            let ops_ =
              List.map
                (fun x ->
                  let shape = (Core.get_aval x).shape in
                  let stopped = b1 Stop_gradient [ x ] in
                  let pred = bcast shape predicate in
                  b1 Select_n [ pred; stopped; x ])
                moved
            in
            vmap_flat
              (fun a -> Jaxpr.eval_closed_jaxpr branch a)
              (List.map (fun _ -> Some 0) ops_)
              ops_)
          branches
      in
      let n_out = List.length (List.hd branch_outs) in
      let outs =
        List.init n_out (fun k ->
            let cases = List.map (fun o -> List.nth o k) branch_outs in
            let shape = (Core.get_aval (List.hd cases)).shape in
            b1 Select_n (bcast shape index :: cases))
      in
      (outs, List.map (fun _ -> Some 0) outs)
  | pred :: ops, None :: op_bdims, _ ->
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
      let outs =
        Core.bind
          (Cond { branches = List.map batch_branch branches; platforms })
          (pred :: moved)
      in
      (outs, List.map (fun _ -> Some 0) outs)
  | _ -> failwith "batching: cond expects a predicate"

let scan_batch (axis_size : int) ~(length : int) ~(reverse : bool)
    ~(num_consts : int) ~(num_carry : int) ~(unroll : int)
    (jaxpr : closed_jaxpr) (vals : value list) (bdims : int option list) :
    value list * int option list =
  let split3 l =
    match Util.split_list l [ num_consts; num_carry ] with
    | [ a; b; c ] -> (a, b, c)
    | _ -> failwith "batching: scan split"
  in
  let consts, init, xs = split3 vals in
  let const_b, init_b, xs_b = split3 bdims in
  let flag = Option.is_some in
  let const_batched = List.map flag const_b in
  let init_batched = List.map flag init_b in
  let xs_batched = List.map flag xs_b in
  let num_ys = List.length jaxpr.jaxpr.outs - num_carry in
  let batched_aval is_batched (a : aval) =
    if is_batched then { a with shape = Array.append [| axis_size |] a.shape }
    else a
  in
  let binder_avals =
    List.map (fun (v : var) -> v.vaval) jaxpr.jaxpr.in_binders
  in
  let build carry_batched =
    let flags = const_batched @ carry_batched @ xs_batched in
    let out_flags = ref [] in
    let instantiate = carry_batched @ List.init num_ys (fun _ -> false) in
    let closed =
      Jaxpr.make_jaxpr (List.map2 batched_aval flags binder_avals) (fun inner ->
          Core.with_new_main KBatch (GAxisSize axis_size) (fun main ->
              let ins =
                List.map2
                  (fun v is_batched ->
                    if is_batched then Tracer (new_batch_tracer main v (Some 0))
                    else v)
                  inner flags
              in
              let outs = Jaxpr.eval_closed_jaxpr jaxpr ins in
              let pairs =
                List.map2
                  (fun o inst ->
                    let v, b = to_batch_info main (Core.full_raise main o) in
                    match (b, inst) with
                    | None, false -> (v, false)
                    | None, true -> (move_batch_axis axis_size None 0 v, true)
                    | Some _, _ -> (move_batch_axis axis_size b 0 v, true))
                  outs instantiate
              in
              out_flags := List.map snd pairs;
              List.map fst pairs))
    in
    (closed, !out_flags)
  in
  let split_carry l =
    match Util.split_list l [ num_carry ] with
    | [ a; b ] -> (a, b)
    | _ -> failwith "batching: scan split"
  in
  let rec fixpoint carry_batched =
    let closed, out_flags = build carry_batched in
    let carry_out, ys_batched = split_carry out_flags in
    if carry_out = carry_batched then (closed, carry_batched, ys_batched)
    else fixpoint (List.map2 ( || ) carry_batched carry_out)
  in
  let new_body, carry_batched, ys_batched = fixpoint init_batched in
  let new_consts =
    List.map2
      (fun v b ->
        match b with
        | Some d when d <> 0 -> move_batch_axis axis_size b 0 v
        | _ -> v)
      consts const_b
  in
  let new_init =
    List.map2
      (fun (v, b) now -> if now then move_batch_axis axis_size b 0 v else v)
      (List.combine init init_b) carry_batched
  in
  let new_xs =
    List.map2
      (fun v b ->
        match b with
        | Some d when d <> 1 -> move_batch_axis axis_size b 1 v
        | _ -> v)
      xs xs_b
  in
  let outs =
    Core.bind
      (Scan { length; reverse; num_consts; num_carry; jaxpr = new_body; unroll })
      (new_consts @ new_init @ new_xs)
  in
  let out_bdims =
    List.map (fun b -> if b then Some 0 else None) carry_batched
    @ List.map (fun b -> if b then Some 1 else None) ys_batched
  in
  (outs, out_bdims)

let unstacks_operands prim (bdims : int option list) =
  let any_batched = List.exists (fun b -> b <> None) in
  match (prim, bdims) with
  | Triangular_solve _, [ None; _ ] -> false
  | Custom_linear_solve _, _ -> false
  | ( ( Conv_general_dilated _ | Reduce_window _ | Reduce_window_max _
      | Reduce_window_min _ | Reduce_window_sum _ | Select_and_gather_add _
      | Select_and_scatter _ | Select_and_scatter_add _ | Threefry2x32
      | Iota_2x32_shape _ | Random_seed | Random_split _ | Random_fold_in
      | Random_bits _ | Random_wrap | Random_unwrap | While _ | Cholesky
      | Householder_product | Lu | Lu_pivots_to_permutation _ | Qr _
      | Triangular_solve _ | Tridiagonal_solve _ | Eig _ | Eigh _ | Hessenberg
      | Schur _ | Svd _ | Tridiagonal _ | Rng_bit_generator _ ),
      _ ) ->
      true
  | Custom_vjp_call id, _ -> (
      match Core.custom_rule id with Core.Callback_rule _ -> true | _ -> false)
  | Dynamic_slice _, _ :: indices -> any_batched indices
  | Dynamic_update_slice, _ :: _ :: indices -> any_batched indices
  | Pad _, [ _; padding ] -> padding <> None
  | Reduce _, _ ->
      let n = List.length bdims / 2 in
      any_batched (List.filteri (fun i _ -> i >= n) bdims)
  | _ -> false

let unstacked_batch axis_size prim vals bdims =
  let per_element =
    List.map2
      (fun v b ->
        match b with
        | None -> List.init axis_size (fun _ -> v)
        | Some d -> Core.bind (Unstack d) [ v ])
      vals bdims
  in
  let results =
    List.init axis_size (fun i ->
        Core.bind prim (List.map (fun column -> List.nth column i) per_element))
  in
  match results with
  | [] -> failwith "batching: empty batch"
  | first :: _ ->
      List.mapi
        (fun j _ ->
          Core.bind1 (Stack 0) (List.map (fun out -> List.nth out j) results))
        first

let batched_custom_rules : (int * int, int) Hashtbl.t = Hashtbl.create 16
let all_mapped l = List.map (fun _ -> Some 0) l
let vmap_all f args = vmap_flat f (all_mapped args) args

let split_at_n l n =
  match Util.split_list l [ n ] with
  | [ a; b ] -> (a, b)
  | _ -> failwith "batching: custom rule arity"

let vmap_keeping_symbolic_zeros f args =
  let inner_aval (a : aval) =
    { a with shape = Array.sub a.shape 1 (Array.length a.shape - 1) }
  in
  let is_zero = List.map Ad_util.is_symbolic_zero args in
  let mapped = List.filter (fun v -> not (Ad_util.is_symbolic_zero v)) args in
  let out_zero = ref [] in
  let rebuild given =
    let rest = ref given in
    List.map2
      (fun zero arg ->
        if zero then
          Ad_util.symbolic_zero_of_aval (inner_aval (Core.get_aval arg))
        else
          match !rest with
          | v :: tl ->
              rest := tl;
              v
          | [] -> failwith "batching: custom rule arity")
      is_zero args
  in
  let outs =
    vmap_all
      (fun given ->
        let outs = f (rebuild given) in
        out_zero := List.map Ad_util.is_symbolic_zero outs;
        outs)
      mapped
  in
  List.map2
    (fun zero o ->
      if zero then Ad_util.symbolic_zero_of_aval (Core.get_aval o) else o)
    !out_zero outs

let batch_custom_rule id =
  let primal_of p args = vmap_all p args in
  let vmap_pairs f args =
    let n_out = ref 0 in
    let out =
      vmap_keeping_symbolic_zeros
        (fun given ->
          let a, b = f given in
          n_out := List.length a;
          a @ b)
        args
    in
    split_at_n out !n_out
  in
  let lift g res xs =
    let n = List.length res in
    vmap_keeping_symbolic_zeros
      (fun all ->
        let r, x = split_at_n all n in
        g r x)
      (res @ xs)
  in
  match Core.custom_rule id with
  | Core.Jvp_rule { primal; jvp } ->
      Core.Jvp_rule
        {
          primal = primal_of primal;
          jvp =
            (fun ps ts ->
              let n = List.length ps in
              vmap_pairs
                (fun all ->
                  let p, t = split_at_n all n in
                  jvp p t)
                (ps @ ts));
        }
  | Core.Vjp_rule { primal; fwd; bwd; fwd_perturbed } ->
      Core.Vjp_rule
        {
          primal = primal_of primal;
          fwd = vmap_pairs fwd;
          bwd = lift bwd;
          fwd_perturbed =
            Option.map
              (fun rule perturbed -> vmap_pairs (rule perturbed))
              fwd_perturbed;
        }
  | Core.Linear_rule { forward; transpose; res_differentiable } ->
      Core.Linear_rule
        {
          forward = lift forward;
          transpose = lift transpose;
          res_differentiable;
        }
  | Core.Callback_rule _ -> failwith "batching: callbacks are unstacked"

let batched_rule_id id axis_size =
  match Hashtbl.find_opt batched_custom_rules (id, axis_size) with
  | Some batched -> batched
  | None ->
      let batched = Core.register_custom_rule (batch_custom_rule id) in
      (match Core.custom_names_of id with
      | Some (name, jvp_name) ->
          Core.register_custom_names batched ~name ~jvp_name
      | None -> ());
      Hashtbl.replace batched_custom_rules (id, axis_size) batched;
      batched

let custom_batch axis_size prim vals bdims =
  let moved =
    List.map2 (fun v b -> move_batch_axis axis_size b 0 v) vals bdims
  in
  let prim' =
    match prim with
    | Custom_jvp_call id -> Custom_jvp_call (batched_rule_id id axis_size)
    | Custom_vjp_call id -> Custom_vjp_call (batched_rule_id id axis_size)
    | Custom_lin { rule; n_res; out_avals } ->
        Custom_lin
          {
            rule = batched_rule_id rule axis_size;
            n_res;
            out_avals =
              List.map
                (fun (a : aval) ->
                  { a with shape = Array.append [| axis_size |] a.shape })
                out_avals;
          }
    | _ -> failwith "batching: not a custom primitive"
  in
  Core.bind prim' moved

let jvp_batch_cache :
    ( int * int * int option list * aval list * bool,
      int * int option list )
    Hashtbl.t =
  Hashtbl.create 16

let batched_jvp_rule id axis_size vals bdims primal jvp =
  let avals = List.map Core.get_aval vals in
  let key = (id, axis_size, bdims, avals, Config.x64_enabled ()) in
  match Hashtbl.find_opt jvp_batch_cache key with
  | Some r -> r
  | None ->
      let subtrace f dims inner record =
        Core.with_new_main KBatch (GAxisSize axis_size) (fun main ->
            let ins =
              List.map2
                (fun v b ->
                  match b with
                  | Some _ -> Tracer (new_batch_tracer main v b)
                  | None -> v)
                inner dims
            in
            let outs = f ins in
            let pairs =
              List.map
                (fun o -> to_batch_info main (Core.full_raise main o))
                outs
            in
            record main (List.map snd pairs);
            List.map fst pairs)
      in
      let out_dims = ref [] in
      let primal' args =
        subtrace primal bdims args (fun _ dims -> out_dims := dims)
      in
      ignore (Jaxpr.make_jaxpr avals primal');
      let dims = !out_dims in
      let inner_zero (t : value) b =
        match b with
        | Some d ->
            Ad_util.symbolic_zero_of_aval (mapped_aval d (Core.get_aval t))
        | None -> t
      in
      let jvp' ps ts =
        let po = ref [] and to_ = ref [] in
        let n_out = ref 0 in
        let result =
          Core.with_new_main KBatch (GAxisSize axis_size) (fun main ->
              let wrap v b =
                match b with
                | Some _ when Ad_util.is_symbolic_zero v -> inner_zero v b
                | Some _ -> Tracer (new_batch_tracer main v b)
                | None -> v
              in
              let ips = List.map2 wrap ps bdims in
              let its = List.map2 wrap ts bdims in
              let a, b = jvp ips its in
              n_out := List.length a;
              let unwrap o dim =
                if Ad_util.is_symbolic_zero o then
                  let av = Core.get_aval o in
                  match dim with
                  | Some d ->
                      Ad_util.symbolic_zero_of_aval
                        (unmapped_aval axis_size d av)
                  | None -> o
                else
                  let v, od = to_batch_info main (Core.full_raise main o) in
                  match (od, dim) with
                  | None, None -> v
                  | Some s, Some d -> if s = d then v else moveaxis s d v
                  | None, Some d -> broadcast_insert axis_size d v
                  | Some _, None ->
                      failwith "batching: custom rule output batch mismatch"
              in
              (List.map2 unwrap a dims, List.map2 unwrap b dims))
        in
        po := fst result;
        to_ := snd result;
        (!po, !to_)
      in
      let nid =
        Core.register_custom_rule
          (Core.Jvp_rule { primal = primal'; jvp = jvp' })
      in
      (match Core.custom_names_of id with
      | Some (name, jvp_name) -> Core.register_custom_names nid ~name ~jvp_name
      | None -> ());
      let r = (nid, dims) in
      Hashtbl.replace jvp_batch_cache key r;
      r

let batch_closed_axes (c : closed_jaxpr) axis_size in_dims out_dims =
  let binder_avals =
    List.filteri
      (fun i _ -> i >= List.length c.consts)
      (List.map (fun (v : var) -> v.vaval) c.jaxpr.in_binders)
  in
  let widened =
    List.map2
      (fun (a : aval) d ->
        match d with
        | Some d -> { a with shape = insert_size a.shape d axis_size }
        | None -> a)
      binder_avals in_dims
  in
  Jaxpr.make_jaxpr widened (fun inner ->
      Core.with_new_main KBatch (GAxisSize axis_size) (fun main ->
          let ins =
            List.map2
              (fun v d ->
                match d with
                | Some _ -> Tracer (new_batch_tracer main v d)
                | None -> v)
              inner in_dims
          in
          let outs = Jaxpr.eval_closed_jaxpr c ins in
          List.map2
            (fun o d ->
              let v, b = to_batch_info main (Core.full_raise main o) in
              match (b, d) with
              | None, None -> v
              | _, Some d -> move_batch_axis axis_size b d v
              | Some _, None -> failwith "batching: while output mismatch")
            outs out_dims))

let batch_flags (c : closed_jaxpr) axis_size flags =
  let binder_avals =
    List.filteri
      (fun i _ -> i >= List.length c.consts)
      (List.map (fun (v : var) -> v.vaval) c.jaxpr.in_binders)
  in
  let out_flags = ref [] in
  let widened =
    List.map2
      (fun (a : aval) f ->
        if f then { a with shape = insert_size a.shape 0 axis_size } else a)
      binder_avals flags
  in
  ignore
    (Jaxpr.make_jaxpr widened (fun inner ->
         Core.with_new_main KBatch (GAxisSize axis_size) (fun main ->
             let ins =
               List.map2
                 (fun v f ->
                   if f then Tracer (new_batch_tracer main v (Some 0)) else v)
                 inner flags
             in
             let outs = Jaxpr.eval_closed_jaxpr c ins in
             let pairs =
               List.map
                 (fun o -> to_batch_info main (Core.full_raise main o))
                 outs
             in
             out_flags := List.map (fun (_, b) -> Option.is_some b) pairs;
             List.map fst pairs)));
  !out_flags

let while_batch axis_size ~cond_nconsts ~body_nconsts (cond : closed_jaxpr)
    (body : closed_jaxpr) vals bdims =
  let split l =
    match Util.split_list l [ cond_nconsts; body_nconsts ] with
    | [ a; b; c ] -> (a, b, c)
    | _ -> failwith "batching: while split"
  in
  let cconsts, bconsts, init = split vals in
  let cconst_dims, bconst_dims, init_dims = split bdims in
  let flag = Option.is_some in
  let cconst_bat = List.map flag cconst_dims in
  let bconst_bat = List.map flag bconst_dims in
  let rec fixpoint carry_bat =
    let out = batch_flags body axis_size (bconst_bat @ carry_bat) in
    let merged = List.map2 ( || ) carry_bat out in
    if merged = carry_bat then carry_bat else fixpoint merged
  in
  let carry_bat = fixpoint (List.map flag init_dims) in
  let pred_bat =
    List.hd (batch_flags cond axis_size (cconst_bat @ carry_bat))
  in
  let carry_dims, cond_out =
    if pred_bat then (List.map (fun _ -> Some 0) carry_bat, [ Some 0 ])
    else
      let rank =
        match cond.jaxpr.outs with
        | [ A_var v ] -> Array.length v.vaval.shape
        | _ -> 0
      in
      (List.map (fun b -> if b then Some rank else None) carry_bat, [ None ])
  in
  let body' =
    batch_closed_axes body axis_size (bconst_dims @ carry_dims) carry_dims
  in
  let cond' =
    batch_closed_axes cond axis_size (cconst_dims @ carry_dims) cond_out
  in
  let new_init =
    List.map2
      (fun (x, old_axis) new_axis ->
        match (old_axis, new_axis) with
        | None, None -> x
        | _, Some d -> move_batch_axis axis_size old_axis d x
        | Some _, None -> failwith "batching: while carry mismatch")
      (List.combine init init_dims)
      carry_dims
  in
  let outs =
    Core.bind
      (While { cond_nconsts; body_nconsts; cond = cond'; body = body' })
      (cconsts @ bconsts @ new_init)
  in
  (outs, carry_dims)

let ffi_shift_layout layout =
  let n = Array.length layout in
  Array.init (n + 1) (fun i -> if i = n then 0 else layout.(i) + 1)

let ffi_batch axis_size (c : ffi_call) vals bdims =
  let moved =
    List.map2
      (fun v b -> match b with None -> v | Some d -> moveaxis d 0 v)
      vals bdims
  in
  let batched_result_avals =
    List.map
      (fun (a : aval) -> { a with shape = insert_size a.shape 0 axis_size })
      c.ffi_result_avals
  in
  let with_layouts inputs =
    Ffi_call
      {
        c with
        ffi_result_avals = batched_result_avals;
        ffi_input_layouts = inputs;
        ffi_output_layouts = List.map ffi_shift_layout c.ffi_output_layouts;
      }
  in
  match c.ffi_vmap_method with
  | Some ("sequential" | "sequential_unrolled") ->
      unstacked_batch axis_size (Ffi_call c) moved
        (List.map (Option.map (fun _ -> 0)) bdims)
  | Some "legacy_vectorized" ->
      let inputs =
        List.map2
          (fun layout b -> if b = None then layout else ffi_shift_layout layout)
          c.ffi_input_layouts bdims
      in
      Core.bind (with_layouts inputs) moved
  | Some (("expand_dims" | "broadcast_all") as m) ->
      let size = if m = "broadcast_all" then axis_size else 1 in
      let broadcast v b =
        if b <> None then v
        else
          let a = Core.get_aval v in
          b1
            (Broadcast_in_dim
               {
                 shape = insert_size a.shape 0 size;
                 dims = Array.init (Array.length a.shape) (fun i -> i + 1);
               })
            [ v ]
      in
      Core.bind
        (with_layouts (List.map ffi_shift_layout c.ffi_input_layouts))
        (List.map2 broadcast moved bdims)
  | _ ->
      failwith
        (Printf.sprintf
           "vmap is only supported for the ffi_call primitive when vmap_method \
            is one of 'sequential', 'sequential_unrolled', 'expand_dims', \
            'broadcast_all', or 'legacy_vectorized'. Got vmap_method=%s."
           (match c.ffi_vmap_method with
           | None -> "None"
           | Some m -> "'" ^ m ^ "'"))

let batch_call_cache :
    ( int * int * int option list * bool,
      closed_jaxpr * int option list )
    Hashtbl.t =
  Hashtbl.create 64

let batch_jaxpr (c : closed_jaxpr) axis_size vals bdims =
  let key = (c.jid, axis_size, bdims, Config.x64_enabled ()) in
  match Hashtbl.find_opt batch_call_cache key with
  | Some r -> r
  | None ->
      let out_dims = ref [] in
      let closed =
        Jaxpr.make_jaxpr (List.map Core.get_aval vals) (fun inner ->
            Core.with_new_main KBatch (GAxisSize axis_size) (fun main ->
                let ins =
                  List.map2
                    (fun v b ->
                      match b with
                      | Some _ -> Tracer (new_batch_tracer main v b)
                      | None -> v)
                    inner bdims
                in
                let outs = Jaxpr.eval_closed_jaxpr c ins in
                let pairs =
                  List.map
                    (fun o -> to_batch_info main (Core.full_raise main o))
                    outs
                in
                out_dims := List.map snd pairs;
                List.map fst pairs))
      in
      Pretty_printer.register_call_name closed.jid
        (Pretty_printer.call_name c.jid);
      Core.inherit_inline ~from:c.jid closed.jid;
      let r = (closed, !out_dims) in
      Hashtbl.replace batch_call_cache key r;
      r

let batch_solve_jaxpr (c : closed_jaxpr) axis_size vals bdims =
  Jaxpr.make_jaxpr (List.map Core.get_aval vals) (fun inner ->
      Core.with_new_main KBatch (GAxisSize axis_size) (fun main ->
          let ins =
            List.map2
              (fun v b ->
                match b with
                | Some _ -> Tracer (new_batch_tracer main v (Some 0))
                | None -> v)
              inner bdims
          in
          List.map
            (fun o ->
              let v, b = to_batch_info main (Core.full_raise main o) in
              bdim_at_front axis_size v b)
            (Jaxpr.eval_closed_jaxpr c ins)))

let custom_linear_solve_batch axis_size solve transpose_solve num_consts vals
    bdims =
  let vals =
    List.mapi
      (fun i (v, b) ->
        if i < num_consts && b = None then v else bdim_at_front axis_size v b)
      (List.combine vals bdims)
  in
  let bdims =
    List.mapi
      (fun i b -> if i < num_consts && b = None then None else Some 0)
      bdims
  in
  let solve = batch_solve_jaxpr solve axis_size vals bdims in
  let transpose_solve =
    Option.map
      (fun ts -> batch_solve_jaxpr ts axis_size vals bdims)
      transpose_solve
  in
  Core.bind (Custom_linear_solve { solve; transpose_solve; num_consts }) vals

let triangular_solve_batch prim vals bdims =
  match (prim, vals, bdims) with
  | Triangular_solve { left_side; _ }, [ x; y ], [ None; Some by ] ->
      let n = Array.length (Core.get_aval y).shape in
      let y = moveaxis by (if left_side then n - 1 else n - 2) y in
      let s = (Core.get_aval y).shape in
      let flat =
        if left_side then
          Array.append (Array.sub s 0 (n - 2)) [| s.(n - 2) * s.(n - 1) |]
        else
          Array.append
            (Array.sub s 0 (n - 3))
            [| s.(n - 3) * s.(n - 2); s.(n - 1) |]
      in
      let out = b1 prim [ x; b1 (Reshape flat) [ y ] ] in
      (b1 (Reshape s) [ out ], Some (if left_side then n - 1 else n - 2))
  | _, [ x; y ], [ None; None ] -> (b1 prim [ x; y ], None)
  | _ -> failwith "batching: triangular_solve arity"

let batch_process_primitive (trace : trace) (prim : primitive)
    (args : value list) : value list =
  let axis_size = axis_size_of trace in
  let pairs = List.map (to_batch_info trace) args in
  let vals = List.map fst pairs and bdims = List.map snd pairs in
  match prim with
  | Xla_call c ->
      let closed, out_dims = batch_jaxpr c axis_size vals bdims in
      let outs = Core.bind (Xla_call closed) vals in
      List.map2 (fun o b -> Tracer (new_batch_tracer trace o b)) outs out_dims
  | Collective { kind; axis_name } ->
      let out, out_bdim =
        collective_batch trace axis_size kind axis_name vals bdims
      in
      [ Tracer (new_batch_tracer trace out out_bdim) ]
  | While { cond_nconsts; body_nconsts; cond; body } ->
      let outs, dims =
        while_batch axis_size ~cond_nconsts ~body_nconsts cond body vals bdims
      in
      List.map2 (fun o b -> Tracer (new_batch_tracer trace o b)) outs dims
  | Ffi_call c ->
      List.map
        (fun o -> Tracer (new_batch_tracer trace o (Some 0)))
        (ffi_batch axis_size c vals bdims)
  | Custom_linear_solve { solve; transpose_solve; num_consts } ->
      List.map
        (fun o -> Tracer (new_batch_tracer trace o (Some 0)))
        (custom_linear_solve_batch axis_size solve transpose_solve num_consts
           vals bdims)
  | Triangular_solve _ when not (unstacks_operands prim bdims) ->
      let out, b = triangular_solve_batch prim vals bdims in
      [ Tracer (new_batch_tracer trace out b) ]
  | Qr_pivoted _ | Geqrf | Geqp3 | Ormqr _ ->
      let args = List.map2 (bdim_at_front axis_size) vals bdims in
      List.map
        (fun o -> Tracer (new_batch_tracer trace o (Some 0)))
        (Core.bind prim args)
  | _ when unstacks_operands prim bdims ->
      List.map
        (fun o -> Tracer (new_batch_tracer trace o (Some 0)))
        (unstacked_batch axis_size prim vals bdims)
  | Split _ | Unstack _ | Optimization_barrier | Sort _ | Top_k _ ->
      let outs, out_bdims = vmap_rule_multi axis_size prim vals bdims in
      List.map2 (fun o b -> Tracer (new_batch_tracer trace o b)) outs out_bdims
  | Cond { branches; platforms } ->
      let outs, out_bdims =
        cond_batch ~platforms axis_size branches vals bdims
      in
      List.map2 (fun o b -> Tracer (new_batch_tracer trace o b)) outs out_bdims
  | Scan { length; reverse; num_consts; num_carry; jaxpr; unroll } ->
      let outs, out_bdims =
        scan_batch axis_size ~length ~reverse ~num_consts ~num_carry ~unroll
          jaxpr vals bdims
      in
      List.map2 (fun o b -> Tracer (new_batch_tracer trace o b)) outs out_bdims
  | Custom_jvp_call id
    when match Core.custom_rule id with Core.Jvp_rule _ -> true | _ -> false
    -> (
      match Core.custom_rule id with
      | Core.Jvp_rule { primal; jvp } ->
          let nid, dims = batched_jvp_rule id axis_size vals bdims primal jvp in
          let outs = Core.bind (Custom_jvp_call nid) vals in
          List.map2 (fun o b -> Tracer (new_batch_tracer trace o b)) outs dims
      | _ -> failwith "batching: expected a jvp rule")
  | Custom_jvp_call _ | Custom_vjp_call _ | Custom_lin _ ->
      List.map
        (fun o -> Tracer (new_batch_tracer trace o (Some 0)))
        (custom_batch axis_size prim vals bdims)
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
