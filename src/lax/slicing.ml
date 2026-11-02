open Types

let resolve_strides n = function Some s -> s | None -> Array.make n 1

let type_error fmt =
  Printf.ksprintf (fun msg -> raise (Errors.Jax_type_error msg)) fmt

let all_ge a b =
  let n = Stdlib.min (Array.length a) (Array.length b) in
  let ok = ref true in
  for i = 0 to n - 1 do
    if a.(i) < b.(i) then ok := false
  done;
  !ok

let check_slice start_indices limit_indices strides ishape =
  let t = Utils.py_tuple in
  Utils.check_shapelike "slice" "start_indices" start_indices;
  Utils.check_shapelike "slice" "limit_indices" limit_indices;
  let n = Array.length ishape in
  if n <> Array.length start_indices then
    type_error
      "slice start_indices must have length equal to the number of dimensions \
       of the operand, got indices %s for operand shape %s."
      (t start_indices) (t ishape);
  if Array.length start_indices <> Array.length limit_indices then
    type_error
      "slice limit_indices must have the same length as start_indices, got \
       start_indices %s and limit_indices %s."
      (t start_indices) (t limit_indices);
  if not (all_ge ishape limit_indices) then
    type_error
      "slice limit_indices must be less than or equal to operand shape, got \
       limit_indices %s for operand shape %s."
      (t limit_indices) (t ishape);
  if Array.exists (fun s -> s < 0) start_indices then
    type_error
      "slice start_indices must be greater than or equal to zero, got \
       start_indices of %s."
      (t start_indices);
  if not (all_ge limit_indices start_indices) then
    type_error
      "slice limit_indices must be greater than or equal to start_indices, got \
       start_indices %s and limit_indices %s."
      (t start_indices) (t limit_indices);
  match strides with
  | Some s when s <> Array.make n 1 ->
      Utils.check_shapelike "slice" "strides" s;
      if Array.length s <> n then
        type_error
          "slice strides must have length equal to the number of dimensions of \
           the operand, got strides %s for operand shape %s."
          (t s) (t ishape);
      if Array.exists (fun x -> x < 0) s then
        type_error "slice strides must be positive, got %s" (t s)
  | _ -> ()

let check_dynamic_slice slice_sizes ishape start_shapes =
  let t = Utils.py_tuple in
  if not (all_ge ishape slice_sizes) then
    type_error
      "slice slice_sizes must be less than or equal to operand shape, got \
       slice_sizes %s for operand shape %s."
      (t slice_sizes) (t ishape);
  if Array.exists (fun s -> s < 0) slice_sizes then
    type_error
      "slice slice_sizes must be greater than or equal to zero, got \
       slice_sizes of %s."
      (t slice_sizes);
  if List.exists (fun s -> Array.length s <> 0) start_shapes then
    type_error
      "start_indices arguments to dynamic_slice must be scalars,  got indices \
       %s"
      (String.concat ", " (List.map t start_shapes))

let check_dynamic_update_slice ishape update_shape start_shapes =
  let t = Utils.py_tuple in
  if Array.length ishape <> Array.length update_shape then
    type_error
      "dynamic_update_slice update must have the same rank as operand, got \
       update shape %s for operand shape %s."
      (t update_shape) (t ishape);
  if Array.length ishape <> List.length start_shapes then
    type_error
      "dynamic_update_slice start_indices must have length equal to the rank \
       of operand, got indices %s for operand shape %s."
      (String.concat ", " (List.map t start_shapes))
      (t ishape);
  if not (all_ge ishape update_shape) then
    type_error
      "dynamic_update_slice update shape must be smaller than operand shape, \
       got update shape %s for operand shape %s."
      (t update_shape) (t ishape);
  if List.exists (fun s -> Array.length s <> 0) start_shapes then
    type_error
      "start_indices arguments to dynamic_update_slice must be scalars, got \
       indices %s"
      (String.concat ", " (List.map t start_shapes))

let slice_shape start_indices limit_indices strides ishape =
  check_slice start_indices limit_indices strides ishape;
  let n = Array.length ishape in
  let strides = resolve_strides n strides in
  Array.init n (fun i ->
      let d = limit_indices.(i) - start_indices.(i) in
      if d <= 0 then 0 else (d + strides.(i) - 1) / strides.(i))

let slice_impl start_indices limit_indices strides operand =
  let os = Ndarray.shape operand in
  let n = Array.length os in
  let strides = resolve_strides n strides in
  let out_shape = slice_shape start_indices limit_indices (Some strides) os in
  let out_n = Utils.prod out_shape in
  let out = Array.make out_n 0.0 in
  for f = 0 to out_n - 1 do
    let oidx = Utils.decode f out_shape in
    let iidx =
      Array.init n (fun d -> start_indices.(d) + (oidx.(d) * strides.(d)))
    in
    out.(f) <- Ndarray.get_f operand iidx
  done;
  Ndarray.of_floats (Ndarray.dtype operand) out_shape out

let clamp_start raw dim_size slice_size =
  let hi = dim_size - slice_size in
  if raw < 0 then 0 else if raw > hi then hi else raw

let read_index nd = Int64.to_int (Ndarray.get_i64 nd [||])

let dynamic_slice_impl slice_sizes inputs =
  match inputs with
  | operand :: idx_nds ->
      let os = Ndarray.shape operand in
      check_dynamic_slice slice_sizes os (List.map Ndarray.shape idx_nds);
      let n = Array.length os in
      let starts =
        Array.init n (fun d ->
            clamp_start (read_index (List.nth idx_nds d)) os.(d) slice_sizes.(d))
      in
      let out_n = Utils.prod slice_sizes in
      let out = Array.make out_n 0.0 in
      for f = 0 to out_n - 1 do
        let oidx = Utils.decode f slice_sizes in
        let iidx = Array.init n (fun d -> starts.(d) + oidx.(d)) in
        out.(f) <- Ndarray.get_f operand iidx
      done;
      Ndarray.of_floats (Ndarray.dtype operand) slice_sizes out
  | [] -> failwith "lax: dynamic_slice expects an operand"

let read_all nd =
  let os = Ndarray.shape nd in
  let n = Utils.prod os in
  let arr = Array.make n 0.0 in
  let _ =
    Ndarray.fold
      (fun i x ->
        arr.(i) <- x;
        i + 1)
      0 nd
  in
  arr

let dynamic_update_slice_impl inputs =
  match inputs with
  | operand :: update :: idx_nds ->
      let os = Ndarray.shape operand in
      let us = Ndarray.shape update in
      check_dynamic_update_slice os us (List.map Ndarray.shape idx_nds);
      let n = Array.length os in
      let ostr = Utils.strides os in
      let starts =
        Array.init n (fun d ->
            clamp_start (read_index (List.nth idx_nds d)) os.(d) us.(d))
      in
      let out = read_all operand in
      let upd = read_all update in
      for f = 0 to Array.length upd - 1 do
        let uidx = Utils.decode f us in
        let flat = ref 0 in
        for d = 0 to n - 1 do
          flat := !flat + ((starts.(d) + uidx.(d)) * ostr.(d))
        done;
        out.(!flat) <- upd.(f)
      done;
      Ndarray.of_floats (Ndarray.dtype operand) os out
  | _ -> failwith "lax: dynamic_update_slice expects operand and update"

let mem arr x = Array.exists (fun y -> y = x) arr
let clamp lo hi v = if v < lo then lo else if v > hi then hi else v
let read_index_at indices multi = Int64.to_int (Ndarray.get_i64 indices multi)

let offset_operand_dims (gd : gather_dims) rank =
  let out = ref [] in
  for o = rank - 1 downto 0 do
    if
      (not (mem gd.collapsed_slice_dims o))
      && not (mem gd.g_operand_batching_dims o)
    then out := o :: !out
  done;
  Array.of_list !out

let gather_output_shape (gd : gather_dims) slice_sizes indices_shape
    operand_rank =
  let idx_rank = Array.length indices_shape in
  let output_rank = Array.length gd.offset_dims + idx_rank - 1 in
  let expanded = Array.sub indices_shape 0 (idx_rank - 1) in
  let ood = offset_operand_dims gd operand_rank in
  let out = Array.make output_rank 0 in
  let joff = ref 0 and kbatch = ref 0 in
  for d = 0 to output_rank - 1 do
    if mem gd.offset_dims d then begin
      out.(d) <- slice_sizes.(ood.(!joff));
      incr joff
    end
    else begin
      out.(d) <- expanded.(!kbatch);
      incr kbatch
    end
  done;
  out

let gather_shape (gd : gather_dims) slice_sizes indices_shape operand_shape =
  gather_output_shape gd slice_sizes indices_shape (Array.length operand_shape)

let gather_impl (gd : gather_dims) slice_sizes operand indices =
  let op_shape = Ndarray.shape operand in
  let op_rank = Array.length op_shape in
  let op_str = Utils.strides op_shape in
  let op_flat = read_all operand in
  let idx_shape = Ndarray.shape indices in
  let idx_rank = Array.length idx_shape in
  let out_shape = gather_output_shape gd slice_sizes idx_shape op_rank in
  let out_n = Utils.prod out_shape in
  let ood = offset_operand_dims gd op_rank in
  let out = Array.make out_n 0.0 in
  for f = 0 to out_n - 1 do
    let oidx = Utils.decode f out_shape in
    let batch_coords = Array.make (max 0 (idx_rank - 1)) 0 in
    let kb = ref 0 in
    Array.iteri
      (fun d v ->
        if not (mem gd.offset_dims d) then begin
          batch_coords.(!kb) <- v;
          incr kb
        end)
      oidx;
    let op_idx = Array.make op_rank 0 in
    Array.iteri
      (fun m o ->
        op_idx.(o) <- batch_coords.(gd.g_start_indices_batching_dims.(m)))
      gd.g_operand_batching_dims;
    Array.iteri
      (fun k o ->
        let full = Array.make idx_rank 0 in
        Array.blit batch_coords 0 full 0 (idx_rank - 1);
        full.(idx_rank - 1) <- k;
        let raw = read_index_at indices full in
        let hi = op_shape.(o) - slice_sizes.(o) in
        op_idx.(o) <- clamp 0 hi raw)
      gd.start_index_map;
    let joff = ref 0 in
    Array.iteri
      (fun d v ->
        if mem gd.offset_dims d then begin
          let o = ood.(!joff) in
          op_idx.(o) <- op_idx.(o) + v;
          incr joff
        end)
      oidx;
    let flat = ref 0 in
    for o = 0 to op_rank - 1 do
      flat := !flat + (op_idx.(o) * op_str.(o))
    done;
    out.(f) <- op_flat.(!flat)
  done;
  Ndarray.of_floats (Ndarray.dtype operand) out_shape out

let window_operand_dims (sd : scatter_dims) rank =
  let out = ref [] in
  for o = rank - 1 downto 0 do
    if
      (not (mem sd.inserted_window_dims o))
      && not (mem sd.s_operand_batching_dims o)
    then out := o :: !out
  done;
  Array.of_list !out

let scatter_impl combiner (sd : scatter_dims) operand indices updates =
  let op_shape = Ndarray.shape operand in
  let op_rank = Array.length op_shape in
  let op_str = Utils.strides op_shape in
  let upd_shape = Ndarray.shape updates in
  let idx_rank = Array.length (Ndarray.shape indices) in
  let out = read_all operand in
  let upd_flat = read_all updates in
  let wod = window_operand_dims sd op_rank in
  let win_size = Array.make op_rank 1 in
  Array.iteri
    (fun j o -> win_size.(o) <- upd_shape.(sd.update_window_dims.(j)))
    wod;
  for f = 0 to Array.length upd_flat - 1 do
    let uidx = Utils.decode f upd_shape in
    let scatter_coords = Array.make (max 0 (idx_rank - 1)) 0 in
    let ks = ref 0 in
    Array.iteri
      (fun d v ->
        if not (mem sd.update_window_dims d) then begin
          scatter_coords.(!ks) <- v;
          incr ks
        end)
      uidx;
    let op_idx = Array.make op_rank 0 in
    let in_bounds = ref true in
    Array.iteri
      (fun m o ->
        op_idx.(o) <- scatter_coords.(sd.s_scatter_indices_batching_dims.(m)))
      sd.s_operand_batching_dims;
    Array.iteri
      (fun k o ->
        let full = Array.make idx_rank 0 in
        Array.blit scatter_coords 0 full 0 (idx_rank - 1);
        full.(idx_rank - 1) <- k;
        let raw = read_index_at indices full in
        let hi = op_shape.(o) - win_size.(o) in
        if raw < 0 || raw > hi then in_bounds := false;
        op_idx.(o) <- clamp 0 hi raw)
      sd.scatter_dims_to_operand_dims;
    let jw = ref 0 in
    Array.iteri
      (fun d v ->
        if mem sd.update_window_dims d then begin
          let o = wod.(!jw) in
          op_idx.(o) <- op_idx.(o) + v;
          incr jw
        end)
      uidx;
    let flat = ref 0 in
    for o = 0 to op_rank - 1 do
      flat := !flat + (op_idx.(o) * op_str.(o))
    done;
    if !in_bounds then out.(!flat) <- combiner out.(!flat) upd_flat.(f)
  done;
  Ndarray.of_floats (Ndarray.dtype operand) op_shape out

let scatter_combiner = function
  | Scatter _ -> fun _ y -> y
  | Scatter_add _ -> ( +. )
  | Scatter_sub _ -> ( -. )
  | Scatter_mul _ -> ( *. )
  | Scatter_min _ -> Float.min
  | Scatter_max _ -> Float.max
  | _ -> failwith "lax: not a scatter primitive"

let default_gather_fill dt sh =
  let n = Array.fold_left ( * ) 1 sh in
  match dt with
  | d when Dtypes.issubdtype d `Inexact ->
      Ndarray.of_floats dt sh (Array.make n Float_format.nan)
  | Dtype.I64 -> Ndarray.of_int64s dt sh (Array.make n Int64.min_int)
  | Dtype.Uint64 -> Ndarray.of_int64s dt sh (Array.make n (-1L))
  | Dtype.Bool -> Ndarray.of_floats dt sh (Array.make n 1.0)
  | Dtype.I8 -> Ndarray.of_floats dt sh (Array.make n (-128.0))
  | Dtype.I16 -> Ndarray.of_floats dt sh (Array.make n (-32768.0))
  | Dtype.I32 -> Ndarray.of_floats dt sh (Array.make n (-2147483648.0))
  | Dtype.Uint8 -> Ndarray.of_floats dt sh (Array.make n 255.0)
  | Dtype.Uint16 -> Ndarray.of_floats dt sh (Array.make n 65535.0)
  | Dtype.Uint32 -> Ndarray.of_floats dt sh (Array.make n 4294967295.0)
  | d when Dtypes.issubdtype d `Signed ->
      Ndarray.of_floats dt sh (Array.make n (Dtypes.iinfo d).imin)
  | d -> Ndarray.of_floats dt sh (Array.make n (Dtypes.iinfo d).imax)

let gather_fill_expanded ?fill_value operand indices (gd : gather_dims)
    slice_sizes =
  let bind1 = Core.bind1 in
  let aval = Core.get_aval in
  let op_shape = (aval operand).shape and op_dt = (aval operand).dtype in
  let idx_aval = aval indices in
  let idt = idx_aval.dtype and ish = idx_aval.shape in
  let batch = Array.length ish - 1 in
  let upper =
    Array.map
      (fun o -> float_of_int (op_shape.(o) - slice_sizes.(o)))
      gd.start_index_map
  in
  let bound = Concrete (Ndarray.of_floats idt [| Array.length upper |] upper) in
  let bound =
    bind1 (Broadcast_in_dim { shape = ish; dims = [| batch |] }) [ bound ]
  in
  let zero =
    Concrete
      (Ndarray.of_floats idt ish (Array.make (Array.fold_left ( * ) 1 ish) 0.0))
  in
  let mask =
    bind1 And [ bind1 Ge [ indices; zero ]; bind1 Le [ indices; bound ] ]
  in
  let mask = bind1 (Reduce_and [| batch |]) [ mask ] in
  let gathered =
    bind1 (Gather { dimension_numbers = gd; slice_sizes }) [ operand; indices ]
  in
  let out_shape = (aval gathered).shape in
  let batch_dims =
    Array.of_list
      (List.filter
         (fun d -> not (Array.exists (fun o -> o = d) gd.offset_dims))
         (List.init (Array.length out_shape) Fun.id))
  in
  let mask =
    if
      (aval mask).shape = out_shape
      && batch_dims = Array.init (Array.length out_shape) Fun.id
    then mask
    else
      bind1 (Broadcast_in_dim { shape = out_shape; dims = batch_dims }) [ mask ]
  in
  let fill =
    match fill_value with
    | Some f ->
        Concrete
          (Ndarray.of_floats op_dt out_shape
             (Array.make (Array.fold_left ( * ) 1 out_shape) f))
    | None -> Concrete (default_gather_fill op_dt out_shape)
  in
  bind1 Select_n [ mask; fill; gathered ]

let fill_repr ?fill_value dt =
  let of_float f =
    if Float.is_integer f && Float.abs f < 1e16 then Printf.sprintf "%.0f" f
    else Printf.sprintf "%.17g" f
  in
  match fill_value with
  | Some f -> of_float f
  | None -> (
      match dt with
      | Dtype.Bool -> "True"
      | d when Dtypes.issubdtype d `Inexact -> "nan"
      | Dtype.I64 -> "-9223372036854775808"
      | Dtype.Uint64 -> "18446744073709551615"
      | d when Dtypes.issubdtype d `Signed -> of_float (Dtypes.iinfo d).imin
      | d -> of_float (Dtypes.iinfo d).imax)

let gather_fill ?(flag = Fun.id) ?fill_value operand indices (gd : gather_dims)
    slice_sizes =
  let dt = (Core.get_aval operand).dtype in
  Core.bind1
    (flag
       (Np_ints.mark_gather_fill
          (Gather { dimension_numbers = gd; slice_sizes })
          (fill_repr ?fill_value dt) fill_value))
    [ operand; indices ]

let gather_fill_lowered prim operand indices =
  let bind1 = Core.bind1 in
  let aval = Core.get_aval in
  let gd, slice_sizes =
    match prim with
    | Gather { dimension_numbers; slice_sizes } ->
        (dimension_numbers, slice_sizes)
    | _ -> invalid_arg "gather_fill_lowered: not a gather"
  in
  let fill_value =
    match Np_ints.gather_fill_of prim with
    | Some (_, fill) -> fill
    | None -> None
  in
  let op_shape = (aval operand).shape and op_dt = (aval operand).dtype in
  let idt =
    if Array.exists (fun d -> d >= 0x7fffffff) op_shape then Dtype.I64
    else Dtype.I32
  in
  let indices =
    if (aval indices).dtype = idt then indices
    else bind1 (Convert_element_type idt) [ indices ]
  in
  let ish = (aval indices).shape in
  let rank = Array.length ish in
  let batch = rank - 1 in
  let upper =
    Array.map
      (fun o -> float_of_int (op_shape.(o) - slice_sizes.(o)))
      gd.start_index_map
  in
  let bound = Concrete (Ndarray.of_floats idt [| Array.length upper |] upper) in
  let ge = bind1 Ge [ indices; Core.splat idt [||] 0.0 ] in
  let bound =
    if batch = 0 then bound
    else
      let expanded =
        bind1
          (Broadcast_in_dim
             {
               shape =
                 Array.init rank (fun d ->
                     if d = batch then Array.length upper else 1);
               dims = [| batch |];
             })
          [ bound ]
      in
      bind1
        (Broadcast_in_dim { shape = ish; dims = Array.init rank Fun.id })
        [ expanded ]
  in
  let mask = bind1 And [ ge; bind1 Le [ indices; bound ] ] in
  let mask = bind1 (Reduce_and [| batch |]) [ mask ] in
  let gathered =
    let inner = Gather { dimension_numbers = gd; slice_sizes } in
    let inner =
      match Np_ints.gather_flags_of prim with
      | Some (sorted, unique) -> Np_ints.mark_gather_flags ~sorted ~unique inner
      | None -> inner
    in
    bind1 inner [ operand; indices ]
  in
  let out_shape = (aval gathered).shape in
  let batch_dims =
    Array.of_list
      (List.filter
         (fun d -> not (Array.exists (fun o -> o = d) gd.offset_dims))
         (List.init (Array.length out_shape) Fun.id))
  in
  let mask =
    if
      (aval mask).shape = out_shape
      && batch_dims = Array.init (Array.length out_shape) Fun.id
    then mask
    else
      bind1 (Broadcast_in_dim { shape = out_shape; dims = batch_dims }) [ mask ]
  in
  let fill =
    match fill_value with
    | Some f -> Ndarray.of_floats op_dt [||] [| f |]
    | None -> default_gather_fill op_dt [||]
  in
  let fill =
    bind1
      (Broadcast_in_dim { shape = out_shape; dims = [||] })
      [ Concrete fill ]
  in
  bind1 Select_n [ mask; fill; gathered ]

let () =
  Core.gather_fill_hook :=
    fun prim args ->
      match (prim, args) with
      | Gather { dimension_numbers; slice_sizes }, [ operand; indices ] ->
          let fill_value =
            match Np_ints.gather_fill_of prim with
            | Some (_, fill) -> fill
            | None -> None
          in
          [
            gather_fill_expanded ?fill_value operand indices dimension_numbers
              slice_sizes;
          ]
      | _ -> invalid_arg "gather_fill_hook: arity"

let clamp_scatter_indices operand indices updates (sd : scatter_dims) =
  let bind1 = Core.bind1 in
  let aval = Core.get_aval in
  let op_shape = (aval operand).shape in
  let upd_shape = (aval updates).shape in
  let rank = Array.length op_shape in
  let idt = (aval indices).dtype and ish = (aval indices).shape in
  let slice_sizes = Array.make rank 1 in
  let pos = ref 0 in
  for i = 0 to rank - 1 do
    if
      not
        (Array.exists (fun d -> d = i) sd.inserted_window_dims
        || Array.exists (fun d -> d = i) sd.s_operand_batching_dims)
    then begin
      slice_sizes.(i) <- upd_shape.(sd.update_window_dims.(!pos));
      incr pos
    end
  done;
  let upper =
    Array.map
      (fun o -> float_of_int (op_shape.(o) - slice_sizes.(o)))
      sd.scatter_dims_to_operand_dims
  in
  let host =
    if Array.exists (fun v -> v > 2147483647.0) upper then Dtype.I64
    else Dtype.I32
  in
  let raw = Concrete (Ndarray.of_floats host [| Array.length upper |] upper) in
  let int_max dt =
    match dt with
    | Dtype.I64 | Dtype.Uint64 -> 9223372036854775807.0
    | Dtype.I32 -> 2147483647.0
    | Dtype.Uint32 -> 4294967295.0
    | Dtype.I16 -> 32767.0
    | Dtype.Uint16 -> 65535.0
    | Dtype.I8 -> 127.0
    | Dtype.Uint8 -> 255.0
    | _ -> 2147483647.0
  in
  let cap =
    bind1
      (Broadcast_in_dim { shape = [| Array.length upper |]; dims = [||] })
      [
        Concrete
          (Ndarray.of_floats host [||]
             [| Float.min (int_max host) (int_max idt) |]);
      ]
  in
  let bound = bind1 Min [ raw; cap ] in
  let bound =
    if host = idt then bound else bind1 (Convert_element_type idt) [ bound ]
  in
  let bound =
    if ish = [| Array.length upper |] then bound
    else
      bind1
        (Broadcast_in_dim { shape = ish; dims = [| Array.length ish - 1 |] })
        [ bound ]
  in
  let zero =
    bind1
      (Broadcast_in_dim { shape = ish; dims = [||] })
      [ Concrete (Ndarray.of_floats idt [||] [| 0.0 |]) ]
  in
  bind1 Clamp [ zero; indices; bound ]

let unclip_scatter p =
  let off (d : scatter_dims) = { d with s_clip = false } in
  let plain =
    match p with
    | Scatter ({ dimension_numbers = d; _ } as o) ->
        Scatter { o with dimension_numbers = off d }
    | Scatter_add { dimension_numbers = d } ->
        Scatter_add { dimension_numbers = off d }
    | Scatter_sub { dimension_numbers = d } ->
        Scatter_sub { dimension_numbers = off d }
    | Scatter_mul ({ dimension_numbers = d; _ } as o) ->
        Scatter_mul { o with dimension_numbers = off d }
    | Scatter_min { dimension_numbers = d } ->
        Scatter_min { dimension_numbers = off d }
    | Scatter_max { dimension_numbers = d } ->
        Scatter_max { dimension_numbers = off d }
    | _ -> invalid_arg "unclip_scatter: not a scatter primitive"
  in
  Np_ints.copy_scatter_marks ~from:p plain
