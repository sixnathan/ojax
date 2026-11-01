open Types

let resolve_strides n = function Some s -> s | None -> Array.make n 1

let slice_shape start_indices limit_indices strides ishape =
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
      Ndarray.of_floats dt sh (Array.make n Float.nan)
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

let gather_fill ?fill_value operand indices (gd : gather_dims) slice_sizes =
  let dt = (Core.get_aval operand).dtype in
  Core.bind1
    (Np_ints.mark_gather_fill
       (Gather { dimension_numbers = gd; slice_sizes })
       (fill_repr ?fill_value dt) fill_value)
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
  let bound = Concrete (Ndarray.of_floats idt [| Array.length upper |] upper) in
  let bound =
    bind1
      (Broadcast_in_dim { shape = ish; dims = [| Array.length ish - 1 |] })
      [ bound ]
  in
  let zero =
    Concrete
      (Ndarray.of_floats idt ish (Array.make (Array.fold_left ( * ) 1 ish) 0.0))
  in
  bind1 Clamp [ zero; indices; bound ]
