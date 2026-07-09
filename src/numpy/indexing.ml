module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Lax_numpy

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let ndim v = Array.length (shape v)
let size v = Array.fold_left ( * ) 1 (shape v)
let bind1 = C.bind1

let canonicalize_axis ax n =
  let a = if ax < 0 then ax + n else ax in
  if a < 0 || a >= n then
    invalid_arg
      (Printf.sprintf "axis %d out of bounds for array of ndim %d" ax n);
  a

let full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (Array.fold_left ( * ) 1 sh) x))

let iota dt sh dim =
  bind1 (T.Iota { dtype = dt; shape = sh; dimension = dim }) []

let convert v dt =
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let broadcast_shapes a b =
  let na = Array.length a and nb = Array.length b in
  let n = max na nb in
  Array.init n (fun i ->
      let da = if i < n - na then 1 else a.(i - (n - na)) in
      let db = if i < n - nb then 1 else b.(i - (n - nb)) in
      if da = db then da
      else if da = 1 then db
      else if db = 1 then da
      else invalid_arg "take_along_axis: incompatible shapes")

let wrap_negative indices axis_size =
  let idt = dtype indices and ish = shape indices in
  NL.where_
    (bind1 T.Lt [ indices; full idt ish 0.0 ])
    (bind1 T.Add [ indices; full idt ish (float_of_int axis_size) ])
    indices

let take ?axis ?mode a indices =
  let a, axis_idx =
    match axis with
    | None -> (NL.ravel a, 0)
    | Some ax -> (a, canonicalize_axis ax (ndim a))
  in
  let ash = shape a in
  let ish = shape indices in
  let rank = Array.length ash in
  let axis_size = ash.(axis_idx) in
  let indices =
    match mode with
    | Some "clip" -> indices
    | Some "wrap" ->
        let idt = dtype indices and ish = shape indices in
        bind1 T.Rem [ indices; full idt ish (float_of_int axis_size) ]
    | None | Some "fill" -> wrap_negative indices axis_size
    | Some m -> invalid_arg ("take: unsupported mode " ^ m)
  in
  let index_dims = Array.length ish in
  let slice_sizes = Array.copy ash in
  slice_sizes.(axis_idx) <- 1;
  let offset_dims =
    Array.of_list
      (List.init axis_idx (fun i -> i)
      @ List.init (rank - 1 - axis_idx) (fun i -> axis_idx + index_dims + i))
  in
  let dnums =
    {
      T.offset_dims;
      collapsed_slice_dims = [| axis_idx |];
      start_index_map = [| axis_idx |];
      g_operand_batching_dims = [||];
      g_start_indices_batching_dims = [||];
    }
  in
  let gather_indices = NL.reshape indices (Array.append ish [| 1 |]) in
  bind1
    (T.Gather { dimension_numbers = dnums; slice_sizes })
    [ a; gather_indices ]

let take_along_axis ?(axis = -1) a indices =
  let rank = ndim a in
  if rank <> ndim indices then
    invalid_arg "take_along_axis: arr and indices must have the same ndim";
  let axis_int = canonicalize_axis axis rank in
  let ash = shape a in
  let idx_shape = shape indices in
  let axis_size = ash.(axis_int) in
  let arr_shape = Array.copy ash in
  arr_shape.(axis_int) <- 1;
  let out_shape = broadcast_shapes idx_shape arr_shape in
  let indices = wrap_negative indices axis_size in
  let index_dims =
    List.filter
      (fun i -> i = axis_int || idx_shape.(i) <> 1)
      (List.init rank Fun.id)
  in
  let gather_index_shape =
    Array.append
      (Array.of_list (List.map (fun i -> out_shape.(i)) index_dims))
      [| 1 |]
  in
  let gather_indices = NL.reshape indices gather_index_shape in
  let slice_sizes = ref []
  and offset_dims = ref []
  and start_index_map = ref []
  and collapsed = ref []
  and op_batch = ref []
  and idx_batch = ref []
  and dims_to_squeeze = ref [] in
  let new_i = ref 0 and j = ref 0 in
  for i = 0 to rank - 1 do
    if i = axis_int then begin
      slice_sizes := 1 :: !slice_sizes;
      start_index_map := !new_i :: !start_index_map;
      collapsed := !new_i :: !collapsed;
      incr new_i;
      incr j
    end
    else if idx_shape.(i) = 1 then begin
      offset_dims := i :: !offset_dims;
      slice_sizes := arr_shape.(i) :: !slice_sizes;
      incr new_i
    end
    else if arr_shape.(i) = 1 then begin
      dims_to_squeeze := i :: !dims_to_squeeze;
      incr j
    end
    else begin
      if arr_shape.(i) = 0 then slice_sizes := 0 :: !slice_sizes
      else slice_sizes := 1 :: !slice_sizes;
      op_batch := !new_i :: !op_batch;
      idx_batch := !j :: !idx_batch;
      incr new_i;
      incr j
    end
  done;
  let asc l = Array.of_list (List.rev l) in
  let a =
    if !dims_to_squeeze = [] then a
    else bind1 (T.Squeeze (asc !dims_to_squeeze)) [ a ]
  in
  let dnums =
    {
      T.offset_dims = asc !offset_dims;
      collapsed_slice_dims = asc !collapsed;
      start_index_map = asc !start_index_map;
      g_operand_batching_dims = asc !op_batch;
      g_start_indices_batching_dims = asc !idx_batch;
    }
  in
  bind1
    (T.Gather { dimension_numbers = dnums; slice_sizes = asc !slice_sizes })
    [ a; gather_indices ]

let tile_to_size v n =
  let m = size v in
  if m = n then v
  else begin
    let reps = (n + m - 1) / m in
    let tiled = NL.concatenate ~axis:0 (List.init reps (fun _ -> v)) in
    if size tiled = n then tiled
    else
      bind1
        (T.Slice
           { start_indices = [| 0 |]; limit_indices = [| n |]; strides = None })
        [ tiled ]
  end

let identity_scatter_dims r =
  {
    T.update_window_dims = [||];
    inserted_window_dims = Array.init r Fun.id;
    scatter_dims_to_operand_dims = Array.init r Fun.id;
    s_operand_batching_dims = [||];
    s_scatter_indices_batching_dims = [||];
  }

let put ?mode a ind v =
  let ash = shape a in
  let asize = Array.fold_left ( * ) 1 ash in
  let ind = NL.ravel ind in
  let vv = NL.ravel v in
  let n = size ind in
  if asize = 0 || n = 0 || size vv = 0 then a
  else begin
    let vv = convert (tile_to_size vv n) (dtype a) in
    let idt = dtype ind in
    let ind =
      match mode with
      | None -> ind
      | Some "clip" -> NL.clip ~min:0.0 ~max:(float_of_int (asize - 1)) ind
      | Some "wrap" ->
          bind1 T.Rem [ ind; full idt (shape ind) (float_of_int asize) ]
      | Some m -> invalid_arg ("put: unsupported mode " ^ m)
    in
    let coords = NL.unravel_index ind ash in
    let r = Array.length ash in
    let scatter_indices =
      NL.concatenate ~axis:1
        (List.map (fun c -> NL.reshape c [| n; 1 |]) coords)
    in
    let dnums = identity_scatter_dims r in
    bind1
      (T.Scatter { dimension_numbers = dnums; unique_indices = false })
      [ a; scatter_indices; vv ]
  end

let put_along_axis ?axis arr indices values =
  let arr, indices, values, axis, orig_shape =
    match axis with
    | None ->
        (NL.ravel arr, NL.ravel indices, NL.ravel values, 0, Some (shape arr))
    | Some ax -> (arr, indices, values, canonicalize_axis ax (ndim arr), None)
  in
  let rank = ndim arr in
  if rank <> ndim indices then
    invalid_arg "put_along_axis: arr and indices must have the same ndim";
  let ash = shape arr in
  let axis_size = ash.(axis) in
  let indices = wrap_negative indices axis_size in
  let ish = shape indices in
  let b = Array.copy ash in
  b.(axis) <- ish.(axis);
  let indices_b = NL.broadcast_to indices b in
  let values_b = convert (NL.broadcast_to values b) (dtype arr) in
  let idt = dtype indices in
  let coords =
    List.init rank (fun d ->
        let c = if d = axis then indices_b else iota idt b d in
        NL.reshape c (Array.append b [| 1 |]))
  in
  let scatter_indices = NL.concatenate ~axis:rank coords in
  let dnums = identity_scatter_dims rank in
  let result =
    bind1
      (T.Scatter { dimension_numbers = dnums; unique_indices = false })
      [ arr; scatter_indices; values_b ]
  in
  match orig_shape with Some s -> NL.reshape result s | None -> result
