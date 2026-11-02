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
let canonicalize_axis = Validate.canonicalize_axis
let full dt sh x = Core.splat dt sh x

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
      else
        Py_exn.value_error
          (Printf.sprintf
             "operands could not be broadcast together with shapes %s %s"
             (Validate.python_tuple a) (Validate.python_tuple b)))

let wrap_negative indices axis_size =
  let idt = dtype indices in
  NL.where_
    (bind1 T.Lt [ indices; full idt [||] 0.0 ])
    (bind1 T.Add
       [
         indices;
         (match full idt [||] (float_of_int axis_size) with
         | T.Concrete nd when (get_aval indices).T.weak_type ->
             T.Concrete (Nd.as_weak nd)
         | c -> c);
       ])
    indices

let take ?axis ?mode ?fill_value ?(unique_indices = false)
    ?(indices_are_sorted = false) a indices =
  let a, axis_idx =
    match axis with
    | None -> (NL.ravel a, 0)
    | Some ax -> (a, canonicalize_axis ax (ndim a))
  in
  let ash = shape a in
  let ish = shape indices in
  let rank = Array.length ash in
  let axis_size = ash.(axis_idx) in
  if axis_size = 0 && Array.fold_left ( * ) 1 ish <> 0 then
    raise
      (Py_exn.Index_error "Cannot do a non-empty jnp.take() from an empty axis.");
  let indices =
    match mode with
    | Some "clip" -> indices
    | Some "wrap" ->
        Ufuncs.mod_ indices (full (dtype indices) [||] (float_of_int axis_size))
    | Some "raise" ->
        Py_exn.not_implemented_error
          "The 'raise' mode to jnp.take is not supported."
    | None | Some "fill" -> wrap_negative indices axis_size
    | Some m ->
        Py_exn.value_error (Printf.sprintf "Invalid mode '%s' for np.take" m)
  in
  let index_dims = Array.length ish in
  let indices_size = Array.fold_left ( * ) 1 ish in
  if axis_size = 0 && indices_size <> 0 then
    raise
      (Py_exn.Index_error "Cannot do a non-empty jnp.take() from an empty axis.");
  if axis_size = 0 then a
  else if indices_size = 0 then
    full (dtype a)
      (Array.concat
         [
           Array.sub ash 0 axis_idx;
           ish;
           Array.sub ash (axis_idx + 1) (rank - axis_idx - 1);
         ])
      0.0
  else
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
        g_sorted_unique = false;
      }
    in
    let gather_indices =
      bind1
        (T.Broadcast_in_dim
           {
             shape = Array.append ish [| 1 |];
             dims = Array.init index_dims Fun.id;
           })
        [ indices ]
    in
    let flag prim =
      if unique_indices || indices_are_sorted then
        Np_ints.mark_gather_flags ~sorted:indices_are_sorted
          ~unique:unique_indices prim
      else prim
    in
    match mode with
    | None | Some "fill" ->
        Lax.Slicing.gather_fill ~flag ?fill_value a gather_indices dnums
          slice_sizes
    | Some "clip" ->
        bind1
          (Np_ints.mark_gather_clip ~sorted:false
             (flag (T.Gather { dimension_numbers = dnums; slice_sizes })))
          [ a; gather_indices ]
    | _ ->
        bind1
          (flag (T.Gather { dimension_numbers = dnums; slice_sizes }))
          [ a; gather_indices ]

let take_along_axis_body ~axis ?mode ?fill_value ?wrap_negative_indices a
    indices =
  let rank = ndim a in
  if rank <> ndim indices then
    Py_exn.value_error
      (Printf.sprintf
         "indices and arr must have the same number of dimensions; %d vs. %d"
         (ndim indices) rank);
  let axis_int = canonicalize_axis axis rank in
  let ash = shape a in
  let idx_shape = shape indices in
  let axis_size = ash.(axis_int) in
  let arr_shape = Array.copy ash in
  arr_shape.(axis_int) <- 1;
  let out_shape = broadcast_shapes idx_shape arr_shape in
  if axis_size = 0 then full (dtype a) out_shape 0.0
  else
    let wrap =
      match wrap_negative_indices with
      | Some w -> w
      | None -> mode <> Some "promise_in_bounds"
    in
    let indices =
      if not wrap then indices
      else begin
        let index_dtype = if axis_size > 2147483647 then D.I64 else D.I32 in
        let indices =
          if dtype indices = index_dtype then indices
          else bind1 (T.Convert_element_type index_dtype) [ indices ]
        in
        let negative = bind1 T.Lt [ indices; full index_dtype [||] 0.0 ] in
        let shifted =
          bind1 T.Add
            [ indices; full index_dtype [||] (float_of_int axis_size) ]
        in
        bind1 T.Select_n [ negative; indices; shifted ]
      end
    in
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
        g_sorted_unique = false;
      }
    in
    match mode with
    | None | Some "fill" ->
        Lax.Slicing.gather_fill ?fill_value a gather_indices dnums
          (asc !slice_sizes)
    | Some ("clip" | "promise_in_bounds") ->
        bind1
          (T.Gather
             { dimension_numbers = dnums; slice_sizes = asc !slice_sizes })
          [ a; gather_indices ]
    | Some m -> Py_exn.value_error ("Unknown gather mode \"" ^ m ^ "\"")

let take_along_axis ?(axis = -1) ?mode ?fill_value ?wrap_negative_indices a
    indices =
  Jnp_util.jit1 "take_along_axis"
    ~key:
      (Digest.to_hex
         (Digest.string
            (Marshal.to_string
               (axis, mode, fill_value, wrap_negative_indices)
               [])))
    (function
      | [ a; indices ] ->
          take_along_axis_body ~axis ?mode ?fill_value ?wrap_negative_indices a
            indices
      | _ -> invalid_arg "take_along_axis: arity")
    [ a; indices ]

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
    s_sorted_unique = false;
    s_clip = false;
  }

let put ?mode ?(inplace = true) a ind v =
  if inplace then
    raise
      (Py_exn.Value_error
         "jax.numpy.put cannot modify arrays in-place, because JAX arrays are \
          immutable. Pass inplace=False to instead return an updated array.");
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
      | Some "raise" ->
          Py_exn.not_implemented_error
            "The 'raise' mode to jnp.put is not supported."
      | Some m ->
          Py_exn.value_error
            (Printf.sprintf
               "mode should be one of 'wrap' or 'clip'; got mode='%s'" m)
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

let put_along_axis ?axis ?mode arr indices values =
  let arr, indices, values, axis, orig_shape =
    match axis with
    | None ->
        (NL.ravel arr, NL.ravel indices, NL.ravel values, 0, Some (shape arr))
    | Some ax -> (arr, indices, values, canonicalize_axis ax (ndim arr), None)
  in
  let rank = ndim arr in
  if rank <> ndim indices then
    Py_exn.value_error
      (Printf.sprintf
         "put_along_axis arguments 'arr' and 'indices' must have same ndim. \
          Got arr.ndim=%d and indices.ndim=%d."
         rank (ndim indices));
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
  let scatter_indices =
    match mode with
    | Some "clip" ->
        Lax.Slicing.clamp_scatter_indices arr scatter_indices values_b dnums
    | None | Some ("fill" | "drop" | "promise_in_bounds") -> scatter_indices
    | Some m -> Py_exn.value_error ("Unknown gather mode \"" ^ m ^ "\"")
  in
  let result =
    bind1
      (T.Scatter { dimension_numbers = dnums; unique_indices = false })
      [ arr; scatter_indices; values_b ]
  in
  match orig_shape with Some s -> NL.reshape result s | None -> result

type dslice = Full | Dslice of { start : T.value; size : int; stride : int }

let ds ?size ?(stride = 1) start =
  if stride < 0 then Py_exn.value_error "`stride` must be >= 0.";
  match (start, size) with
  | None, _ -> Full
  | Some start, Some size -> Dslice { start; size; stride }
  | Some count, None ->
      let size =
        match count with
        | T.Tracer _ ->
            Py_exn.type_error "ds: a one-argument slice needs a static size"
        | T.Concrete nd -> int_of_float (Nd.get_f nd [||])
        | T.Device b -> int_of_float (Nd.get_f (Pjrt.Buffer.to_host b) [||])
      in
      Dslice
        { start = T.Concrete (Nd.of_floats D.I32 [||] [| 0.0 |]); size; stride }

let dslice_index x slices =
  let sh = shape x in
  let nd = Array.length sh in
  let given = List.length slices in
  if given > nd then
    Py_exn.index_error
      (Printf.sprintf
         "Too many indices: %d-dimensional array indexed with %d regular \
          indices."
         nd given);
  let slices = slices @ List.init (nd - given) (fun _ -> Full) in
  let unit_stride = function
    | Full | Dslice { stride = 1; _ } -> true
    | Dslice _ -> false
  in
  let idx_dtype = Nd_index.index_dtype sh in
  if List.for_all (fun s -> s = Full) slices then x
  else if List.for_all unit_stride slices then
    let starts =
      List.map
        (function
          | Full -> T.Concrete (Nd.of_floats D.I32 [||] [| 0.0 |])
          | Dslice { start; _ } -> start)
        slices
    in
    let sizes =
      Array.of_list
        (List.mapi
           (fun axis slice ->
             match slice with Full -> sh.(axis) | Dslice { size; _ } -> size)
           slices)
    in
    let starts =
      if List.length starts > 1 then
        List.map (fun v -> C.convert_like ~weak:false v idx_dtype) starts
      else starts
    in
    let normalised =
      List.mapi
        (fun axis start ->
          let d = dtype start in
          let negative =
            Ufuncs.less start (T.Concrete (Nd.of_floats d [||] [| 0.0 |]))
          in
          let shifted =
            Lax_api.add start
              (T.Concrete (Nd.of_floats d [||] [| float_of_int sh.(axis) |]))
          in
          bind1 T.Select_n [ negative; start; shifted ])
        starts
    in
    Lax_api.dynamic_slice_normalized x normalised sizes
  else
    let gather_indices = ref [] in
    let indices_shape = ref [] in
    let start_map = ref [] in
    let collapsed = ref [] in
    let offset = ref [] in
    let gather_sizes = ref [] in
    let slice_shape = ref [] in
    List.iteri
      (fun axis slice ->
        let y_axis = axis in
        match slice with
        | Full | Dslice { stride = 1; _ } ->
            let size =
              match slice with Dslice { size; _ } -> size | _ -> sh.(axis)
            in
            (match slice with
            | Dslice { start; size; _ } when size <> sh.(axis) ->
                gather_indices :=
                  ( C.convert_like ~weak:false start idx_dtype,
                    List.length !indices_shape )
                  :: !gather_indices;
                start_map := axis :: !start_map
            | _ -> ());
            gather_sizes := size :: !gather_sizes;
            offset := y_axis :: !offset;
            slice_shape := size :: !slice_shape
        | Dslice { start; size; stride } ->
            let begin_at =
              bind1 (T.Convert_element_type idx_dtype)
                [ bind1 T.Copy [ start ] ]
            in
            let steps =
              Lax_api.mul
                (T.Concrete
                   (Nd.of_floats idx_dtype [||] [| float_of_int stride |]))
                (bind1
                   (T.Iota
                      { dtype = idx_dtype; shape = [| size |]; dimension = 0 })
                   [])
            in
            let indices = Lax_api.add begin_at steps in
            gather_sizes := 1 :: !gather_sizes;
            gather_indices :=
              (indices, List.length !indices_shape) :: !gather_indices;
            start_map := axis :: !start_map;
            indices_shape := size :: !indices_shape;
            collapsed := axis :: !collapsed;
            slice_shape := size :: !slice_shape)
      slices;
    let items = List.rev !gather_indices in
    let shape_list = List.rev !indices_shape in
    let gather_array =
      match items with
      | [] -> T.Concrete (Nd.of_floats idx_dtype [| 0 |] [||])
      | [ (g, _) ] -> Lax_api.expand_dims g [| ndim g |]
      | _ ->
          let full = Array.of_list (shape_list @ [ 1 ]) in
          let last = List.length shape_list in
          Lax_api.concatenate
            (List.map
               (fun (g, i) ->
                 Lax_api.broadcast_in_dim g full
                   (Array.init (ndim g) (fun k -> i + k)))
               items)
            last
    in
    let prim =
      T.Gather
        {
          dimension_numbers =
            {
              offset_dims = Array.of_list (List.rev !offset);
              collapsed_slice_dims =
                Array.of_list (List.sort compare !collapsed);
              start_index_map = Array.of_list (List.rev !start_map);
              g_operand_batching_dims = [||];
              g_start_indices_batching_dims = [||];
              g_sorted_unique = false;
            };
          slice_sizes = Array.of_list (List.rev !gather_sizes);
        }
    in
    bind1
      (Np_ints.mark_gather_flags ~sorted:true ~unique:true prim)
      [ x; gather_array ]

let put_along_axis ?axis ?mode ?(inplace = true) a0 a1 a2 =
  if inplace then
    raise
      (Py_exn.Value_error
         "jax.numpy.put_along_axis cannot modify arrays in-place, because JAX \
          arraysare immutable. Pass inplace=False to instead return an updated \
          array.");
  Eager_jit.one ~jit:"put_along_axis" "numpy.indexing.put_along_axis"
    (axis, mode) [] [ a0; a1; a2 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1; a2 ] -> put_along_axis ?axis ?mode a0 a1 a2
      | _ -> assert false)

let take_index ?axis ?mode ?fill_value a i = take ?axis ?mode ?fill_value a i

let take ?axis ?mode ?fill_value ?unique_indices ?indices_are_sorted a0 a1 =
  Eager_jit.one ~jit:"_take" "numpy.indexing.take"
    (axis, mode, fill_value, unique_indices, indices_are_sorted) [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] ->
          take ?axis ?mode ?fill_value ?unique_indices ?indices_are_sorted a0 a1
      | _ -> assert false)

let is_weak v = (get_aval v).T.weak_type

let weak_literal v x =
  T.Concrete
    (Ndarray.as_weak (Ndarray.of_floats (dtype v) [||] [| float_of_int x |]))

let strong_index i =
  if is_weak i then bind1 (T.Convert_element_type (dtype i)) [ i ] else i

let select_negative i n =
  let negative = bind1 T.Lt [ i; weak_literal i 0 ] in
  let shifted = bind1 T.Add [ i; weak_literal i n ] in
  bind1 T.Select_n [ negative; i; shifted ]

let check_nonempty_axes sh count =
  for axis = 0 to count - 1 do
    if sh.(axis) = 0 then
      raise
        (Py_exn.Index_error
           (Printf.sprintf "index is out of bounds for axis %d with size 0" axis))
  done

let dynamic_at x i =
  let sh = shape x in
  check_nonempty_axes sh 1;
  let negative = bind1 T.Lt [ i; weak_literal i 0 ] in
  let shifted = bind1 T.Add [ strong_index i; weak_literal i sh.(0) ] in
  let start = bind1 T.Select_n [ negative; i; shifted ] in
  let sizes = Array.mapi (fun d s -> if d = 0 then 1 else s) sh in
  let zeros =
    List.init
      (Array.length sh - 1)
      (fun _ -> T.Concrete (Ndarray.of_floats (dtype i) [||] [| 0.0 |]))
  in
  bind1 (T.Squeeze [| 0 |])
    [ bind1 (T.Dynamic_slice { slice_sizes = sizes }) (x :: start :: zeros) ]

let dynamic_at2 x i j =
  let sh = shape x in
  check_nonempty_axes sh 2;
  let dt = if Array.exists (fun d -> d > 2147483647) sh then D.I64 else D.I32 in
  let to_dt v =
    if is_weak v || dtype v <> dt then bind1 (T.Convert_element_type dt) [ v ]
    else v
  in
  let i = to_dt i in
  let j = to_dt j in
  let start_i = select_negative i sh.(0) in
  let start_j = select_negative j sh.(1) in
  let sizes = Array.mapi (fun d s -> if d < 2 then 1 else s) sh in
  bind1
    (T.Squeeze [| 0; 1 |])
    [ bind1 (T.Dynamic_slice { slice_sizes = sizes }) [ x; start_i; start_j ] ]

let set_at2 x i j value =
  let sh = shape x in
  check_nonempty_axes sh 2;
  let wi = select_negative i sh.(0) in
  let wj = select_negative j sh.(1) in
  let as_i32 v =
    if is_weak v || dtype v <> D.I32 then
      bind1 (T.Convert_element_type D.I32) [ v ]
    else v
  in
  let wi = as_i32 wi in
  let wj = as_i32 wj in
  let lift v =
    bind1 (T.Broadcast_in_dim { shape = [| 1 |]; dims = [||] }) [ v ]
  in
  let li = lift wi in
  let lj = lift wj in
  let index = bind1 (T.Concatenate 0) [ li; lj ] in
  bind1
    (T.Scatter
       {
         dimension_numbers =
           {
             T.update_window_dims = [||];
             inserted_window_dims = [| 0; 1 |];
             scatter_dims_to_operand_dims = [| 0; 1 |];
             s_operand_batching_dims = [||];
             s_scatter_indices_batching_dims = [||];
             s_sorted_unique = true;
             s_clip = false;
           };
         unique_indices = true;
       })
    [ x; index; value ]

let () =
  Interpreters.Batching.take_on_axis :=
    fun v indices axis ->
      take ~axis v
        (T.Concrete
           (Nd.of_floats
              (if Config.value Config.enable_x64 then D.I64 else D.I32)
              [| Array.length indices |]
              (Array.map float_of_int indices)))

let () =
  (Reductions.take_along_axis_hook := fun ~axis a i -> take_along_axis ~axis a i);
  Reductions.getitem_hook :=
    fun a idx -> Nd_index.take a (List.map (fun i -> Nd_index.Arr i) idx)
