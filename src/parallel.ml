open Types
module B = Interpreters.Batching
module L = Lax_api

let bound_axis_size axis_name =
  match B.find_axis axis_name with
  | _, _, size -> size
  | exception Invalid_argument msg -> (
      match List.assoc_opt axis_name !Core.mesh_axes with
      | Some size -> size
      | None -> invalid_arg msg)

let collective kind axis_name v =
  ignore (bound_axis_size axis_name);
  Core.bind1 (Collective { kind; axis_name }) [ v ]

let is_vmap_axis axis_name =
  match B.find_axis axis_name with
  | _ -> true
  | exception Invalid_argument _ -> false

let validate_reduce_groups = function
  | None -> ()
  | Some groups ->
      let total = List.fold_left (fun n g -> n + List.length g) 0 groups in
      let seen = List.sort_uniq compare (List.concat groups) in
      if seen <> List.init total Fun.id then
        raise
          (Py_exn.Value_error
             "axis_index_groups must cover all indices exactly once")

let check_groups ~message axis_name = function
  | None -> ()
  | Some groups ->
      if is_vmap_axis axis_name then
        raise (Py_exn.Not_implemented_error message);
      if List.length groups <> 1 then
        raise
          (Py_exn.Not_implemented_error
             "axis_index_groups with more than one group needs a mapped axis \
              lowered with replica groups")

let reduction_message =
  "axis_index_groups not supported in vmap collectives. Please open a feature \
   request!"

let reduce_with kind ?axis_index_groups v ~axis_name =
  validate_reduce_groups axis_index_groups;
  check_groups ~message:reduction_message axis_name axis_index_groups;
  collective kind axis_name v

let psum ?axis_index_groups v ~axis_name =
  reduce_with Psum ?axis_index_groups v ~axis_name

let pmax ?axis_index_groups v ~axis_name =
  reduce_with Pmax ?axis_index_groups v ~axis_name

let pmin ?axis_index_groups v ~axis_name =
  reduce_with Pmin ?axis_index_groups v ~axis_name

let axis_size ~axis_name = bound_axis_size axis_name

let pmean ?axis_index_groups v ~axis_name =
  let total = psum ?axis_index_groups v ~axis_name in
  Lax_api.div total
    (Lax_api.full_like total (float_of_int (axis_size ~axis_name)))

let all_gather ?axis_index_groups ?(axis = 0) ?(tiled = false)
    ?(to_ = "varying") v ~axis_name =
  if not (List.mem to_ [ "varying"; "reduced"; "invarying" ]) then
    raise
      (Py_exn.Value_error
         "Got unexpected `to` value for `jax.lax.all_gather`. Allowed `to` \
          values are: {'varying', 'reduced', 'invarying'}");
  if to_ <> "varying" && axis_index_groups <> None then
    raise (Py_exn.Not_implemented_error "");
  check_groups ~message:"axis_index_groups not supported in vmap" axis_name
    axis_index_groups;
  let rank = Array.length (Core.get_aval v).shape in
  let dim =
    Validate.canonicalize_axis axis (if tiled then rank else rank + 1)
  in
  let size = axis_size ~axis_name in
  let gathered = collective (All_gather { axis = dim; size }) axis_name v in
  if not tiled then gathered
  else
    let sh = (Core.get_aval gathered).shape in
    let folded =
      Array.concat
        [
          Array.sub sh 0 dim;
          [| sh.(dim) * sh.(dim + 1) |];
          Array.sub sh (dim + 2) (Array.length sh - dim - 2);
        ]
    in
    L.reshape gathered folded

let pbroadcast ?(source = 0) v ~axis_name =
  collective (Pbroadcast source) axis_name v

let ppermute v ~axis_name ~perm =
  collective (Ppermute (Array.of_list perm)) axis_name v

let perm_string perm =
  match perm with
  | [ (a, b) ] -> Printf.sprintf "((%d, %d),)" a b
  | _ ->
      "("
      ^ String.concat ", "
          (List.map (fun (a, b) -> Printf.sprintf "(%d, %d)" a b) perm)
      ^ ")"

let single_side name axis_name params ~out v =
  ignore (bound_axis_size axis_name);
  Core.bind1
    (Decomposable
       {
         dname = name;
         dparams = params;
         dimpl =
           Core.register_decomposition (fun _ ->
               raise (Failure (name ^ " is currently only implemented on GPU")));
         davals = [ out ];
       })
    [ v ]

let psend x ~axis_name ~perm =
  single_side "psend" axis_name
    [ ("axis_name", "('" ^ axis_name ^ "',)"); ("perm", perm_string perm) ]
    ~out:{ shape = [| 0 |]; dtype = Dtype.F32; weak_type = false }
    x

let precv token ~out_shape ~axis_name ~perm =
  let out = Shape_dtype_struct.traced_aval out_shape in
  single_side "precv" axis_name
    [
      ("axis_name", "('" ^ axis_name ^ "',)");
      ( "out_shape",
        Ref_state.dtype_name out.dtype
        ^ "["
        ^ String.concat "," (List.map string_of_int (Array.to_list out.shape))
        ^ "]" );
      ("perm", perm_string perm);
    ]
    ~out token

let axis_index ~axis_name =
  ignore (bound_axis_size axis_name);
  Core.bind1 (Collective { kind = Axis_index; axis_name }) []

let chunk_of v ~axis ~index ~size = L.dynamic_slice_in_dim ~axis v index size

let tiled_all_to_all x ~axis_name ~split_axis ~concat_axis =
  let n = axis_size ~axis_name in
  let sh = (Core.get_aval x).shape in
  if sh.(split_axis) mod n <> 0 then
    Py_exn.value_error
      (Printf.sprintf
         "The size of all_to_all split_axis (%d) has to be divisible by the \
          size of the named axis %s (%d)"
         sh.(split_axis) axis_name n);
  let size = sh.(split_axis) / n in
  let gathered = all_gather x ~axis_name in
  let me = axis_index ~axis_name in
  let start =
    L.mul me (L.full [||] (float_of_int size) (Core.get_aval me).dtype)
  in
  let pieces =
    List.init n (fun j ->
        chunk_of
          (L.index_in_dim ~axis:0 ~keepdims:false gathered j)
          ~axis:split_axis ~index:start ~size)
  in
  L.concatenate pieces concat_axis

let on_vmap_axis axis_name =
  match B.find_axis axis_name with
  | _ -> true
  | exception Invalid_argument _ -> false

let all_to_all ?axis_index_groups ?(tiled = false) x ~axis_name ~split_axis
    ~concat_axis =
  check_groups ~message:"Please open a feature request!" axis_name
    axis_index_groups;
  if on_vmap_axis axis_name then begin
    let size = axis_size ~axis_name in
    let sh = (Core.get_aval x).shape in
    let bind x split_axis concat_axis =
      collective (All_to_all { split_axis; concat_axis; size }) axis_name x
    in
    if tiled then begin
      if sh.(split_axis) mod size <> 0 then
        Py_exn.value_error
          (Printf.sprintf
             "The size of all_to_all split_axis (%d) has to be divisible by \
              the size of the named axis %s (%d)"
             sh.(split_axis) axis_name size);
      bind x split_axis concat_axis
    end
    else begin
      if size <> sh.(split_axis) then
        invalid_arg
          (Printf.sprintf
             "all_to_all requires the size of the mapped axis axis_name to \
              equal x.shape[split_axis], but they are %d and %d respectively."
             size sh.(split_axis));
      if split_axis = concat_axis then bind x split_axis concat_axis
      else
        let concat, split =
          if split_axis < concat_axis then (concat_axis + 1, split_axis)
          else (concat_axis, split_axis + 1)
        in
        L.squeeze (bind (L.expand_dims x [| concat |]) split concat) [| split |]
    end
  end
  else if tiled then tiled_all_to_all x ~axis_name ~split_axis ~concat_axis
  else begin
    let n = axis_size ~axis_name in
    let sh = (Core.get_aval x).shape in
    if n <> sh.(split_axis) then
      invalid_arg
        (Printf.sprintf
           "all_to_all requires the size of the mapped axis axis_name to equal \
            x.shape[split_axis], but they are %d and %d respectively."
           n sh.(split_axis));
    if split_axis = concat_axis then
      tiled_all_to_all x ~axis_name ~split_axis ~concat_axis
    else
      let concat, split =
        if split_axis < concat_axis then (concat_axis + 1, split_axis)
        else (concat_axis, split_axis + 1)
      in
      let expanded = L.expand_dims x [| concat |] in
      L.squeeze
        (tiled_all_to_all expanded ~axis_name ~split_axis:split
           ~concat_axis:concat)
        [| split |]
  end

let pswapaxes ?axis_index_groups x ~axis_name ~axis =
  all_to_all ?axis_index_groups x ~axis_name ~split_axis:axis ~concat_axis:axis

let psum_scatter ?(scatter_dimension = 0) ?axis_index_groups ?(tiled = false) x
    ~axis_name =
  check_groups ~message:"axis_index_groups not supported in vmap" axis_name
    axis_index_groups;
  let n = axis_size ~axis_name in
  let sh = (Core.get_aval x).shape in
  if
    on_vmap_axis axis_name
    &&
    if tiled then sh.(scatter_dimension) mod n = 0
    else sh.(scatter_dimension) = n
  then
    collective
      (Reduce_scatter { scatter_dimension; size = n; tiled })
      axis_name x
  else
    let summed = psum x ~axis_name in
    let me = axis_index ~axis_name in
    if tiled then begin
      if sh.(scatter_dimension) mod n <> 0 then
        invalid_arg
          "psum_scatter: scatter_dimension must divide by the axis size";
      let size = sh.(scatter_dimension) / n in
      let start =
        L.mul me (L.full [||] (float_of_int size) (Core.get_aval me).dtype)
      in
      chunk_of summed ~axis:scatter_dimension ~index:start ~size
    end
    else begin
      if sh.(scatter_dimension) <> n then
        invalid_arg
          "psum_scatter: scatter_dimension must equal the axis size when not \
           tiled";
      L.squeeze
        (chunk_of summed ~axis:scatter_dimension ~index:me ~size:1)
        [| scatter_dimension |]
    end

let pshuffle x ~axis_name ~perm =
  let sorted = List.sort compare perm in
  if sorted <> List.init (List.length perm) Fun.id then
    invalid_arg "`perm` does not represent a permutation";
  ppermute x ~axis_name ~perm:(List.mapi (fun i src -> (src, i)) perm)

let ragged_all_to_all ?axis_index_groups operand output input_offsets send_sizes
    output_offsets recv_sizes ~axis_name =
  check_groups ~message:"Please open a feature request!" axis_name
    axis_index_groups;
  let check name v =
    let aval = Core.get_aval v in
    if not (Dtypes.issubdtype aval.dtype `Integer) then
      invalid_arg
        (Printf.sprintf "ragged_all_to_all %s must be integer type." name);
    if Array.length aval.shape <> 1 || aval.shape.(0) < 1 then
      invalid_arg
        (Printf.sprintf
           "ragged_all_to_all %s must be rank 1 with positive dimension size"
           name)
  in
  check "input_offsets" input_offsets;
  check "send_sizes" send_sizes;
  check "output_offsets" output_offsets;
  check "recv_sizes" recv_sizes;
  (try ignore (B.find_axis axis_name)
   with Invalid_argument _ ->
     failwith
       "ragged_all_to_all must be used within a mapped context like vmap or \
        shard_map.");
  let n = axis_size ~axis_name in
  let slots = (Core.get_aval input_offsets).shape.(0) in
  if slots mod n <> 0 then
    invalid_arg
      "ragged_all_to_all offsets length must be divisible by the size of the \
       mapped axis";
  let per_device = slots / n in
  let i32 v = L.convert_element_type v Dtype.I32 in
  let exchange v =
    tiled_all_to_all (i32 v) ~axis_name ~split_axis:0 ~concat_axis:0
  in
  let received_at = exchange output_offsets in
  let source_at = exchange input_offsets in
  let recv = i32 recv_sizes in
  let gathered = all_gather operand ~axis_name in
  let out_shape = (Core.get_aval output).shape in
  let rows = out_shape.(0) in
  let positions = L.iota Dtype.I32 rows in
  let across v = L.broadcast v [| rows |] in
  let dt = (Core.get_aval output).dtype in
  let scalar v i = L.index_in_dim ~axis:0 ~keepdims:false v i in
  List.fold_left
    (fun result i ->
      let start = across (scalar received_at i) in
      let stop = L.add start (across (scalar recv i)) in
      let mask = L.bitwise_and (L.le start positions) (L.lt positions stop) in
      let wide = L.broadcast_in_dim mask out_shape [| 0 |] in
      let from_row =
        L.add (L.sub positions start) (across (scalar source_at i))
      in
      let sender = scalar gathered (i / per_device) in
      let taken =
        Numpy.Indexing.take_index ~axis:0 ~mode:"clip" sender from_row
      in
      L.select wide (L.convert_element_type taken dt) result)
    output (List.init slots Fun.id)
