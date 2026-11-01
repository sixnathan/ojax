module T = Types
module D = Dtype
module L = Lax_api
module NL = Numpy.Lax_numpy

type rng_algorithm = Lax_config.random_algorithm =
  | Rng_default
  | Three_fry
  | Philox

let aval = Core.get_aval
let shape v = (aval v).T.shape
let dtype v = (aval v).T.dtype

let approx_max_k ?(reduction_dimension = -1) operand k =
  L.top_k ~axis:reduction_dimension operand k

let approx_min_k ?(reduction_dimension = -1) operand k =
  let values, indices = L.top_k ~axis:reduction_dimension (L.neg operand) k in
  (L.neg values, indices)

let broadcast_like arr like = NL.broadcast_to arr (shape like)
let create_token () = Core.bind1 T.Create_token []
let after_all tokens = Core.bind1 T.After_all tokens

type ragged_mode = Ragged_noncontracting | Ragged_contracting | Ragged_batch

let remaining rank groups =
  List.filter
    (fun i -> not (List.exists (fun g -> Array.mem i g) groups))
    (List.init rank Fun.id)

let take_until target items =
  let rec go acc = function
    | [] -> List.rev acc
    | x :: _ when x = target -> List.rev acc
    | x :: rest -> go (x :: acc) rest
  in
  go [] items

let ragged_prefix mode rank ragged batch contract =
  let batch_l = Array.to_list batch in
  match mode with
  | Ragged_noncontracting ->
      batch_l @ take_until ragged (remaining rank [ contract; batch ])
  | Ragged_contracting -> batch_l @ take_until ragged (Array.to_list contract)
  | Ragged_batch -> take_until ragged batch_l

let ragged_to_dense x group_sizes dim axes =
  let gs_shape = shape group_sizes in
  let gs_rank = Array.length gs_shape in
  let g = gs_shape.(gs_rank - 1) in
  let dt = dtype group_sizes in
  let x_shape = shape x in
  let rank = Array.length x_shape in
  let dense_shape = Array.append [| g |] x_shape in
  let widened =
    L.broadcast_in_dim x dense_shape (Array.init rank (fun i -> i + 1))
  in
  let positions = L.broadcasted_iota dt dense_shape (dim + 1) in
  let last = gs_rank - 1 in
  let ends = L.cumsum ~axis:last group_sizes in
  let first =
    L.full (Array.mapi (fun i s -> if i = last then 1 else s) gs_shape) 0.0 dt
  in
  let shifted = L.slice_in_dim ~axis:last ends 0 (g - 1) in
  let starts = L.concatenate [ first; shifted ] last in
  let order =
    List.sort
      (fun a b -> compare (List.nth axes a) (List.nth axes b))
      (List.init (List.length axes) Fun.id)
  in
  let perm = Array.of_list (last :: order) in
  let place v =
    let sorted_axes = List.map (List.nth axes) order in
    let dims = Array.of_list (0 :: List.map (fun a -> a + 1) sorted_axes) in
    L.broadcast_in_dim (L.transpose v perm) dense_shape dims
  in
  let mask =
    L.bitwise_and (L.le (place starts) positions) (L.lt positions (place ends))
  in
  L.select mask widened (L.full_like widened 0.0)

let ragged_dot_general ?precision ?preferred_element_type ?group_offset lhs rhs
    group_sizes (dims : Lax_config.ragged_dot_dimension_numbers) =
  if group_offset <> None then failwith "Unimplemented group_offset support.";
  if Array.length dims.lhs_ragged_dimensions <> 1 then
    invalid_arg "ragged_dot_general expects exactly one lhs ragged dimension.";
  let ragged = dims.lhs_ragged_dimensions.(0) in
  let lhs_rank = Array.length (shape lhs) in
  let rhs_rank = Array.length (shape rhs) in
  let check_range dim rank name arg =
    if dim < 0 || dim >= rank then
      invalid_arg
        (Printf.sprintf
           "ragged_dot_general requires %s numbers to be nonnegative and less \
            than the number of axes of the %s value, got %d for %s of rank %d."
           name arg dim arg rank)
  in
  check_range ragged lhs_rank "lhs ragged dimension" "lhs";
  let dn = dims.dot_dimension_numbers in
  let l_contract = dn.lhs_contract and r_contract = dn.rhs_contract in
  let l_batch = dn.lhs_batch and r_batch = dn.rhs_batch in
  let mode =
    if Array.mem ragged l_contract then Ragged_contracting
    else if Array.mem ragged l_batch then Ragged_batch
    else Ragged_noncontracting
  in
  (match dtype group_sizes with
  | D.I8 | D.I16 | D.I32 | D.I64 | D.Uint8 | D.Uint16 | D.Uint32 | D.Uint64 ->
      ()
  | _ ->
      invalid_arg
        "ragged_dot_general requires that group_sizes.dtype is subtype of \
         np.integer.");
  let gs_shape = shape group_sizes in
  let gs_rank = Array.length gs_shape in
  if gs_rank = 0 then invalid_arg "expected rank of group_sizes to be >=1.";
  let l_prefix = ragged_prefix mode lhs_rank ragged l_batch l_contract in
  let prefix_shape =
    Array.of_list (List.map (fun i -> (shape lhs).(i)) l_prefix)
  in
  let num_groups = gs_shape.(gs_rank - 1) in
  if gs_rank <> 1 && gs_shape <> Array.append prefix_shape [| num_groups |] then
    invalid_arg
      "expected group_sizes to have the shape of the lhs prefix dimensions \
       followed by the group count.";
  let rhs_group = dims.rhs_group_dimensions in
  (match mode with
  | Ragged_contracting | Ragged_batch ->
      if Array.length rhs_group <> 0 then
        invalid_arg
          "ragged_dot_general requires zero group dimensions in the rhs when \
           lhs ragged dimension is contracting or batch."
  | Ragged_noncontracting ->
      if Array.length rhs_group <> 1 then
        invalid_arg
          "ragged_dot_general requires exactly one rhs group dimension when \
           lhs ragged dimension is noncontracting.";
      let group = rhs_group.(0) in
      check_range group rhs_rank "rhs group dimension" "rhs";
      if Array.mem group r_batch || Array.mem group r_contract then
        invalid_arg
          "ragged_dot_general requires rhs group dimension numbers to be \
           distinct from contracting and batch dimensions.";
      if (shape rhs).(group) <> num_groups then
        invalid_arg
          (Printf.sprintf "expected rhs group dimension size to be %d, got %d."
             num_groups
             (shape rhs).(group)));
  let group_sizes =
    if gs_rank = 1 then L.broadcast group_sizes prefix_shape else group_sizes
  in
  let incr_all = Array.map (fun d -> d + 1) in
  let dot = L.dot_general ?precision ?preferred_element_type in
  match mode with
  | Ragged_noncontracting ->
      dot
        (ragged_to_dense lhs group_sizes ragged l_prefix)
        rhs
        ~lhs_contract:(Array.append (incr_all l_contract) [| 0 |])
        ~rhs_contract:(Array.append r_contract [| rhs_group.(0) |])
        ~lhs_batch:(incr_all l_batch) ~rhs_batch:r_batch
  | Ragged_contracting ->
      let position =
        let rec go i = if l_contract.(i) = ragged then i else go (i + 1) in
        go 0
      in
      let rhs_ragged = r_contract.(position) in
      let r_prefix =
        ragged_prefix mode rhs_rank rhs_ragged r_batch r_contract
      in
      dot
        (ragged_to_dense lhs group_sizes ragged l_prefix)
        (ragged_to_dense rhs group_sizes rhs_ragged r_prefix)
        ~lhs_contract:(incr_all l_contract) ~rhs_contract:(incr_all r_contract)
        ~lhs_batch:(Array.append [| 0 |] (incr_all l_batch))
        ~rhs_batch:(Array.append [| 0 |] (incr_all r_batch))
  | Ragged_batch ->
      dot lhs rhs ~lhs_contract:l_contract ~rhs_contract:r_contract
        ~lhs_batch:l_batch ~rhs_batch:r_batch

let ragged_dot ?precision ?preferred_element_type ?group_offset lhs rhs
    group_sizes =
  ragged_dot_general ?precision ?preferred_element_type ?group_offset lhs rhs
    group_sizes
    {
      Lax_config.dot_dimension_numbers =
        {
          lhs_contract = [| 1 |];
          rhs_contract = [| 1 |];
          lhs_batch = [||];
          rhs_batch = [||];
          untyped_dot = false;
        };
      lhs_ragged_dimensions = [| 0 |];
      rhs_group_dimensions = [| 0 |];
    }

let validate_operand_scale side operand scale contracting =
  Array.iteri
    (fun i size ->
      let scale_size = (shape scale).(i) in
      if Array.mem i contracting then begin
        if size mod scale_size <> 0 then
          invalid_arg
            (Printf.sprintf
               "%s contracting dim %d of size %d must be divisible by its \
                scale's dim size %d."
               side i size scale_size);
        let ratio = size / scale_size in
        if ratio < 2 then
          invalid_arg
            (Printf.sprintf
               "The ratio of %s contracting dim %d to its scale's dim size \
                (%d) must be at least 2."
               side i ratio)
      end
      else if scale_size <> size then
        invalid_arg
          (Printf.sprintf
             "%s dim %d of size %d does not match scale dim size %d." side i
             size scale_size))
    (shape operand)

let validate_scaled_dot lhs rhs lhs_scale rhs_scale
    (dn : Lax_config.dot_dimension_numbers) =
  let ranks =
    List.map
      (fun v -> Array.length (shape v))
      [ lhs; rhs; lhs_scale; rhs_scale ]
  in
  if List.exists (( <> ) (List.hd ranks)) ranks then
    invalid_arg "All input tensors must have the same rank.";
  if Array.length dn.lhs_batch <> Array.length dn.rhs_batch then
    invalid_arg
      (Printf.sprintf
         "LHS and RHS must have the same number of batch dimensions, got %d \
          and %d."
         (Array.length dn.lhs_batch)
         (Array.length dn.rhs_batch));
  if Array.length dn.lhs_contract <> Array.length dn.rhs_contract then
    invalid_arg
      (Printf.sprintf
         "LHS and RHS must have the same number of contracting dimensions, got \
          %d and %d."
         (Array.length dn.lhs_contract)
         (Array.length dn.rhs_contract));
  Array.iteri
    (fun k l ->
      let r = dn.rhs_batch.(k) in
      let sizes =
        [
          (shape lhs).(l);
          (shape rhs).(r);
          (shape lhs_scale).(l);
          (shape rhs_scale).(r);
        ]
      in
      if List.exists (( <> ) (List.hd sizes)) sizes then
        invalid_arg
          (Printf.sprintf
             "All input tensors must have the same batch dimension size for \
              batch dims (%d, %d)."
             l r))
    dn.lhs_batch;
  Array.iteri
    (fun k l ->
      let r = dn.rhs_contract.(k) in
      if (shape lhs).(l) <> (shape rhs).(r) then
        invalid_arg
          (Printf.sprintf
             "LHS contracting dim %d of size %d does not match RHS contracting \
              dim %d of size %d."
             l
             (shape lhs).(l)
             r
             (shape rhs).(r)))
    dn.lhs_contract;
  validate_operand_scale "LHS" lhs lhs_scale dn.lhs_contract;
  validate_operand_scale "RHS" rhs rhs_scale dn.rhs_contract

let scale_broadcast scale operand_shape contracting =
  Array.fold_left
    (fun scale i ->
      let sh = shape scale in
      if sh.(i) = operand_shape.(i) then scale
      else begin
        let multiplier = operand_shape.(i) / sh.(i) in
        let rank = Array.length sh in
        let wide_shape =
          Array.init (rank + 1) (fun d ->
              if d <= i then sh.(d)
              else if d = i + 1 then multiplier
              else sh.(d - 1))
        in
        let dims = Array.init rank (fun d -> if d <= i then d else d + 1) in
        let wide = L.broadcast_in_dim scale wide_shape dims in
        let flat =
          Array.mapi (fun d s -> if d = i then s * multiplier else s) sh
        in
        L.reshape wide flat
      end)
    scale contracting

let scaled_dot ?lhs_scale ?rhs_scale ?dimension_numbers ?preferred_element_type
    lhs rhs =
  let dn =
    match dimension_numbers with
    | Some dn -> dn
    | None ->
        let n = Array.length (shape lhs) in
        if n < 2 then
          invalid_arg "scaled_dot does not support 0-rank and 1-rank operands.";
        let batch = Array.init (n - 2) Fun.id in
        {
          Lax_config.lhs_contract = [| n - 1 |];
          rhs_contract = [| n - 2 |];
          lhs_batch = batch;
          rhs_batch = batch;
          untyped_dot = false;
        }
  in
  let dummy operand contracting =
    let sh =
      Array.mapi
        (fun d s -> if Array.mem d contracting then 1 else s)
        (shape operand)
    in
    L.full sh 1.0 D.Bf16
  in
  let lhs_scale = Option.value lhs_scale ~default:(dummy lhs dn.lhs_contract) in
  let rhs_scale = Option.value rhs_scale ~default:(dummy rhs dn.rhs_contract) in
  validate_scaled_dot lhs rhs lhs_scale rhs_scale dn;
  let preferred = Option.value preferred_element_type ~default:D.Bf16 in
  let scaled operand scale contracting =
    let scale = scale_broadcast scale (shape operand) contracting in
    let operand = NL.astype operand D.Bf16 and scale = NL.astype scale D.Bf16 in
    match preferred with
    | D.F32 | D.F64 ->
        L.mul (NL.astype operand preferred) (NL.astype scale preferred)
    | _ -> L.mul operand scale
  in
  L.dot_general ~preferred_element_type:preferred
    (scaled lhs lhs_scale dn.lhs_contract)
    (scaled rhs rhs_scale dn.rhs_contract)
    ~lhs_contract:dn.lhs_contract ~rhs_contract:dn.rhs_contract
    ~lhs_batch:dn.lhs_batch ~rhs_batch:dn.rhs_batch

let conv_general_dilated_local ?window_strides ?padding ?lhs_dilation
    ?rhs_dilation ?dimension_numbers lhs rhs ~filter_shape =
  let rank = Array.length (shape lhs) in
  let identity = Array.init rank Fun.id in
  let dn =
    match dimension_numbers with
    | Some d -> d
    | None ->
        { T.lhs_spec = identity; rhs_spec = identity; out_spec = identity }
  in
  let patches =
    L.conv_general_dilated_patches ?window_strides ?padding ?lhs_dilation
      ?rhs_dilation ~dimension_numbers:dn lhs ~filter_shape
  in
  let spatial = Array.length filter_shape in
  let lhs_b = Array.sub dn.out_spec 2 spatial in
  let rhs_b = Array.sub dn.rhs_spec 2 spatial in
  let order = Array.init spatial Fun.id in
  Array.sort (fun a b -> compare lhs_b.(a) lhs_b.(b)) order;
  let out =
    L.dot_general patches rhs
      ~lhs_contract:[| dn.out_spec.(1) |]
      ~rhs_contract:[| dn.rhs_spec.(1) |]
      ~lhs_batch:(Array.map (fun i -> lhs_b.(i)) order)
      ~rhs_batch:(Array.map (fun i -> rhs_b.(i)) order)
  in
  NL.moveaxis [| -2; -1 |] [| dn.out_spec.(0); dn.out_spec.(1) |] out

let rng_bit_generator ?(dtype = D.Uint32) ?(algorithm = Rng_default) key
    out_shape =
  (match dtype with
  | D.Uint8 | D.Uint16 | D.Uint32 | D.Uint64 -> ()
  | d ->
      raise
        (Invalid_argument
           (Printf.sprintf "rng_bit_generator: unsupported dtype %s"
              (D.numpy_name d))));
  let key_aval = aval key in
  (match (key_aval.T.dtype, key_aval.T.shape) with
  | D.Uint32, [| 4 |] | D.Uint64, [| 2 |] -> ()
  | _ -> invalid_arg "rng_bit_generator: key must be u32[4] or u64[2]");
  let rbg_algorithm =
    match algorithm with
    | Rng_default -> "DEFAULT"
    | Three_fry -> "THREE_FRY"
    | Philox -> "PHILOX"
  in
  match
    Core.bind
      (T.Rng_bit_generator
         { rbg_shape = out_shape; rbg_dtype = dtype; rbg_algorithm })
      [ key ]
  with
  | [ next_key; values ] -> (next_key, values)
  | _ -> failwith "rng_bit_generator: arity"
