open Types

let tuple = Validate.python_tuple
let at = Validate.tuple_index

let aval_string (a : aval) =
  Validate.aval_short ~dtype_name:(Dtype.numpy_name a.dtype) ~weak:a.weak_type
    a.shape

let check_reduce_axes (a : aval) axes =
  let listed = Array.to_list axes in
  if List.length (List.sort_uniq compare listed) <> List.length listed then
    Py_exn.value_error ("duplicate value in 'axes' of reduction: " ^ tuple axes);
  let n = Array.length a.shape in
  if List.exists (fun d -> d < 0 || d >= n) listed then
    Py_exn.value_error
      (Printf.sprintf "reduction axes %s contains out-of-bounds indices for %s."
         (tuple axes) (aval_string a))

let check_top_k (a : aval) k axis =
  if Dtypes.issubdtype a.dtype `Complex then
    Py_exn.value_error "top_k is not compatible with complex inputs.";
  if k < 0 then
    Py_exn.value_error
      (Printf.sprintf "k argument to top_k must be nonnegative, got %d" k);
  let n = Array.length a.shape in
  if n = 0 then
    Py_exn.type_error
      ("top_k operand must have >= 1 dimension, got " ^ tuple a.shape);
  if axis < 0 || axis >= n then
    Py_exn.value_error
      (Printf.sprintf "axis argument out of range: axis=%d for operand.shape=%s"
         axis (tuple a.shape));
  if a.shape.(axis) < k then
    Py_exn.value_error
      (Printf.sprintf
         "k argument to top_k must be no larger than size along axis; got k=%d \
          with shape=%s and axis=%d"
         k
         (Validate.python_list (Array.to_list a.shape))
         axis)

let dilate_dim d r = if d = 0 then 0 else 1 + (r * (d - 1))

let check_conv lhs rhs (dn : conv_dims) ~window_strides ~padding ~lhs_dilation
    ~rhs_dilation ~feature_group_count ~batch_group_count =
  let value_error = Py_exn.value_error in
  let fmt = Printf.sprintf in
  if Array.length lhs <> Array.length rhs then
    value_error
      (fmt
         "conv_general_dilated lhs and rhs must have the same number of \
          dimensions, but got %s and %s."
         (tuple lhs) (tuple rhs));
  if feature_group_count <= 0 then
    value_error
      (fmt
         "conv_general_dilated feature_group_count must be a positive integer, \
          got %d."
         feature_group_count);
  let lhs_feature = at lhs (at dn.lhs_spec 1) in
  let quot = Validate.py_floordiv lhs_feature feature_group_count in
  if Validate.py_mod lhs_feature feature_group_count <> 0 then
    value_error
      (fmt
         "conv_general_dilated feature_group_count must divide lhs feature \
          dimension size, but %d does not divide %d."
         feature_group_count lhs_feature);
  let rhs_in = at rhs (at dn.rhs_spec 1) in
  if quot <> rhs_in then
    value_error
      (fmt
         "conv_general_dilated lhs feature dimension size divided by \
          feature_group_count must equal the rhs input feature dimension size, \
          but %d // %d != %d."
         lhs_feature feature_group_count rhs_in);
  let rhs_out = at rhs (at dn.rhs_spec 0) in
  if Validate.py_mod rhs_out feature_group_count <> 0 then
    value_error
      (fmt
         "conv_general_dilated rhs output feature dimension size must be a \
          multiple of feature_group_count, but %d is not a multiple of %d."
         rhs_out feature_group_count);
  if batch_group_count <= 0 then
    value_error
      (fmt
         "conv_general_dilated batch_group_count must be a positive integer, \
          got %d."
         batch_group_count);
  let lhs_batch = at lhs (at dn.lhs_spec 0) in
  if batch_group_count > 1 && lhs_batch mod batch_group_count <> 0 then
    value_error
      (fmt
         "conv_general_dilated batch_group_count must divide lhs batch \
          dimension size, but %d does not divide %d."
         batch_group_count lhs_batch);
  if rhs_out mod batch_group_count <> 0 then
    value_error
      (fmt
         "conv_general_dilated rhs output feature dimension size must be a \
          multiple of batch_group_count, but %d is not a multiple of %d."
         rhs_out batch_group_count);
  if batch_group_count > 1 && feature_group_count > 1 then
    value_error
      (fmt
         "At most one of batch_group_count and feature_group_count may be > 1, \
          got batch_group_count=%d and feature_group_count=%d"
         batch_group_count feature_group_count);
  let sdims = Array.length dn.rhs_spec - 2 in
  if sdims <> Array.length window_strides then
    value_error
      (fmt
         "conv_general_dilated window and window_strides must have the same \
          number of dimensions, but got %d and %d"
         sdims
         (Array.length window_strides));
  if Array.length padding <> sdims then
    Py_exn.type_error
      (fmt "Wrong number of explicit pads for convolution: expected %d, got %d."
         sdims (Array.length padding));
  let dilated shape spec dilation =
    if Array.exists (fun d -> d <= 0) dilation then
      Py_exn.type_error
        ("All dilations must be positive, got " ^ tuple dilation ^ ".");
    let n = Array.length spec in
    let pad = max 0 (n - Array.length dilation) in
    Array.init
      (min n (pad + Array.length dilation))
      (fun i ->
        let r = if i < pad then 1 else dilation.(i - pad) in
        dilate_dim shape.(spec.(i)) r)
  in
  let lhs_trans = dilated lhs dn.lhs_spec lhs_dilation in
  ignore (dilated rhs dn.rhs_spec rhs_dilation);
  if Array.length padding <> Array.length lhs_trans - 2 then
    Py_exn.type_error
      (fmt "Wrong number of explicit pads for convolution: expected %d, got %d."
         (Array.length lhs_trans - 2)
         (Array.length padding));
  let spatial = Array.sub lhs_trans 2 (Array.length padding) in
  if
    Array.exists Fun.id
      (Array.mapi (fun i (lo, hi) -> spatial.(i) + lo + hi < 0) padding)
  then
    value_error
      ("Negative padding is larger than the size of the corresponding \
        dimension: got padding=("
      ^ String.concat ", "
          (List.map (fun (l, h) -> fmt "(%d, %d)" l h) (Array.to_list padding))
      ^ (if Array.length padding = 1 then ",)" else ")")
      ^ " for lhs_shape[2:]=["
      ^ String.concat " " (List.map string_of_int (Array.to_list spatial))
      ^ "]")

let check prim avals =
  match prim with
  | Reduce_sum axes
  | Reduce_max axes
  | Reduce_min axes
  | Reduce_prod axes
  | Reduce_and axes
  | Reduce_or axes
  | Reduce_xor axes -> (
      match avals () with a :: _ -> check_reduce_axes a axes | [] -> ())
  | Top_k { k; axis } -> (
      match avals () with a :: _ -> check_top_k a k axis | [] -> ())
  | Broadcast_in_dim { shape; _ } ->
      Validate.check_shapelike ~name:"broadcast_in_dim" ~arg:"shape" shape
  | Conv_general_dilated
      {
        window_strides;
        padding;
        lhs_dilation;
        rhs_dilation;
        dimension_numbers;
        feature_group_count;
        batch_group_count;
        _;
      } -> (
      match avals () with
      | [ lhs; rhs ] ->
          check_conv lhs.shape rhs.shape dimension_numbers ~window_strides
            ~padding ~lhs_dilation ~rhs_dilation ~feature_group_count
            ~batch_group_count
      | _ -> ())
  | _ -> ()
