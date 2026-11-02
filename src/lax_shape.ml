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
      ^ " for lhs_shape[2:]=("
      ^ String.concat ", "
          (List.map (fmt "np.int64(%d)") (Array.to_list spatial))
      ^ if Array.length spatial = 1 then ",)" else ")")

let verify_conv rhs (dn : conv_dims) ~window_strides ~lhs_dilation ~rhs_dilation
    =
  let sdims = Array.length dn.rhs_spec - 2 in
  let cannot_lower line =
    Py_exn.value_error ("Cannot lower jaxpr with verifier errors:\n\t" ^ line)
  in
  let same_size what given =
    if Array.length given <> sdims then
      cannot_lower
        (Printf.sprintf
           "expects %s factors to have same dimension-size as size of window \
            dimensions (%d), but got: %d."
           what sdims (Array.length given))
  in
  same_size "base-dilation" lhs_dilation;
  same_size "window-dilation" rhs_dilation;
  for i = 0 to sdims - 1 do
    let w = rhs.(dn.rhs_spec.(i + 2)) in
    if w <= 0 then
      cannot_lower
        (Printf.sprintf
           "expects window to have positive value for %d-th window dimension, \
            but got %d."
           i w);
    if window_strides.(i) <= 0 then
      cannot_lower
        (Printf.sprintf
           "expects window to have positive stride for %d-th window dimension, \
            but got %d."
           i window_strides.(i))
  done

let shapes_text shapes = String.concat ", " (List.map tuple shapes)

let check_concatenate shapes dimension =
  match shapes with
  | [] -> Py_exn.type_error "concatenate expects at least one operand, got 0."
  | first :: rest ->
      if List.exists (fun s -> Array.length s <> Array.length first) rest then
        Py_exn.type_error
          ("Cannot concatenate arrays with different numbers of dimensions: \
            got " ^ shapes_text shapes ^ ".");
      if dimension < 0 || dimension >= Array.length first then
        Py_exn.type_error
          (Printf.sprintf
             "concatenate dimension out of bounds: dimension %d for shapes %s."
             dimension (shapes_text shapes));
      let without s =
        List.filteri (fun i _ -> i <> dimension) (Array.to_list s)
      in
      if List.exists (fun s -> without s <> without first) rest then
        Py_exn.type_error
          (Printf.sprintf
             "Cannot concatenate arrays with shapes that differ in dimensions \
              other than the one being concatenated: concatenating along \
              dimension %d for shapes %s."
             dimension (shapes_text shapes))

let padding_text config =
  let one (l, h, i) = Printf.sprintf "(%d, %d, %d)" l h i in
  match Array.to_list config with
  | [ c ] -> "(" ^ one c ^ ",)"
  | items -> "(" ^ String.concat ", " (List.map one items) ^ ")"

let check_pad operand value config =
  if Array.length value <> 0 then
    Py_exn.value_error
      ("padding_value must be a scalar; got np.shape(padding_value)="
     ^ tuple value);
  if Array.length config <> Array.length operand then
    Py_exn.value_error
      (Printf.sprintf
         "length of padding_config must equal the number of axes of operand, \
          got padding_config %s for operand shape %s"
         (padding_text config) (tuple operand));
  if Array.exists (fun (_, _, i) -> i < 0) config then
    Py_exn.value_error
      ("interior padding in padding_config must be nonnegative, got \
        padding_config " ^ padding_text config);
  let result =
    Array.mapi
      (fun n (l, h, i) -> l + h + dilate_dim operand.(n) (i + 1))
      config
  in
  if Array.exists (fun d -> d < 0) result then
    Py_exn.value_error
      (Printf.sprintf
         "Dimension size after padding is not at least 0, got result shape %s, \
          for padding_config %s and operand shape %s"
         (tuple result) (padding_text config) (tuple operand))

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
  | Rng_bit_generator { rbg_shape; _ }
    when Array.exists (fun d -> d < 0) rbg_shape ->
      raise
        (Py_exn.Mlir_error
           "Invalid type:\nerror: unknown: invalid tensor dimension size")
  | Concatenate dimension ->
      check_concatenate
        (List.map (fun (a : aval) -> a.shape) (avals ()))
        dimension
  | Pad config -> (
      match avals () with
      | [ operand; value ] -> check_pad operand.shape value.shape config
      | _ -> ())
  | Sort _ -> (
      match avals () with
      | first :: rest
        when List.exists (fun (a : aval) -> a.shape <> first.shape) rest ->
          Py_exn.type_error
            ("Arguments to sort must have equal shapes, got: "
            ^ String.concat " "
                (List.map (fun (a : aval) -> tuple a.shape) (first :: rest)))
      | _ -> ())
  | _ -> ()

let verify prim avals =
  match prim with
  | Conv_general_dilated
      { window_strides; lhs_dilation; rhs_dilation; dimension_numbers; _ } -> (
      match avals () with
      | [ _; (rhs : aval) ] ->
          verify_conv rhs.shape dimension_numbers ~window_strides ~lhs_dilation
            ~rhs_dilation
      | _ -> ())
  | _ -> ()
