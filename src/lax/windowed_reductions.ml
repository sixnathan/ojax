open Types

let dilate_dim d f = if d = 0 then 0 else ((d - 1) * f) + 1
let stride_dim d w s = if d < w then 0 else ((d - w) / s) + 1

let out_shape operand_shape (w : window_dims) =
  Array.mapi
    (fun i d ->
      let lo, hi = w.w_padding.(i) in
      let dd = dilate_dim d w.base_dilation.(i) in
      let wdim = dilate_dim w.window_dimensions.(i) w.window_dilation.(i) in
      stride_dim (dd + lo + hi) wdim w.window_strides.(i))
    operand_shape

let source_index operand_shape (w : window_dims) oidx widx =
  let n = Array.length operand_shape in
  let idx = Array.make n 0 in
  let ok = ref true in
  let k = ref 0 in
  while !ok && !k < n do
    let lo, _ = w.w_padding.(!k) in
    let p =
      (oidx.(!k) * w.window_strides.(!k))
      + (widx.(!k) * w.window_dilation.(!k))
      - lo
    in
    let bd = w.base_dilation.(!k) in
    if p < 0 || p mod bd <> 0 then ok := false
    else begin
      let orig = p / bd in
      if orig >= operand_shape.(!k) then ok := false else idx.(!k) <- orig
    end;
    incr k
  done;
  if !ok then Some idx else None

let window_pool ~op ~init (w : window_dims) operand =
  let operand_shape = Ndarray.shape operand in
  let osh = out_shape operand_shape w in
  let n_out = Utils.prod osh in
  let wsizes = w.window_dimensions in
  let nwin = Utils.prod wsizes in
  let out = Array.make n_out init in
  for of_ = 0 to n_out - 1 do
    let oidx = Utils.decode of_ osh in
    let acc = ref init in
    for wf = 0 to nwin - 1 do
      let widx = Utils.decode wf wsizes in
      match source_index operand_shape w oidx widx with
      | Some idx -> acc := op !acc (Ndarray.get_f operand idx)
      | None -> acc := op !acc init
    done;
    out.(of_) <- !acc
  done;
  Ndarray.of_floats (Ndarray.dtype operand) osh out

let reduce_window_sum w operand = window_pool ~op:( +. ) ~init:0.0 w operand

let reduce_window_max w operand =
  window_pool ~op:Float.max ~init:neg_infinity w operand

let reduce_window_min w operand =
  window_pool ~op:Float.min ~init:infinity w operand

let reduce_window_general ~reducer ~init w operand =
  window_pool ~op:reducer ~init w operand

let better select k best = match select with Wge -> k > best | Wle -> k < best

let select_and_gather_add select w tangents operand =
  let operand_shape = Ndarray.shape operand in
  let osh = out_shape operand_shape w in
  let n_out = Utils.prod osh in
  let wsizes = w.window_dimensions in
  let nwin = Utils.prod wsizes in
  let out = Array.make n_out 0.0 in
  for of_ = 0 to n_out - 1 do
    let oidx = Utils.decode of_ osh in
    let best_key =
      ref (match select with Wge -> neg_infinity | Wle -> infinity)
    in
    let best_val = ref 0.0 in
    let found = ref false in
    for wf = 0 to nwin - 1 do
      let widx = Utils.decode wf wsizes in
      match source_index operand_shape w oidx widx with
      | Some idx ->
          let k = Ndarray.get_f operand idx in
          if (not !found) || better select k !best_key then begin
            best_key := k;
            best_val := Ndarray.get_f tangents idx;
            found := true
          end
      | None -> ()
    done;
    out.(of_) <- !best_val
  done;
  Ndarray.of_floats (Ndarray.dtype tangents) osh out

let select_and_scatter_add select w source operand =
  let operand_shape = Ndarray.shape operand in
  let osh = out_shape operand_shape w in
  let n_out = Utils.prod osh in
  let wsizes = w.window_dimensions in
  let nwin = Utils.prod wsizes in
  let op_str = Utils.strides operand_shape in
  let result = Array.make (Utils.prod operand_shape) 0.0 in
  for of_ = 0 to n_out - 1 do
    let oidx = Utils.decode of_ osh in
    let best_key =
      ref (match select with Wge -> neg_infinity | Wle -> infinity)
    in
    let best_idx = ref [||] in
    let found = ref false in
    for wf = 0 to nwin - 1 do
      let widx = Utils.decode wf wsizes in
      match source_index operand_shape w oidx widx with
      | Some idx ->
          let k = Ndarray.get_f operand idx in
          if (not !found) || better select k !best_key then begin
            best_key := k;
            best_idx := Array.copy idx;
            found := true
          end
      | None -> ()
    done;
    if !found then begin
      let flat = ref 0 in
      Array.iteri (fun d v -> flat := !flat + (v * op_str.(d))) !best_idx;
      result.(!flat) <- result.(!flat) +. Ndarray.get_f source oidx
    end
  done;
  Ndarray.of_floats (Ndarray.dtype operand) operand_shape result
