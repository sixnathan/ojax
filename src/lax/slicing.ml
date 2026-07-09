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
