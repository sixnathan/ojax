open Types

let to_array nd =
  let n = Utils.prod (Ndarray.shape nd) in
  let arr = Array.make n 0.0 in
  let _ =
    Ndarray.fold
      (fun i x ->
        arr.(i) <- x;
        i + 1)
      0 nd
  in
  arr

let sign_f x = if x > 0.0 then 1.0 else if x < 0.0 then -1.0 else x
let bool_of b = if b then 1.0 else 0.0
let exp2_f x = Float.pow 2.0 x

let clz_nbits = function
  | Dtype.I32 -> 32
  | Dtype.I64 -> 64
  | _ -> failwith "lax: clz requires an integer operand"

let clz_bits nbits x =
  let m = Int64.of_float x in
  let m = if nbits = 32 then Int64.logand m 0xFFFFFFFFL else m in
  if Int64.equal m 0L then float_of_int nbits
  else
    let rec loop k =
      if k < 0 then float_of_int nbits
      else if Int64.equal (Int64.logand (Int64.shift_right_logical m k) 1L) 1L
      then float_of_int (nbits - 1 - k)
      else loop (k - 1)
    in
    loop (nbits - 1)

let int_nbits = function
  | Dtype.I32 -> 32
  | Dtype.I64 -> 64
  | _ -> failwith "lax: integer operand required"

let popcount_bits nbits x =
  let m = Int64.of_float x in
  let m = if nbits = 32 then Int64.logand m 0xFFFFFFFFL else m in
  let rec loop k acc =
    if k >= nbits then acc
    else
      loop (k + 1)
        (acc + Int64.to_int (Int64.logand (Int64.shift_right_logical m k) 1L))
  in
  float_of_int (loop 0 0)

let not_f dt x =
  match dt with
  | Dtype.Bool -> if x = 0.0 then 1.0 else 0.0
  | Dtype.I32 | Dtype.I64 -> Int64.to_float (Int64.lognot (Int64.of_float x))
  | _ -> failwith "lax: not requires a boolean or integer operand"

let and_f dt x y =
  match dt with
  | Dtype.Bool -> if x <> 0.0 && y <> 0.0 then 1.0 else 0.0
  | Dtype.I32 | Dtype.I64 ->
      Int64.to_float (Int64.logand (Int64.of_float x) (Int64.of_float y))
  | _ -> failwith "lax: and requires a boolean or integer operand"

let umulhi a b =
  let m = 0xFFFFFFFFL in
  let al = Int64.logand a m and ah = Int64.shift_right_logical a 32 in
  let bl = Int64.logand b m and bh = Int64.shift_right_logical b 32 in
  let lo_lo = Int64.mul al bl in
  let hi_lo = Int64.mul ah bl in
  let lo_hi = Int64.mul al bh in
  let hi_hi = Int64.mul ah bh in
  let cross =
    Int64.add
      (Int64.add (Int64.shift_right_logical lo_lo 32) (Int64.logand hi_lo m))
      (Int64.logand lo_hi m)
  in
  Int64.add hi_hi
    (Int64.add
       (Int64.shift_right_logical hi_lo 32)
       (Int64.add
          (Int64.shift_right_logical lo_hi 32)
          (Int64.shift_right_logical cross 32)))

let mulhi_i64 a b =
  let u = umulhi a b in
  let u = if a < 0L then Int64.sub u b else u in
  if b < 0L then Int64.sub u a else u

let mulhi_f dt x y =
  match dt with
  | Dtype.I32 ->
      Int64.to_float
        (Int64.shift_right (Int64.mul (Int64.of_float x) (Int64.of_float y)) 32)
  | Dtype.I64 ->
      Int64.to_float (mulhi_i64 (Int64.of_float x) (Int64.of_float y))
  | _ -> failwith "lax: mulhi requires an integer operand"

let or_f dt x y =
  match dt with
  | Dtype.Bool -> if x <> 0.0 || y <> 0.0 then 1.0 else 0.0
  | Dtype.I32 | Dtype.I64 ->
      Int64.to_float (Int64.logor (Int64.of_float x) (Int64.of_float y))
  | _ -> failwith "lax: or requires a boolean or integer operand"

let xor_f dt x y =
  match dt with
  | Dtype.Bool -> if x <> 0.0 <> (y <> 0.0) then 1.0 else 0.0
  | Dtype.I32 | Dtype.I64 ->
      Int64.to_float (Int64.logxor (Int64.of_float x) (Int64.of_float y))
  | _ -> failwith "lax: xor requires a boolean or integer operand"

let rem_f dt x y =
  match dt with
  | Dtype.F32 | Dtype.F64 -> Float.rem x y
  | Dtype.I32 | Dtype.I64 ->
      Int64.to_float (Int64.rem (Int64.of_float x) (Int64.of_float y))
  | _ -> failwith "lax: rem requires an integer or float operand"

let shift_left_f x y =
  Int64.to_float (Int64.shift_left (Int64.of_float x) (int_of_float y))

let shift_right_arithmetic_f x y =
  Int64.to_float (Int64.shift_right (Int64.of_float x) (int_of_float y))

let shift_right_logical_f dt x y =
  let s = int_of_float y in
  match dt with
  | Dtype.I32 ->
      let u = Int64.logand (Int64.of_float x) 0xFFFFFFFFL in
      Int64.to_float (Int64.shift_right_logical u s)
  | Dtype.I64 -> Int64.to_float (Int64.shift_right_logical (Int64.of_float x) s)
  | _ -> failwith "lax: shift_right_logical requires an integer operand"

let f64_nextafter x y =
  if x <> x || y <> y then Float.nan
  else if x = y then y
  else if y > x then Float.succ x
  else Float.pred x

let f32_next_up x =
  if x = infinity then infinity
  else if x = 0.0 then Int32.float_of_bits 1l
  else
    let b = Int32.bits_of_float x in
    Int32.float_of_bits (if x > 0.0 then Int32.add b 1l else Int32.sub b 1l)

let f32_next_down x =
  if x = neg_infinity then neg_infinity
  else if x = 0.0 then Int32.float_of_bits (Int32.logor Int32.min_int 1l)
  else
    let b = Int32.bits_of_float x in
    Int32.float_of_bits (if x > 0.0 then Int32.sub b 1l else Int32.add b 1l)

let f32_nextafter x y =
  if x <> x || y <> y then Float.nan
  else if x = y then y
  else if y > x then f32_next_up x
  else f32_next_down x

let nextafter_f dt x y =
  match dt with
  | Dtype.F32 -> f32_nextafter x y
  | Dtype.F64 -> f64_nextafter x y
  | _ -> failwith "lax: nextafter requires a float operand"

let integer_pow_f y x = Float.pow x (float_of_int y)
let logistic_f x = 1.0 /. (1.0 +. Float.exp (-.x))
let rsqrt_f x = 1.0 /. Float.sqrt x
let square_f x = x *. x
let un f = function [ a ] -> [ f a ] | _ -> failwith "lax: expected 1 operand"

let bin f = function
  | [ a; b ] -> [ f a b ]
  | _ -> failwith "lax: expected 2 operands"

let broadcast_in_dim_impl shape dims a =
  let os = Ndarray.shape a in
  let ostr = Utils.strides os in
  let av = to_array a in
  let out_n = Utils.prod shape in
  let out = Array.make out_n 0.0 in
  for f = 0 to out_n - 1 do
    let oidx = Utils.decode f shape in
    let iflat = ref 0 in
    Array.iteri
      (fun i od ->
        let ii = if os.(i) = 1 then 0 else oidx.(od) in
        iflat := !iflat + (ii * ostr.(i)))
      dims;
    out.(f) <- av.(!iflat)
  done;
  Ndarray.of_floats (Ndarray.dtype a) shape out

let reshape_impl ns a = Ndarray.of_floats (Ndarray.dtype a) ns (to_array a)

let reduce_sum_impl axes a =
  let os = Ndarray.shape a in
  let out_shape = Utils.reduce_shape os axes in
  let out_str = Utils.strides out_shape in
  let ndim = Array.length os in
  let is_red = Array.make ndim false in
  Array.iter (fun ax -> is_red.(ax) <- true) axes;
  let src = to_array a in
  let out = Array.make (Utils.prod out_shape) 0.0 in
  for flat = 0 to Array.length src - 1 do
    let oidx = Utils.decode flat os in
    let of_ = ref 0 and k = ref 0 in
    for d = 0 to ndim - 1 do
      if not is_red.(d) then begin
        of_ := !of_ + (oidx.(d) * out_str.(!k));
        incr k
      end
    done;
    out.(!of_) <- out.(!of_) +. src.(flat)
  done;
  Ndarray.of_floats (Ndarray.dtype a) out_shape out

let dot_general_impl (dd : dot_dims) lhs rhs =
  let ls = Ndarray.shape lhs and rs = Ndarray.shape rhs in
  let la = to_array lhs and ra = to_array rhs in
  let lstr = Utils.strides ls and rstr = Utils.strides rs in
  let lhs_free =
    Utils.free_axes (Array.length ls) dd.lhs_batch dd.lhs_contract
  in
  let rhs_free =
    Utils.free_axes (Array.length rs) dd.rhs_batch dd.rhs_contract
  in
  let contract_sizes = Array.map (fun a -> ls.(a)) dd.lhs_contract in
  let out_shape = Utils.dot_general_shape dd ls rs in
  let out_n = Utils.prod out_shape in
  let out = Array.make out_n 0.0 in
  let nb = Array.length dd.lhs_batch in
  let nlf = Array.length lhs_free in
  let ncon = Array.length dd.lhs_contract in
  let cprod = Utils.prod contract_sizes in
  for f = 0 to out_n - 1 do
    let oidx = Utils.decode f out_shape in
    let acc = ref 0.0 in
    for cf = 0 to cprod - 1 do
      let cidx = Utils.decode cf contract_sizes in
      let lflat = ref 0 and rflat = ref 0 in
      for k = 0 to nb - 1 do
        lflat := !lflat + (oidx.(k) * lstr.(dd.lhs_batch.(k)));
        rflat := !rflat + (oidx.(k) * rstr.(dd.rhs_batch.(k)))
      done;
      for j = 0 to nlf - 1 do
        lflat := !lflat + (oidx.(nb + j) * lstr.(lhs_free.(j)))
      done;
      for j = 0 to Array.length rhs_free - 1 do
        rflat := !rflat + (oidx.(nb + nlf + j) * rstr.(rhs_free.(j)))
      done;
      for m = 0 to ncon - 1 do
        lflat := !lflat + (cidx.(m) * lstr.(dd.lhs_contract.(m)));
        rflat := !rflat + (cidx.(m) * rstr.(dd.rhs_contract.(m)))
      done;
      acc := !acc +. (la.(!lflat) *. ra.(!rflat))
    done;
    out.(f) <- !acc
  done;
  Ndarray.of_floats (Ndarray.dtype lhs) out_shape out

let select_n_impl = function
  | which :: cases ->
      let wa = to_array which in
      let carr = Array.of_list (List.map to_array cases) in
      let head = List.hd cases in
      let out_dt = Ndarray.dtype head in
      let shp = Ndarray.shape head in
      let n = Utils.prod shp in
      let out = Array.init n (fun i -> carr.(int_of_float wa.(i)).(i)) in
      [ Ndarray.of_floats out_dt shp out ]
  | [] -> failwith "lax: select_n expects at least a predicate"

let concatenate_impl dim operands =
  let shapes = List.map Ndarray.shape operands in
  let out_shape = Utils.concatenate_shape dim shapes in
  let out_str = Utils.strides out_shape in
  let out = Array.make (Utils.prod out_shape) 0.0 in
  let dt = Ndarray.dtype (List.hd operands) in
  let offset = ref 0 in
  List.iter
    (fun op ->
      let os = Ndarray.shape op in
      let av = to_array op in
      for f = 0 to Array.length av - 1 do
        let oidx = Utils.decode f os in
        let flat = ref 0 in
        Array.iteri
          (fun d idx ->
            let od = if d = dim then idx + !offset else idx in
            flat := !flat + (od * out_str.(d)))
          oidx;
        out.(!flat) <- av.(f)
      done;
      offset := !offset + os.(dim))
    operands;
  Ndarray.of_floats dt out_shape out

let pad_impl cfg operand pv =
  let in_shape = Ndarray.shape operand in
  let n = Array.length in_shape in
  let pad_v = Ndarray.get_f pv [||] in
  let inter =
    Array.init n (fun i ->
        let _, _, interior = cfg.(i) in
        Utils.dilate_dim in_shape.(i) (interior + 1))
  in
  let lo_pos =
    Array.init n (fun i ->
        let lo, _, _ = cfg.(i) in
        max lo 0)
  in
  let hi_pos =
    Array.init n (fun i ->
        let _, hi, _ = cfg.(i) in
        max hi 0)
  in
  let pos_shape = Array.init n (fun i -> lo_pos.(i) + hi_pos.(i) + inter.(i)) in
  let pos_str = Utils.strides pos_shape in
  let pos = Array.make (Utils.prod pos_shape) pad_v in
  let av = to_array operand in
  for f = 0 to Array.length av - 1 do
    let oidx = Utils.decode f in_shape in
    let flat = ref 0 in
    for d = 0 to n - 1 do
      let _, _, interior = cfg.(d) in
      let p = lo_pos.(d) + (oidx.(d) * (interior + 1)) in
      flat := !flat + (p * pos_str.(d))
    done;
    pos.(!flat) <- av.(f)
  done;
  let final_shape = Utils.pad_shape cfg in_shape in
  let start =
    Array.init n (fun i ->
        let lo, _, _ = cfg.(i) in
        max (-lo) 0)
  in
  let out = Array.make (Utils.prod final_shape) 0.0 in
  for f = 0 to Array.length out - 1 do
    let fidx = Utils.decode f final_shape in
    let flat = ref 0 in
    for d = 0 to n - 1 do
      flat := !flat + ((fidx.(d) + start.(d)) * pos_str.(d))
    done;
    out.(f) <- pos.(!flat)
  done;
  Ndarray.of_floats (Ndarray.dtype operand) final_shape out

let rev_impl dims operand =
  let os = Ndarray.shape operand in
  let n = Array.length os in
  let is_rev = Array.make n false in
  Array.iter (fun d -> is_rev.(d) <- true) dims;
  let av = to_array operand in
  let str = Utils.strides os in
  let out = Array.make (Array.length av) 0.0 in
  for f = 0 to Array.length av - 1 do
    let oidx = Utils.decode f os in
    let flat = ref 0 in
    for d = 0 to n - 1 do
      let id = if is_rev.(d) then os.(d) - 1 - oidx.(d) else oidx.(d) in
      flat := !flat + (id * str.(d))
    done;
    out.(f) <- av.(!flat)
  done;
  Ndarray.of_floats (Ndarray.dtype operand) os out

let split_impl sizes axis operand =
  let os = Ndarray.shape operand in
  let str = Utils.strides os in
  let av = to_array operand in
  let dt = Ndarray.dtype operand in
  let _, rev_out =
    Array.fold_left
      (fun (offset, acc) size ->
        let out_shape =
          Array.mapi (fun i d -> if i = axis then size else d) os
        in
        let out = Array.make (Utils.prod out_shape) 0.0 in
        for f = 0 to Array.length out - 1 do
          let oidx = Utils.decode f out_shape in
          let flat = ref 0 in
          for d = 0 to Array.length os - 1 do
            let id = if d = axis then oidx.(d) + offset else oidx.(d) in
            flat := !flat + (id * str.(d))
          done;
          out.(f) <- av.(!flat)
        done;
        (offset + size, Ndarray.of_floats dt out_shape out :: acc))
      (0, []) sizes
  in
  List.rev rev_out

let squeeze_impl dims operand =
  reshape_impl (Utils.squeeze_shape dims (Ndarray.shape operand)) operand

let stack_impl axis operands =
  let in_shape = Ndarray.shape (List.hd operands) in
  let out_shape = Utils.stack_shape axis (List.length operands) in_shape in
  let out_str = Utils.strides out_shape in
  let dt = Ndarray.dtype (List.hd operands) in
  let out = Array.make (Utils.prod out_shape) 0.0 in
  List.iteri
    (fun k op ->
      let av = to_array op in
      let os = Ndarray.shape op in
      for f = 0 to Array.length av - 1 do
        let iidx = Utils.decode f os in
        let flat = ref 0 and src = ref 0 in
        for d = 0 to Array.length out_shape - 1 do
          let idx =
            if d = axis then k
            else
              let v = iidx.(!src) in
              incr src;
              v
          in
          flat := !flat + (idx * out_str.(d))
        done;
        out.(!flat) <- av.(f)
      done)
    operands;
  Ndarray.of_floats dt out_shape out

let tile_impl reps operand =
  let os = Ndarray.shape operand in
  let out_shape = Utils.tile_shape reps os in
  let av = to_array operand in
  let str = Utils.strides os in
  let out = Array.make (Utils.prod out_shape) 0.0 in
  for f = 0 to Array.length out - 1 do
    let oidx = Utils.decode f out_shape in
    let flat = ref 0 in
    for d = 0 to Array.length os - 1 do
      flat := !flat + (oidx.(d) mod os.(d) * str.(d))
    done;
    out.(f) <- av.(!flat)
  done;
  Ndarray.of_floats (Ndarray.dtype operand) out_shape out

let transpose_impl perm operand =
  let os = Ndarray.shape operand in
  let out_shape = Utils.transpose_shape perm os in
  let av = to_array operand in
  let str = Utils.strides os in
  let out = Array.make (Utils.prod out_shape) 0.0 in
  for f = 0 to Array.length out - 1 do
    let oidx = Utils.decode f out_shape in
    let flat = ref 0 in
    Array.iteri (fun d p -> flat := !flat + (oidx.(d) * str.(p))) perm;
    out.(f) <- av.(!flat)
  done;
  Ndarray.of_floats (Ndarray.dtype operand) out_shape out

let unstack_impl axis operand =
  let os = Ndarray.shape operand in
  let str = Utils.strides os in
  let av = to_array operand in
  let out_shape = Utils.remove_int os axis in
  let out_n = Utils.prod out_shape in
  let dt = Ndarray.dtype operand in
  List.init os.(axis) (fun k ->
      let out = Array.make out_n 0.0 in
      for f = 0 to out_n - 1 do
        let oidx = Utils.decode f out_shape in
        let flat = ref 0 and src = ref 0 in
        for d = 0 to Array.length os - 1 do
          let idx =
            if d = axis then k
            else
              let v = oidx.(!src) in
              incr src;
              v
          in
          flat := !flat + (idx * str.(d))
        done;
        out.(f) <- av.(!flat)
      done;
      Ndarray.of_floats dt out_shape out)

let reduce_op_impl combine identity axes a =
  let os = Ndarray.shape a in
  let out_shape = Utils.reduce_shape os axes in
  let out_str = Utils.strides out_shape in
  let ndim = Array.length os in
  let is_red = Array.make ndim false in
  Array.iter (fun ax -> is_red.(ax) <- true) axes;
  let src = to_array a in
  let out = Array.make (Utils.prod out_shape) identity in
  for flat = 0 to Array.length src - 1 do
    let oidx = Utils.decode flat os in
    let of_ = ref 0 and k = ref 0 in
    for d = 0 to ndim - 1 do
      if not is_red.(d) then begin
        of_ := !of_ + (oidx.(d) * out_str.(!k));
        incr k
      end
    done;
    out.(!of_) <- combine out.(!of_) src.(flat)
  done;
  Ndarray.of_floats (Ndarray.dtype a) out_shape out

let and_identity = function Dtype.Bool -> 1.0 | _ -> -1.0

let reduce_and_impl axes a =
  let dt = Ndarray.dtype a in
  reduce_op_impl (and_f dt) (and_identity dt) axes a

let reduce_or_impl axes a = reduce_op_impl (or_f (Ndarray.dtype a)) 0.0 axes a
let reduce_xor_impl axes a = reduce_op_impl (xor_f (Ndarray.dtype a)) 0.0 axes a

let argminmax_impl is_max axis index_dtype operand =
  let os = Ndarray.shape operand in
  let str = Utils.strides os in
  let out_shape = Utils.remove_int os axis in
  let out_n = Utils.prod out_shape in
  let av = to_array operand in
  let n = os.(axis) in
  let out = Array.make out_n 0.0 in
  for of_ = 0 to out_n - 1 do
    let oidx = Utils.decode of_ out_shape in
    let base = ref 0 and k = ref 0 in
    for d = 0 to Array.length os - 1 do
      if d <> axis then begin
        base := !base + (oidx.(!k) * str.(d));
        incr k
      end
    done;
    let best = ref av.(!base) and best_i = ref 0 in
    for i = 1 to n - 1 do
      let v = av.(!base + (i * str.(axis))) in
      let pick = (if is_max then v > !best else v < !best) || v <> v in
      if pick then begin
        best := v;
        best_i := i
      end
    done;
    out.(of_) <- float_of_int !best_i
  done;
  Ndarray.of_floats index_dtype out_shape out

let clamp_impl mn x mx =
  let a = to_array mn and b = to_array x and c = to_array mx in
  let out =
    Array.init (Array.length b) (fun i ->
        Float.max (Float.min a.(i) c.(i)) (Float.min b.(i) c.(i)))
  in
  Ndarray.of_floats (Ndarray.dtype x) (Ndarray.shape x) out

let beta_impl = function
  | [ av; bv; xv ] ->
      let dt = Ndarray.dtype av in
      let a = to_array av and b = to_array bv and x = to_array xv in
      let out =
        Array.init (Array.length a) (fun i ->
            Special.regularized_incomplete_beta dt a.(i) b.(i) x.(i))
      in
      [ Ndarray.of_floats dt (Ndarray.shape xv) out ]
  | _ -> failwith "lax: regularized_incomplete_beta expects 3 operands"

let bitcast_impl new_dt operand =
  let f =
    match (Ndarray.dtype operand, new_dt) with
    | Dtype.F32, Dtype.I32 -> fun x -> Int32.to_float (Int32.bits_of_float x)
    | Dtype.I32, Dtype.F32 -> fun x -> Int32.float_of_bits (Int32.of_float x)
    | Dtype.F64, Dtype.I64 -> fun x -> Int64.to_float (Int64.bits_of_float x)
    | Dtype.I64, Dtype.F64 -> fun x -> Int64.float_of_bits (Int64.of_float x)
    | a, b when a = b -> fun x -> x
    | _ ->
        failwith
          "lax: bitcast_convert_type only supports same-width real reinterpret \
           in M1"
  in
  Ndarray.of_floats new_dt (Ndarray.shape operand)
    (Array.map f (to_array operand))

let iota_impl dtype shape dimension =
  let n = Utils.prod shape in
  let out =
    Array.init n (fun f -> float_of_int (Utils.decode f shape).(dimension))
  in
  Ndarray.of_floats dtype shape out

let empty_impl dtype shape =
  Ndarray.of_floats dtype shape (Array.make (Utils.prod shape) 0.0)

let platform_cpu_index (platforms : string array option array) : int =
  let n = Array.length platforms in
  let rec find_cpu i =
    if i >= n then find_default 0
    else
      match platforms.(i) with
      | Some names when Array.exists (fun p -> p = "cpu") names -> i
      | _ -> find_cpu (i + 1)
  and find_default i =
    if i >= n then failwith "lax: platform_index has no cpu or default branch"
    else match platforms.(i) with None -> i | Some _ -> find_default (i + 1)
  in
  find_cpu 0

let composite_impl (cj : closed_jaxpr) inputs =
  let outs =
    Jaxpr.eval_closed_jaxpr cj (List.map (fun nd -> Concrete nd) inputs)
  in
  List.map
    (function
      | Concrete nd -> nd
      | Tracer _ -> failwith "lax: composite reducer produced a tracer")
    outs

let reduce_impl (cj : closed_jaxpr) dimensions operands inits =
  let os = Ndarray.shape (List.hd operands) in
  let out_shape = Utils.reduce_shape os dimensions in
  let out_str = Utils.strides out_shape in
  let ndim = Array.length os in
  let is_red = Array.make ndim false in
  Array.iter (fun ax -> is_red.(ax) <- true) dimensions;
  let op_dts = List.map Ndarray.dtype operands in
  let out_n = Utils.prod out_shape in
  let op_arrays = List.map to_array operands in
  let accs =
    List.map (fun nd -> Array.make out_n (Ndarray.get_f nd [||])) inits
  in
  let src0 = List.hd op_arrays in
  let mk dt x = Concrete (Ndarray.of_floats dt [||] [| x |]) in
  let as_nd = function
    | Concrete nd -> nd
    | Tracer _ -> failwith "lax: reduce reducer produced a tracer"
  in
  for flat = 0 to Array.length src0 - 1 do
    let oidx = Utils.decode flat os in
    let of_ = ref 0 and k = ref 0 in
    for d = 0 to ndim - 1 do
      if not is_red.(d) then begin
        of_ := !of_ + (oidx.(d) * out_str.(!k));
        incr k
      end
    done;
    let acc_args = List.map2 (fun dt acc -> mk dt acc.(!of_)) op_dts accs in
    let elem_args = List.map2 (fun dt a -> mk dt a.(flat)) op_dts op_arrays in
    let res = Jaxpr.eval_closed_jaxpr cj (acc_args @ elem_args) in
    List.iteri
      (fun i v -> (List.nth accs i).(!of_) <- Ndarray.get_f (as_nd v) [||])
      res
  done;
  List.map2 (fun dt acc -> Ndarray.of_floats dt out_shape acc) op_dts accs

let reduce_precision_bits32 mant_bits bits =
  if mant_bits >= 23 then bits
  else
    let shift = 23 - mant_bits in
    let last = Int32.shift_left 1l shift in
    let base = Int32.sub (Int32.shift_right_logical last 1) 1l in
    let x_last = Int32.logand (Int32.shift_right_logical bits shift) 1l in
    let bias = Int32.add x_last base in
    let rounded = Int32.add bits bias in
    let trunc = Int32.lognot (Int32.sub last 1l) in
    Int32.logand rounded trunc

let reduce_precision_bits64 mant_bits bits =
  if mant_bits >= 52 then bits
  else
    let shift = 52 - mant_bits in
    let last = Int64.shift_left 1L shift in
    let base = Int64.sub (Int64.shift_right_logical last 1) 1L in
    let x_last = Int64.logand (Int64.shift_right_logical bits shift) 1L in
    let bias = Int64.add x_last base in
    let rounded = Int64.add bits bias in
    let trunc = Int64.lognot (Int64.sub last 1L) in
    Int64.logand rounded trunc

let reduce_precision_f dt exponent_bits mantissa_bits x =
  match dt with
  | Dtype.F32 ->
      if exponent_bits < 8 then
        failwith "lax: reduce_precision exponent reduction deferred (M5)"
      else if not (Float.is_finite x) then x
      else
        Int32.float_of_bits
          (reduce_precision_bits32 mantissa_bits (Int32.bits_of_float x))
  | Dtype.F64 ->
      if exponent_bits < 11 then
        failwith "lax: reduce_precision exponent reduction deferred (M5)"
      else if not (Float.is_finite x) then x
      else
        Int64.float_of_bits
          (reduce_precision_bits64 mantissa_bits (Int64.bits_of_float x))
  | _ -> failwith "lax: reduce_precision requires a floating-point operand"

let line_base os str dimension lidx =
  let base = ref 0 and k = ref 0 in
  for d = 0 to Array.length os - 1 do
    if d <> dimension then begin
      base := !base + (lidx.(!k) * str.(d));
      incr k
    end
  done;
  !base

let sort_impl dimension num_keys operands =
  let os = Ndarray.shape (List.hd operands) in
  let str = Utils.strides os in
  let n = os.(dimension) in
  let stride = str.(dimension) in
  let arrs = Array.of_list (List.map to_array operands) in
  let outs = Array.map Array.copy arrs in
  let line_shape = Utils.remove_int os dimension in
  let nlines = Utils.prod line_shape in
  for line = 0 to nlines - 1 do
    let lidx = Utils.decode line line_shape in
    let base = line_base os str dimension lidx in
    let perm = Array.init n (fun i -> i) in
    let cmp i j =
      let rec go ki =
        if ki >= num_keys then 0
        else
          let ka = arrs.(ki) in
          let vi = ka.(base + (i * stride)) and vj = ka.(base + (j * stride)) in
          if vi < vj then -1 else if vi > vj then 1 else go (ki + 1)
      in
      go 0
    in
    Array.stable_sort cmp perm;
    Array.iteri
      (fun opi src ->
        let dst = outs.(opi) in
        Array.iteri
          (fun newpos oldi ->
            dst.(base + (newpos * stride)) <- src.(base + (oldi * stride)))
          perm)
      arrs
  done;
  List.mapi
    (fun i op -> Ndarray.of_floats (Ndarray.dtype op) os outs.(i))
    operands

let top_k_impl k axis operand =
  let os = Ndarray.shape operand in
  let str = Utils.strides os in
  let n = os.(axis) in
  let stride = str.(axis) in
  let a = to_array operand in
  let out_shape = Array.mapi (fun i d -> if i = axis then k else d) os in
  let out_str = Utils.strides out_shape in
  let vals = Array.make (Utils.prod out_shape) 0.0 in
  let idxs = Array.make (Utils.prod out_shape) 0.0 in
  let line_shape = Utils.remove_int os axis in
  let nlines = Utils.prod line_shape in
  for line = 0 to nlines - 1 do
    let lidx = Utils.decode line line_shape in
    let base = line_base os str axis lidx in
    let obase = line_base out_shape out_str axis lidx in
    let perm = Array.init n (fun i -> i) in
    let cmp i j =
      let vi = a.(base + (i * stride)) and vj = a.(base + (j * stride)) in
      if vi > vj then -1 else if vi < vj then 1 else 0
    in
    Array.stable_sort cmp perm;
    for t = 0 to k - 1 do
      let oi = perm.(t) in
      vals.(obase + (t * out_str.(axis))) <- a.(base + (oi * stride));
      idxs.(obase + (t * out_str.(axis))) <- float_of_int oi
    done
  done;
  [
    Ndarray.of_floats (Ndarray.dtype operand) out_shape vals;
    Ndarray.of_floats Dtype.I32 out_shape idxs;
  ]

let scatter_dnums = function
  | Scatter { dimension_numbers; _ }
  | Scatter_add { dimension_numbers }
  | Scatter_sub { dimension_numbers }
  | Scatter_mul { dimension_numbers; _ }
  | Scatter_min { dimension_numbers }
  | Scatter_max { dimension_numbers } ->
      dimension_numbers
  | _ -> failwith "lax: not a scatter primitive"

let impl prim inputs =
  match prim with
  | Neg -> un (fun a -> Ndarray.map (Ndarray.dtype a) (fun x -> -.x) a) inputs
  | Sin -> un (fun a -> Ndarray.map (Ndarray.dtype a) sin a) inputs
  | Cos -> un (fun a -> Ndarray.map (Ndarray.dtype a) cos a) inputs
  | Exp -> un (fun a -> Ndarray.map (Ndarray.dtype a) exp a) inputs
  | Log -> un (fun a -> Ndarray.map (Ndarray.dtype a) log a) inputs
  | Tanh -> un (fun a -> Ndarray.map (Ndarray.dtype a) tanh a) inputs
  | Abs -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.abs a) inputs
  | Sign -> un (fun a -> Ndarray.map (Ndarray.dtype a) sign_f a) inputs
  | Add -> bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) ( +. ) a b) inputs
  | Sub -> bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) ( -. ) a b) inputs
  | Mul -> bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) ( *. ) a b) inputs
  | Div -> bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) ( /. ) a b) inputs
  | Max -> bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) Float.max a b) inputs
  | Min -> bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) Float.min a b) inputs
  | Pow -> bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) Float.pow a b) inputs
  | Eq ->
      bin (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x = y)) a b) inputs
  | Lt ->
      bin (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x < y)) a b) inputs
  | Gt ->
      bin (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x > y)) a b) inputs
  | Select_n -> select_n_impl inputs
  | Convert_element_type dt ->
      un (fun a -> Ndarray.map dt (fun x -> x) a) inputs
  | Broadcast_in_dim { shape; dims } ->
      un (broadcast_in_dim_impl shape dims) inputs
  | Reshape ns -> un (reshape_impl ns) inputs
  | Reduce_sum axes -> un (reduce_sum_impl axes) inputs
  | Dot_general dd -> bin (dot_general_impl dd) inputs
  | Acos -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.acos a) inputs
  | Acosh -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.acosh a) inputs
  | Asin -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.asin a) inputs
  | Asinh -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.asinh a) inputs
  | Atan -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.atan a) inputs
  | Atanh -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.atanh a) inputs
  | Cbrt -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.cbrt a) inputs
  | Ceil -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.ceil a) inputs
  | Clz ->
      un
        (fun a ->
          Ndarray.map (Ndarray.dtype a)
            (clz_bits (clz_nbits (Ndarray.dtype a)))
            a)
        inputs
  | Conj -> un (fun a -> a) inputs
  | Copy -> un (fun a -> a) inputs
  | Cosh -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.cosh a) inputs
  | Exp2 -> un (fun a -> Ndarray.map (Ndarray.dtype a) exp2_f a) inputs
  | Expm1 -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.expm1 a) inputs
  | Floor -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.floor a) inputs
  | Imag -> un (fun a -> Ndarray.map (Ndarray.dtype a) (fun _ -> 0.0) a) inputs
  | Integer_pow y ->
      un (fun a -> Ndarray.map (Ndarray.dtype a) (integer_pow_f y) a) inputs
  | Is_finite ->
      un
        (fun a -> Ndarray.map Bool (fun x -> bool_of (Float.is_finite x)) a)
        inputs
  | Log1p -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.log1p a) inputs
  | Logistic -> un (fun a -> Ndarray.map (Ndarray.dtype a) logistic_f a) inputs
  | Not ->
      un
        (fun a -> Ndarray.map (Ndarray.dtype a) (not_f (Ndarray.dtype a)) a)
        inputs
  | Population_count ->
      un
        (fun a ->
          Ndarray.map (Ndarray.dtype a)
            (popcount_bits (int_nbits (Ndarray.dtype a)))
            a)
        inputs
  | Real -> un (fun a -> a) inputs
  | Round -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.round a) inputs
  | Rsqrt -> un (fun a -> Ndarray.map (Ndarray.dtype a) rsqrt_f a) inputs
  | Sinh -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.sinh a) inputs
  | Sqrt -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.sqrt a) inputs
  | Square -> un (fun a -> Ndarray.map (Ndarray.dtype a) square_f a) inputs
  | Tan -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.tan a) inputs
  | And ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (and_f (Ndarray.dtype a)) a b)
        inputs
  | Atan2 ->
      bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) Float.atan2 a b) inputs
  | Complex -> bin (fun a _ -> a) inputs
  | Eq_to ->
      bin (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x = y)) a b) inputs
  | Ge ->
      bin
        (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x >= y)) a b)
        inputs
  | Le ->
      bin
        (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x <= y)) a b)
        inputs
  | Le_to ->
      bin
        (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x <= y)) a b)
        inputs
  | Lt_to ->
      bin (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x < y)) a b) inputs
  | Mulhi ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (mulhi_f (Ndarray.dtype a)) a b)
        inputs
  | Ne ->
      bin
        (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x <> y)) a b)
        inputs
  | Nextafter ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (nextafter_f (Ndarray.dtype a)) a b)
        inputs
  | Or ->
      bin
        (fun a b -> Ndarray.map2 (Ndarray.dtype a) (or_f (Ndarray.dtype a)) a b)
        inputs
  | Rem ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (rem_f (Ndarray.dtype a)) a b)
        inputs
  | Shift_left ->
      bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) shift_left_f a b) inputs
  | Shift_right_arithmetic ->
      bin
        (fun a b -> Ndarray.map2 (Ndarray.dtype a) shift_right_arithmetic_f a b)
        inputs
  | Shift_right_logical ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a)
            (shift_right_logical_f (Ndarray.dtype a))
            a b)
        inputs
  | Xor ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (xor_f (Ndarray.dtype a)) a b)
        inputs
  | Concatenate dim -> [ concatenate_impl dim inputs ]
  | Pad cfg -> bin (fun a pv -> pad_impl cfg a pv) inputs
  | Rev dims -> un (rev_impl dims) inputs
  | Split { sizes; axis } -> (
      match inputs with
      | [ a ] -> split_impl sizes axis a
      | _ -> failwith "lax: split expects 1 operand")
  | Squeeze dims -> un (squeeze_impl dims) inputs
  | Stack axis -> [ stack_impl axis inputs ]
  | Tile reps -> un (tile_impl reps) inputs
  | Transpose perm -> un (transpose_impl perm) inputs
  | Unstack axis -> (
      match inputs with
      | [ a ] -> unstack_impl axis a
      | _ -> failwith "lax: unstack expects 1 operand")
  | Reduce_max axes -> un (reduce_op_impl Float.max neg_infinity axes) inputs
  | Reduce_min axes -> un (reduce_op_impl Float.min infinity axes) inputs
  | Reduce_prod axes -> un (reduce_op_impl ( *. ) 1.0 axes) inputs
  | Reduce_and axes -> un (reduce_and_impl axes) inputs
  | Reduce_or axes -> un (reduce_or_impl axes) inputs
  | Reduce_xor axes -> un (reduce_xor_impl axes) inputs
  | Argmax { axis; index_dtype } ->
      un (argminmax_impl true axis index_dtype) inputs
  | Argmin { axis; index_dtype } ->
      un (argminmax_impl false axis index_dtype) inputs
  | Reduce { jaxpr; dimensions } -> (
      let num = List.length inputs / 2 in
      match Util.split_list inputs [ num ] with
      | [ operands; inits ] -> reduce_impl jaxpr dimensions operands inits
      | _ -> failwith "lax: reduce expects operands and init values")
  | Clamp -> (
      match inputs with
      | [ mn; x; mx ] -> [ clamp_impl mn x mx ]
      | _ -> failwith "lax: clamp expects 3 operands")
  | Bitcast_convert_type new_dt -> un (bitcast_impl new_dt) inputs
  | Iota { dtype; shape; dimension } -> [ iota_impl dtype shape dimension ]
  | Empty { shape; dtype } -> [ empty_impl dtype shape ]
  | Empty2 dtype -> [ empty_impl dtype [||] ]
  | Composite cj -> composite_impl cj inputs
  | Dce_sink -> []
  | After_all | Create_token ->
      failwith "lax: token primitives are not represented in M1"
  | From_edtype _ ->
      failwith "lax: from_edtype (extended dtypes) deferred to M5"
  | Optimization_barrier -> inputs
  | Reduce_precision { exponent_bits; mantissa_bits } ->
      un
        (fun a ->
          Ndarray.map (Ndarray.dtype a)
            (reduce_precision_f (Ndarray.dtype a) exponent_bits mantissa_bits)
            a)
        inputs
  | Sort { dimension; num_keys; _ } -> sort_impl dimension num_keys inputs
  | Tie -> bin (fun _ b -> b) inputs
  | Top_k { k; axis } -> (
      match inputs with
      | [ a ] -> top_k_impl k axis a
      | _ -> failwith "lax: top_k expects 1 operand")
  | Ragged_dot_general ->
      failwith "lax: ragged_dot_general deferred (needs group_sizes machinery)"
  | Rng_bit_generator | Rng_uniform ->
      failwith "lax: rng primitives deferred to M3 (PRNG)"
  | To_edtype _ -> failwith "lax: to_edtype (extended dtypes) deferred to M5"
  | Slice { start_indices; limit_indices; strides } ->
      un (Slicing.slice_impl start_indices limit_indices strides) inputs
  | Dynamic_slice { slice_sizes } ->
      [ Slicing.dynamic_slice_impl slice_sizes inputs ]
  | Dynamic_update_slice -> [ Slicing.dynamic_update_slice_impl inputs ]
  | Gather { dimension_numbers; slice_sizes } -> (
      match inputs with
      | [ operand; indices ] ->
          [ Slicing.gather_impl dimension_numbers slice_sizes operand indices ]
      | _ -> failwith "lax: gather expects operand and indices")
  | ( Scatter _ | Scatter_add _ | Scatter_sub _ | Scatter_mul _ | Scatter_min _
    | Scatter_max _ ) as p -> (
      match inputs with
      | [ operand; indices; updates ] ->
          [
            Slicing.scatter_impl
              (Slicing.scatter_combiner p)
              (scatter_dnums p) operand indices updates;
          ]
      | _ -> failwith "lax: scatter expects operand, indices and updates")
  | Conv_general_dilated
      {
        window_strides;
        padding;
        lhs_dilation;
        rhs_dilation;
        dimension_numbers;
        feature_group_count;
        batch_group_count;
      } ->
      bin
        (fun lhs rhs ->
          Convolution.conv_impl dimension_numbers window_strides padding
            lhs_dilation rhs_dilation feature_group_count batch_group_count lhs
            rhs)
        inputs
  | Reduce_window_sum window ->
      un (Windowed_reductions.reduce_window_sum window) inputs
  | Reduce_window_max window ->
      un (Windowed_reductions.reduce_window_max window) inputs
  | Reduce_window_min window ->
      un (Windowed_reductions.reduce_window_min window) inputs
  | Reduce_window { reducer; window } -> (
      match inputs with
      | [ operand; init ] ->
          let dt = Ndarray.dtype operand in
          let mk x = Concrete (Ndarray.of_floats dt [||] [| x |]) in
          let reducer_f a b =
            match Jaxpr.eval_closed_jaxpr reducer [ mk a; mk b ] with
            | [ Concrete nd ] -> Ndarray.get_f nd [||]
            | _ ->
                failwith "lax: reduce_window reducer produced unexpected output"
          in
          [
            Windowed_reductions.reduce_window_general ~reducer:reducer_f
              ~init:(Ndarray.get_f init [||]) window operand;
          ]
      | _ -> failwith "lax: reduce_window expects operand and init value")
  | Select_and_gather_add { select; window } -> (
      match inputs with
      | [ tangents; operand ] ->
          [
            Windowed_reductions.select_and_gather_add select window tangents
              operand;
          ]
      | _ -> failwith "lax: select_and_gather_add expects tangents and operand")
  | Select_and_scatter_add { select; window } -> (
      match inputs with
      | [ source; operand ] ->
          [
            Windowed_reductions.select_and_scatter_add select window source
              operand;
          ]
      | _ -> failwith "lax: select_and_scatter_add expects source and operand")
  | Select_and_scatter _ ->
      failwith "lax: select_and_scatter (general two-jaxpr form) deferred (M2)"
  | Bessel_i0e ->
      un
        (fun a ->
          Ndarray.map (Ndarray.dtype a) (Special.bessel_i0e (Ndarray.dtype a)) a)
        inputs
  | Bessel_i1e ->
      un
        (fun a ->
          Ndarray.map (Ndarray.dtype a) (Special.bessel_i1e (Ndarray.dtype a)) a)
        inputs
  | Digamma ->
      un (fun a -> Ndarray.map (Ndarray.dtype a) Special.digamma a) inputs
  | Erf -> un (fun a -> Ndarray.map (Ndarray.dtype a) Special.erf a) inputs
  | Erf_inv ->
      un (fun a -> Ndarray.map (Ndarray.dtype a) Special.erf_inv a) inputs
  | Erfc -> un (fun a -> Ndarray.map (Ndarray.dtype a) Special.erfc a) inputs
  | Lgamma ->
      un (fun a -> Ndarray.map (Ndarray.dtype a) Special.lgamma a) inputs
  | Igamma ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (Special.igamma (Ndarray.dtype a)) a b)
        inputs
  | Igammac ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (Special.igammac (Ndarray.dtype a)) a b)
        inputs
  | Igamma_grad_a ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a)
            (Special.igamma_grad_a (Ndarray.dtype a))
            a b)
        inputs
  | Polygamma ->
      bin
        (fun m x ->
          Ndarray.map2 (Ndarray.dtype x)
            (Special.polygamma (Ndarray.dtype x))
            m x)
        inputs
  | Zeta ->
      bin
        (fun x q ->
          Ndarray.map2 (Ndarray.dtype x) (Special.zeta (Ndarray.dtype x)) x q)
        inputs
  | Regularized_incomplete_beta -> beta_impl inputs
  | Platform_index platforms ->
      [
        Ndarray.of_floats Dtype.I32 [||]
          [| float_of_int (platform_cpu_index platforms) |];
      ]
  | Cond { t; f } -> (
      match inputs with
      | pred :: ops ->
          let branch = if Ndarray.get_f pred [||] <> 0.0 then t else f in
          let outs =
            Jaxpr.eval_closed_jaxpr branch
              (List.map (fun nd -> Concrete nd) ops)
          in
          List.map
            (function
              | Concrete nd -> nd
              | Tracer _ -> failwith "lax: cond branch produced a tracer")
            outs
      | [] -> failwith "lax: cond expects a predicate")
  | Scan { length; reverse; num_carry; jaxpr } ->
      let inputs_v = List.map (fun nd -> Concrete nd) inputs in
      Control_flow.Loops.scan_impl ~length ~reverse ~num_carry jaxpr inputs_v
      |> List.map (function
        | Concrete nd -> nd
        | Tracer _ -> failwith "lax: scan produced a tracer")
  | While { cond; body } ->
      let inputs_v = List.map (fun nd -> Concrete nd) inputs in
      Control_flow.Loops.while_impl cond body inputs_v
      |> List.map (function
        | Concrete nd -> nd
        | Tracer _ -> failwith "lax: while produced a tracer")
  | Cumsum { axis; reverse } ->
      un (Control_flow.Loops.cumred_impl ~axis ~reverse ~combine:( +. )) inputs
  | Cumprod { axis; reverse } ->
      un (Control_flow.Loops.cumred_impl ~axis ~reverse ~combine:( *. )) inputs
  | Cummax { axis; reverse } ->
      un
        (Control_flow.Loops.cumred_impl ~axis ~reverse ~combine:Float.max)
        inputs
  | Cummin { axis; reverse } ->
      un
        (Control_flow.Loops.cumred_impl ~axis ~reverse ~combine:Float.min)
        inputs
  | Cumlogsumexp { axis; reverse } ->
      un
        (Control_flow.Loops.cumred_impl ~axis ~reverse
           ~combine:Control_flow.Loops.logaddexp)
        inputs
  | Custom_linear_solve { solve; _ } ->
      let inputs_v = List.map (fun nd -> Concrete nd) inputs in
      Control_flow.Solves.solve_impl solve inputs_v
      |> List.map (function
        | Concrete nd -> nd
        | Tracer _ -> failwith "lax: custom_linear_solve produced a tracer")
  | Xla_call _ -> failwith "lax: xla_call handled by interpreters"

let shaped shape dtype weak_type = { shape; dtype; weak_type }

let aval_of_atom = function
  | A_var v -> v.vaval
  | A_lit nd ->
      { shape = Ndarray.shape nd; dtype = Ndarray.dtype nd; weak_type = false }
  | DropVar a -> a

let un_aval f = function
  | [ a ] -> [ f a ]
  | _ -> failwith "lax: expected 1 aval"

let bin_aval f = function
  | [ a; b ] -> [ f a b ]
  | _ -> failwith "lax: expected 2 avals"

let abstract_eval prim avals =
  match prim with
  | Neg | Sin | Cos | Exp | Log | Tanh | Abs | Sign | Acos | Acosh | Asin
  | Asinh | Atan | Atanh | Cbrt | Ceil | Clz | Conj | Copy | Cosh | Exp2 | Expm1
  | Floor | Imag | Integer_pow _ | Log1p | Logistic | Not | Population_count
  | Real | Round | Rsqrt | Sinh | Sqrt | Square | Tan | Bessel_i0e | Bessel_i1e
  | Digamma | Erf | Erf_inv | Erfc | Lgamma ->
      un_aval (fun a -> a) avals
  | Is_finite -> un_aval (fun a -> shaped a.shape Bool false) avals
  | Add | Sub | Mul | Div | Max | Min | Pow | And | Atan2 | Complex | Mulhi
  | Nextafter | Or | Rem | Shift_left | Shift_right_arithmetic
  | Shift_right_logical | Xor ->
      bin_aval
        (fun a b -> shaped a.shape a.dtype (a.weak_type && b.weak_type))
        avals
  | Eq | Lt | Gt | Ge | Le | Eq_to | Le_to | Lt_to | Ne ->
      bin_aval (fun a _ -> shaped a.shape Bool false) avals
  | Select_n -> (
      match avals with
      | _ :: (c :: _ as cases) ->
          [ shaped c.shape c.dtype (Utils.all_weak cases) ]
      | _ -> failwith "lax: select_n expects a predicate and cases")
  | Convert_element_type dt -> un_aval (fun a -> shaped a.shape dt false) avals
  | Broadcast_in_dim { shape; _ } ->
      un_aval (fun a -> shaped shape a.dtype a.weak_type) avals
  | Reshape ns -> un_aval (fun a -> shaped ns a.dtype a.weak_type) avals
  | Reduce_sum axes ->
      un_aval
        (fun a -> shaped (Utils.reduce_shape a.shape axes) a.dtype a.weak_type)
        avals
  | Dot_general dd ->
      bin_aval
        (fun l r ->
          shaped
            (Utils.dot_general_shape dd l.shape r.shape)
            l.dtype (Utils.all_weak avals))
        avals
  | Concatenate dim -> (
      match avals with
      | [] -> failwith "lax: concatenate expects at least one operand"
      | first :: _ ->
          [
            shaped
              (Utils.concatenate_shape dim (List.map (fun a -> a.shape) avals))
              first.dtype (Utils.all_weak avals);
          ])
  | Pad cfg ->
      bin_aval
        (fun a pv ->
          shaped
            (Utils.pad_shape cfg a.shape)
            a.dtype
            (a.weak_type && pv.weak_type))
        avals
  | Rev _ -> un_aval (fun a -> a) avals
  | Split { sizes; axis } -> (
      match avals with
      | [ a ] ->
          List.map
            (fun s -> shaped s a.dtype a.weak_type)
            (Utils.split_shapes sizes axis a.shape)
      | _ -> failwith "lax: split expects 1 operand")
  | Squeeze dims ->
      un_aval
        (fun a -> shaped (Utils.squeeze_shape dims a.shape) a.dtype a.weak_type)
        avals
  | Stack axis -> (
      match avals with
      | [] -> failwith "lax: stack expects at least one operand"
      | first :: _ ->
          [
            shaped
              (Utils.stack_shape axis (List.length avals) first.shape)
              first.dtype (Utils.all_weak avals);
          ])
  | Tile reps ->
      un_aval
        (fun a -> shaped (Utils.tile_shape reps a.shape) a.dtype a.weak_type)
        avals
  | Transpose perm ->
      un_aval
        (fun a ->
          shaped (Utils.transpose_shape perm a.shape) a.dtype a.weak_type)
        avals
  | Unstack axis -> (
      match avals with
      | [ a ] ->
          List.map
            (fun s -> shaped s a.dtype a.weak_type)
            (Utils.unstack_shapes axis a.shape)
      | _ -> failwith "lax: unstack expects 1 operand")
  | Reduce_max axes | Reduce_min axes | Reduce_prod axes ->
      un_aval
        (fun a -> shaped (Utils.reduce_shape a.shape axes) a.dtype a.weak_type)
        avals
  | Reduce_and axes | Reduce_or axes | Reduce_xor axes ->
      un_aval
        (fun a -> shaped (Utils.reduce_shape a.shape axes) a.dtype false)
        avals
  | Argmax { axis; index_dtype } | Argmin { axis; index_dtype } ->
      un_aval
        (fun a -> shaped (Utils.remove_int a.shape axis) index_dtype false)
        avals
  | Reduce { dimensions; _ } -> (
      let num = List.length avals / 2 in
      match Util.split_list avals [ num ] with
      | [ operands; inits ] ->
          List.map2
            (fun (op : aval) (init : aval) ->
              shaped
                (Utils.reduce_shape op.shape dimensions)
                op.dtype
                (op.weak_type && init.weak_type))
            operands inits
      | _ -> failwith "lax: reduce expects operands and init values")
  | Clamp -> (
      match avals with
      | [ _; x; _ ] -> [ shaped x.shape x.dtype (Utils.all_weak avals) ]
      | _ -> failwith "lax: clamp expects 3 avals")
  | Bitcast_convert_type new_dt ->
      un_aval (fun a -> shaped a.shape new_dt false) avals
  | Iota { dtype; shape; _ } -> [ shaped shape dtype false ]
  | Empty { shape; dtype } -> [ shaped shape dtype false ]
  | Empty2 dtype -> [ shaped [||] dtype false ]
  | Composite cj -> List.map aval_of_atom cj.jaxpr.outs
  | Dce_sink -> []
  | After_all | Create_token ->
      failwith "lax: token primitives have no shaped aval in M1"
  | From_edtype _ ->
      failwith "lax: from_edtype (extended dtypes) deferred to M5"
  | Optimization_barrier -> avals
  | Reduce_precision _ -> un_aval (fun a -> a) avals
  | Sort _ -> avals
  | Tie -> bin_aval (fun _ b -> b) avals
  | Top_k { k; axis } -> (
      match avals with
      | [ a ] ->
          let s = Array.mapi (fun i d -> if i = axis then k else d) a.shape in
          [ shaped s a.dtype a.weak_type; shaped s Dtype.I32 a.weak_type ]
      | _ -> failwith "lax: top_k expects 1 operand")
  | Ragged_dot_general ->
      failwith "lax: ragged_dot_general deferred (needs group_sizes machinery)"
  | Rng_bit_generator | Rng_uniform ->
      failwith "lax: rng primitives deferred to M3 (PRNG)"
  | To_edtype _ -> failwith "lax: to_edtype (extended dtypes) deferred to M5"
  | Slice { start_indices; limit_indices; strides } ->
      un_aval
        (fun a ->
          shaped
            (Slicing.slice_shape start_indices limit_indices strides a.shape)
            a.dtype a.weak_type)
        avals
  | Dynamic_slice { slice_sizes } -> (
      match avals with
      | operand :: _ -> [ shaped slice_sizes operand.dtype operand.weak_type ]
      | [] -> failwith "lax: dynamic_slice expects an operand")
  | Dynamic_update_slice -> (
      match avals with
      | operand :: _ -> [ operand ]
      | [] -> failwith "lax: dynamic_update_slice expects an operand")
  | Gather { dimension_numbers; slice_sizes } -> (
      match avals with
      | [ operand; indices ] ->
          [
            shaped
              (Slicing.gather_shape dimension_numbers slice_sizes indices.shape
                 operand.shape)
              operand.dtype operand.weak_type;
          ]
      | _ -> failwith "lax: gather expects operand and indices avals")
  | Scatter _ | Scatter_add _ | Scatter_sub _ | Scatter_mul _ | Scatter_min _
  | Scatter_max _ -> (
      match avals with
      | operand :: _ -> [ operand ]
      | [] -> failwith "lax: scatter expects an operand aval")
  | Conv_general_dilated
      {
        window_strides;
        padding;
        lhs_dilation;
        rhs_dilation;
        dimension_numbers;
        feature_group_count;
        batch_group_count;
      } ->
      bin_aval
        (fun l r ->
          shaped
            (Convolution.conv_shape dimension_numbers window_strides padding
               lhs_dilation rhs_dilation feature_group_count batch_group_count
               l.shape r.shape)
            l.dtype (Utils.all_weak avals))
        avals
  | Reduce_window_sum window
  | Reduce_window_max window
  | Reduce_window_min window ->
      un_aval
        (fun a ->
          shaped
            (Windowed_reductions.out_shape a.shape window)
            a.dtype a.weak_type)
        avals
  | Reduce_window { window; _ } -> (
      match avals with
      | [ operand; init ] ->
          [
            shaped
              (Windowed_reductions.out_shape operand.shape window)
              operand.dtype
              (operand.weak_type && init.weak_type);
          ]
      | _ -> failwith "lax: reduce_window expects operand and init aval")
  | Select_and_gather_add { window; _ } -> (
      match avals with
      | [ tangents; operand ] ->
          [
            shaped
              (Windowed_reductions.out_shape operand.shape window)
              operand.dtype tangents.weak_type;
          ]
      | _ -> failwith "lax: select_and_gather_add expects two avals")
  | Select_and_scatter_add _ -> (
      match avals with
      | [ _source; operand ] -> [ operand ]
      | _ -> failwith "lax: select_and_scatter_add expects two avals")
  | Select_and_scatter _ -> (
      match avals with
      | operand :: _ -> [ operand ]
      | [] -> failwith "lax: select_and_scatter expects an operand aval")
  | Igamma | Igamma_grad_a | Igammac | Polygamma | Zeta ->
      bin_aval
        (fun a b -> shaped a.shape a.dtype (a.weak_type && b.weak_type))
        avals
  | Regularized_incomplete_beta -> (
      match avals with
      | [ a; b; x ] ->
          [ shaped x.shape x.dtype (a.weak_type && b.weak_type && x.weak_type) ]
      | _ -> failwith "lax: regularized_incomplete_beta expects 3 avals")
  | Platform_index _ -> [ shaped [||] Dtype.I32 false ]
  | Cond { t; _ } -> List.map aval_of_atom t.jaxpr.outs
  | Scan { length; num_carry; jaxpr; _ } ->
      Control_flow.Loops.scan_out_avals ~length ~num_carry jaxpr
  | While { body; _ } -> Control_flow.Loops.while_out_avals body
  | Cumsum { axis; _ }
  | Cumprod { axis; _ }
  | Cummax { axis; _ }
  | Cummin { axis; _ }
  | Cumlogsumexp { axis; _ } ->
      un_aval
        (fun a ->
          if axis < 0 || axis >= Array.length a.shape then
            failwith "lax: cumulative axis out of bounds";
          a)
        avals
  | Custom_linear_solve { solve; _ } ->
      Control_flow.Solves.solve_out_avals solve
  | Xla_call _ -> failwith "lax: xla_call handled by interpreters"

let install () =
  Core.rules.impl <- impl;
  Core.rules.abstract_eval <- abstract_eval

let () = install ()
let cond = Control_flow.Conditionals.cond
let platform_index = Control_flow.Conditionals.platform_index
let scan = Control_flow.Loops.scan
let while_loop = Control_flow.Loops.while_loop
let cumsum = Control_flow.Loops.cumsum
let cumprod = Control_flow.Loops.cumprod
let cummax = Control_flow.Loops.cummax
let cummin = Control_flow.Loops.cummin
let cumlogsumexp = Control_flow.Loops.cumlogsumexp
let custom_linear_solve = Control_flow.Solves.custom_linear_solve
