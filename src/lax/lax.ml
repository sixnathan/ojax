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
  | Dtype.I2 | Dtype.Uint2 -> 2
  | Dtype.I4 | Dtype.Uint4 -> 4
  | Dtype.I8 | Dtype.Uint8 -> 8
  | Dtype.I16 | Dtype.Uint16 -> 16
  | Dtype.I32 | Dtype.Uint32 -> 32
  | Dtype.I64 | Dtype.Uint64 -> 64
  | _ -> failwith "lax: clz requires an integer operand"

let low_bits nbits m =
  if nbits >= 64 then m
  else Int64.logand m (Int64.sub (Int64.shift_left 1L nbits) 1L)

let sign_extended nbits m =
  let unused = 64 - nbits in
  Int64.shift_right (Int64.shift_left m unused) unused

let clz_bits nbits x =
  let m = low_bits nbits (Int64.of_float x) in
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
  | Dtype.I2 | Dtype.Uint2 -> 2
  | Dtype.Uint4 -> 4
  | Dtype.I4 -> 8
  | Dtype.I8 | Dtype.Uint8 -> 8
  | Dtype.I16 | Dtype.Uint16 -> 16
  | Dtype.I32 | Dtype.Uint32 -> 32
  | Dtype.I64 | Dtype.Uint64 -> 64
  | _ -> failwith "lax: integer operand required"

let popcount_bits nbits x =
  let m = low_bits nbits (Int64.of_float x) in
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
  | Dtype.I8 | Dtype.I16 | Dtype.I2 | Dtype.I4 | Dtype.I32 | Dtype.I64
  | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4 | Dtype.Uint32 ->
      Int64.to_float (Int64.lognot (Int64.of_float x))
  | _ -> failwith "lax: not requires a boolean or integer operand"

let uint32_mask_i64 = 0xFFFFFFFFL
let uint32_wrap v = Int64.to_float (Int64.logand v uint32_mask_i64)

let add_f dt x y =
  match dt with
  | Dtype.Uint32 ->
      uint32_wrap (Int64.add (Int64.of_float x) (Int64.of_float y))
  | _ -> x +. y

let sub_f dt x y =
  match dt with
  | Dtype.Uint32 ->
      uint32_wrap (Int64.sub (Int64.of_float x) (Int64.of_float y))
  | _ -> x -. y

let mul_f dt x y =
  match dt with
  | Dtype.Uint32 ->
      uint32_wrap (Int64.mul (Int64.of_float x) (Int64.of_float y))
  | Dtype.I8 | Dtype.I16 | Dtype.I32 | Dtype.Uint8 | Dtype.Uint16 ->
      let nbits = int_nbits dt in
      let low =
        low_bits nbits (Int64.mul (Int64.of_float x) (Int64.of_float y))
      in
      Int64.to_float
        (match dt with
        | Dtype.Uint8 | Dtype.Uint16 -> low
        | _ -> sign_extended nbits low)
  | _ -> x *. y

let and_f dt x y =
  match dt with
  | Dtype.Bool -> if x <> 0.0 && y <> 0.0 then 1.0 else 0.0
  | Dtype.I8 | Dtype.I16 | Dtype.I2 | Dtype.I4 | Dtype.I32 | Dtype.I64
  | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4 | Dtype.Uint32 ->
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
  | Dtype.I8 | Dtype.I16 | Dtype.I2 | Dtype.I4 | Dtype.I32 | Dtype.I64
  | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4 | Dtype.Uint32 ->
      Int64.to_float (Int64.logor (Int64.of_float x) (Int64.of_float y))
  | _ -> failwith "lax: or requires a boolean or integer operand"

let xor_f dt x y =
  match dt with
  | Dtype.Bool -> if x <> 0.0 <> (y <> 0.0) then 1.0 else 0.0
  | Dtype.I8 | Dtype.I16 | Dtype.I2 | Dtype.I4 | Dtype.I32 | Dtype.I64
  | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4 | Dtype.Uint32 ->
      Int64.to_float (Int64.logxor (Int64.of_float x) (Int64.of_float y))
  | _ -> failwith "lax: xor requires a boolean or integer operand"

let rem_f dt x y =
  match dt with
  | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn
  | Dtype.F8_e3m4 | Dtype.F8_e4m3 | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn
  | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2 | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu
  | Dtype.F32 | Dtype.F64 ->
      Float.rem x y
  | Dtype.I8 | Dtype.I16 | Dtype.I2 | Dtype.I4 | Dtype.I32 | Dtype.I64
  | Dtype.Uint8 | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4 | Dtype.Uint32
  | Dtype.Uint64 ->
      if y = 0.0 then x
      else Int64.to_float (Int64.rem (Int64.of_float x) (Int64.of_float y))
  | _ -> failwith "lax: rem requires an integer or float operand"

let shift_amount nbits y =
  let s = Int64.of_float y in
  if Int64.compare s 0L < 0 || Int64.compare s (Int64.of_int nbits) >= 0 then
    None
  else Some (Int64.to_int s)

let shift_result dt nbits m =
  let low = low_bits nbits m in
  Int64.to_float
    (match dt with
    | Dtype.I2 | Dtype.I4 | Dtype.I8 | Dtype.I16 | Dtype.I32 | Dtype.I64 ->
        sign_extended nbits low
    | _ -> low)

let shift_left_f dt x y =
  let nbits = clz_nbits dt in
  match shift_amount nbits y with
  | None -> 0.0
  | Some s -> shift_result dt nbits (Int64.shift_left (Int64.of_float x) s)

let shift_right_arithmetic_f dt x y =
  let nbits = clz_nbits dt in
  let signed = sign_extended nbits (low_bits nbits (Int64.of_float x)) in
  let s = match shift_amount nbits y with None -> nbits - 1 | Some s -> s in
  shift_result dt nbits (Int64.shift_right signed s)

let shift_right_logical_f dt x y =
  let nbits = clz_nbits dt in
  match shift_amount nbits y with
  | None -> 0.0
  | Some s ->
      shift_result dt nbits
        (Int64.shift_right_logical (low_bits nbits (Int64.of_float x)) s)

let f64_nextafter x y =
  if x <> x || y <> y then Float_format.nan
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
  if x <> x || y <> y then Float_format.nan
  else if x = y then y
  else if y > x then f32_next_up x
  else f32_next_down x

let narrow_nextafter ~precision ~min_exp x y =
  if x <> x || y <> y then Float_format.nan
  else if x = y then y
  else
    let smallest = Float.ldexp 1.0 (min_exp - precision + 1) in
    if x = 0.0 then Float.copy_sign smallest y
    else if Float.abs x = Float.infinity then
      Float.copy_sign (Float.ldexp 1.0 (-min_exp + 2) -. 0.0) x
      *. (1.0 -. Float.ldexp 1.0 (-precision))
    else
      let _, e = Float.frexp x in
      let exponent = e - 1 in
      let ulp = Float.ldexp 1.0 (Stdlib.max exponent min_exp - precision + 1) in
      let away = y > x = (x > 0.0) in
      let at_power =
        Float.abs x = Float.ldexp 1.0 exponent && exponent > min_exp
      in
      let step = if (not away) && at_power then ulp /. 2.0 else ulp in
      if y > x then x +. step else x -. step

let nextafter_f dt x y =
  match dt with
  | Dtype.F32 -> f32_nextafter x y
  | Dtype.F64 -> f64_nextafter x y
  | Dtype.F16 -> narrow_nextafter ~precision:11 ~min_exp:(-14) x y
  | Dtype.Bf16 -> narrow_nextafter ~precision:8 ~min_exp:(-126) x y
  | d when Option.is_some (Float_format.of_dtype d) ->
      Float_format.next_after (Option.get (Float_format.of_dtype d)) x y
  | _ -> failwith "lax: nextafter requires a float operand"

let integer_pow_f y x = Float.pow x (float_of_int y)
let logistic_f x = 1.0 /. (1.0 +. Float.exp (-.x))
let rsqrt_f x = 1.0 /. Float.sqrt x

let round_nearest_even_f x =
  let r = Float.round x in
  if Float.abs (x -. r) = 0.5 then 2.0 *. Float.round (x /. 2.0) else r

let square_f x = x *. x
let un f = function [ a ] -> [ f a ] | _ -> failwith "lax: expected 1 operand"

let bin f = function
  | [ a; b ] -> [ f a b ]
  | _ -> failwith "lax: expected 2 operands"

let is_complex_dtype = function
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let complex_basetype = function
  | Dtype.Complex64 -> Dtype.F32
  | Dtype.Complex128 -> Dtype.F64
  | d -> d

let complexify = function
  | Dtype.F32 -> Dtype.Complex64
  | Dtype.F64 -> Dtype.Complex128
  | d -> d

let to_carray nd =
  let sh = Ndarray.shape nd in
  let n = Utils.prod sh in
  Array.init n (fun i -> Ndarray.get_c nd (Utils.decode i sh))

let map_c dt f nd =
  Ndarray.of_complex dt (Ndarray.shape nd) (Array.map f (to_carray nd))

let map_c2r dt f nd =
  Ndarray.of_floats dt (Ndarray.shape nd) (Array.map f (to_carray nd))

let map_r2c dt f nd =
  Ndarray.of_complex dt (Ndarray.shape nd) (Array.map f (to_array nd))

let map2_c dt f a b =
  let ca = to_carray a and cb = to_carray b in
  Ndarray.of_complex dt (Ndarray.shape a)
    (Array.init (Array.length ca) (fun i -> f ca.(i) cb.(i)))

let csin (z : Complex.t) =
  {
    Complex.re = Stdlib.sin z.re *. Stdlib.cosh z.im;
    im = Stdlib.cos z.re *. Stdlib.sinh z.im;
  }

let ccos (z : Complex.t) =
  {
    Complex.re = Stdlib.cos z.re *. Stdlib.cosh z.im;
    im = -.(Stdlib.sin z.re *. Stdlib.sinh z.im);
  }

let complex_of_re_im a b =
  let ra = to_array a and rb = to_array b in
  Array.init (Array.length ra) (fun i -> { Complex.re = ra.(i); im = rb.(i) })

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

let small_step dt =
  match Float_format.of_dtype dt with
  | Some f -> Float_format.round f
  | None -> Fun.id

let reduce_sum_impl axes a =
  let step = small_step (Ndarray.dtype a) in
  let os = Ndarray.shape a in
  let out_shape = Utils.reduce_shape os axes in
  let out_str = Utils.strides out_shape in
  let ndim = Array.length os in
  let is_red = Array.make ndim false in
  Array.iter (fun ax -> is_red.(ax) <- true) axes;
  let src = to_array a in
  let out = Array.make (Utils.prod out_shape) (step 0.0) in
  for flat = 0 to Array.length src - 1 do
    let oidx = Utils.decode flat os in
    let of_ = ref 0 and k = ref 0 in
    for d = 0 to ndim - 1 do
      if not is_red.(d) then begin
        of_ := !of_ + (oidx.(d) * out_str.(!k));
        incr k
      end
    done;
    out.(!of_) <- step (out.(!of_) +. src.(flat))
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

let dot_general_complex_impl (dd : dot_dims) lhs rhs =
  let ls = Ndarray.shape lhs and rs = Ndarray.shape rhs in
  let la = to_carray lhs and ra = to_carray rhs in
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
  let out = Array.make out_n Complex.zero in
  let nb = Array.length dd.lhs_batch in
  let nlf = Array.length lhs_free in
  let ncon = Array.length dd.lhs_contract in
  let cprod = Utils.prod contract_sizes in
  for f = 0 to out_n - 1 do
    let oidx = Utils.decode f out_shape in
    let acc = ref Complex.zero in
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
      acc := Complex.add !acc (Complex.mul la.(!lflat) ra.(!rflat))
    done;
    out.(f) <- !acc
  done;
  Ndarray.of_complex (Ndarray.dtype lhs) out_shape out

let reduce_sum_complex_impl axes a =
  let os = Ndarray.shape a in
  let out_shape = Utils.reduce_shape os axes in
  let out_str = Utils.strides out_shape in
  let ndim = Array.length os in
  let is_red = Array.make ndim false in
  Array.iter (fun ax -> is_red.(ax) <- true) axes;
  let src = to_carray a in
  let out = Array.make (Utils.prod out_shape) Complex.zero in
  for flat = 0 to Array.length src - 1 do
    let oidx = Utils.decode flat os in
    let of_ = ref 0 and k = ref 0 in
    for d = 0 to ndim - 1 do
      if not is_red.(d) then begin
        of_ := !of_ + (oidx.(d) * out_str.(!k));
        incr k
      end
    done;
    out.(!of_) <- Complex.add out.(!of_) src.(flat)
  done;
  Ndarray.of_complex (Ndarray.dtype a) out_shape out

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
  let step = small_step (Ndarray.dtype a) in
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
    out.(!of_) <- step (combine out.(!of_) src.(flat))
  done;
  Ndarray.of_floats (Ndarray.dtype a) out_shape out

let small_identity pick a =
  let dt = Ndarray.dtype a in
  match Float_format.of_dtype dt with
  | Some f -> Float_format.round f (pick dt)
  | None -> if pick Dtype.F64 > 0.0 then infinity else neg_infinity

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
      let pick =
        Ndarray.dtype operand <> Dtype.F8_e8m0fnu
        && ((if is_max then v > !best else v < !best) || v <> v)
      in
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
  let at v i = if Array.length v = 1 then v.(0) else v.(i) in
  let out =
    Array.init (Array.length b) (fun i ->
        Float.max (Float.min (at a i) (at c i)) (Float.min b.(i) (at c i)))
  in
  Ndarray.of_floats (Ndarray.dtype x) (Ndarray.shape x) out

let beta_impl = function
  | [ av; bv; xv ] ->
      let dt = Ndarray.dtype av in
      let a = to_array av and b = to_array bv and x = to_array xv in
      let at v i = if Array.length v = 1 then v.(0) else v.(i) in
      let n =
        Stdlib.max (Array.length a)
          (Stdlib.max (Array.length b) (Array.length x))
      in
      let out =
        Array.init n (fun i ->
            Special.regularized_incomplete_beta dt (at a i) (at b i) (at x i))
      in
      let sh =
        List.fold_left
          (fun acc v ->
            let s = Ndarray.shape v in
            if Array.length s > Array.length acc then s else acc)
          [||] [ av; bv; xv ]
      in
      [ Ndarray.of_floats dt sh out ]
  | _ -> failwith "lax: regularized_incomplete_beta expects 3 operands"

let small_bit_width = function
  | Dtype.I2 | Dtype.Uint2 -> Some 2
  | Dtype.I4 | Dtype.Uint4 | Dtype.F4_e2m1fn -> Some 4
  | Dtype.I8 | Dtype.Uint8 | Dtype.F8_e3m4 | Dtype.F8_e4m3
  | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2
  | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu ->
      Some 8
  | _ -> None

let small_to_bits dt v =
  match Float_format.of_dtype dt with
  | Some f -> Float_format.encode f v
  | None -> int_of_float v land ((1 lsl Option.get (small_bit_width dt)) - 1)

let small_of_bits dt bits =
  match Float_format.of_dtype dt with
  | Some f -> Float_format.decode f bits
  | None -> float_of_int bits

let bitcast_impl new_dt operand =
  let shape = Ndarray.shape operand in
  let n = Utils.prod shape in
  let word i = Ndarray.get_i64 operand (Utils.decode i shape) in
  let value i = Ndarray.get_f operand (Utils.decode i shape) in
  let as_wide =
    match Ndarray.dtype operand with
    | Dtype.F64 -> fun i -> Int64.bits_of_float (value i)
    | Dtype.I64 | Dtype.Uint64 -> word
    | _ -> fun _ -> 0L
  in
  let as_narrow =
    match Ndarray.dtype operand with
    | Dtype.F32 -> fun i -> Ndarray.f32_bits_of_float (value i)
    | Dtype.I32 | Dtype.Uint32 -> fun i -> Int64.to_int32 (word i)
    | _ -> fun _ -> 0l
  in
  match (Ndarray.dtype operand, new_dt) with
  | (Dtype.F64 | Dtype.I64 | Dtype.Uint64), Dtype.F64 ->
      Ndarray.of_floats new_dt shape
        (Array.init n (fun i -> Int64.float_of_bits (as_wide i)))
  | (Dtype.F64 | Dtype.I64 | Dtype.Uint64), (Dtype.I64 | Dtype.Uint64) ->
      Ndarray.of_int64s new_dt shape (Array.init n as_wide)
  | (Dtype.F32 | Dtype.I32 | Dtype.Uint32), Dtype.F32 ->
      Ndarray.of_floats new_dt shape
        (Array.init n (fun i -> Ndarray.float_of_f32_bits (as_narrow i)))
  | (Dtype.F32 | Dtype.I32 | Dtype.Uint32), Dtype.I32 ->
      Ndarray.of_floats new_dt shape
        (Array.init n (fun i -> Int32.to_float (as_narrow i)))
  | (Dtype.F32 | Dtype.I32 | Dtype.Uint32), Dtype.Uint32 ->
      Ndarray.of_floats new_dt shape
        (Array.init n (fun i ->
             Int64.to_float
               (Int64.logand (Int64.of_int32 (as_narrow i)) 0xFFFFFFFFL)))
  | a, b when a = b -> operand
  | a, b when small_bit_width a <> None && small_bit_width a = small_bit_width b
    ->
      let raw = Array.init n (fun i -> small_to_bits a (value i)) in
      Ndarray.of_floats new_dt shape (Array.map (small_of_bits b) raw)
  | ( (Dtype.F16 | Dtype.Bf16 | Dtype.I16 | Dtype.Uint16),
      (Dtype.F16 | Dtype.Bf16 | Dtype.I16 | Dtype.Uint16) ) ->
      let bits i =
        let v = value i in
        match Ndarray.dtype operand with
        | Dtype.F16 -> Ndarray.bits_of_half v
        | Dtype.Bf16 -> Ndarray.bits_of_bfloat v
        | _ -> int_of_float v land 0xFFFF
      in
      Ndarray.of_floats new_dt shape
        (Array.init n (fun i ->
             let b = bits i in
             match new_dt with
             | Dtype.F16 -> Ndarray.half_of_bits b
             | Dtype.Bf16 -> Ndarray.bfloat_of_bits b
             | Dtype.I16 ->
                 float_of_int (if b >= 0x8000 then b - 0x10000 else b)
             | _ -> float_of_int b))
  | _ ->
      invalid_arg
        "lax.bitcast_convert_type: operand and target must have the same width"

let iota_impl dtype shape dimension =
  let n = Utils.prod shape in
  let out =
    Array.init n (fun f -> float_of_int (Utils.decode f shape).(dimension))
  in
  Ndarray.of_floats dtype shape out

let empty_impl dtype shape =
  Ndarray.of_floats dtype shape (Array.make (Utils.prod shape) 0.0)

let platform_branch_index (platforms : string array option array) : int =
  let current = Config.current_platform () in
  let n = Array.length platforms in
  let rec find_named i =
    if i >= n then find_default 0
    else
      match platforms.(i) with
      | Some names when Array.exists (fun p -> p = current) names -> i
      | _ -> find_named (i + 1)
  and find_default i =
    if i >= n then
      failwith
        (Printf.sprintf "lax: platform_index has no %s or default branch"
           current)
    else match platforms.(i) with None -> i | Some _ -> find_default (i + 1)
  in
  find_named 0

let composite_impl (cj : closed_jaxpr) inputs =
  let outs =
    Jaxpr.eval_closed_jaxpr cj (List.map (fun nd -> Concrete nd) inputs)
  in
  List.map
    (function
      | Concrete nd -> nd
      | Tracer _ -> failwith "lax: composite reducer produced a tracer"
      | Device _ -> failwith "lax: composite reducer produced a tracer")
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
    | Device _ -> failwith "lax: reduce reducer produced a tracer"
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

let limit_exponent exponent_bits x =
  if (not (Float.is_finite x)) || x = 0.0 then x
  else
    let _, e = Float.frexp x in
    let exponent = e - 1 in
    let bias = (1 lsl (exponent_bits - 1)) - 1 in
    if exponent > bias then Float.copy_sign Float.infinity x
    else if exponent < 1 - bias then Float.copy_sign 0.0 x
    else x

let rec reduce_precision_f dt exponent_bits mantissa_bits x =
  match dt with
  | Dtype.F32 ->
      if exponent_bits < 8 then
        limit_exponent exponent_bits
          (Int32.float_of_bits
             (reduce_precision_bits32 mantissa_bits (Int32.bits_of_float x)))
      else if not (Float.is_finite x) then x
      else
        Int32.float_of_bits
          (reduce_precision_bits32 mantissa_bits (Int32.bits_of_float x))
  | Dtype.F64 ->
      if exponent_bits < 11 then
        limit_exponent exponent_bits
          (Int64.float_of_bits
             (reduce_precision_bits64 mantissa_bits (Int64.bits_of_float x)))
      else if not (Float.is_finite x) then x
      else
        Int64.float_of_bits
          (reduce_precision_bits64 mantissa_bits (Int64.bits_of_float x))
  | d when Option.is_some (Float_format.of_dtype d) ->
      Float_format.round
        (Option.get (Float_format.of_dtype d))
        (reduce_precision_f Dtype.F64 exponent_bits mantissa_bits x)
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
          if Float.is_nan vi then if Float.is_nan vj then go (ki + 1) else 1
          else if Float.is_nan vj then -1
          else if vi < vj then -1
          else if vi > vj then 1
          else go (ki + 1)
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

let custom_output = function
  | Concrete nd -> nd
  | Device b -> Pjrt.Buffer.to_host b
  | Tracer _ ->
      failwith
        "custom derivative rule closed over a tracer; pass it as an argument"

let is_float_source = function
  | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn | Dtype.F6_e2m3fn | Dtype.F6_e3m2fn
  | Dtype.F8_e3m4 | Dtype.F8_e4m3 | Dtype.F8_e4m3b11fnuz | Dtype.F8_e4m3fn
  | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2 | Dtype.F8_e5m2fnuz | Dtype.F8_e8m0fnu
  | Dtype.F32 | Dtype.F64 ->
      true
  | _ -> false

let to_wide_int dt v =
  if Float.is_nan v then 0L
  else
    match dt with
    | Dtype.I64 ->
        if v >= 9223372036854775808.0 then Int64.max_int
        else if v <= -9223372036854775808.0 then Int64.min_int
        else Int64.of_float v
    | _ ->
        if v <= 0.0 then 0L
        else if v >= 18446744073709551616.0 then -1L
        else if v >= 9223372036854775808.0 then
          Int64.of_float (v -. 18446744073709551616.0)
        else Int64.of_float v

let is_int_dtype = function
  | Dtype.I8 | Dtype.I16 | Dtype.I32 | Dtype.I64 | Dtype.Uint8 | Dtype.Uint16
  | Dtype.Uint32 | Dtype.Uint64 ->
      true
  | _ -> false

let wrap_int_to dt m =
  let nbits = int_nbits dt in
  let low = low_bits nbits m in
  match dt with
  | Dtype.I8 | Dtype.I16 | Dtype.I32 | Dtype.I64 -> sign_extended nbits low
  | _ -> low

let saturate_to dt v =
  let clamp lo hi =
    if Float.is_nan v then 0.0 else Float.min hi (Float.max lo (Float.trunc v))
  in
  match dt with
  | Dtype.I8 -> clamp (-128.0) 127.0
  | Dtype.I16 -> clamp (-32768.0) 32767.0
  | Dtype.I32 -> clamp (-2147483648.0) 2147483647.0
  | Dtype.I64 -> clamp (-9223372036854775808.0) 9223372036854774784.0
  | Dtype.Uint8 -> clamp 0.0 255.0
  | Dtype.Uint16 -> clamp 0.0 65535.0
  | Dtype.I2 -> clamp (-2.0) 1.0
  | Dtype.I4 -> clamp (-8.0) 7.0
  | Dtype.Uint2 -> clamp 0.0 3.0
  | Dtype.Uint4 -> clamp 0.0 15.0
  | Dtype.Float0 -> 0.0
  | Dtype.Uint32 -> clamp 0.0 4294967295.0
  | Dtype.Uint64 -> clamp 0.0 18446744073709549568.0
  | Dtype.Bool | Dtype.F16 | Dtype.Bf16 | Dtype.F4_e2m1fn | Dtype.F6_e2m3fn
  | Dtype.F6_e3m2fn | Dtype.F8_e3m4 | Dtype.F8_e4m3 | Dtype.F8_e4m3b11fnuz
  | Dtype.F8_e4m3fn | Dtype.F8_e4m3fnuz | Dtype.F8_e5m2 | Dtype.F8_e5m2fnuz
  | Dtype.F8_e8m0fnu | Dtype.F32 | Dtype.F64 | Dtype.Complex64
  | Dtype.Complex128 ->
      v

let convert_impl dt a =
  let src = Ndarray.dtype a in
  let sh = Ndarray.shape a in
  let n = Array.fold_left ( * ) 1 sh in
  let of_ints values =
    match dt with
    | Dtype.I64 | Dtype.Uint64 -> Ndarray.of_int64s dt sh values
    | _ -> Ndarray.of_floats dt sh (Array.map Int64.to_float values)
  in
  let reals () =
    let out = Array.make n 0.0 and i = ref 0 in
    ignore
      (Ndarray.fold
         (fun () v ->
           out.(!i) <- v;
           incr i)
         () a);
    out
  in
  if is_complex_dtype src && is_complex_dtype dt then map_c dt (fun z -> z) a
  else if is_complex_dtype src && dt = Dtype.Bool then
    let parts = to_carray a in
    Ndarray.of_floats dt sh
      (Array.map
         (fun (z : Complex.t) -> if z.re <> 0.0 then 1.0 else 0.0)
         parts)
  else if is_int_dtype src && is_int_dtype dt then
    of_ints (Array.map (wrap_int_to dt) (Ndarray.to_int64s a))
  else if
    (dt = Dtype.I64 || dt = Dtype.Uint64)
    && (is_float_source src || is_complex_dtype src)
  then of_ints (Array.map (to_wide_int dt) (reals ()))
  else if is_float_source src || is_complex_dtype src then
    Ndarray.map dt (saturate_to dt) a
  else Ndarray.map dt (fun x -> x) a

let impl_core_ref : (primitive -> Ndarray.t list -> Ndarray.t list) ref =
  ref (fun _ _ -> failwith "lax: impl not installed")

let rec impl_base prim inputs =
  match prim with
  | Neg -> un (fun a -> Ndarray.map (Ndarray.dtype a) (fun x -> -.x) a) inputs
  | Sin ->
      un
        (fun a ->
          let dt = Ndarray.dtype a in
          if is_complex_dtype dt then map_c dt csin a else Ndarray.map dt sin a)
        inputs
  | Cos ->
      un
        (fun a ->
          let dt = Ndarray.dtype a in
          if is_complex_dtype dt then map_c dt ccos a else Ndarray.map dt cos a)
        inputs
  | Exp ->
      un
        (fun a ->
          let dt = Ndarray.dtype a in
          if is_complex_dtype dt then map_c dt Complex.exp a
          else Ndarray.map dt exp a)
        inputs
  | Log ->
      un
        (fun a ->
          let dt = Ndarray.dtype a in
          if is_complex_dtype dt then map_c dt Complex.log a
          else Ndarray.map dt log a)
        inputs
  | Tanh -> un (fun a -> Ndarray.map (Ndarray.dtype a) tanh a) inputs
  | Abs ->
      un
        (fun a ->
          let dt = Ndarray.dtype a in
          if is_complex_dtype dt then
            map_c2r (complex_basetype dt) Complex.norm a
          else Ndarray.map dt Float.abs a)
        inputs
  | Sign -> un (fun a -> Ndarray.map (Ndarray.dtype a) sign_f a) inputs
  | Add | Add_any ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (add_f (Ndarray.dtype a)) a b)
        inputs
  | Sub ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (sub_f (Ndarray.dtype a)) a b)
        inputs
  | Mul ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (mul_f (Ndarray.dtype a)) a b)
        inputs
  | Div ->
      bin
        (fun a b ->
          let dt = Ndarray.dtype a in
          match dt with
          | Dtype.I8 | Dtype.I16 | Dtype.I2 | Dtype.I4 | Dtype.I32 | Dtype.Uint8
          | Dtype.Uint16 | Dtype.Uint2 | Dtype.Uint4 | Dtype.Uint32 ->
              Ndarray.map2 dt
                (fun x y -> if y = 0.0 then -1.0 else Float.trunc (x /. y))
                a b
          | _ -> Ndarray.map2 dt ( /. ) a b)
        inputs
  | Max -> bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) Float.max a b) inputs
  | Min -> bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) Float.min a b) inputs
  | Pow ->
      bin
        (fun a b ->
          let dt = Ndarray.dtype a in
          if is_complex_dtype dt then map2_c dt Complex.pow a b
          else Ndarray.map2 dt Float.pow a b)
        inputs
  | Eq ->
      bin (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x = y)) a b) inputs
  | Lt ->
      bin (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x < y)) a b) inputs
  | Gt ->
      bin (fun a b -> Ndarray.map2 Bool (fun x y -> bool_of (x > y)) a b) inputs
  | Select_n -> select_n_impl inputs
  | Convert_element_type dt -> un (convert_impl dt) inputs
  | Broadcast_in_dim { shape; dims } ->
      un
        (fun a ->
          let r = broadcast_in_dim_impl shape dims a in
          if Ndarray.weak a then Ndarray.as_weak r else r)
        inputs
  | Reshape ns -> un (reshape_impl ns) inputs
  | Decomposable { dimpl; _ } ->
      List.map
        (function
          | Concrete nd -> nd
          | Device b -> Pjrt.Buffer.to_host b
          | Tracer _ -> failwith "lax: decomposition produced a tracer")
        (Core.decomposition dimpl (List.map (fun nd -> Concrete nd) inputs))
  | Reshape_perm { new_sizes; dimensions } ->
      un (fun a -> reshape_impl new_sizes (transpose_impl dimensions a)) inputs
  | Reduce_sum axes ->
      un
        (fun a ->
          if is_complex_dtype (Ndarray.dtype a) then
            reduce_sum_complex_impl axes a
          else reduce_sum_impl axes a)
        inputs
  | Dot_general dd ->
      bin
        (fun l r ->
          if is_complex_dtype (Ndarray.dtype l) then
            dot_general_complex_impl dd l r
          else dot_general_impl dd l r)
        inputs
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
  | Conj ->
      un
        (fun a ->
          let dt = Ndarray.dtype a in
          if is_complex_dtype dt then map_c dt Complex.conj a
          else map_r2c (complexify dt) (fun x -> { Complex.re = x; im = 0.0 }) a)
        inputs
  | Copy -> un (fun a -> a) inputs
  | Cosh -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.cosh a) inputs
  | Exp2 -> un (fun a -> Ndarray.map (Ndarray.dtype a) exp2_f a) inputs
  | Expm1 -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.expm1 a) inputs
  | Floor -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.floor a) inputs
  | Imag ->
      un
        (fun a ->
          let dt = Ndarray.dtype a in
          if is_complex_dtype dt then
            map_c2r (complex_basetype dt) (fun z -> z.Complex.im) a
          else Ndarray.map (complex_basetype dt) (fun _ -> 0.0) a)
        inputs
  | Integer_pow y ->
      un
        (fun a ->
          let dt = Ndarray.dtype a in
          if is_int_dtype dt then
            let rec power acc base e =
              if e = 0 then acc
              else
                power
                  (if e land 1 = 1 then Int64.mul acc base else acc)
                  (Int64.mul base base) (e lsr 1)
            in
            let values =
              Array.map
                (fun m -> wrap_int_to dt (power 1L m y))
                (Ndarray.to_int64s a)
            in
            match dt with
            | Dtype.I64 | Dtype.Uint64 ->
                Ndarray.of_int64s dt (Ndarray.shape a) values
            | _ ->
                Ndarray.of_floats dt (Ndarray.shape a)
                  (Array.map Int64.to_float values)
          else Ndarray.map dt (integer_pow_f y) a)
        inputs
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
  | Real ->
      un
        (fun a ->
          let dt = Ndarray.dtype a in
          if is_complex_dtype dt then
            map_c2r (complex_basetype dt) (fun z -> z.Complex.re) a
          else a)
        inputs
  | Round -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.round a) inputs
  | Round_nearest_even ->
      un (fun a -> Ndarray.map (Ndarray.dtype a) round_nearest_even_f a) inputs
  | Rsqrt -> un (fun a -> Ndarray.map (Ndarray.dtype a) rsqrt_f a) inputs
  | Sinh -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.sinh a) inputs
  | Sqrt ->
      un
        (fun a ->
          let dt = Ndarray.dtype a in
          if is_complex_dtype dt then map_c dt Complex.sqrt a
          else Ndarray.map dt Float.sqrt a)
        inputs
  | Square -> un (fun a -> Ndarray.map (Ndarray.dtype a) square_f a) inputs
  | Tan -> un (fun a -> Ndarray.map (Ndarray.dtype a) Float.tan a) inputs
  | And ->
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (and_f (Ndarray.dtype a)) a b)
        inputs
  | Atan2 ->
      bin (fun a b -> Ndarray.map2 (Ndarray.dtype a) Float.atan2 a b) inputs
  | Complex ->
      bin
        (fun a b ->
          let dt = complexify (Ndarray.dtype a) in
          Ndarray.of_complex dt (Ndarray.shape a) (complex_of_re_im a b))
        inputs
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
      bin
        (fun a b ->
          Ndarray.map2 (Ndarray.dtype a) (shift_left_f (Ndarray.dtype a)) a b)
        inputs
  | Shift_right_arithmetic ->
      bin
        (fun a b ->
          let dt = Ndarray.dtype a in
          Ndarray.map2 dt (shift_right_arithmetic_f dt) a b)
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
  | Reduce_max axes ->
      un
        (fun a ->
          reduce_op_impl Float.max (small_identity Dtypes.max_identity a) axes a)
        inputs
  | Reduce_min axes ->
      un
        (fun a ->
          reduce_op_impl Float.min (small_identity Dtypes.min_identity a) axes a)
        inputs
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
  | After_all | Create_token -> [ Ndarray.of_floats Dtype.F32 [| 0 |] [||] ]
  | Ffi_call _ -> !Core.ffi_impl prim inputs
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
  | Rng_bit_generator _ ->
      failwith "lax: rng_bit_generator runs through the compiled backend"
  | Rng_uniform -> failwith "lax: rng primitives deferred to M3 (PRNG)"
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
  | Scatter ({ dimension_numbers = { s_clip = true; _ } as d; _ } as o) -> (
      match inputs with
      | [ operand; indices; updates ] ->
          let plain = { d with s_clip = false } in
          let clamped =
            Slicing.clamp_scatter_indices (Concrete operand) (Concrete indices)
              (Concrete updates) plain
          in
          let clamped_nd =
            match clamped with
            | Concrete nd -> nd
            | _ -> failwith "lax: scatter clip"
          in
          impl_base
            (Scatter { o with dimension_numbers = plain })
            [ operand; clamped_nd; updates ]
      | _ -> failwith "lax: scatter expects operand, indices and updates")
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
        _;
      } ->
      let conv lhs rhs =
        Convolution.conv_impl dimension_numbers window_strides padding
          lhs_dilation rhs_dilation feature_group_count batch_group_count lhs
          rhs
      in
      bin
        (fun lhs rhs ->
          match Ndarray.dtype lhs with
          | Dtype.Complex64 | Dtype.Complex128 ->
              let part prim v =
                match impl_core_ref.contents prim [ v ] with
                | [ out ] -> out
                | _ -> failwith "lax: conv complex parts"
              in
              let lift2 prim a b =
                match impl_core_ref.contents prim [ a; b ] with
                | [ out ] -> out
                | _ -> failwith "lax: conv complex parts"
              in
              let x_re = part Real lhs and x_im = part Imag lhs in
              let y_re = part Real rhs and y_im = part Imag rhs in
              let k1 = conv (lift2 Add x_re x_im) y_re in
              let k2 = conv x_re (lift2 Sub y_im y_re) in
              let k3 = conv x_im (lift2 Add y_re y_im) in
              lift2 Complex (lift2 Sub k1 k3) (lift2 Add k1 k2)
          | _ -> conv lhs rhs)
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
          [| float_of_int (platform_branch_index platforms) |];
      ]
  | Cond { branches; _ } -> (
      match inputs with
      | pred :: ops ->
          let last = List.length branches - 1 in
          let chosen = int_of_float (Ndarray.get_f pred [||]) in
          let branch = List.nth branches (max 0 (min last chosen)) in
          let outs =
            Jaxpr.eval_closed_jaxpr branch
              (List.map (fun nd -> Concrete nd) ops)
          in
          List.map
            (function
              | Concrete nd -> nd
              | Tracer _ -> failwith "lax: cond branch produced a tracer"
              | Device _ -> failwith "lax: cond branch produced a tracer")
            outs
      | [] -> failwith "lax: cond expects a predicate")
  | Scan { length; reverse; num_consts; num_carry; jaxpr; _ } ->
      let inputs_v = List.map (fun nd -> Concrete nd) inputs in
      Control_flow.Loops.scan_impl ~length ~reverse ~num_consts ~num_carry jaxpr
        inputs_v
      |> List.map (function
        | Concrete nd -> nd
        | Tracer _ -> failwith "lax: scan produced a tracer"
        | Device _ -> failwith "lax: scan produced a tracer")
  | While { cond_nconsts; body_nconsts; cond; body } ->
      let inputs_v = List.map (fun nd -> Concrete nd) inputs in
      let n = cond_nconsts + body_nconsts in
      let cond, body =
        Jaxpr.lift_while ~cond_nconsts ~body_nconsts cond body
          (List.map Core.get_aval inputs_v)
      in
      Control_flow.Loops.while_impl cond body inputs_v
      |> List.filteri (fun i _ -> i >= n)
      |> List.map (function
        | Concrete nd -> nd
        | Tracer _ -> failwith "lax: while produced a tracer"
        | Device _ -> failwith "lax: while produced a tracer")
  | Cumsum { axis; reverse } ->
      un
        (fun a ->
          if Ndarray.dtype a = Dtype.F8_e8m0fnu then
            Ndarray.map Dtype.F8_e8m0fnu (fun _ -> Float_format.nan) a
          else
            let step = small_step (Ndarray.dtype a) in
            Control_flow.Loops.cumred_impl ~axis ~reverse
              ~combine:(fun x y -> step (x +. y))
              a)
        inputs
  | Cumprod { axis; reverse } ->
      un
        (fun a ->
          let step = small_step (Ndarray.dtype a) in
          Control_flow.Loops.cumred_impl ~axis ~reverse
            ~combine:(fun x y -> step (x *. y))
            a)
        inputs
  | Cummax { axis; reverse } ->
      un
        (fun a ->
          if Ndarray.dtype a = Dtype.F8_e8m0fnu then
            Ndarray.map Dtype.F8_e8m0fnu (fun _ -> Float_format.nan) a
          else
            Control_flow.Loops.cumred_impl ~axis ~reverse ~combine:Float.max a)
        inputs
  | Cummin { axis; reverse } ->
      un
        (fun a ->
          if Ndarray.dtype a = Dtype.F8_e8m0fnu then
            Ndarray.map Dtype.F8_e8m0fnu (fun _ -> Float_format.nan) a
          else
            Control_flow.Loops.cumred_impl ~axis ~reverse ~combine:Float.min a)
        inputs
  | Cumlogsumexp { axis; reverse } ->
      un
        (Control_flow.Loops.cumred_impl ~axis ~reverse
           ~combine:Control_flow.Loops.logaddexp)
        inputs
  | Custom_linear_solve { solve; const_lengths; _ } ->
      let inputs_v = List.map (fun nd -> Concrete nd) inputs in
      Control_flow.Solves.solve_impl const_lengths solve inputs_v
      |> List.map (function
        | Concrete nd -> nd
        | Tracer _ -> failwith "lax: custom_linear_solve produced a tracer"
        | Device _ -> failwith "lax: custom_linear_solve produced a tracer")
  | Threefry2x32 | Iota_2x32_shape _ | Random_seed | Random_split _
  | Random_fold_in | Random_bits _ | Random_wrap | Random_unwrap
  | Random_gamma _ ->
      failwith "lax: prng primitive handled by the random module"
  | Cholesky -> Linalg.cholesky_impl inputs
  | Cholesky_update | Symmetric_product _ ->
      List.map
        (function
          | Concrete nd -> nd
          | Device b -> Pjrt.Buffer.to_host b
          | Tracer _ -> failwith "lax: lowered linalg produced a tracer")
        (!Core.lowered_linalg_hook prim
           (List.map (fun nd -> Concrete nd) inputs))
  | Lu -> Linalg.lu_impl inputs
  | Qr full_matrices -> Linalg.qr_impl full_matrices inputs
  | Qr_pivoted full_matrices -> Linalg.qr_pivoted_impl full_matrices inputs
  | Geqrf -> Linalg.geqrf_impl (List.hd inputs)
  | Geqp3 -> (
      match inputs with
      | [ a; jpvt ] -> Linalg.geqp3_impl a jpvt
      | _ -> failwith "linalg: geqp3 expects two operands")
  | Ormqr { ormqr_left; ormqr_transpose } -> (
      match inputs with
      | [ a; taus; c ] ->
          Linalg.ormqr_impl ~left:ormqr_left ~transpose:ormqr_transpose a taus c
      | _ -> failwith "linalg: ormqr expects three operands")
  | Householder_product -> Linalg.householder_product_impl inputs
  | Lu_pivots_to_permutation permutation_size ->
      Linalg.lu_pivots_to_permutation_impl permutation_size inputs
  | Triangular_solve
      { left_side; lower; transpose_a; conjugate_a; unit_diagonal } ->
      Linalg.triangular_solve_impl left_side lower transpose_a conjugate_a
        unit_diagonal inputs
  | Tridiagonal_solve { perturb_singular } ->
      Linalg.tridiagonal_solve_impl ~perturb_singular inputs
  | Eig { compute_left; compute_right; _ } ->
      Linalg.eig_impl compute_left compute_right inputs
  | Eigh { lower } -> Linalg.eigh_impl lower inputs
  | Hessenberg ->
      if Config.current_platform () = "cuda" then
        failwith
          "MLIR translation rule for primitive 'hessenberg' not found for \
           platform cuda";
      Linalg.hessenberg_impl inputs
  | Schur { compute_vectors } ->
      if Config.current_platform () = "cuda" then
        failwith
          "MLIR translation rule for primitive 'schur' not found for platform \
           cuda";
      Linalg.schur_impl compute_vectors inputs
  | Svd { full_matrices; compute_uv; gesvd } ->
      Linalg.svd_impl ~gesvd full_matrices compute_uv inputs
  | Tridiagonal { lower } -> Linalg.tridiagonal_impl lower inputs
  | Collective { axis_name; _ } ->
      invalid_arg ("unbound axis name: " ^ axis_name)
  | Fft { fft_type; fft_lengths } -> (
      match inputs with
      | [ x ] -> [ Fft.fft_impl fft_type fft_lengths x ]
      | _ -> failwith "lax: fft expects one operand")
  | Custom_jvp_call id | Custom_vjp_call id ->
      let primal =
        match Core.custom_rule id with
        | Core.Jvp_rule { primal; _ } | Core.Vjp_rule { primal; _ } -> primal
        | Core.Callback_rule { host; _ } -> host
        | Core.Linear_rule _ -> failwith "lax: linear rule has no primal call"
      in
      List.map custom_output (primal (List.map (fun nd -> Concrete nd) inputs))
  | Custom_lin { rule; n_res; _ } -> (
      match Core.custom_rule rule with
      | Core.Linear_rule { forward; _ } ->
          let args = List.map (fun nd -> Concrete nd) inputs in
          let res, linear =
            match Util.split_list args [ n_res ] with
            | [ a; b ] -> (a, b)
            | _ -> failwith "lax: custom_lin arity"
          in
          List.map custom_output (forward res linear)
      | Core.Jvp_rule _ | Core.Vjp_rule _ | Core.Callback_rule _ ->
          failwith
            "can't apply forward-mode autodiff (jvp) to a custom_vjp function.")
  | Checkpoint_name _ | Stop_gradient -> inputs
  | Xla_call _ -> failwith "lax: xla_call handled by interpreters"

and complex_parts nd =
  let dt = Ndarray.dtype nd in
  let shape = Ndarray.shape nd in
  let data = to_carray nd in
  let base = complex_basetype dt in
  ( Ndarray.of_floats base shape (Array.map (fun c -> c.Complex.re) data),
    Ndarray.of_floats base shape (Array.map (fun c -> c.Complex.im) data) )

and impl_by_parts prim inputs complex_dtype =
  let re_inputs, im_inputs =
    List.split
      (List.map
         (fun nd ->
           if is_complex_dtype (Ndarray.dtype nd) then complex_parts nd
           else (nd, nd))
         inputs)
  in
  let res = impl_base prim re_inputs and ims = impl_base prim im_inputs in
  List.map2
    (fun r i ->
      let shape = Ndarray.shape r in
      let n = Array.fold_left ( * ) 1 shape in
      Ndarray.of_complex complex_dtype shape
        (Array.init n (fun k ->
             let idx = Utils.decode k shape in
             { Complex.re = Ndarray.get_f r idx; im = Ndarray.get_f i idx })))
    res ims

let is_wide_int = function Dtype.I64 | Dtype.Uint64 -> true | _ -> false

let wide_int_parts nd =
  let shape = Ndarray.shape nd in
  let n = Utils.prod shape in
  let read i = Ndarray.get_i64 nd (Utils.decode i shape) in
  ( Ndarray.of_floats Dtype.F64 shape
      (Array.init n (fun i -> Int64.to_float (Int64.shift_right (read i) 32))),
    Ndarray.of_floats Dtype.F64 shape
      (Array.init n (fun i ->
           Int64.to_float (Int64.logand (read i) 0xFFFFFFFFL))) )

let wide_int_join dtype hi lo =
  let shape = Ndarray.shape hi in
  let n = Utils.prod shape in
  Ndarray.of_int64s dtype shape
    (Array.init n (fun i ->
         let idx = Utils.decode i shape in
         Int64.logor
           (Int64.shift_left (Int64.of_float (Ndarray.get_f hi idx)) 32)
           (Int64.of_float (Ndarray.get_f lo idx))))

let structural_only = function
  | Select_n | Broadcast_in_dim _ | Reshape _ | Reshape_perm _ | Copy
  | Checkpoint_name _ | Concatenate _ | Pad _ | Rev _ | Split _ | Squeeze _
  | Stack _ | Tile _ | Transpose _ | Unstack _ | Slice _ | Dynamic_slice _
  | Dynamic_update_slice | Gather _ | Scatter _ ->
      true
  | _ -> false

let wide_int_elementwise prim inputs =
  let dt = match inputs with a :: _ -> Ndarray.dtype a | [] -> Dtype.I64 in
  if not (List.for_all (fun nd -> Ndarray.dtype nd = dt) inputs) then None
  else
    let binary f =
      match inputs with
      | [ a; b ] ->
          let shape = Ndarray.shape a in
          let n = Utils.prod shape in
          Some
            [
              Ndarray.of_int64s dt shape
                (Array.init n (fun i ->
                     let idx = Utils.decode i shape in
                     f (Ndarray.get_i64 a idx) (Ndarray.get_i64 b idx)));
            ]
      | _ -> None
    in
    let unary f =
      match inputs with
      | [ a ] ->
          let shape = Ndarray.shape a in
          let n = Utils.prod shape in
          Some
            [
              Ndarray.of_int64s dt shape
                (Array.init n (fun i ->
                     f (Ndarray.get_i64 a (Utils.decode i shape))));
            ]
      | _ -> None
    in
    let compare_with test =
      match inputs with
      | [ a; b ] ->
          let shape = Ndarray.shape a in
          let n = Utils.prod shape in
          let cmp =
            if dt = Dtype.Uint64 then Int64.unsigned_compare else Int64.compare
          in
          Some
            [
              Ndarray.of_floats Dtype.Bool shape
                (Array.init n (fun i ->
                     let idx = Utils.decode i shape in
                     if
                       test
                         (cmp (Ndarray.get_i64 a idx) (Ndarray.get_i64 b idx))
                     then 1.0
                     else 0.0));
            ]
      | _ -> None
    in
    let shift_amount s = Int64.to_int s land 63 in
    let unsigned = dt = Dtype.Uint64 in
    match prim with
    | Add | Add_any -> binary Int64.add
    | Sub -> binary Int64.sub
    | Mul -> binary Int64.mul
    | Neg -> unary Int64.neg
    | And -> binary Int64.logand
    | Or -> binary Int64.logor
    | Xor -> binary Int64.logxor
    | Not -> unary Int64.lognot
    | Div ->
        binary (fun x y ->
            if y = 0L then -1L
            else if unsigned then Int64.unsigned_div x y
            else Int64.div x y)
    | Rem ->
        binary (fun x y ->
            if y = 0L then x
            else if unsigned then Int64.unsigned_rem x y
            else Int64.rem x y)
    | Max ->
        binary (fun x y ->
            let c =
              if unsigned then Int64.unsigned_compare x y else Int64.compare x y
            in
            if c >= 0 then x else y)
    | Min ->
        binary (fun x y ->
            let c =
              if unsigned then Int64.unsigned_compare x y else Int64.compare x y
            in
            if c <= 0 then x else y)
    | Shift_left ->
        binary (fun x s ->
            if Int64.unsigned_compare s 64L >= 0 then 0L
            else Int64.shift_left x (shift_amount s))
    | Shift_right_logical ->
        binary (fun x s ->
            if Int64.unsigned_compare s 64L >= 0 then 0L
            else Int64.shift_right_logical x (shift_amount s))
    | Shift_right_arithmetic ->
        binary (fun x s ->
            if Int64.unsigned_compare s 64L >= 0 then Int64.shift_right x 63
            else Int64.shift_right x (shift_amount s))
    | Eq -> compare_with (fun c -> c = 0)
    | Ne -> compare_with (fun c -> c <> 0)
    | Lt -> compare_with (fun c -> c < 0)
    | Le -> compare_with (fun c -> c <= 0)
    | Gt -> compare_with (fun c -> c > 0)
    | Ge -> compare_with (fun c -> c >= 0)
    | _ -> None

let wide_data_positions prim inputs =
  let n = List.length inputs in
  match prim with
  | Select_n -> List.init (n - 1) (fun i -> i + 1)
  | Gather _ | Dynamic_slice _ -> [ 0 ]
  | Scatter _ -> [ 0; 2 ]
  | Dynamic_update_slice -> [ 0; 1 ]
  | _ -> List.init n Fun.id

let wide_data_operands prim inputs =
  let positions = wide_data_positions prim inputs in
  positions <> []
  && List.for_all
       (fun i -> is_wide_int (Ndarray.dtype (List.nth inputs i)))
       positions

let wide_int_by_parts prim inputs =
  let positions = wide_data_positions prim inputs in
  let split =
    List.mapi
      (fun i nd -> if List.mem i positions then wide_int_parts nd else (nd, nd))
      inputs
  in
  let dt = Ndarray.dtype (List.nth inputs (List.hd positions)) in
  let his = impl_base prim (List.map fst split) in
  let los = impl_base prim (List.map snd split) in
  List.map2 (wide_int_join dt) his los

let acts_on_parts = function
  | Select_n | Broadcast_in_dim _ | Reshape _ | Reshape_perm _ | Copy
  | Checkpoint_name _ | Concatenate _ | Pad _ | Rev _ | Split _ | Squeeze _
  | Stack _ | Tile _ | Transpose _ | Unstack _ | Slice _ | Dynamic_slice _
  | Dynamic_update_slice | Gather _ | Scatter _ | Reduce_sum _ | Cumsum _ | Neg
  | Add | Add_any | Sub | Scatter_add _ | Scatter_sub _ | Reduce_window_sum _
  | Round | Round_nearest_even ->
      true
  | _ -> false

let complex_divide (lhs : Complex.t) (rhs : Complex.t) =
  let inf = Float.infinity in
  let a = lhs.re and b = lhs.im and c = rhs.re and d = rhs.im in
  let re, im =
    if Float.abs c < Float.abs d then
      let ratio = c /. d in
      let denom = d +. (c *. ratio) in
      (((a *. ratio) +. b) /. denom, ((b *. ratio) -. a) /. denom)
    else
      let ratio = d /. c in
      let denom = c +. (d *. ratio) in
      ((a +. (b *. ratio)) /. denom, (b -. (a *. ratio)) /. denom)
  in
  if not (Float.is_nan re && Float.is_nan im) then { Complex.re; im }
  else
    let finite = Float.is_finite in
    let is_inf v = Float.abs v = inf in
    let flag v = Float.copy_sign (if is_inf v then 1.0 else 0.0) v in
    if ((not (Float.is_nan a)) || not (Float.is_nan b)) && c = 0.0 && d = 0.0
    then
      let scale = Float.copy_sign inf c in
      { Complex.re = scale *. a; im = scale *. b }
    else if (is_inf a || is_inf b) && finite c && finite d then
      let fa = flag a and fb = flag b in
      {
        Complex.re = inf *. ((fa *. c) +. (fb *. d));
        im = inf *. ((fb *. c) -. (fa *. d));
      }
    else if finite a && finite b && (is_inf c || is_inf d) then
      let fc = flag c and fd = flag d in
      {
        Complex.re = 0.0 *. ((fc *. a) +. (fd *. b));
        im = 0.0 *. ((fc *. b) -. (fd *. a));
      }
    else { Complex.re; im }

let complex_order (a : Complex.t) (b : Complex.t) =
  match Float.compare a.re b.re with 0 -> Float.compare a.im b.im | c -> c

let complex_reduce combine identity axes a =
  let os = Ndarray.shape a in
  let out_shape = Utils.reduce_shape os axes in
  let out_str = Utils.strides out_shape in
  let ndim = Array.length os in
  let is_red = Array.make ndim false in
  Array.iter (fun ax -> is_red.(ax) <- true) axes;
  let src = to_carray a in
  let count = Utils.prod out_shape in
  let out = Array.make count identity in
  let seen = Array.make count false in
  Array.iteri
    (fun flat value ->
      let oidx = Utils.decode flat os in
      let of_ = ref 0 and k = ref 0 in
      for d = 0 to ndim - 1 do
        if not is_red.(d) then begin
          of_ := !of_ + (oidx.(d) * out_str.(!k));
          incr k
        end
      done;
      out.(!of_) <- (if seen.(!of_) then combine out.(!of_) value else value);
      seen.(!of_) <- true)
    src;
  Ndarray.of_complex (Ndarray.dtype a) out_shape out

let complex_cumulative combine ~axis ~reverse a =
  let os = Ndarray.shape a in
  let str = Utils.strides os in
  let src = to_carray a in
  let out = Array.copy src in
  let n = os.(axis) in
  let step = str.(axis) in
  let visit flat =
    let idx = Utils.decode flat os in
    let position = if reverse then n - 1 - idx.(axis) else idx.(axis) in
    if position > 0 then
      let previous = if reverse then flat + step else flat - step in
      out.(flat) <- combine out.(previous) out.(flat)
  in
  let total = Array.length src in
  if reverse then
    for flat = total - 1 downto 0 do
      visit flat
    done
  else
    for flat = 0 to total - 1 do
      visit flat
    done;
  Ndarray.of_complex (Ndarray.dtype a) os out

let complex_sort dimension num_keys operands =
  let split nd =
    if is_complex_dtype (Ndarray.dtype nd) then
      let re, im = complex_parts nd in
      [ re; im ]
    else [ nd ]
  in
  let keys_after =
    List.fold_left ( + ) 0
      (List.mapi
         (fun i nd ->
           if i >= num_keys then 0
           else if is_complex_dtype (Ndarray.dtype nd) then 2
           else 1)
         operands)
  in
  let sorted =
    sort_impl dimension keys_after (List.concat_map split operands)
  in
  let rec join remaining sorted =
    match (remaining, sorted) with
    | [], _ -> []
    | nd :: rest, re :: im :: tail when is_complex_dtype (Ndarray.dtype nd) ->
        let shape = Ndarray.shape nd in
        let n = Utils.prod shape in
        Ndarray.of_complex (Ndarray.dtype nd) shape
          (Array.init n (fun k ->
               let idx = Utils.decode k shape in
               { Complex.re = Ndarray.get_f re idx; im = Ndarray.get_f im idx }))
        :: join rest tail
    | _ :: rest, x :: tail -> x :: join rest tail
    | _ :: _, [] -> failwith "lax: complex sort arity"
  in
  join operands sorted

let complex_elementwise prim inputs =
  let one = Complex.one in
  let chain f = function
    | [ a ] -> Some [ map_c (Ndarray.dtype a) f a ]
    | _ -> None
  in
  let chain2 f = function
    | [ a; b ] -> Some [ map2_c (Ndarray.dtype a) f a b ]
    | _ -> None
  in
  let ccosh (z : Complex.t) =
    {
      Complex.re = Stdlib.cosh z.re *. Stdlib.cos z.im;
      im = Stdlib.sinh z.re *. Stdlib.sin z.im;
    }
  in
  let csinh (z : Complex.t) =
    {
      Complex.re = Stdlib.sinh z.re *. Stdlib.cos z.im;
      im = Stdlib.cosh z.re *. Stdlib.sin z.im;
    }
  in
  let compare_with test = function
    | [ a; b ] ->
        let ca = to_carray a and cb = to_carray b in
        Some
          [
            Ndarray.of_floats Dtype.Bool (Ndarray.shape a)
              (Array.init (Array.length ca) (fun i ->
                   if test ca.(i) cb.(i) then 1.0 else 0.0));
          ]
    | _ -> None
  in
  let cz re im = { Complex.re; im } in
  let casin z =
    let iz = Complex.mul Complex.i z in
    let root = Complex.sqrt (Complex.sub one (Complex.mul z z)) in
    Complex.mul (cz 0.0 (-1.0)) (Complex.log (Complex.add iz root))
  in
  let half_pi = cz (Float.pi /. 2.0) 0.0 in
  let half = cz 0.5 0.0 in
  let cacos z = Complex.sub half_pi (casin z) in
  let catan (z : Complex.t) =
    if z.re = 0.0 && Float.abs z.im = 1.0 then
      cz z.re (Float.copy_sign Float.infinity z.im)
    else
      Complex.mul (cz 0.0 0.5)
        (Complex.log
           (Complex.div (Complex.add Complex.i z) (Complex.sub Complex.i z)))
  in
  let casinh z =
    Complex.log
      (Complex.add z (Complex.sqrt (Complex.add (Complex.mul z z) one)))
  in
  let cacosh z =
    Complex.log
      (Complex.add z
         (Complex.mul
            (Complex.sqrt (Complex.add z one))
            (Complex.sqrt (Complex.sub z one))))
  in
  let catanh (z : Complex.t) =
    if z.im = 0.0 && Float.abs z.re = 1.0 then
      cz (Float.copy_sign Float.infinity z.re) z.im
    else
      Complex.mul half
        (Complex.log (Complex.div (Complex.add one z) (Complex.sub one z)))
  in
  let pick keep (a : Complex.t) (b : Complex.t) =
    let keeps =
      if a.re <> b.re then
        a.re = a.re && b.re = b.re && keep (compare a.re b.re)
      else keep (compare a.im b.im) && a.im = a.im && b.im = b.im
    in
    if keeps then a else b
  in
  let reduce_with combine identity axes = function
    | [ a ] -> Some [ complex_reduce combine identity axes a ]
    | _ -> None
  in
  match prim with
  | Mul -> chain2 Complex.mul inputs
  | Div -> chain2 complex_divide inputs
  | Atan2 ->
      let i = { Complex.re = 0.0; im = 1.0 } in
      chain2
        (fun x y ->
          let numerator = Complex.add y (Complex.mul i x) in
          let root =
            Complex.sqrt (Complex.add (Complex.mul y y) (Complex.mul x x))
          in
          Complex.mul (Complex.neg i)
            (Complex.log (complex_divide numerator root)))
        inputs
  | Exp2 ->
      let ln2 = { Complex.re = Float.log 2.0; im = 0.0 } in
      chain (fun z -> Complex.exp (Complex.mul z ln2)) inputs
  | Logistic ->
      chain
        (fun z ->
          Complex.div one (Complex.add one (Complex.exp (Complex.neg z))))
        inputs
  | Asin -> chain casin inputs
  | Acos -> chain cacos inputs
  | Atan -> chain catan inputs
  | Asinh -> chain casinh inputs
  | Acosh -> chain cacosh inputs
  | Atanh -> chain catanh inputs
  | Max -> chain2 (pick (fun c -> c >= 0)) inputs
  | Min -> chain2 (pick (fun c -> c <= 0)) inputs
  | Reduce_prod axes -> reduce_with Complex.mul one axes inputs
  | Reduce_max axes ->
      reduce_with (pick (fun c -> c >= 0)) Complex.zero axes inputs
  | Reduce_min axes ->
      reduce_with (pick (fun c -> c <= 0)) Complex.zero axes inputs
  | Cumprod { axis; reverse } -> (
      match inputs with
      | [ a ] -> Some [ complex_cumulative Complex.mul ~axis ~reverse a ]
      | _ -> None)
  | Sort { dimension; num_keys; _ } ->
      Some (complex_sort dimension num_keys inputs)
  | Square -> chain (fun z -> Complex.mul z z) inputs
  | Cosh -> chain ccosh inputs
  | Sinh -> chain csinh inputs
  | Tanh -> chain (fun z -> Complex.div (csinh z) (ccosh z)) inputs
  | Tan -> chain (fun z -> Complex.div (csin z) (ccos z)) inputs
  | Expm1 -> chain (fun z -> Complex.sub (Complex.exp z) one) inputs
  | Log1p -> chain (fun z -> Complex.log (Complex.add z one)) inputs
  | Rsqrt -> chain (fun z -> complex_divide one (Complex.sqrt z)) inputs
  | Integer_pow y ->
      chain
        (fun z ->
          let rec go acc i =
            if i = 0 then acc else go (Complex.mul acc z) (i - 1)
          in
          if y >= 0 then go one y else complex_divide one (go one (-y)))
        inputs
  | Sign ->
      chain
        (fun z ->
          let m = Complex.norm z in
          if m = 0.0 then Complex.zero
          else { Complex.re = z.re /. m; im = z.im /. m })
        inputs
  | Eq -> compare_with (fun a b -> a = b) inputs
  | Ne -> compare_with (fun a b -> a <> b) inputs
  | _ -> None

let shape_tuple shape =
  match Array.to_list shape with
  | [ d ] -> Printf.sprintf "(%d,)" d
  | ds -> "(" ^ String.concat ", " (List.map string_of_int ds) ^ ")"

let linalg_core_ranks = function
  | Cholesky | Lu | Qr _ | Qr_pivoted _ | Geqrf | Eigh _ | Eig _ | Hessenberg
  | Schur _ | Svd _ | Tridiagonal _ ->
      Some [ 2 ]
  | Householder_product | Geqp3 -> Some [ 2; 1 ]
  | Ormqr _ -> Some [ 2; 1; 2 ]
  | Lu_pivots_to_permutation _ -> Some [ 1 ]
  | Triangular_solve _ -> Some [ 2; 2 ]
  | Tridiagonal_solve _ -> Some [ 1; 1; 1; 2 ]
  | Symmetric_product _ -> Some [ 2; 2 ]
  | _ -> None

let batch_prefix ranks shapes =
  match (ranks, shapes) with
  | r :: _, s :: _ when Array.length s > r ->
      Some (Array.sub s 0 (Array.length s - r))
  | _ -> None

let element_slice nd batch_rank batch_index =
  let shape = Ndarray.shape nd in
  let rank = Array.length shape in
  let core = Array.sub shape batch_rank (rank - batch_rank) in
  let n = Utils.prod core in
  let full i =
    Array.append
      (Utils.decode batch_index (Array.sub shape 0 batch_rank))
      (Utils.decode i core)
  in
  let dt = Ndarray.dtype nd in
  if is_complex_dtype dt then
    Ndarray.of_complex dt core
      (Array.init n (fun i -> Ndarray.get_c nd (full i)))
  else
    Ndarray.of_floats dt core
      (Array.init n (fun i -> Ndarray.get_f nd (full i)))

let stack_elements batch elements =
  match elements with
  | [] -> failwith "lax: empty linalg batch"
  | first :: _ ->
      let core = Ndarray.shape first in
      let dt = Ndarray.dtype first in
      let shape = Array.append batch core in
      let cell = Utils.prod core in
      let total = List.length elements * cell in
      let arr = Array.of_list elements in
      let index k =
        (arr.(k / max cell 1), Utils.decode (k mod max cell 1) core)
      in
      if is_complex_dtype dt then
        Ndarray.of_complex dt shape
          (Array.init total (fun k ->
               let e, i = index k in
               Ndarray.get_c e i))
      else
        Ndarray.of_floats dt shape
          (Array.init total (fun k ->
               let e, i = index k in
               Ndarray.get_f e i))

let impl_core prim inputs =
  let wide = List.exists (fun nd -> is_wide_int (Ndarray.dtype nd)) inputs in
  match if wide then wide_int_elementwise prim inputs else None with
  | Some results -> results
  | None when wide && structural_only prim && wide_data_operands prim inputs ->
      wide_int_by_parts prim inputs
  | None -> (
      let has_complex =
        List.exists (fun nd -> is_complex_dtype (Ndarray.dtype nd)) inputs
      in
      match if has_complex then complex_elementwise prim inputs else None with
      | Some results -> results
      | None -> (
          match
            if acts_on_parts prim then
              List.find_opt
                (fun nd -> is_complex_dtype (Ndarray.dtype nd))
                inputs
            else None
          with
          | Some witness -> impl_by_parts prim inputs (Ndarray.dtype witness)
          | None -> impl_base prim inputs))

let shaped shape dtype weak_type = { shape; dtype; weak_type }

let aval_of_atom = function
  | A_var v -> v.vaval
  | A_lit nd ->
      {
        shape = Ndarray.shape nd;
        dtype = Ndarray.dtype nd;
        weak_type = Ndarray.weak nd;
      }
  | DropVar a -> a

let un_aval f = function
  | [ a ] -> [ f a ]
  | _ -> failwith "lax: expected 1 aval"

let bin_aval f = function
  | [ a; b ] -> [ f a b ]
  | _ -> failwith "lax: expected 2 avals"

let abstract_eval prim avals =
  match prim with
  | Xla_call c ->
      List.map
        (function
          | A_var (v : var) -> v.vaval
          | A_lit nd -> shaped (Ndarray.shape nd) (Ndarray.dtype nd) false
          | DropVar a -> a)
        c.jaxpr.outs
  | Neg | Sin | Cos | Exp | Log | Tanh | Sign | Acos | Acosh | Asin | Asinh
  | Atan | Atanh | Cbrt | Ceil | Clz | Copy | Cosh | Exp2 | Expm1 | Floor
  | Integer_pow _ | Log1p | Logistic | Not | Population_count | Round
  | Round_nearest_even | Rsqrt | Sinh | Sqrt | Square | Tan | Bessel_i0e
  | Bessel_i1e | Digamma | Erf | Erf_inv | Erfc | Lgamma ->
      un_aval (fun a -> a) avals
  | Abs | Real | Imag ->
      un_aval
        (fun a -> shaped a.shape (complex_basetype a.dtype) a.weak_type)
        avals
  | Conj ->
      un_aval (fun a -> shaped a.shape (complexify a.dtype) a.weak_type) avals
  | Complex ->
      bin_aval
        (fun a b ->
          shaped
            (Core.elementwise_shape [ a.shape; b.shape ])
            (complexify a.dtype)
            (a.weak_type && b.weak_type))
        avals
  | Is_finite -> un_aval (fun a -> shaped a.shape Bool false) avals
  | Add | Add_any | Sub | Mul | Div | Max | Min | Pow | And | Atan2 | Mulhi
  | Nextafter | Or | Rem | Shift_left | Shift_right_arithmetic
  | Shift_right_logical | Xor ->
      bin_aval
        (fun a b ->
          shaped
            (Core.elementwise_shape [ a.shape; b.shape ])
            a.dtype
            (a.weak_type && b.weak_type))
        avals
  | Eq | Lt | Gt | Ge | Le | Eq_to | Le_to | Lt_to | Ne ->
      bin_aval
        (fun a b ->
          shaped
            (Core.elementwise_shape [ a.shape; b.shape ])
            Bool
            (a.weak_type && b.weak_type))
        avals
  | Select_n -> (
      match avals with
      | _ :: (c :: _ as cases) ->
          [ shaped c.shape c.dtype (Utils.all_weak cases) ]
      | _ -> failwith "lax: select_n expects a predicate and cases")
  | Convert_element_type dt -> un_aval (fun a -> shaped a.shape dt false) avals
  | Broadcast_in_dim { shape; _ } ->
      un_aval (fun a -> shaped shape a.dtype a.weak_type) avals
  | Reshape ns -> un_aval (fun a -> shaped ns a.dtype a.weak_type) avals
  | Reshape_perm { new_sizes; _ } ->
      un_aval (fun a -> shaped new_sizes a.dtype a.weak_type) avals
  | Decomposable { davals; _ } -> davals
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
  | After_all | Create_token -> [ shaped [| 0 |] Dtype.F32 false ]
  | Ffi_call c -> c.ffi_result_avals
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
  | Rng_bit_generator { rbg_shape; rbg_dtype; _ } -> (
      match avals with
      | [ key ] ->
          [
            { key with weak_type = key.weak_type };
            { shape = rbg_shape; dtype = rbg_dtype; weak_type = false };
          ]
      | _ -> failwith "lax: rng_bit_generator expects 1 operand")
  | Rng_uniform -> failwith "lax: rng primitives deferred to M3 (PRNG)"
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
        _;
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
        (fun a b ->
          shaped
            (Core.elementwise_shape [ a.shape; b.shape ])
            a.dtype
            (a.weak_type && b.weak_type))
        avals
  | Regularized_incomplete_beta -> (
      match avals with
      | [ a; b; x ] ->
          let widest =
            List.fold_left
              (fun acc (v : aval) ->
                if Array.length v.shape > Array.length acc then v.shape else acc)
              [||] [ a; b; x ]
          in
          [ shaped widest x.dtype (a.weak_type && b.weak_type && x.weak_type) ]
      | _ -> failwith "lax: regularized_incomplete_beta expects 3 avals")
  | Platform_index _ -> [ shaped [||] Dtype.I32 false ]
  | Cond { branches; _ } -> List.map aval_of_atom (List.hd branches).jaxpr.outs
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
  | Threefry2x32 | Iota_2x32_shape _ | Random_seed | Random_split _
  | Random_fold_in | Random_bits _ | Random_wrap | Random_unwrap
  | Random_gamma _ ->
      failwith "lax: prng primitive handled by the random module"
  | Cholesky -> Linalg.cholesky_aval avals
  | Cholesky_update -> (
      match avals with
      | [ r; w ] ->
          let rank name (a : aval) i expected =
            if Array.length a.shape <> expected then
              invalid_arg
                (Printf.sprintf
                   "Input %d to %s must have a rank of exactly %d, but got \
                    shape=%s"
                   i name expected (shape_tuple a.shape))
          in
          rank "cholesky_update" r 0 2;
          rank "cholesky_update" w 1 1;
          if r.shape.(0) <> r.shape.(1) || w.shape.(0) <> r.shape.(1) then
            invalid_arg
              (Printf.sprintf
                 "Rank-1 update to Cholesky decomposition takes a square \
                  matrix and a vector of the same size as input. Got shapes %s \
                  and %s instead"
                 (shape_tuple r.shape) (shape_tuple w.shape));
          [ r ]
      | _ -> failwith "linalg: cholesky_update expects two avals")
  | Symmetric_product _ -> (
      match avals with
      | [ a; c ] ->
          let ra = Array.length a.shape and rc = Array.length c.shape in
          if
            ra < 2 || rc < 2
            || a.shape.(ra - 2) <> c.shape.(rc - 1)
            || c.shape.(rc - 2) <> c.shape.(rc - 1)
          then
            invalid_arg
              (Printf.sprintf
                 "symmetric_update expects a rectangular matrix of shape (m, \
                  n) and a square matrix of shape (n, n). Got shapes %s and \
                  %s."
                 (shape_tuple a.shape) (shape_tuple c.shape));
          [ c ]
      | _ -> failwith "linalg: symmetric_product expects two avals")
  | Lu -> Linalg.lu_aval avals
  | Qr full_matrices -> Linalg.qr_aval full_matrices avals
  | Qr_pivoted full_matrices -> Linalg.qr_pivoted_aval full_matrices avals
  | Geqrf -> Linalg.geqrf_aval avals
  | Geqp3 -> Linalg.geqp3_aval avals
  | Ormqr { ormqr_left; _ } -> Linalg.ormqr_aval ~left:ormqr_left avals
  | Householder_product -> Linalg.householder_product_aval avals
  | Lu_pivots_to_permutation permutation_size ->
      Linalg.lu_pivots_to_permutation_aval permutation_size avals
  | Triangular_solve _ -> Linalg.triangular_solve_aval avals
  | Tridiagonal_solve _ -> Linalg.tridiagonal_solve_aval avals
  | Eig { compute_left; compute_right; _ } ->
      Linalg.eig_aval compute_left compute_right avals
  | Eigh _ -> Linalg.eigh_aval avals
  | Hessenberg -> Linalg.hessenberg_aval avals
  | Schur { compute_vectors } -> Linalg.schur_aval compute_vectors avals
  | Svd { full_matrices; compute_uv; _ } ->
      Linalg.svd_aval full_matrices compute_uv avals
  | Tridiagonal _ -> Linalg.tridiagonal_aval avals
  | Collective { kind; _ } -> (
      match (kind, avals) with
      | (Psum | Pmax | Pmin | Pbroadcast _ | Ppermute _), [ a ] -> [ a ]
      | All_to_all { split_axis; concat_axis; size }, [ a ] ->
          let shape = Array.copy a.shape in
          shape.(split_axis) <- shape.(split_axis) / size;
          shape.(concat_axis) <- shape.(concat_axis) * size;
          [ { a with shape } ]
      | Reduce_scatter { scatter_dimension; size; tiled }, [ a ] ->
          let shape =
            if tiled then
              Array.mapi
                (fun i d -> if i = scatter_dimension then d / size else d)
                a.shape
            else Utils.squeeze_shape [| scatter_dimension |] a.shape
          in
          [ { a with shape } ]
      | All_gather { axis; size }, [ a ] ->
          let n = Array.length a.shape in
          let shape =
            Array.init (n + 1) (fun i ->
                if i < axis then a.shape.(i)
                else if i = axis then size
                else a.shape.(i - 1))
          in
          [ { a with shape } ]
      | Axis_index, _ -> [ shaped [||] Dtype.I32 false ]
      | _ -> failwith "lax: collective arity")
  | Fft { fft_type; fft_lengths } -> (
      match avals with
      | [ a ] -> [ Fft.fft_aval fft_type fft_lengths a ]
      | _ -> failwith "lax: fft expects one aval")
  | Custom_jvp_call id | Custom_vjp_call id -> (
      match Core.custom_rule id with
      | Core.Callback_rule { result_avals; _ } -> result_avals
      | Core.Jvp_rule _ | Core.Vjp_rule _ ->
          let traced = Option.get (Jaxpr.custom_call_jaxpr id avals) in
          List.map aval_of_atom traced.jaxpr.outs
      | Core.Linear_rule _ -> failwith "lax: linear rule has no primal call")
  | Custom_lin { out_avals; _ } -> out_avals
  | Checkpoint_name _ | Stop_gradient -> avals

let () = impl_core_ref := impl_core

let abstract_eval_batched prim avals =
  let batch =
    match linalg_core_ranks prim with
    | Some ranks ->
        batch_prefix ranks (List.map (fun (a : aval) -> a.shape) avals)
    | None -> None
  in
  match batch with
  | None -> abstract_eval prim avals
  | Some batch ->
      let k = Array.length batch in
      let core =
        List.map
          (fun (a : aval) ->
            { a with shape = Array.sub a.shape k (Array.length a.shape - k) })
          avals
      in
      List.map
        (fun (a : aval) -> { a with shape = Array.append batch a.shape })
        (abstract_eval prim core)

let impl prim inputs =
  let batch =
    match linalg_core_ranks prim with
    | Some ranks -> batch_prefix ranks (List.map Ndarray.shape inputs)
    | None -> None
  in
  match batch with
  | None -> impl_core prim inputs
  | Some batch ->
      let batch_rank = Array.length batch in
      let count = Utils.prod batch in
      if count = 0 then
        List.map
          (fun (out : aval) ->
            Ndarray.of_floats out.dtype out.shape
              (Array.make (Utils.prod out.shape) 0.0))
          (abstract_eval_batched prim
             (List.map
                (fun nd ->
                  {
                    shape = Ndarray.shape nd;
                    dtype = Ndarray.dtype nd;
                    weak_type = false;
                  })
                inputs))
      else
        let results =
          List.init count (fun b ->
              impl_core prim
                (List.map (fun nd -> element_slice nd batch_rank b) inputs))
        in
        let n_out = List.length (List.hd results) in
        List.init n_out (fun j ->
            stack_elements batch (List.map (fun out -> List.nth out j) results))

let install () =
  Core.rules.impl <- impl;
  Core.rules.abstract_eval <- abstract_eval_batched

let () = install ()
let cond = Control_flow.Conditionals.cond
let switch = Control_flow.Conditionals.switch
let platform_index = Control_flow.Conditionals.platform_index
let branch_call = Control_flow.Conditionals.branch_call
let scan = Control_flow.Loops.scan
let lower_scan = Control_flow.Loops.lower_scan
let dynamic_index_in_dim = Control_flow.Loops.dynamic_index_in_dim
let dynamic_update_index_in_dim = Control_flow.Loops.dynamic_update_index_in_dim
let while_loop = Control_flow.Loops.while_loop
let while_loop_keyed = Control_flow.Loops.while_loop_keyed
let cumsum = Control_flow.Loops.cumsum
let cumprod = Control_flow.Loops.cumprod
let cummax = Control_flow.Loops.cummax
let cummin = Control_flow.Loops.cummin
let cumlogsumexp = Control_flow.Loops.cumlogsumexp
let custom_linear_solve = Control_flow.Solves.custom_linear_solve

let broadcasted_matvec a b =
  let sa = (Core.get_aval a).shape and sb = (Core.get_aval b).shape in
  let an = Array.length sa and bn = Array.length sb in
  let batch =
    let la = Array.sub sa 0 (an - 2) and lb = Array.sub sb 0 (bn - 1) in
    let n = max (Array.length la) (Array.length lb) in
    let at l i =
      let k = i - (n - Array.length l) in
      if k < 0 then 1 else l.(k)
    in
    Array.init n (fun i -> max (at la i) (at lb i))
  in
  let nb = Array.length batch in
  let to_shape x sh =
    let have = (Core.get_aval x).shape in
    if have = sh then x
    else
      let n = Array.length sh and h = Array.length have in
      Core.bind1
        (Broadcast_in_dim
           { shape = sh; dims = Array.init h (fun i -> n - h + i) })
        [ x ]
  in
  let a = to_shape a (Array.append batch (Array.sub sa (an - 2) 2)) in
  let b = to_shape b (Array.append batch [| sb.(bn - 1) |]) in
  let range = Array.init nb Fun.id in
  Core.bind1
    (Dot_general
       {
         lhs_contract = [| nb + 1 |];
         rhs_contract = [| nb |];
         lhs_batch = range;
         rhs_batch = range;
         dot_precision = Some ("HIGHEST", "HIGHEST");
         untyped_dot = true;
       })
    [ a; b ]

let solve_with_lu a b =
  let sa = (Core.get_aval a).shape and sb = (Core.get_aval b).shape in
  let an = Array.length sa and bn = Array.length sb in
  let out_shape =
    Array.init bn (fun i ->
        let d_a = if i = an - 1 then 1 else sa.(i) in
        if sb.(i) = 1 then d_a else sb.(i))
  in
  let b =
    if out_shape = sb then b
    else
      Core.bind1
        (Broadcast_in_dim { shape = out_shape; dims = Array.init bn Fun.id })
        [ b ]
  in
  let lu_, _, permutation = Linalg.lu (Custom_derivatives.stop_gradient a) in
  let one bs =
    Control_flow.Solves.custom_linear_solve
      ~linear_matvec:(function
        | [ x ] ->
            let sh = (Core.get_aval x).shape in
            let widened =
              Core.bind1
                (Broadcast_in_dim
                   { shape = sh; dims = Array.init (Array.length sh) Fun.id })
                [ x ]
            in
            [ broadcasted_matvec a widened ]
        | _ -> invalid_arg "solve: arity")
      (function
        | [ x ] -> [ broadcasted_matvec a x ] | _ -> invalid_arg "solve: arity")
      bs
      (fun _ -> function
        | [ x ] -> [ Linalg.lu_solve_vector ~transpose:false lu_ permutation x ]
        | _ -> invalid_arg "solve: arity")
      ~transpose_solve:(fun _ -> function
        | [ x ] -> [ Linalg.lu_solve_vector ~transpose:true lu_ permutation x ]
        | _ -> invalid_arg "solve: arity")
  in
  if an = bn + 1 then List.hd (one [ b ])
  else
    match
      Interpreters.Batching.vmap_out one [ Some (bn - 1) ] [ b ] (max an bn - 1)
    with
    | [ out ] -> out
    | _ -> invalid_arg "solve: arity"

let () = Interpreters.Ad.solve_ref := solve_with_lu

module Linalg = Linalg
module Special = Special
module Slicing = Slicing
module Fft = Fft

let complex_bilinear = Utils.complex_bilinear
