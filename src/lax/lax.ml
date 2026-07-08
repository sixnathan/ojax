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
  | Xla_call _ | Cond _ ->
      failwith "lax: control primitives handled by interpreters"

let shaped shape dtype weak_type = { shape; dtype; weak_type }

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
  | Real | Round | Rsqrt | Sinh | Sqrt | Square | Tan ->
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
  | Xla_call _ | Cond _ ->
      failwith "lax: control primitives handled by interpreters"

let install () =
  Core.rules.impl <- impl;
  Core.rules.abstract_eval <- abstract_eval

let () = install ()
