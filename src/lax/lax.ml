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
  | Floor ->
      un_aval (fun a -> a) avals
  | Add | Sub | Mul | Div | Max | Min | Pow ->
      bin_aval
        (fun a b -> shaped a.shape a.dtype (a.weak_type && b.weak_type))
        avals
  | Eq | Lt | Gt -> bin_aval (fun a _ -> shaped a.shape Bool false) avals
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
