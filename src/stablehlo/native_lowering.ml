open Types
module L = Lax_api

let weak v = (Core.get_aval v).weak_type

let promoted op a b =
  if weak a = weak b then op a b
  else
    let dt = (Core.get_aval a).dtype in
    let a' = Core.convert_like ~weak:false a dt in
    let b' = Core.convert_like ~weak:false b dt in
    op a' b'

let ( +: ) = promoted L.add
let ( -: ) = promoted L.sub
let ( *: ) = promoted L.mul
let ( /: ) = promoted L.div
let ( &&: ) = L.bitwise_and
let ( ||: ) = L.bitwise_or
let shape v = (Core.get_aval v).shape
let dtype v = (Core.get_aval v).dtype

let like v c =
  let a = Core.get_aval v in
  if a.weak_type then
    let lit =
      Concrete (Ndarray.as_weak (Ndarray.of_floats a.dtype [||] [| c |]))
    in
    if Array.length a.shape = 0 then lit
    else Core.bind1 (Broadcast_in_dim { shape = a.shape; dims = [||] }) [ lit ]
  else L.full_like v c

let select = L.select
let not_ = L.bitwise_not
let is_nan x = L.ne x x
let lgamma x = Core.bind1 Lgamma [ x ]
let digamma x = Core.bind1 Digamma [ x ]
let any pred = L.reduce_or pred (Array.init (Array.length (shape pred)) Fun.id)
let int32 x = Concrete (Ndarray.of_floats Dtype.I32 [||] [| float_of_int x |])
let finfo_eps = Lax.Special.finfo_eps
let finfo_max = Lax.Special.finfo_max
let finfo_tiny = Lax.Special.finfo_tiny

let up_and_broadcast doit args =
  let target = L.broadcast_shapes (List.map shape args) in
  let widened =
    List.map
      (fun a ->
        if shape a = target then a
        else
          L.broadcast_in_dim a target
            (Array.init
               (Array.length (shape a))
               (fun i -> Array.length target - Array.length (shape a) + i)))
      args
  in
  let source = dtype (List.hd widened) in
  match source with
  | Dtype.F16 | Dtype.Bf16 ->
      let result =
        doit Dtype.F32
          (List.map (fun a -> L.convert_element_type a Dtype.F32) widened)
      in
      [ L.convert_element_type result source ]
  | _ -> [ doit source widened ]

type igamma_mode = Value | Derivative | Sample_derivative

let scalar_const v x = Concrete (Ndarray.of_floats (dtype v) [||] [| x |])

let igamma_series ax x a enabled eps mode =
  let cond = function
    | enabled :: _ -> any enabled
    | [] -> invalid_arg "Native_lowering: igamma series state"
  in
  let body = function
    | [ enabled; r0; c0; ans0; x0; dc_da0; dans_da0 ] ->
        let r = r0 +: scalar_const r0 1.0 in
        let t1 = dc_da0 *: (x0 /: r) in
        let t2n = c0 *: x0 in
        let t2 = t2n /: (r *: r) in
        let dc_da = t1 -: t2 in
        let dans_da = dans_da0 +: dc_da in
        let c = c0 *: (x0 /: r) in
        let ans = ans0 +: c in
        let keep_going =
          match mode with
          | Value ->
              let ratio = c /: ans in
              L.gt ratio (scalar_const c eps)
          | Derivative | Sample_derivative ->
              let ratio = L.abs (dc_da /: dans_da) in
              L.gt ratio (scalar_const dc_da eps)
        in
        let cond_out = enabled &&: keep_going in
        let s1 = select enabled r r0 in
        let s2 = select enabled c c0 in
        let s3 = select enabled ans ans0 in
        let s4 = select enabled x0 x0 in
        let s5 = select enabled dc_da dc_da0 in
        let s6 = select enabled dans_da dans_da0 in
        [ cond_out; s1; s2; s3; s4; s5; s6 ]
    | _ -> invalid_arg "Native_lowering: igamma series state"
  in
  let one1 = like a 1.0 in
  let one2 = like a 1.0 in
  let zero1 = like a 0.0 in
  let zero2 = like a 0.0 in
  let init = [ enabled; a; one1; one2; x; zero1; zero2 ] in
  match L.while_loop cond body init with
  | [ _; _; _; ans; _; _; dans_da ] -> (
      match mode with
      | Value -> ans *: ax /: a
      | Derivative ->
          let lx = L.log x in
          let dlogax_da = lx -: digamma (a +: scalar_const a 1.0) in
          let inner = (ans *: dlogax_da) +: dans_da in
          ax *: inner /: a
      | Sample_derivative ->
          let lx = L.log x in
          let dlogax_da = lx -: digamma (a +: scalar_const a 1.0) in
          let inner = dans_da +: (ans *: dlogax_da) in
          L.neg inner *: x /: a)
  | _ -> invalid_arg "Native_lowering: igamma series state"

let igammac_continued_fraction ax x a enabled eps mode =
  let eps_of v = scalar_const v eps in
  let cond = function
    | enabled :: _ :: _ :: _ :: _ :: c :: _ ->
        let below = L.lt c (scalar_const c 2000.0) in
        below &&: any enabled
    | _ -> invalid_arg "Native_lowering: igammac state"
  in
  let body = function
    | [
        enabled;
        ans0;
        t0;
        y0;
        z0;
        c0;
        pkm1;
        qkm1;
        pkm2;
        qkm2;
        dpkm2_da;
        dqkm2_da;
        dpkm1_da;
        dqkm1_da;
        dans_da0;
      ] ->
        let c = c0 +: scalar_const c0 1.0 in
        let y = y0 +: scalar_const y0 1.0 in
        let z = z0 +: scalar_const z0 2.0 in
        let yc = y *: c in
        let p1 = pkm1 *: z in
        let pk = p1 -: (pkm2 *: yc) in
        let q1 = qkm1 *: z in
        let qk = q1 -: (qkm2 *: yc) in
        let nonzero = L.ne qk (scalar_const qk 0.0) in
        let r = pk /: qk in
        let tv = L.abs ((ans0 -: r) /: r) in
        let t = select nonzero tv (like r 1.0) in
        let ans = select nonzero r ans0 in
        let d1 = dpkm1_da *: z in
        let d2 = d1 -: pkm1 in
        let d3 = d2 -: (dpkm2_da *: yc) in
        let dpk_da = d3 +: (pkm2 *: c) in
        let e1 = dqkm1_da *: z in
        let e2 = e1 -: qkm1 in
        let e3 = e2 -: (dqkm2_da *: yc) in
        let dqk_da = e3 +: (qkm2 *: c) in
        let dn = dpk_da -: (ans *: dqk_da) in
        let dans_da_new = select nonzero (L.div dn qk) dans_da0 in
        let gabs = L.abs (dans_da_new -: dans_da0) in
        let grad_conditional = select nonzero gabs (like dans_da0 1.0) in
        let npkm2 = pkm1
        and npkm1 = pk
        and nqkm2 = qkm1
        and nqkm1 = qk
        and ndpkm2 = dpkm1_da
        and ndqkm2 = dqkm1_da
        and ndpkm1 = dpk_da
        and ndqkm1 = dqk_da in
        let apk = L.abs pk in
        let rescale = L.gt apk (L.reciprocal (eps_of pk)) in
        let scaled v =
          let m = L.mul v (eps_of v) in
          select rescale m v
        in
        let npkm2 = scaled npkm2 in
        let npkm1 = scaled npkm1 in
        let nqkm2 = scaled nqkm2 in
        let nqkm1 = scaled nqkm1 in
        let ndpkm2 = scaled ndpkm2 in
        let ndqkm2 = scaled ndqkm2 in
        let ndpkm1 = scaled ndpkm1 in
        let ndqkm1 = scaled ndqkm1 in
        let keep_going =
          match mode with
          | Value -> L.gt t (eps_of t)
          | Derivative | Sample_derivative ->
              L.gt grad_conditional (eps_of grad_conditional)
        in
        let cond_out = enabled &&: keep_going in
        let s1 = select enabled ans ans0 in
        let s2 = select enabled t t0 in
        let s3 = select enabled y y0 in
        let s4 = select enabled z z0 in
        let s6 = select enabled npkm1 pkm1 in
        let s7 = select enabled nqkm1 qkm1 in
        let s8 = select enabled npkm2 pkm2 in
        let s9 = select enabled nqkm2 qkm2 in
        let s10 = select enabled ndpkm2 dpkm2_da in
        let s11 = select enabled ndqkm2 dqkm2_da in
        let s12 = select enabled ndpkm1 dpkm1_da in
        let s13 = select enabled ndqkm1 dqkm1_da in
        let s14 = select enabled dans_da_new dans_da0 in
        [ cond_out; s1; s2; s3; s4; c; s6; s7; s8; s9; s10; s11; s12; s13; s14 ]
    | _ -> invalid_arg "Native_lowering: igammac state"
  in
  let y = scalar_const a 1.0 -: a in
  let xy = x +: y in
  let z = xy +: scalar_const x 1.0 in
  let c = scalar_const x 0.0 in
  let pkm2 = like x 1.0 in
  let qkm2 = x in
  let pkm1 = x +: scalar_const x 1.0 in
  let qkm1 = z *: x in
  let ans = pkm1 /: qkm1 in
  let t = like x 1.0 in
  let dpkm2_da = like x 0.0 in
  let dqkm2_da = like x 0.0 in
  let dpkm1_da = like x 0.0 in
  let dqkm1_da = L.neg x in
  let dn = dpkm1_da -: (ans *: dqkm1_da) in
  let dans_da = dn /: qkm1 in
  let init =
    [
      enabled;
      ans;
      t;
      y;
      z;
      c;
      pkm1;
      qkm1;
      pkm2;
      qkm2;
      dpkm2_da;
      dqkm2_da;
      dpkm1_da;
      dqkm1_da;
      dans_da;
    ]
  in
  match L.while_loop cond body init with
  | _ :: ans :: rest -> (
      match mode with
      | Value -> ans *: ax
      | Derivative ->
          let dans_da = List.nth rest 12 in
          let lx = L.log x in
          let dlogax_da = lx -: digamma a in
          ax *: ((ans *: dlogax_da) +: dans_da)
      | Sample_derivative ->
          let dans_da = List.nth rest 12 in
          let lx = L.log x in
          let dlogax_da = lx -: digamma a in
          let inner = dans_da +: (ans *: dlogax_da) in
          L.neg (inner *: x))
  | _ -> invalid_arg "Native_lowering: igammac state"

let log_ax a x =
  let lx = L.log x in
  let t = a *: lx in
  let d = t -: x in
  d -: lgamma a

let underflow dt ax =
  let top = Concrete (Ndarray.of_floats dt [||] [| finfo_max dt |]) in
  L.lt ax (L.neg (L.log top))

let igamma_impl dt = function
  | [ a; x ] ->
      let eps = finfo_eps dt in
      let na = is_nan a in
      let nan = na ||: is_nan x in
      let x_inf = L.eq x (scalar_const x Float.infinity) in
      let a_zero = L.eq a (scalar_const a 0.0) in
      let x_zero = L.eq x (scalar_const x 0.0) in
      let d1 = L.lt x (scalar_const x 0.0) in
      let d2 = L.lt a (scalar_const a 0.0) in
      let d3 = a_zero &&: x_zero in
      let domain_error = d1 ||: d2 in
      let domain_error = domain_error ||: d3 in
      let domain_error = domain_error ||: nan in
      let u1 = L.ge x (scalar_const x 1.0) in
      let use_igammac = u1 &&: L.gt x a in
      let log_axv = log_ax a x in
      let under = underflow dt log_axv in
      let ax = L.exp log_axv in
      let e = x_zero ||: domain_error in
      let e = e ||: under in
      let e = e ||: x_inf in
      let enabled = not_ e in
      let en_cf = enabled &&: use_igammac in
      let cf0 = igammac_continued_fraction ax x a en_cf eps Value in
      let cf = scalar_const a 1.0 -: cf0 in
      let en_s = enabled &&: not_ use_igammac in
      let series = igamma_series ax x a en_s eps Value in
      let output = select use_igammac cf series in
      let output = select x_zero (like a 0.0) output in
      let output = select x_inf (like a 1.0) output in
      select domain_error (like a Float.nan) output
  | _ -> invalid_arg "Native_lowering: igamma arity"

let igammac_impl dt = function
  | [ a; x ] ->
      let eps = finfo_eps dt in
      let na = is_nan a in
      let nan = na ||: is_nan x in
      let a_zero = L.eq a (scalar_const a 0.0) in
      let x_zero = L.eq x (scalar_const x 0.0) in
      let x_inf = L.eq x (scalar_const x Float.infinity) in
      let d1 = L.lt x (scalar_const x 0.0) in
      let d2 = L.lt a (scalar_const a 0.0) in
      let d3 = a_zero &&: x_zero in
      let domain_error = d1 ||: d2 in
      let domain_error = domain_error ||: d3 in
      let domain_error = domain_error ||: nan in
      let u1 = L.lt x (scalar_const x 1.0) in
      let use_igamma = u1 ||: L.lt x a in
      let log_axv = log_ax a x in
      let under = underflow dt log_axv in
      let e = domain_error ||: under in
      let e = e ||: x_inf in
      let e = e ||: a_zero in
      let enabled = not_ e in
      let ax = L.exp log_axv in
      let en_s = enabled &&: use_igamma in
      let series = igamma_series ax x a en_s eps Value in
      let en_cf = enabled &&: not_ use_igamma in
      let cf = igammac_continued_fraction ax x a en_cf eps Value in
      let one_minus = scalar_const a 1.0 -: series in
      let output = select use_igamma one_minus cf in
      let inf_or = x_inf ||: a_zero in
      let output = select inf_or (like output 0.0) output in
      select domain_error (like a Float.nan) output
  | _ -> invalid_arg "Native_lowering: igammac arity"

let grad_impl mode dt = function
  | [ a; x ] ->
      let eps = finfo_eps dt in
      let na = is_nan a in
      let nan = na ||: is_nan x in
      let x_zero = L.eq x (like x 0.0) in
      let d1 = L.lt x (like x 0.0) in
      let domain_error = d1 ||: L.le a (like a 0.0) in
      let u1 = L.gt x (like x 1.0) in
      let use_igammac = u1 &&: L.gt x a in
      let log_axv = log_ax a x in
      let under = underflow dt log_axv in
      let ax = L.exp log_axv in
      let e = x_zero ||: domain_error in
      let e = e ||: under in
      let e = e ||: nan in
      let enabled = not_ e in
      let en_cf = enabled &&: use_igammac in
      let cf = L.neg (igammac_continued_fraction ax x a en_cf eps mode) in
      let en_s = enabled &&: not_ use_igammac in
      let series = igamma_series ax x a en_s eps mode in
      let output = select use_igammac cf series in
      let output = select x_zero (like output 0.0) output in
      let bad = domain_error ||: nan in
      select bad (like a Float.nan) output
  | _ -> invalid_arg "Native_lowering: igamma_grad_a arity"

let igamma_grad_a_impl dt args = grad_impl Derivative dt args
let random_gamma_grad_impl dt args = grad_impl Sample_derivative dt args
let igamma = up_and_broadcast igamma_impl
let igammac = up_and_broadcast igammac_impl
let igamma_grad_a = up_and_broadcast igamma_grad_a_impl

let () =
  L.random_gamma_grad_hook :=
    fun a x -> List.hd (up_and_broadcast random_gamma_grad_impl [ a; x ])

let jnp_remainder : (value -> value -> value) ref =
  ref (fun _ _ -> invalid_arg "Native_lowering: remainder not installed")

let jnp_floor_divide : (value -> value -> value) ref =
  ref (fun _ _ -> invalid_arg "Native_lowering: floor_divide not installed")

let lentz ~num_iterations ~small ~threshold ~numerator ~denominator =
  let weak_int x =
    Concrete
      (Ndarray.as_weak
         (Ndarray.of_floats
            (Dtypes.default_int_dtype ())
            [||]
            [| float_of_int x |]))
  in
  let cond = function
    | iteration :: unconverged :: _ ->
        let remain = L.lt iteration (weak_int num_iterations) in
        remain &&: unconverged
    | _ -> invalid_arg "Native_lowering: lentz state"
  in
  let body = function
    | [ iteration; _; c0; d0; h0 ] ->
        let num = numerator iteration in
        let den = denominator iteration in
        let q = num /: c0 in
        let c = den +: q in
        let small_constant = like c small in
        let abs_c = L.abs c in
        let c_small = L.lt abs_c small_constant in
        let c = select c_small small_constant c in
        let p = num *: d0 in
        let d = den +: p in
        let abs_d = L.abs d in
        let d_small = L.lt abs_d small_constant in
        let d = select d_small small_constant d in
        let d = L.reciprocal d in
        let delta = c *: d in
        let h = h0 *: delta in
        let next = iteration +: weak_int 1 in
        let shifted = delta -: like delta 1.0 in
        let magnitude = L.abs shifted in
        let unconverged = any (L.ge magnitude (like delta threshold)) in
        [ next; unconverged; c; d; h ]
    | _ -> invalid_arg "Native_lowering: lentz state"
  in
  let first = denominator (weak_int 0) in
  let abs_first = L.abs first in
  let first_small = L.lt abs_first (like first small) in
  let h = select first_small (like first small) first in
  let zero = like h 0.0 in
  let init =
    [
      weak_int 1;
      Concrete (Ndarray.of_floats Dtype.Bool [||] [| 1.0 |]);
      h;
      zero;
      h;
    ]
  in
  List.nth (L.while_loop cond body init) 4

let betainc_impl dt = function
  | [ a0; b0; x0 ] ->
      let target = shape a0 in
      let eq_like v c = L.eq v (like v c) in
      let either p q =
        let l = p () in
        let r = q () in
        l ||: r
      in
      let a_zero =
        either (fun () -> eq_like a0 0.0) (fun () -> eq_like b0 Float.infinity)
      in
      let b_zero =
        either (fun () -> eq_like b0 0.0) (fun () -> eq_like a0 Float.infinity)
      in
      let x_zero = eq_like x0 0.0 in
      let x_one = eq_like x0 1.0 in
      let x_not_zero = not_ x_zero in
      let x_not_one = not_ x_one in
      let nan_a = is_nan a0 in
      let nan_b = is_nan b0 in
      let nan_ab = nan_a ||: nan_b in
      let nan_x = is_nan x0 in
      let nan = nan_ab ||: nan_x in
      let zero_l = b_zero &&: x_not_one in
      let zero_r = a_zero &&: x_zero in
      let result_zero = zero_l ||: zero_r in
      let one_l = a_zero &&: x_not_zero in
      let one_r = b_zero &&: x_one in
      let result_one = one_l ||: one_r in
      let lt_a = L.lt a0 (like a0 0.0) in
      let lt_b = L.lt b0 (like b0 0.0) in
      let lt_ab = lt_a ||: lt_b in
      let lt_x = L.lt x0 (like x0 0.0) in
      let lt_abx = lt_ab ||: lt_x in
      let gt_x = L.gt x0 (like x0 1.0) in
      let out_of_range = lt_abx ||: gt_x in
      let both_zero = a_zero &&: b_zero in
      let degenerate = both_zero ||: nan in
      let result_nan = out_of_range ||: degenerate in
      let one_a = like a0 1.0 in
      let num = a0 +: one_a in
      let ab = a0 +: b0 in
      let den = ab +: like b0 2.0 in
      let ratio = num /: den in
      let converges = L.lt x0 ratio in
      let a = select converges a0 b0 in
      let b = select converges b0 a0 in
      let one_x = like x0 1.0 in
      let flipped = one_x -: x0 in
      let x = select converges x0 flipped in
      let numerator iteration =
        let i = L.broadcast_in_dim iteration target [||] in
        let int_like c =
          L.broadcast_in_dim
            (Concrete
               (Ndarray.as_weak
                  (Ndarray.of_floats (Dtypes.default_int_dtype ()) [||] [| c |])))
            target [||]
        in
        let two_i = int_like 2.0 in
        let r = !jnp_remainder i two_i in
        let even = L.eq r (int_like 0.0) in
        let is_one = L.eq i (int_like 1.0) in
        let i_minus = i -: int_like 1.0 in
        let two_m = int_like 2.0 in
        let m_int = !jnp_floor_divide i_minus two_m in
        let m_zero = L.eq m_int (int_like 0.0) in
        let m = L.convert_element_type m_int dt in
        let one = like a 1.0 in
        let two = like a 2.0 in
        let zero_numerator =
          let s = a +: b in
          let n = L.neg s in
          let t = n *: x in
          let d = a +: one in
          t /: d
        in
        let even_numerator =
          let am = a +: m in
          let n = L.neg am in
          let ab = a +: b in
          let abm = ab +: m in
          let t = n *: abm in
          let t = t *: x in
          let tm = two *: m in
          let l = a +: tm in
          let tm2 = two *: m in
          let r = a +: tm2 in
          let r = r +: one in
          let d = l *: r in
          t /: d
        in
        let even_value = select m_zero zero_numerator even_numerator in
        let odd_numerator =
          let bm = b -: m in
          let t = m *: bm in
          let t = t *: x in
          let tm = two *: m in
          let l = a +: tm in
          let l = l -: one in
          let tm2 = two *: m in
          let r = a +: tm2 in
          let d = l *: r in
          t /: d
        in
        let one_numerator = like x 1.0 in
        let n = select even even_value odd_numerator in
        select is_one one_numerator n
      in
      let denominator iteration =
        let i = L.broadcast_in_dim iteration target [||] in
        let is_zero = L.eq i (like i 0.0) in
        let zero = like x 0.0 in
        let one = like x 1.0 in
        select is_zero zero one
      in
      let half_eps = finfo_eps dt /. 2.0 in
      let continued =
        lentz
          ~num_iterations:(if dt = Dtype.F32 then 200 else 600)
          ~small:half_eps ~threshold:half_eps ~numerator ~denominator
      in
      let very_small = finfo_tiny dt *. 2.0 in
      let lg_b = lgamma b in
      let ab = a +: b in
      let lg_ab = lgamma ab in
      let lbeta_small = lg_b -: lg_ab in
      let lg_a = lgamma a in
      let lbeta = lg_a +: lbeta_small in
      let tiny_a = L.lt a (like a very_small) in
      let small_branch =
        let n = L.neg x in
        let l = L.log1p n in
        let t = l *: b in
        let t = t -: lbeta_small in
        L.exp t
      in
      let large_branch =
        let lx = L.log x in
        let t = lx *: a in
        let n = L.neg x in
        let l = L.log1p n in
        let u = l *: b in
        let v = t +: u in
        let v = v -: lbeta in
        let e = L.exp v in
        e /: a
      in
      let factor = select tiny_a small_branch large_branch in
      let result = continued *: factor in
      let one_r = like result 1.0 in
      let complement = one_r -: result in
      let result = select converges result complement in
      let result = select result_zero (like a 0.0) result in
      let result = select result_one (like a 1.0) result in
      select result_nan (like a Float.nan) result
  | _ -> invalid_arg "Native_lowering: betainc arity"

let betainc = up_and_broadcast betainc_impl

let chebyshev x coefficients =
  let b0, _, b2 =
    Array.fold_left
      (fun (b0, b1, _) c ->
        let next = (x *: b0) -: b1 +: like x c in
        (next, b0, b1))
      (like x 0.0, like x 0.0, like x 0.0)
      coefficients
  in
  let diff = b0 -: b2 in
  like x 0.5 *: diff

let bessel_i0e = function
  | [ x ] ->
      let source = dtype x in
      let working, coeffs_a, coeffs_b =
        match source with
        | Dtype.F64 ->
            (x, Lax.Special.i0e_coeffs_a64, Lax.Special.i0e_coeffs_b64)
        | Dtype.F32 ->
            (x, Lax.Special.i0e_coeffs_a32, Lax.Special.i0e_coeffs_b32)
        | _ ->
            ( L.convert_element_type x Dtype.F32,
              Lax.Special.i0e_coeffs_a32,
              Lax.Special.i0e_coeffs_b32 )
      in
      let w = L.abs working in
      let half = like w 0.5 in
      let two = like w 2.0 in
      let thirty_two = like w 32.0 in
      let le8 = chebyshev ((half *: w) -: two) coeffs_a in
      let gt8 = chebyshev ((thirty_two /: w) -: two) coeffs_b /: L.sqrt w in
      let result = select (L.le w (like w 8.0)) le8 gt8 in
      [
        (if source = dtype result then result
         else L.convert_element_type result source);
      ]
  | _ -> invalid_arg "Native_lowering: bessel_i0e arity"

let is_complex v =
  match dtype v with Dtype.Complex64 | Dtype.Complex128 -> true | _ -> false

let complex_minmax ~is_max = function
  | [ x; y ] ->
      let pick a b = if is_max then L.gt a b else L.lt a b in
      let rx = L.real x and ry = L.real y in
      let same = L.eq rx ry in
      let by_real = pick rx ry in
      let ix = L.imag x in
      let iy = L.imag y in
      let by_imag = pick ix iy in
      let pick_x = select same by_imag by_real in
      [ select pick_x x y ]
  | _ -> invalid_arg "Native_lowering: complex minmax arity"

let complex_reduce ~is_max ~axes = function
  | [ x ] ->
      let init =
        Concrete
          (Ndarray.of_complex (dtype x) [||]
             [|
               {
                 Complex.re =
                   (if is_max then Float.neg_infinity else Float.infinity);
                 im = 0.0;
               };
             |])
      in
      [
        L.reduce x init
          (fun a b -> List.hd (complex_minmax ~is_max [ a; b ]))
          axes;
      ]
  | _ -> invalid_arg "Native_lowering: complex reduce arity"

let canonical_key x =
  let zeroed = select (L.eq x (like x 0.0)) (like x 0.0) x in
  select (is_nan x) (like x Float.nan) zeroed

let complex_sort ~dimension ~is_stable ~num_keys operands =
  let indexed = List.mapi (fun i v -> (i, v)) operands in
  let keys =
    List.concat_map
      (fun (i, v) ->
        if i >= num_keys then []
        else if is_complex v then
          [ canonical_key (L.real v); canonical_key (L.imag v) ]
        else [ v ])
      indexed
  in
  let payload =
    List.concat_map
      (fun v -> if is_complex v then [ L.real v; L.imag v ] else [ v ])
      operands
  in
  let sorted =
    Core.bind
      (Sort { dimension; is_stable; num_keys = List.length keys })
      (keys @ payload)
  in
  let rec drop n l = if n = 0 then l else drop (n - 1) (List.tl l) in
  let rec rebuild originals rest =
    match (originals, rest) with
    | [], _ -> []
    | v :: more, re :: im :: tail when is_complex v ->
        L.complex re im :: rebuild more tail
    | _ :: more, r :: tail -> r :: rebuild more tail
    | _ -> invalid_arg "Native_lowering: complex sort arity"
  in
  rebuild operands (drop (List.length keys) sorted)

let rec batched core rank args =
  match args with
  | first :: _ when Array.length (shape first) > rank ->
      Interpreters.Batching.vmap_flat (batched core rank)
        (List.map (fun _ -> Some 0) args)
        args
  | _ -> core args

let lu_unblocked a =
  let m = (shape a).(0) and n = (shape a).(1) in
  let k_limit = min m n in
  let m_idx = L.iota Dtype.I32 m in
  let n_idx = L.iota Dtype.I32 n in
  let body k = function
    | [ pivot; perm; a ] ->
        let column =
          L.reshape (L.dynamic_slice a [ int32 0; k ] [| m; 1 |]) [| m |]
        in
        let magnitude =
          if is_complex column then
            L.abs (L.real column) +: L.abs (L.imag column)
          else L.abs column
        in
        let bounded =
          select
            (L.ge m_idx (L.broadcast_in_dim k [| m |] [||]))
            magnitude
            (like magnitude Float.neg_infinity)
        in
        let i = L.argmax bounded ~axis:0 ~index_dtype:Dtype.I32 in
        let pivot = L.dynamic_update_slice pivot (L.reshape i [| 1 |]) [ k ] in
        let row_at r = L.dynamic_slice a [ r; int32 0 ] [| 1; n |] in
        let row_k = row_at k and row_i = row_at i in
        let a = L.dynamic_update_slice a row_i [ k; int32 0 ] in
        let a = L.dynamic_update_slice a row_k [ i; int32 0 ] in
        let perm_at r = L.dynamic_slice perm [ r ] [| 1 |] in
        let perm_k = perm_at k and perm_i = perm_at i in
        let perm = L.dynamic_update_slice perm perm_i [ k ] in
        let perm = L.dynamic_update_slice perm perm_k [ i ] in
        let column =
          L.reshape (L.dynamic_slice a [ int32 0; k ] [| m; 1 |]) [| m |]
        in
        let x = L.reshape (L.dynamic_slice a [ k; k ] [| 1; 1 |]) [||] in
        let scale =
          L.broadcast_in_dim (L.ne x (like x 0.0)) [| m |] [||]
          &&: L.gt m_idx (L.broadcast_in_dim k [| m |] [||])
        in
        let scaled_col =
          select scale (column /: L.broadcast_in_dim x [| m |] [||]) column
        in
        let a =
          L.dynamic_update_slice a
            (L.reshape scaled_col [| m; 1 |])
            [ int32 0; k ]
        in
        let row = L.dynamic_slice a [ k; int32 0 ] [| 1; n |] in
        let outer =
          L.broadcast_in_dim scaled_col [| m; n |] [| 0 |]
          *: L.broadcast_in_dim (L.reshape row [| n |]) [| m; n |] [| 1 |]
        in
        let mask =
          L.broadcast_in_dim
            (L.gt m_idx (L.broadcast_in_dim k [| m |] [||]))
            [| m; n |] [| 0 |]
          &&: L.broadcast_in_dim
                (L.gt n_idx (L.broadcast_in_dim k [| n |] [||]))
                [| m; n |] [| 1 |]
        in
        [ pivot; perm; a -: select mask outer (like outer 0.0) ]
    | _ -> invalid_arg "Native_lowering: lu state"
  in
  let pivot = L.full [| k_limit |] 0.0 Dtype.I32 in
  let perm = m_idx in
  L.fori_loop (int32 0) (int32 k_limit) body [ pivot; perm; a ]

let lu = function
  | [ a ] ->
      let rank = Array.length (shape a) in
      let m = (shape a).(rank - 2) and n = (shape a).(rank - 1) in
      if m = 0 || n = 0 then
        failwith "Stablehlo.Emit: an empty lu runs on the host";
      let core = function
        | [ a ] -> (
            match lu_unblocked a with
            | [ pivot; perm; lu ] -> [ lu; pivot; perm ]
            | _ -> invalid_arg "Native_lowering: lu state")
        | _ -> invalid_arg "Native_lowering: lu arity"
      in
      batched core 2 [ a ]
  | _ -> invalid_arg "Native_lowering: lu arity"

let scalar_like_index i x =
  Concrete (Ndarray.as_weak (Ndarray.of_floats (dtype i) [||] [| x |]))

let is_weak v = (Core.get_aval v).weak_type

let rewriting_index i n =
  let negative = L.lt i (scalar_like_index i 0.0) in
  let strong =
    if is_weak i then Core.bind1 (Convert_element_type (dtype i)) [ i ] else i
  in
  let shifted =
    Core.bind1 Add [ strong; scalar_like_index i (float_of_int n) ]
  in
  Core.bind1 Select_n [ negative; i; shifted ]

let scatter_index i n =
  let negative = L.lt i (scalar_like_index i 0.0) in
  let shifted = Core.bind1 Add [ i; scalar_like_index i (float_of_int n) ] in
  let picked = Core.bind1 Select_n [ negative; i; shifted ] in
  let strong =
    if is_weak i then Core.bind1 (Convert_element_type (dtype i)) [ picked ]
    else picked
  in
  L.broadcast_in_dim strong [| 1 |] [||]

let lu_pivots_body_inner i permutation swaps =
  let m = (shape permutation).(0) in
  let take v at =
    let start = rewriting_index at (shape v).(0) in
    let sliced =
      Core.bind1 (Dynamic_slice { slice_sizes = [| 1 |] }) [ v; start ]
    in
    Core.bind1 (Squeeze [| 0 |]) [ sliced ]
  in
  let j = take swaps i in
  let x = take permutation i in
  let y = take permutation j in
  let set v at value =
    Core.bind1
      (Scatter
         {
           dimension_numbers =
             {
               update_window_dims = [||];
               inserted_window_dims = [| 0 |];
               scatter_dims_to_operand_dims = [| 0 |];
               s_operand_batching_dims = [||];
               s_scatter_indices_batching_dims = [||];
               s_sorted_unique = true;
             };
           unique_indices = true;
         })
      [ v; scatter_index at m; value ]
  in
  set (set permutation i y) j x

let lu_pivots_to_permutation ~permutation_size = function
  | [ swaps ] ->
      let sh = shape swaps in
      let nb = Array.length sh - 1 in
      let k = sh.(nb) in
      let batch = Array.sub sh 0 nb in
      let permutation =
        L.broadcasted_iota Dtype.I32
          (Array.append batch [| permutation_size |])
          nb
      in
      if k = 0 || permutation_size = 0 then [ permutation ]
      else
        let body i = function
          | [ permutation; swaps ] ->
              let rec lifted d args =
                if d = 0 then
                  match args with
                  | [ p; s ] -> [ lu_pivots_body_inner i p s ]
                  | _ -> invalid_arg "Native_lowering: pivots state"
                else
                  Interpreters.Batching.vmap_flat
                    (lifted (d - 1))
                    [ Some 0; Some 0 ] args
              in
              lifted nb [ permutation; swaps ] @ [ swaps ]
          | _ -> invalid_arg "Native_lowering: pivots state"
        in
        [
          List.hd (L.fori_loop (int32 0) (int32 k) body [ permutation; swaps ]);
        ]
  | _ -> invalid_arg "Native_lowering: pivots arity"

let tridiagonal_solve_core = function
  | [ dl; d; du; b ] ->
      let n = (shape d).(0) and nrhs = (shape b).(1) in
      let rhs_at row =
        L.reshape (L.dynamic_slice b [ row; int32 0 ] [| 1; nrhs |]) [| nrhs |]
      in
      let scalar v i = L.reshape (L.dynamic_slice v [ i ] [| 1 |]) [||] in
      let d0 = scalar d (int32 0) in
      let cp0 = scalar du (int32 0) /: d0 in
      let dp0 = rhs_at (int32 0) /: L.broadcast_in_dim d0 [| nrhs |] [||] in
      let empty_rows = L.full [| n; nrhs |] 0.0 (dtype b) in
      let empty_c = L.full [| n |] 0.0 (dtype d) in
      let store_cp cps i v =
        L.dynamic_update_slice cps (L.reshape v [| 1 |]) [ i ]
      in
      let store_dp dps i v =
        L.dynamic_update_slice dps (L.reshape v [| 1; nrhs |]) [ i; int32 0 ]
      in
      let fwd i = function
        | [ cps; dps; cp; dp ] ->
            let cps = store_cp cps (i -: int32 1) cp in
            let dps = store_dp dps (i -: int32 1) dp in
            let a = scalar dl i and diag = scalar d i and c = scalar du i in
            let denom = diag -: (a *: cp) in
            let cp_next = c /: denom in
            let widen v = L.broadcast_in_dim v [| nrhs |] [||] in
            let dp_next = (rhs_at i -: (widen a *: dp)) /: widen denom in
            [ cps; dps; cp_next; dp_next ]
        | _ -> invalid_arg "Native_lowering: tridiagonal state"
      in
      let cps, dps, final =
        match
          L.fori_loop (int32 1) (int32 n) fwd [ empty_c; empty_rows; cp0; dp0 ]
        with
        | [ cps; dps; _; dp ] -> (cps, dps, dp)
        | _ -> invalid_arg "Native_lowering: tridiagonal state"
      in
      let bwd step = function
        | [ out; xn ] ->
            let i = int32 (n - 2) -: step in
            let cp = scalar cps i in
            let dp =
              L.reshape
                (L.dynamic_slice dps [ i; int32 0 ] [| 1; nrhs |])
                [| nrhs |]
            in
            let x = dp -: (L.broadcast_in_dim cp [| nrhs |] [||] *: xn) in
            [ store_dp out i x; x ]
        | _ -> invalid_arg "Native_lowering: tridiagonal state"
      in
      let seeded = store_dp empty_rows (int32 (n - 1)) final in
      [ List.hd (L.fori_loop (int32 0) (int32 (n - 1)) bwd [ seeded; final ]) ]
  | _ -> invalid_arg "Native_lowering: tridiagonal arity"

let tridiagonal_solve args =
  match args with
  | d :: _ ->
      let rank = Array.length (shape d) in
      if (shape d).(rank - 1) = 0 then
        failwith "Stablehlo.Emit: an empty tridiagonal_solve runs on the host";
      batched tridiagonal_solve_core 1 args
  | [] -> invalid_arg "Native_lowering: tridiagonal arity"

let triu x =
  let dims = shape x in
  let rank = Array.length dims in
  let core = Array.sub dims (rank - 2) 2 in
  let int_dt = Dtypes.default_int_dtype () in
  let iota dimension =
    Core.bind1 (Iota { dtype = int_dt; shape = core; dimension }) []
  in
  let rows = iota 0 in
  let shifted =
    L.add rows (Concrete (Ndarray.of_floats int_dt [||] [| -1.0 |]))
  in
  let cols = iota 1 in
  let mask = Core.bind1 Ge [ shifted; cols ] in
  let mask =
    if rank = 2 then mask
    else
      Core.bind1
        (Broadcast_in_dim { shape = dims; dims = [| rank - 2; rank - 1 |] })
        [ mask ]
  in
  let zero = like x 0.0 in
  select mask zero x

let qr_from_factors ~full_matrices = function
  | [ r; taus ] ->
      let dims = shape r in
      let rank = Array.length dims in
      let m = dims.(rank - 2) and n = dims.(rank - 1) in
      let cut_to rows cols v =
        let starts = Array.make rank 0 in
        let limits = Array.copy dims in
        limits.(rank - 2) <- rows;
        limits.(rank - 1) <- cols;
        L.slice v starts limits
      in
      let product v = Core.bind1 Householder_product [ v; taus ] in
      if m < n then [ product (cut_to m m r); triu r ]
      else if full_matrices then
        let pads =
          Array.init rank (fun d ->
              if d = rank - 1 then (0, m - n, 0) else (0, 0, 0))
        in
        [ product (L.pad r (L.full [||] 0.0 (dtype r)) pads); triu r ]
      else
        let q = product r in
        [ q; triu (cut_to n n r) ]
  | _ -> invalid_arg "Native_lowering: qr arity"

let qr_pivoted ~full_matrices a =
  let dims = shape a in
  let rank = Array.length dims in
  let batch = Array.sub dims 0 (rank - 2) in
  let m = dims.(rank - 2) and n = dims.(rank - 1) in
  let jpvt = L.full (Array.append batch [| n |]) 0.0 Dtype.I32 in
  match Core.bind Geqp3 [ a; jpvt ] with
  | [ r; p; taus ] ->
      let p = L.sub p (L.full (shape p) 1.0 Dtype.I32) in
      let cut_to rows cols v =
        let starts = Array.make rank 0 in
        let limits = Array.copy dims in
        limits.(rank - 2) <- rows;
        limits.(rank - 1) <- cols;
        L.slice v starts limits
      in
      let product v = Core.bind1 Householder_product [ v; taus ] in
      if m < n then
        let q = product (cut_to m m r) in
        [ q; triu r; p ]
      else if full_matrices then begin
        let pads =
          Array.init rank (fun d ->
              if d = rank - 1 then (0, m - n, 0) else (0, 0, 0))
        in
        let q = product (L.pad r (L.full [||] 0.0 (dtype r)) pads) in
        [ q; triu r; p ]
      end
      else
        let q = product r in
        [ q; triu (cut_to n n r); p ]
  | _ -> invalid_arg "Native_lowering: geqp3 arity"

let adjoint a =
  let rank = Array.length (shape a) in
  let swapped =
    L.transpose a
      (Array.init rank (fun i ->
           if i = rank - 2 then rank - 1
           else if i = rank - 1 then rank - 2
           else i))
  in
  if is_complex a then L.conj swapped else swapped

let qr_op ~full_matrices a =
  match Core.bind (Qr full_matrices) [ a ] with
  | [ q; r ] -> (q, r)
  | _ -> invalid_arg "Native_lowering: qr arity"

let eigh_op a =
  match Core.bind (Eigh { lower = true }) [ a ] with
  | [ v; w ] -> (v, w)
  | _ -> invalid_arg "Native_lowering: eigh arity"

let hermitian_from ~lower a =
  let dims = shape a in
  let rank = Array.length dims in
  let rows = dims.(rank - 2) and cols = dims.(rank - 1) in
  let total = Array.fold_left ( * ) 1 dims in
  let mask =
    Concrete
      (Ndarray.of_floats Dtype.Bool dims
         (Array.init total (fun i ->
              let local = i mod (rows * cols) in
              let row = local / cols and col = local mod cols in
              if if lower then col <= row else col >= row then 1.0 else 0.0)))
  in
  select mask a (adjoint a)

let eigh_refine_core ~lower = function
  | [ a; v ] ->
      let dims = shape v in
      let n = dims.(1) in
      let rotated = L.dot (adjoint v) (L.dot (hermitian_from ~lower a) v) in
      let diagonal =
        Concrete
          (Ndarray.of_floats Dtype.Bool dims
             (Array.init (n * n) (fun i ->
                  if i mod (n + 1) = 0 then 1.0 else 0.0)))
      in
      let real_rotated =
        if is_complex rotated then L.real rotated else rotated
      in
      let w_zeroed = select diagonal real_rotated (like real_rotated 0.0) in
      let w = L.reduce_sum w_zeroed [| 1 |] in
      let wi = L.broadcast_in_dim w dims [| 0 |] in
      let wj = L.broadcast_in_dim w dims [| 1 |] in
      let gap = wj -: wi in
      let abs_w = L.abs w in
      let scale = L.reduce_max abs_w [| 0 |] in
      let threshold = like scale (Float.sqrt (finfo_eps (dtype w))) *: scale in
      let well_separated =
        L.gt (L.abs gap) (L.broadcast_in_dim threshold dims [||])
      in
      let gap_c = L.convert_element_type gap (dtype rotated) in
      let safe_gap = select well_separated gap_c (like gap_c 1.0) in
      let correction =
        select
          (not_ diagonal &&: well_separated)
          (rotated /: safe_gap) (like rotated 0.0)
      in
      [ v +: L.dot v correction; w ]
  | _ -> invalid_arg "Native_lowering: eigh arity"

let eigh_refined ~lower args = batched (eigh_refine_core ~lower) 2 args
let real_part v = if is_complex v then L.real v else v

let correct_rank_deficiency u =
  let q, r = qr_op ~full_matrices:false u in
  let n = (shape r).(0) in
  let mask =
    Concrete
      (Ndarray.of_floats Dtype.Bool [| n; n |]
         (Array.init (n * n) (fun i -> if i mod (n + 1) = 0 then 1.0 else 0.0)))
  in
  let diag =
    L.reduce_sum (select mask (real_part r) (like (real_part r) 0.0)) [| 1 |]
  in
  let signs =
    select (L.ge diag (like diag 0.0)) (like diag 1.0) (like diag (-1.0))
  in
  let signs = L.convert_element_type signs (dtype q) in
  q *: L.broadcast_in_dim signs (shape q) [| 1 |]

let svd_tall ~compute_uv a =
  let u_p, h, _, _ = Qdwh.qdwh a in
  let v, ascending = eigh_op h in
  let clamped = L.max ascending (like ascending 0.0) in
  let n = (shape a).(1) in
  let s = L.rev clamped [| 0 |] in
  if not compute_uv then (None, s, None)
  else
    let v_out = L.rev v [| 1 |] in
    let u_out = L.dot u_p v_out in
    let eps = finfo_eps (dtype s) in
    let first = L.reshape (L.slice s [| 0 |] [| 1 |]) [||] in
    let last = L.reshape (L.slice s [| n - 1 |] [| n |]) [||] in
    let needs_correction =
      L.le last (like first (float_of_int n *. eps) *: first)
    in
    let corrected =
      List.hd
        (L.while_loop
           (function
             | [ _; pending ] -> pending
             | _ -> invalid_arg "Native_lowering: svd correction state")
           (function
             | [ u; pending ] -> [ correct_rank_deficiency u; like pending 0.0 ]
             | _ -> invalid_arg "Native_lowering: svd correction state")
           [ u_out; needs_correction ])
    in
    (Some corrected, s, Some v_out)

let all_finite a =
  let axes = Array.init (Array.length (shape a)) Fun.id in
  let finite =
    if is_complex a then L.is_finite (L.real a) &&: L.is_finite (L.imag a)
    else L.is_finite a
  in
  L.reduce_and finite axes

let nan_unless finite v =
  select (L.broadcast_in_dim finite (shape v) [||]) v (like v Float.nan)

let svd_core ~full_matrices ~compute_uv = function
  | [ operand ] -> (
      let m0 = (shape operand).(0) and n0 = (shape operand).(1) in
      let flip = m0 < n0 in
      let tall = if flip then adjoint operand else operand in
      let m = (shape tall).(0) and n = (shape tall).(1) in
      let q, null, reduced =
        if full_matrices && m > n then
          let q_full, r_full = qr_op ~full_matrices:true tall in
          ( Some (L.slice q_full [| 0; 0 |] [| m; n |]),
            Some (L.slice q_full [| 0; n |] [| m; m |]),
            L.slice r_full [| 0; 0 |] [| n; n |] )
        else if float_of_int m > 1.15 *. float_of_int n then
          let q, r = qr_op ~full_matrices:false tall in
          (Some q, None, r)
        else (None, None, tall)
      in
      match svd_tall ~compute_uv reduced with
      | u, s, v -> (
          let finite = all_finite reduced in
          let s = nan_unless finite s in
          match (u, v) with
          | Some u, Some v ->
              let u = match q with Some q -> L.dot q u | None -> u in
              let u =
                match null with
                | Some rest -> L.concatenate [ u; rest ] 1
                | None -> u
              in
              let u = nan_unless finite u and v = nan_unless finite v in
              if flip then [ v; s; adjoint u ] else [ u; s; adjoint v ]
          | _ -> [ s ]))
  | _ -> invalid_arg "Native_lowering: svd arity"

let svd ~full_matrices ~compute_uv = function
  | [ a ] ->
      if Array.exists (fun d -> d = 0) (shape a) then
        failwith "Stablehlo.Emit: an empty svd runs on the host";
      batched (svd_core ~full_matrices ~compute_uv) 2 [ a ]
  | _ -> invalid_arg "Native_lowering: svd arity"

let real_basetype v =
  match dtype v with
  | Dtype.Complex64 -> Dtype.F32
  | Dtype.Complex128 -> Dtype.F64
  | d -> d

let nan_full v =
  if is_complex v then
    let nan = { Complex.re = Float.nan; im = Float.nan } in
    L.broadcast_in_dim
      (Concrete (Ndarray.of_complex (dtype v) [||] [| nan |]))
      (shape v) [||]
  else like v Float.nan

let nan_where_bad ok v =
  let batch = shape ok in
  let nb = Array.length batch in
  let rank = Array.length (shape v) in
  let select_shape = Array.append batch (Array.make (rank - nb) 1) in
  let which = L.broadcast_in_dim ok select_shape (Array.init nb Fun.id) in
  let nans = nan_full v in
  let spread = L.broadcast_in_dim which (shape v) (Array.init rank Fun.id) in
  select spread v nans

let full_info info x =
  L.broadcast_in_dim
    (Concrete (Ndarray.of_floats (dtype info) [||] [| x |]))
    (shape info) [||]

let info_ok info = L.eq info (full_info info 0.0)

let lapack_cholesky = function
  | [ r; info ] -> [ nan_where_bad (info_ok info) r ]
  | _ -> invalid_arg "Native_lowering: potrf arity"

let lapack_lu ~permutation_size = function
  | [ lu; pivots; info ] ->
      let pivots = pivots -: full_info pivots 1.0 in
      let ok = L.ge info (full_info info 0.0) in
      let guarded = nan_where_bad ok lu in
      let perm =
        List.hd (lu_pivots_to_permutation ~permutation_size [ pivots ])
      in
      [ guarded; pivots; perm ]
  | _ -> invalid_arg "Native_lowering: getrf arity"

let lapack_guarded = function
  | [] -> invalid_arg "Native_lowering: empty lapack result"
  | outs ->
      let rev = List.rev outs in
      let info = List.hd rev and values = List.rev (List.tl rev) in
      let ok = info_ok info in
      List.map (nan_where_bad ok) values

let lapack_svd ~compute_uv = function
  | [ _; s; u; vt; info ] ->
      let ok = info_ok info in
      let s = nan_where_bad ok s in
      if compute_uv then
        let u = nan_where_bad ok u in
        let vt = nan_where_bad ok vt in
        [ u; s; vt ]
      else [ s ]
  | _ -> invalid_arg "Native_lowering: gesdd arity"

let lapack_eigen_values_real = function
  | wr :: wi :: rest -> L.complex wr wi :: rest
  | _ -> invalid_arg "Native_lowering: geev arity"

let lapack_eig ~real ~compute_left ~compute_right outs =
  match if real then lapack_eigen_values_real outs else outs with
  | [ w; vl; vr; info ] ->
      lapack_guarded
        ((w :: (if compute_left then [ vl ] else []))
        @ (if compute_right then [ vr ] else [])
        @ [ info ])
  | _ -> invalid_arg "Native_lowering: geev arity"

let lapack_schur ~compute_vectors outs =
  let keep =
    match outs with
    | t :: vs :: tail ->
        let info = List.hd (List.rev tail) in
        [ t; vs; info ]
    | _ -> invalid_arg "Native_lowering: gees arity"
  in
  match lapack_guarded keep with
  | [ t; vs ] -> if compute_vectors then [ t; vs ] else [ t ]
  | _ -> invalid_arg "Native_lowering: gees arity"

let complex_sin_parts x =
  let a = Core.bind1 Real [ x ] in
  let b = Core.bind1 Imag [ x ] in
  let dt = (Core.get_aval a).dtype in
  let zero = Core.splat dt [||] 0.0 in
  let a_is_zero = Core.bind1 Eq [ a; zero ] in
  let two = Core.splat dt [||] 2.0 in
  let sn = Core.bind1 Sin [ a ] in
  let cs = Core.bind1 Cos [ a ] in
  let e1m = Core.bind1 Expm1 [ b ] in
  let nb = Core.bind1 Neg [ b ] in
  let e2m = Core.bind1 Expm1 [ nb ] in
  let d = Core.bind1 Sub [ e1m; e2m ] in
  let snh = Core.bind1 Div [ d; two ] in
  let s1 = Core.bind1 Add [ e1m; e2m ] in
  let s2 = Core.bind1 Add [ s1; two ] in
  let csh = Core.bind1 Div [ s2; two ] in
  (a_is_zero, zero, sn, cs, snh, csh)

let complex_sin = function
  | [ x ] ->
      let a_is_zero, zero, sn, cs, snh, csh = complex_sin_parts x in
      let re = Core.bind1 Mul [ sn; csh ] in
      let im = Core.bind1 Mul [ cs; snh ] in
      let on_zero = Core.bind1 Complex [ zero; im ] in
      let other = Core.bind1 Complex [ re; im ] in
      [ Core.bind1 Select_n [ a_is_zero; other; on_zero ] ]
  | _ -> invalid_arg "native_lowering: complex_sin arity"

let complex_cos = function
  | [ x ] ->
      let a_is_zero, zero, sn, cs, snh, csh = complex_sin_parts x in
      let re = Core.bind1 Mul [ cs; csh ] in
      let nsn = Core.bind1 Neg [ sn ] in
      let im = Core.bind1 Mul [ nsn; snh ] in
      let on_zero = Core.bind1 Complex [ re; zero ] in
      let other = Core.bind1 Complex [ re; im ] in
      [ Core.bind1 Select_n [ a_is_zero; other; on_zero ] ]
  | _ -> invalid_arg "native_lowering: complex_cos arity"
