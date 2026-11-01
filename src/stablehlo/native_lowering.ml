open Types
module L = Lax_api

let ( +: ) = L.add
let ( -: ) = L.sub
let ( *: ) = L.mul
let ( /: ) = L.div
let ( &&: ) = L.bitwise_and
let ( ||: ) = L.bitwise_or
let shape v = (Core.get_aval v).shape
let dtype v = (Core.get_aval v).dtype
let like = L.full_like
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

type igamma_mode = Value | Derivative

let igamma_series ax x a enabled eps mode =
  let cond = function
    | enabled :: _ -> any enabled
    | [] -> invalid_arg "Native_lowering: igamma series state"
  in
  let body = function
    | [ enabled; r0; c0; ans0; x0; dc_da0; dans_da0 ] ->
        let r = r0 +: like r0 1.0 in
        let dc_da = (dc_da0 *: (x0 /: r)) -: (c0 *: x0 /: (r *: r)) in
        let dans_da = dans_da0 +: dc_da in
        let c = c0 *: (x0 /: r) in
        let ans = ans0 +: c in
        let keep_going =
          match mode with
          | Value -> L.gt (c /: ans) (like c eps)
          | Derivative -> L.gt (L.abs (dc_da /: dans_da)) (like dc_da eps)
        in
        [
          enabled &&: keep_going;
          select enabled r r0;
          select enabled c c0;
          select enabled ans ans0;
          select enabled x0 x0;
          select enabled dc_da dc_da0;
          select enabled dans_da dans_da0;
        ]
    | _ -> invalid_arg "Native_lowering: igamma series state"
  in
  let init =
    [ enabled; a; like a 1.0; like a 1.0; x; like a 0.0; like a 0.0 ]
  in
  match L.while_loop cond body init with
  | [ _; _; _; ans; _; _; dans_da ] -> (
      match mode with
      | Value -> ans *: ax /: a
      | Derivative ->
          let dlogax_da = L.log x -: digamma (a +: like a 1.0) in
          ax *: ((ans *: dlogax_da) +: dans_da) /: a)
  | _ -> invalid_arg "Native_lowering: igamma series state"

let igammac_continued_fraction ax x a enabled eps mode =
  let eps_of v = like v eps in
  let cond = function
    | enabled :: _ :: _ :: _ :: _ :: c :: _ ->
        any (L.lt c (like c 2000.0)) &&: any enabled
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
        let c = c0 +: like c0 1.0 in
        let y = y0 +: like y0 1.0 in
        let z = z0 +: like z0 2.0 in
        let yc = y *: c in
        let pk = (pkm1 *: z) -: (pkm2 *: yc) in
        let qk = (qkm1 *: z) -: (qkm2 *: yc) in
        let nonzero = L.ne qk (like qk 0.0) in
        let r = pk /: qk in
        let t = select nonzero (L.abs ((ans0 -: r) /: r)) (like r 1.0) in
        let ans = select nonzero r ans0 in
        let dpk_da =
          (dpkm1_da *: z) -: pkm1 -: (dpkm2_da *: yc) +: (pkm2 *: c)
        in
        let dqk_da =
          (dqkm1_da *: z) -: qkm1 -: (dqkm2_da *: yc) +: (qkm2 *: c)
        in
        let dans_da_new =
          select nonzero ((dpk_da -: (ans *: dqk_da)) /: qk) dans_da0
        in
        let grad_conditional =
          select nonzero (L.abs (dans_da_new -: dans_da0)) (like dans_da0 1.0)
        in
        let rescale = L.gt (L.abs pk) (L.reciprocal (eps_of pk)) in
        let scaled v = select rescale (v *: eps_of v) v in
        let keep_going =
          match mode with
          | Value -> L.gt t (eps_of t)
          | Derivative -> L.gt grad_conditional (eps_of grad_conditional)
        in
        [
          enabled &&: keep_going;
          select enabled ans ans0;
          select enabled t t0;
          select enabled y y0;
          select enabled z z0;
          c;
          select enabled (scaled pk) pkm1;
          select enabled (scaled qk) qkm1;
          select enabled (scaled pkm1) pkm2;
          select enabled (scaled qkm1) qkm2;
          select enabled (scaled dpkm1_da) dpkm2_da;
          select enabled (scaled dqkm1_da) dqkm2_da;
          select enabled (scaled dpk_da) dpkm1_da;
          select enabled (scaled dqk_da) dqkm1_da;
          select enabled dans_da_new dans_da0;
        ]
    | _ -> invalid_arg "Native_lowering: igammac state"
  in
  let y = like a 1.0 -: a in
  let z = x +: y +: like x 1.0 in
  let c = like x 0.0 in
  let pkm2 = like x 1.0 in
  let qkm2 = x in
  let pkm1 = x +: like x 1.0 in
  let qkm1 = z *: x in
  let ans = pkm1 /: qkm1 in
  let t = like x 1.0 in
  let dpkm2_da = like x 0.0 in
  let dqkm2_da = like x 0.0 in
  let dpkm1_da = like x 0.0 in
  let dqkm1_da = L.neg x in
  let dans_da = (dpkm1_da -: (ans *: dqkm1_da)) /: qkm1 in
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
          let dlogax_da = L.log x -: digamma a in
          ax *: ((ans *: dlogax_da) +: dans_da))
  | _ -> invalid_arg "Native_lowering: igammac state"

let log_ax a x = (a *: L.log x) -: x -: lgamma a
let underflow dt ax = L.lt ax (like ax (-.Float.log (finfo_max dt)))

let igamma_impl dt = function
  | [ a; x ] ->
      let eps = finfo_eps dt in
      let nan = is_nan a ||: is_nan x in
      let x_inf = L.eq x (like x Float.infinity) in
      let a_zero = L.eq a (like a 0.0) in
      let x_zero = L.eq x (like x 0.0) in
      let domain_error =
        L.lt x (like x 0.0)
        ||: L.lt a (like a 0.0)
        ||: (a_zero &&: x_zero) ||: nan
      in
      let use_igammac = L.ge x (like x 1.0) &&: L.gt x a in
      let log_axv = log_ax a x in
      let under = underflow dt log_axv in
      let ax = L.exp log_axv in
      let enabled = not_ (x_zero ||: domain_error ||: under ||: x_inf) in
      let cf =
        like a 1.0
        -: igammac_continued_fraction ax x a (enabled &&: use_igammac) eps Value
      in
      let series =
        igamma_series ax x a (enabled &&: not_ use_igammac) eps Value
      in
      let output = select use_igammac cf series in
      let output = select x_zero (like a 0.0) output in
      let output = select x_inf (like a 1.0) output in
      select domain_error (like a Float.nan) output
  | _ -> invalid_arg "Native_lowering: igamma arity"

let igammac_impl dt = function
  | [ a; x ] ->
      let eps = finfo_eps dt in
      let nan = is_nan a ||: is_nan x in
      let a_zero = L.eq a (like a 0.0) in
      let x_zero = L.eq x (like x 0.0) in
      let x_inf = L.eq x (like x Float.infinity) in
      let domain_error =
        L.lt x (like x 0.0)
        ||: L.lt a (like a 0.0)
        ||: (a_zero &&: x_zero) ||: nan
      in
      let use_igamma = L.lt x (like x 1.0) ||: L.lt x a in
      let log_axv = log_ax a x in
      let under = underflow dt log_axv in
      let enabled = not_ (domain_error ||: under ||: x_inf ||: a_zero) in
      let ax = L.exp log_axv in
      let series = igamma_series ax x a (enabled &&: use_igamma) eps Value in
      let cf =
        igammac_continued_fraction ax x a
          (enabled &&: not_ use_igamma)
          eps Value
      in
      let output = select use_igamma (like a 1.0 -: series) cf in
      let output = select (x_inf ||: a_zero) (like output 0.0) output in
      select domain_error (like a Float.nan) output
  | _ -> invalid_arg "Native_lowering: igammac arity"

let igamma_grad_a_impl dt = function
  | [ a; x ] ->
      let eps = finfo_eps dt in
      let nan = is_nan a ||: is_nan x in
      let x_zero = L.eq x (like x 0.0) in
      let domain_error = L.lt x (like x 0.0) ||: L.le a (like a 0.0) in
      let use_igammac = L.gt x (like x 1.0) &&: L.gt x a in
      let log_axv = log_ax a x in
      let under = underflow dt log_axv in
      let ax = L.exp log_axv in
      let enabled = not_ (x_zero ||: domain_error ||: under ||: nan) in
      let cf =
        L.neg
          (igammac_continued_fraction ax x a (enabled &&: use_igammac) eps
             Derivative)
      in
      let series =
        igamma_series ax x a (enabled &&: not_ use_igammac) eps Derivative
      in
      let output = select use_igammac cf series in
      let output = select x_zero (like output 0.0) output in
      select (domain_error ||: nan) (like a Float.nan) output
  | _ -> invalid_arg "Native_lowering: igamma_grad_a arity"

let igamma = up_and_broadcast igamma_impl
let igammac = up_and_broadcast igammac_impl
let igamma_grad_a = up_and_broadcast igamma_grad_a_impl

let lentz ~num_iterations ~small ~threshold ~numerator ~denominator ~first =
  let cond = function
    | iteration :: unconverged :: _ ->
        L.lt iteration (int32 num_iterations) &&: unconverged
    | _ -> invalid_arg "Native_lowering: lentz state"
  in
  let body = function
    | [ iteration; _; c0; d0; h0 ] ->
        let num = numerator iteration in
        let den = denominator iteration in
        let small_of v = like v small in
        let c = den +: (num /: c0) in
        let c = select (L.lt (L.abs c) (small_of c)) (small_of c) c in
        let d = den +: (num *: d0) in
        let d = select (L.lt (L.abs d) (small_of d)) (small_of d) d in
        let d = L.reciprocal d in
        let delta = c *: d in
        let h = h0 *: delta in
        [
          iteration +: int32 1;
          any (L.ge (L.abs (delta -: like delta 1.0)) (like delta threshold));
          c;
          d;
          h;
        ]
    | _ -> invalid_arg "Native_lowering: lentz state"
  in
  let h =
    select (L.lt (L.abs first) (like first small)) (like first small) first
  in
  let init =
    [
      int32 1;
      Concrete (Ndarray.of_floats Dtype.Bool [||] [| 1.0 |]);
      h;
      like h 0.0;
      h;
    ]
  in
  List.nth (L.while_loop cond body init) 4

let betainc_impl dt = function
  | [ a0; b0; x0 ] ->
      let target = shape a0 in
      let a_zero = L.eq a0 (like a0 0.0) ||: L.eq b0 (like b0 Float.infinity) in
      let b_zero = L.eq b0 (like b0 0.0) ||: L.eq a0 (like a0 Float.infinity) in
      let x_zero = L.eq x0 (like x0 0.0) in
      let x_one = L.eq x0 (like x0 1.0) in
      let nan = is_nan a0 ||: is_nan b0 ||: is_nan x0 in
      let result_zero = b_zero &&: not_ x_one ||: (a_zero &&: x_zero) in
      let result_one = a_zero &&: not_ x_zero ||: (b_zero &&: x_one) in
      let result_nan =
        L.lt a0 (like a0 0.0)
        ||: L.lt b0 (like b0 0.0)
        ||: L.lt x0 (like x0 0.0)
        ||: L.gt x0 (like x0 1.0)
        ||: (a_zero &&: b_zero) ||: nan
      in
      let converges =
        L.lt x0 ((a0 +: like a0 1.0) /: (a0 +: b0 +: like b0 2.0))
      in
      let a = select converges a0 b0 in
      let b = select converges b0 a0 in
      let x = select converges x0 (like x0 1.0 -: x0) in
      let numerator iteration =
        let i = L.broadcast_in_dim iteration target [||] in
        let even = L.eq (L.rem i (like i 2.0)) (like i 0.0) in
        let is_one = L.eq i (like i 1.0) in
        let m_int = (i -: like i 1.0) /: like i 2.0 in
        let m_zero = L.eq m_int (like m_int 0.0) in
        let m = L.convert_element_type m_int dt in
        let one = like a 1.0 in
        let two = like a 2.0 in
        let zero_numerator = L.neg (a +: b) *: x /: (a +: one) in
        let even_numerator =
          select m_zero zero_numerator
            (L.neg (a +: m)
            *: (a +: b +: m)
            *: x
            /: ((a +: (two *: m)) *: (a +: (two *: m) +: one)))
        in
        let odd_numerator =
          m *: (b -: m) *: x /: ((a +: (two *: m) -: one) *: (a +: (two *: m)))
        in
        select is_one (like x 1.0) (select even even_numerator odd_numerator)
      in
      let denominator iteration =
        let i = L.broadcast_in_dim iteration target [||] in
        select (L.eq i (like i 0.0)) (like x 0.0) (like x 1.0)
      in
      let half_eps = finfo_eps dt /. 2.0 in
      let continued =
        lentz
          ~num_iterations:(if dt = Dtype.F32 then 200 else 600)
          ~small:half_eps ~threshold:half_eps ~numerator ~denominator
          ~first:(like x 0.0)
      in
      let very_small = finfo_tiny dt *. 2.0 in
      let lbeta_small = lgamma b -: lgamma (a +: b) in
      let lbeta = lgamma a +: lbeta_small in
      let log1p_neg_x = L.log1p (L.neg x) in
      let factor =
        select
          (L.lt a (like a very_small))
          (L.exp ((log1p_neg_x *: b) -: lbeta_small))
          (L.exp ((L.log x *: a) +: (log1p_neg_x *: b) -: lbeta) /: a)
      in
      let result = continued *: factor in
      let result = select converges result (like result 1.0 -: result) in
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
  like x 0.5 *: (b0 -: b2)

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
      let le8 = chebyshev ((like w 0.5 *: w) -: like w 2.0) coeffs_a in
      let gt8 =
        chebyshev ((like w 32.0 /: w) -: like w 2.0) coeffs_b /: L.sqrt w
      in
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
      let pick_x =
        select (L.eq rx ry) (pick (L.imag x) (L.imag y)) (pick rx ry)
      in
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

let lu_pivots_to_permutation ~permutation_size = function
  | [ pivots ] ->
      let core = function
        | [ swaps ] ->
            let k = (shape swaps).(0) in
            let permutation = L.iota Dtype.I32 permutation_size in
            if k = 0 || permutation_size = 0 then [ permutation ]
            else
              let body i = function
                | [ permutation ] ->
                    let j = L.dynamic_slice swaps [ i ] [| 1 |] in
                    let j = L.reshape j [||] in
                    let x = L.dynamic_slice permutation [ i ] [| 1 |] in
                    let y = L.dynamic_slice permutation [ j ] [| 1 |] in
                    let p = L.dynamic_update_slice permutation y [ i ] in
                    [ L.dynamic_update_slice p x [ j ] ]
                | _ -> invalid_arg "Native_lowering: pivots state"
              in
              L.fori_loop (int32 0) (int32 k) body [ permutation ]
        | _ -> invalid_arg "Native_lowering: pivots arity"
      in
      batched core 1 [ pivots ]
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

let triu_mask shape_ =
  let rank = Array.length shape_ in
  let rows = shape_.(rank - 2) and cols = shape_.(rank - 1) in
  let total = Array.fold_left ( * ) 1 shape_ in
  Concrete
    (Ndarray.of_floats Dtype.Bool shape_
       (Array.init total (fun i ->
            let local = i mod (rows * cols) in
            if local mod cols >= local / cols then 1.0 else 0.0)))

let triu x = select (triu_mask (shape x)) x (like x 0.0)

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
      else [ product r; triu (cut_to n n r) ]
  | _ -> invalid_arg "Native_lowering: qr arity"

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
    let part = L.full (shape v) Float.nan (real_basetype v) in
    L.complex part part
  else like v Float.nan

let nan_where_bad ok v =
  let dims = Array.init (Array.length (shape ok)) Fun.id in
  select (L.broadcast_in_dim ok (shape v) dims) v (nan_full v)

let info_ok info = L.eq info (like info 0.0)

let tril_mask shape_ =
  let rank = Array.length shape_ in
  let rows = shape_.(rank - 2) and cols = shape_.(rank - 1) in
  let total = Array.fold_left ( * ) 1 shape_ in
  Concrete
    (Ndarray.of_floats Dtype.Bool shape_
       (Array.init total (fun i ->
            let local = i mod max 1 (rows * cols) in
            if local mod max cols 1 <= local / max cols 1 then 1.0 else 0.0)))

let lapack_cholesky = function
  | [ r; info ] ->
      let guarded = nan_where_bad (info_ok info) r in
      [ select (tril_mask (shape r)) guarded (like r 0.0) ]
  | _ -> invalid_arg "Native_lowering: potrf arity"

let lapack_lu ~permutation_size = function
  | [ lu; pivots; info ] ->
      let ok = L.ge info (like info 0.0) in
      let pivots = pivots -: like pivots 1.0 in
      let perm =
        List.hd (lu_pivots_to_permutation ~permutation_size [ pivots ])
      in
      [ nan_where_bad ok lu; pivots; perm ]
  | _ -> invalid_arg "Native_lowering: getrf arity"

let lapack_guarded = function
  | [] -> invalid_arg "Native_lowering: empty lapack result"
  | outs ->
      let rev = List.rev outs in
      let info = List.hd rev and values = List.rev (List.tl rev) in
      let ok = info_ok info in
      List.map (nan_where_bad ok) values

let lapack_svd ~compute_uv outs =
  match lapack_guarded outs with
  | [ _; s; u; vt ] -> if compute_uv then [ u; s; vt ] else [ s ]
  | _ -> invalid_arg "Native_lowering: gesdd arity"

let lapack_eigen_values_real = function
  | wr :: wi :: rest -> L.complex wr wi :: rest
  | _ -> invalid_arg "Native_lowering: geev arity"

let lapack_eig ~real ~compute_left ~compute_right outs =
  let outs = if real then lapack_eigen_values_real outs else outs in
  match lapack_guarded outs with
  | [ w; vl; vr ] ->
      (w :: (if compute_left then [ vl ] else []))
      @ if compute_right then [ vr ] else []
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
