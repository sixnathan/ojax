open Types
module Nd = Ndarray
module C = Ojax__Core
module D = Dtype
module PR = Prng
module TF = Threefry

let prod a = Array.fold_left ( * ) 1 a
let float_override : D.t option ref = ref None
let int_override : D.t option ref = ref None

let fdt () =
  match !float_override with
  | Some d -> d
  | None -> Dtypes.default_float_dtype ()

let idt () =
  match !int_override with Some d -> d | None -> Dtypes.default_int_dtype ()

let with_dtypes ?float_dtype ?int_dtype thunk =
  let saved_f = !float_override and saved_i = !int_override in
  (match float_dtype with
  | Some _ -> float_override := float_dtype
  | None -> ());
  (match int_dtype with Some _ -> int_override := int_dtype | None -> ());
  Fun.protect
    ~finally:(fun () ->
      float_override := saved_f;
      int_override := saved_i)
    thunk

let f32 x =
  match fdt () with
  | D.F64 -> x
  | _ -> Int32.float_of_bits (Int32.bits_of_float x)

let concrete = function
  | Concrete nd -> nd
  | Tracer _ -> failwith "random/core: sampler on a tracer not supported in M3"
  | Device b -> Pjrt.Buffer.to_host b

let unravel shape i =
  let n = Array.length shape in
  let idx = Array.make n 0 in
  let r = ref i in
  for k = n - 1 downto 0 do
    idx.(k) <- !r mod shape.(k);
    r := !r / shape.(k)
  done;
  idx

let geti nd i = Int64.to_int (Nd.get_i64 nd (unravel (Nd.shape nd) i))

let const dt shape x =
  Concrete
    (Nd.canonicalize dt (Nd.of_floats dt shape (Array.make (prod shape) x)))

let wrap k = PR.random_wrap k
let unwrap k = PR.random_unwrap k
let mask32 = 0xFFFFFFFFL
let default_prng_impl () = TF.threefry_prng_impl

let resolve_prng_impl = function
  | None -> TF.threefry_prng_impl
  | Some "threefry2x32" -> TF.threefry_prng_impl
  | Some name ->
      invalid_arg
        (Printf.sprintf "unrecognized PRNG implementation \"%s\"" name)

let key_impl _keys = TF.threefry_prng_impl.PR.name
let key_dtype _impl_spec = D.Uint32

let copy_nd nd =
  let shape = Nd.shape nd in
  let n = prod shape in
  Nd.of_floats (Nd.dtype nd) shape
    (Array.init n (fun i -> Nd.get_f nd (unravel shape i)))

let raw_copy = function
  | Concrete nd -> Concrete (copy_nd nd)
  | Device b -> Concrete (copy_nd (Pjrt.Buffer.to_host b))
  | Tracer _ as v -> C.bind1 Copy [ v ]

let key seed = Key_reuse.mark_typed (PR.random_seed seed)
let key_data keys = Key_reuse.suspended (fun () -> raw_copy (unwrap keys))

let wrap_key_data data =
  Key_reuse.mark_typed (Key_reuse.suspended (fun () -> raw_copy (wrap data)))

let clone k = Key_reuse.mark_typed (Key_reuse.suspended (fun () -> raw_copy k))

let fold_in k data =
  unwrap
    (PR.random_fold_in (wrap k)
       (C.bind1 (Convert_element_type D.Uint32) [ data ]))

let split k num = unwrap (PR.random_split (wrap k) [| num |])
let bits k ~shape = PR.random_bits (wrap k) ~bit_width:32 ~shape

let uniform_unit k ~shape =
  match fdt () with
  | D.F64
    when (match k with Tracer _ -> true | _ -> false)
         || not (Config.value Config.threefry_partitionable) ->
      let raw = PR.random_bits (wrap k) ~bit_width:64 ~shape in
      let twelve =
        Concrete (Nd.of_int64s D.Uint64 shape (Array.make (prod shape) 12L))
      in
      let exponent =
        Concrete
          (Nd.of_int64s D.Uint64 shape
             (Array.make (prod shape) 0x3FF0000000000000L))
      in
      let mantissa =
        C.bind1 Or [ C.bind1 Shift_right_logical [ raw; twelve ]; exponent ]
      in
      C.bind1 (Bitcast_convert_type D.F64) [ mantissa ]
  | D.F64 ->
      let raw = TF.bits64 (concrete (wrap k)) shape in
      let exponent = 0x3FF0000000000000L in
      Concrete
        (Nd.of_floats D.F64 shape
           (Array.map
              (fun b ->
                Int64.float_of_bits
                  (Int64.logor (Int64.shift_right_logical b 12) exponent))
              raw))
  | dt ->
      let raw = PR.random_bits (wrap k) ~bit_width:32 ~shape in
      let shifted =
        C.bind1 Shift_right_logical [ raw; const D.Uint32 shape 9.0 ]
      in
      let exponent = const D.Uint32 shape (float_of_int 0x3F800000) in
      let mantissa = C.bind1 Or [ shifted; exponent ] in
      let as_int = C.bind1 (Convert_element_type D.I32) [ mantissa ] in
      C.bind1 (Bitcast_convert_type dt) [ as_int ]

let uniform k ~shape ~minval ~maxval =
  let dt = fdt () in
  let floats_v = uniform_unit k ~shape in
  let minv = f32 minval and maxv = f32 maxval in
  let span = f32 (maxv -. minv) in
  let u = C.bind1 Sub [ floats_v; const dt shape 1.0 ] in
  let scaled = C.bind1 Mul [ u; const dt shape span ] in
  let y = C.bind1 Add [ scaled; const dt shape minv ] in
  C.bind1 Max [ const dt shape minv; y ]

let scalar_f v = Nd.get_f (concrete v) [||]
let eagerly thunk = C.with_eval_dynamic thunk

let normal k ~shape =
  let dt = fdt () in
  let lo =
    eagerly (fun () ->
        scalar_f (C.bind1 Nextafter [ const dt [||] (-1.0); const dt [||] 0.0 ]))
  in
  let u = uniform k ~shape ~minval:lo ~maxval:1.0 in
  let sqrt2 = f32 (sqrt 2.0) in
  C.bind1 Mul [ const dt shape sqrt2; C.bind1 Erf_inv [ u ] ]

let truncated_normal k ~lower ~upper ~shape =
  let dt = fdt () in
  let sqrt2 = f32 (sqrt 2.0) in
  let lower_f = f32 lower and upper_f = f32 upper in
  let erf_scaled b =
    scalar_f
      (C.bind1 Erf [ C.bind1 Div [ const dt [||] b; const dt [||] sqrt2 ] ])
  in
  let a = erf_scaled lower_f and b = erf_scaled upper_f in
  let u = uniform k ~shape ~minval:a ~maxval:b in
  let out = C.bind1 Mul [ const dt shape sqrt2; C.bind1 Erf_inv [ u ] ] in
  let lo =
    scalar_f
      (C.bind1 Nextafter [ const dt [||] lower_f; const dt [||] infinity ])
  in
  let hi =
    scalar_f
      (C.bind1 Nextafter [ const dt [||] upper_f; const dt [||] neg_infinity ])
  in
  C.bind1 Min [ const dt shape hi; C.bind1 Max [ const dt shape lo; out ] ]

let randint_base k ~shape ~minval ~maxval =
  let keys = concrete (PR.random_split (wrap k) [| 2 |]) in
  let subkey j =
    Concrete
      (Nd.of_floats D.Uint32 [| 2 |]
         [|
           Int64.to_float (Nd.get_i64 keys [| j; 0 |]);
           Int64.to_float (Nd.get_i64 keys [| j; 1 |]);
         |])
  in
  let hb = concrete (PR.random_bits (wrap (subkey 0)) ~bit_width:32 ~shape) in
  let lb = concrete (PR.random_bits (wrap (subkey 1)) ~bit_width:32 ~shape) in
  let span =
    if maxval <= minval then 1L
    else Int64.logand (Int64.of_int (maxval - minval)) mask32
  in
  let m1 = Int64.rem 0x10000L span in
  let mult = Int64.rem (Int64.mul m1 m1) span in
  let n = prod shape in
  let out =
    Array.init n (fun i ->
        let h = Int64.of_int (geti hb i) and l = Int64.of_int (geti lb i) in
        let t1 = Int64.rem h span in
        let t2 = Int64.logand (Int64.mul t1 mult) mask32 in
        let t3 = Int64.logand (Int64.add t2 (Int64.rem l span)) mask32 in
        let off = Int64.rem t3 span in
        float_of_int (minval + Int64.to_int off))
  in
  Concrete (Nd.of_floats D.I32 shape out)

let randint64 k ~shape ~minval ~maxval =
  let keys = concrete (PR.random_split (wrap k) [| 2 |]) in
  let subkey j =
    Nd.of_floats D.Uint32 [| 2 |]
      [|
        Int64.to_float (Nd.get_i64 keys [| j; 0 |]);
        Int64.to_float (Nd.get_i64 keys [| j; 1 |]);
      |]
  in
  let higher = TF.bits64 (subkey 0) shape in
  let lower = TF.bits64 (subkey 1) shape in
  let minv = Int64.of_int minval and maxv = Int64.of_int maxval in
  let span = if maxv <= minv then 1L else Int64.sub maxv minv in
  let urem = Int64.unsigned_rem in
  let m = urem 0x100000000L span in
  let m = urem (Int64.mul m m) span in
  Concrete
    (Nd.of_int64s D.I64 shape
       (Array.mapi
          (fun i h ->
            let offset =
              Int64.add (Int64.mul (urem h span) m) (urem lower.(i) span)
            in
            Int64.add minv (urem offset span))
          higher))

let randint k ~shape ~minval ~maxval =
  match idt () with
  | D.I64 -> randint64 k ~shape ~minval ~maxval
  | _ -> randint_base k ~shape ~minval ~maxval

let const_like_value v x =
  let a = C.get_aval v in
  const a.dtype a.shape x

let uint32max = 4294967295.0

let permutation k n =
  let dt = Dtypes.default_int_dtype () in
  let x0 = C.bind1 (Iota { dtype = dt; shape = [| n |]; dimension = 0 }) [] in
  let rounds =
    int_of_float (ceil (3.0 *. log (float_of_int (max 1 n)) /. log uint32max))
  in
  let subkey keys j =
    Concrete
      (Nd.of_floats D.Uint32 [| 2 |]
         [|
           Int64.to_float (Nd.get_i64 keys [| j; 0 |]);
           Int64.to_float (Nd.get_i64 keys [| j; 1 |]);
         |])
  in
  let rec loop cur x r =
    if r = 0 then x
    else
      let keys = concrete (PR.random_split (wrap cur) [| 2 |]) in
      let nextk = subkey keys 0 and subk = subkey keys 1 in
      let sort_keys = PR.random_bits (wrap subk) ~bit_width:32 ~shape:[| n |] in
      let outs =
        C.bind
          (Sort { dimension = 0; is_stable = true; num_keys = 1 })
          [ sort_keys; x ]
      in
      loop nextk (List.nth outs 1) (r - 1)
  in
  loop k x0 rounds

let choice k ~n ~shape ~replace =
  let dt = Dtypes.default_int_dtype () in
  let n_draws = prod shape in
  if n_draws = 0 then Concrete (Nd.of_floats dt shape [||])
  else if replace then
    C.bind1 (Convert_element_type dt) [ randint k ~shape ~minval:0 ~maxval:n ]
  else
    let perm = permutation k n in
    let sliced =
      C.bind1
        (Slice
           {
             start_indices = [| 0 |];
             limit_indices = [| n_draws |];
             strides = None;
           })
        [ perm ]
    in
    C.bind1 (Reshape shape) [ sliced ]

let split_n k num =
  let keys = concrete (PR.random_split (wrap k) [| num |]) in
  Array.init num (fun j ->
      Concrete
        (Nd.of_floats D.Uint32 [| 2 |]
           [|
             Int64.to_float (Nd.get_i64 keys [| j; 0 |]);
             Int64.to_float (Nd.get_i64 keys [| j; 1 |]);
           |]))

let split2 k =
  let a = split_n k 2 in
  (a.(0), a.(1))

let s x = const (fdt ()) [||] x
let cf shape x = const (fdt ()) shape x
let e_add a b = C.bind1 Add [ a; b ]
let e_sub a b = C.bind1 Sub [ a; b ]
let e_mul a b = C.bind1 Mul [ a; b ]
let e_div a b = C.bind1 Div [ a; b ]
let e_neg a = C.bind1 Neg [ a ]
let e_log a = C.bind1 Log [ a ]
let e_log1p a = C.bind1 Log1p [ a ]
let e_exp a = C.bind1 Exp [ a ]
let e_sqrt a = C.bind1 Sqrt [ a ]
let e_abs a = C.bind1 Abs [ a ]
let e_sign a = C.bind1 Sign [ a ]
let e_tan a = C.bind1 Tan [ a ]
let e_floor a = C.bind1 Floor [ a ]
let e_pow a b = C.bind1 Pow [ a; b ]
let e_ipow a n = C.bind1 (Integer_pow n) [ a ]
let e_lt a b = C.bind1 Lt [ a; b ]
let e_le a b = C.bind1 Le [ a; b ]
let e_eq a b = C.bind1 Eq [ a; b ]
let e_max a b = C.bind1 Max [ a; b ]

let e_select pred on_true on_false =
  C.bind1 Select_n [ pred; on_false; on_true ]

let is_double () = fdt () = D.F64
let float_eps () = if is_double () then 2.0 ** -52.0 else 2.0 ** -23.0
let float_epsneg () = if is_double () then 2.0 ** -53.0 else 2.0 ** -24.0
let float_tiny () = if is_double () then 2.0 ** -1022.0 else 2.0 ** -126.0
let float_pi () = f32 Float.pi
let default_int () = Dtypes.default_int_dtype ()

let softmax_last v full =
  let last = Array.length full - 1 in
  let dims = Array.init last (fun i -> i) in
  let m = C.bind1 (Reduce_max [| last |]) [ v ] in
  let m_b = C.bind1 (Broadcast_in_dim { shape = full; dims }) [ m ] in
  let e = e_exp (e_sub v m_b) in
  let total = C.bind1 (Reduce_sum [| last |]) [ e ] in
  let total_b = C.bind1 (Broadcast_in_dim { shape = full; dims }) [ total ] in
  e_div e total_b

let unif01 k ~shape = uniform k ~shape ~minval:0.0 ~maxval:1.0

let exponential k ~shape =
  let u = unif01 k ~shape in
  e_neg (e_log1p (e_neg u))

let exponential_s k = e_neg (e_log1p (e_neg (unif01 k ~shape:[||])))

let cauchy k ~shape =
  let u = uniform k ~shape ~minval:(float_eps ()) ~maxval:1.0 in
  e_tan (e_mul (cf shape (float_pi ())) (e_sub u (cf shape 0.5)))

let laplace k ~shape =
  let u = uniform k ~shape ~minval:(-1.0 +. float_epsneg ()) ~maxval:1.0 in
  e_mul (e_sign u) (e_log1p (e_neg (e_abs u)))

let logistic k ~shape =
  let x = uniform k ~shape ~minval:(float_tiny ()) ~maxval:1.0 in
  e_sub (e_log x) (e_log1p (e_neg x))

let gumbel k ~shape =
  let u = uniform k ~shape ~minval:(float_tiny ()) ~maxval:1.0 in
  e_neg (e_log (e_neg (e_log u)))

let pareto k ~shape ~b =
  let e = exponential k ~shape in
  e_exp (e_div e (cf shape b))

let rayleigh k ~shape ~scale =
  let u = unif01 k ~shape in
  e_mul (cf shape scale) (e_sqrt (e_mul (e_log u) (cf shape (-2.0))))

let weibull_min k ~shape ~scale ~concentration =
  let u = unif01 k ~shape in
  e_mul
    (e_pow (e_neg (e_log1p (e_neg u))) (cf shape (f32 (1.0 /. concentration))))
    (cf shape scale)

let lognormal k ~shape ~sigma = e_exp (e_mul (normal k ~shape) (cf shape sigma))

let triangular k ~shape ~left ~mode ~right =
  let l = cf shape left and r = cf shape right and m = cf shape mode in
  let fc = e_div (e_sub m l) (e_sub r l) in
  let u = unif01 k ~shape in
  let out1 = e_add l (e_sqrt (e_mul (e_mul u (e_sub r l)) (e_sub m l))) in
  let out2 =
    e_sub r
      (e_sqrt (e_mul (e_mul (e_sub (cf shape 1.0) u) (e_sub r l)) (e_sub r m)))
  in
  e_select (e_lt u fc) out1 out2

let wald k ~shape ~mean =
  let k1, k2 = split2 k in
  let mn = cf shape mean in
  let v = normal k1 ~shape in
  let z = unif01 k2 ~shape in
  let y = e_ipow v 2 in
  let y_sq = e_ipow y 2 in
  let mean_sq = e_ipow mn 2 in
  let sqrt_term =
    e_sqrt (e_add (e_mul (e_mul (cf shape 4.0) mn) y) (e_mul mean_sq y_sq))
  in
  let x =
    e_sub
      (e_add mn (e_div (e_mul mean_sq y) (cf shape 2.0)))
      (e_mul (e_div mn (cf shape 2.0)) sqrt_term)
  in
  e_select (e_le z (e_div mn (e_add mn x))) x (e_div mean_sq x)

let geometric k ~shape ~p =
  let u0 = unif01 k ~shape in
  let u = e_select (e_eq u0 (cf shape 0.0)) (cf shape 1.0) u0 in
  let log_u = e_log u in
  let log1mp = e_log1p (e_neg (cf shape p)) in
  let g = e_add (e_floor (e_div log_u log1mp)) (cf shape 1.0) in
  C.bind1 (Convert_element_type (default_int ())) [ g ]

let bernoulli k ~shape ~p =
  let u = unif01 k ~shape in
  e_lt u (cf shape p)

let rademacher_f k ~shape =
  let u = unif01 k ~shape in
  let b = C.bind1 (Convert_element_type (fdt ())) [ e_lt u (cf shape 0.5) ] in
  e_sub (e_mul (cf shape 2.0) b) (cf shape 1.0)

let rademacher k ~shape =
  C.bind1 (Convert_element_type (default_int ())) [ rademacher_f k ~shape ]

let categorical k ~logits ~axis =
  let ls = Nd.shape (concrete logits) in
  let ndim = Array.length ls in
  let ax = if axis < 0 then axis + ndim else axis in
  let g = gumbel k ~shape:ls in
  let scores = e_add g logits in
  C.bind1 (Argmax { axis = ax; index_dtype = default_int () }) [ scores ]

let gamma_one key alpha_f log_space =
  let boost = alpha_f >= 1.0 in
  let alpha = if boost then s alpha_f else s (alpha_f +. 1.0) in
  let one = s 1.0 in
  let third = s (1.0 /. 3.0) in
  let d = e_sub alpha third in
  let c = e_div third (e_sqrt d) in
  let squeeze = s 0.0331 in
  let half = s 0.5 in
  let rec inner ikey =
    let sub = split_n ikey 2 in
    let x = normal sub.(1) ~shape:[||] in
    let v = e_add one (e_mul x c) in
    if scalar_f v <= 0.0 then inner sub.(0) else (x, v)
  in
  let rec loop lkey =
    let sub = split_n lkey 3 in
    let x, v = inner sub.(1) in
    let big_x = e_mul x x in
    let big_v = e_mul (e_mul v v) v in
    let big_u = unif01 sub.(2) ~shape:[||] in
    let c1 =
      scalar_f big_u >= scalar_f (e_sub one (e_mul squeeze (e_mul big_x big_x)))
    in
    let c2 =
      scalar_f (e_log big_u)
      >= scalar_f
           (e_add (e_mul big_x half)
              (e_mul d (e_add (e_sub one big_v) (e_log big_v))))
    in
    if c1 && c2 then loop sub.(0) else big_v
  in
  let sub0 = split_n key 2 in
  let big_v = loop sub0.(0) in
  if log_space then
    let ls = e_neg (exponential_s sub0.(1)) in
    let log_boost =
      if boost || scalar_f ls = 0.0 then s 0.0
      else e_mul ls (e_div one (s alpha_f))
    in
    e_add (e_add (e_log d) (e_log big_v)) log_boost
  else
    let samp = e_sub one (unif01 sub0.(1) ~shape:[||]) in
    let bst = if boost then one else e_pow samp (e_div one (s alpha_f)) in
    e_mul (e_mul d big_v) bst

let gamma_arr key ~shape ~alphas ~log_space =
  let num = prod shape in
  let keys = split_n key num in
  let out =
    Array.init num (fun i -> scalar_f (gamma_one keys.(i) alphas.(i) log_space))
  in
  Concrete (Nd.of_floats (fdt ()) shape out)

let const_alphas shape a = Array.make (prod shape) a

let gamma k ~shape ~a =
  gamma_arr k ~shape ~alphas:(const_alphas shape a) ~log_space:false

let loggamma k ~shape ~a =
  gamma_arr k ~shape ~alphas:(const_alphas shape a) ~log_space:true

let beta k ~shape ~a ~b =
  let ka, kb = split2 k in
  let lga =
    gamma_arr ka ~shape ~alphas:(const_alphas shape a) ~log_space:true
  in
  let lgb =
    gamma_arr kb ~shape ~alphas:(const_alphas shape b) ~log_space:true
  in
  let log_max = e_max lga lgb in
  let ga = e_exp (e_sub lga log_max) in
  let gb = e_exp (e_sub lgb log_max) in
  e_div ga (e_add ga gb)

let chisquare k ~shape ~df =
  let half = df /. 2.0 in
  let lg =
    gamma_arr k ~shape ~alphas:(const_alphas shape half) ~log_space:true
  in
  e_mul (e_exp lg) (cf shape 2.0)

let t k ~shape ~df =
  let kn, kg = split2 k in
  let n = normal kn ~shape in
  let half = df /. 2.0 in
  let g =
    gamma_arr kg ~shape ~alphas:(const_alphas shape half) ~log_space:false
  in
  e_mul n (e_sqrt (e_div (cf shape (f32 half)) g))

let f k ~shape ~dfnum ~dfden =
  let kd, kn = split2 k in
  let chi2_dfn = chisquare kn ~shape ~df:dfnum in
  let chi2_dfd = chisquare kd ~shape ~df:dfden in
  let num = e_div chi2_dfn (cf shape dfnum) in
  let den = e_div chi2_dfd (cf shape dfden) in
  e_div num den

let generalized_normal k ~shape ~p =
  let keys = split_n k 2 in
  let g =
    gamma_arr keys.(0) ~shape
      ~alphas:(const_alphas shape (1.0 /. p))
      ~log_space:false
  in
  let r = rademacher_f keys.(1) ~shape in
  e_mul r (e_pow g (cf shape (f32 (1.0 /. p))))

let dirichlet k ~alpha ~shape =
  let alpha_nd = concrete alpha in
  let ash = Nd.shape alpha_nd in
  let n = ash.(Array.length ash - 1) in
  let full = Array.append shape [| n |] in
  let num = prod full in
  let alphas =
    Array.init num (fun i -> Nd.get_f alpha_nd (unravel ash (i mod n)))
  in
  let lg = gamma_arr k ~shape:full ~alphas ~log_space:true in
  softmax_last lg full

let lgamma_host x = scalar_f (C.bind1 Lgamma [ const (fdt ()) [||] x ])

let poisson_rejection k ~shape ~lams =
  let n = prod shape in
  let lams = Array.map f32 lams in
  let log_lam = Array.map (fun l -> f32 (log l)) lams in
  let b = Array.map (fun l -> f32 (0.931 +. (2.53 *. f32 (sqrt l)))) lams in
  let a = Array.map (fun bi -> f32 (-0.059 +. (0.02483 *. bi))) b in
  let inv_alpha =
    Array.map (fun bi -> f32 (1.1239 +. (1.1328 /. (bi -. 3.4)))) b
  in
  let v_r = Array.map (fun bi -> f32 (0.9277 -. (3.6224 /. (bi -. 2.0)))) b in
  let out = Array.make n (-1.0) in
  let accepted = Array.make n false in
  let key = ref k in
  while Array.exists not accepted do
    let keys = split_n !key 3 in
    key := keys.(0);
    let u = concrete (unif01 keys.(1) ~shape) in
    let v = concrete (unif01 keys.(2) ~shape) in
    for i = 0 to n - 1 do
      let ui = f32 (Nd.get_f u (unravel shape i) -. 0.5) in
      let vi = Nd.get_f v (unravel shape i) in
      let us = f32 (0.5 -. Float.abs ui) in
      let cand =
        Float.floor
          (f32 ((((2.0 *. a.(i) /. us) +. b.(i)) *. ui) +. lams.(i) +. 0.43))
      in
      let s_ =
        f32 (log (vi *. inv_alpha.(i) /. ((a.(i) /. (us *. us)) +. b.(i))))
      in
      let t_ =
        f32 (-.lams.(i) +. (cand *. log_lam.(i)) -. lgamma_host (cand +. 1.0))
      in
      let accept1 = us >= 0.07 && vi <= v_r.(i) in
      let reject = cand < 0.0 || (us < 0.013 && vi > us) in
      let accept = accept1 || ((not reject) && s_ <= t_) in
      if accept then out.(i) <- cand;
      if accept then accepted.(i) <- true
    done
  done;
  out

let stirling_tail x =
  let table =
    [|
      0.0810614667953272;
      0.0413406959554092;
      0.0276779256849983;
      0.02079067210376509;
      0.0166446911898211;
      0.0138761288230707;
      0.0118967099458917;
      0.0104112652619720;
      0.00925546218271273;
      0.00833056343336287;
    |]
  in
  if x < 10.0 && x >= 0.0 then table.(int_of_float x)
  else
    let kp1 = x +. 1.0 in
    let sq = kp1 *. kp1 in
    ((1.0 /. 12.0) -. (((1.0 /. 360.0) -. (1.0 /. 1260.0 /. sq)) /. sq)) /. kp1

let btrs k ~shape ~counts ~probs =
  let n = prod shape in
  let stddev =
    Array.mapi
      (fun i c -> f32 (sqrt (c *. probs.(i) *. (1.0 -. probs.(i)))))
      counts
  in
  let b = Array.map (fun d -> f32 (1.15 +. (2.53 *. d))) stddev in
  let a =
    Array.mapi
      (fun i bi -> f32 (-0.0873 +. (0.0248 *. bi) +. (0.01 *. probs.(i))))
      b
  in
  let c = Array.mapi (fun i ci -> f32 ((ci *. probs.(i)) +. 0.5)) counts in
  let v_r = Array.map (fun bi -> f32 (0.92 -. (4.2 /. bi))) b in
  let r = Array.map (fun p -> f32 (p /. (1.0 -. p))) probs in
  let alpha =
    Array.mapi (fun i bi -> f32 ((2.83 +. (5.1 /. bi)) *. stddev.(i))) b
  in
  let m =
    Array.mapi (fun i ci -> Float.floor (f32 ((ci +. 1.0) *. probs.(i)))) counts
  in
  let out = Array.make n (-1.0) in
  let accepted = Array.make n false in
  let key = ref k in
  while Array.exists not accepted do
    let keys = split_n !key 3 in
    key := keys.(0);
    let u = concrete (unif01 keys.(1) ~shape) in
    let v = concrete (unif01 keys.(2) ~shape) in
    for i = 0 to n - 1 do
      let count = counts.(i) and mi = m.(i) and ri = r.(i) in
      let ui = f32 (Nd.get_f u (unravel shape i) -. 0.5) in
      let vi = Nd.get_f v (unravel shape i) in
      let us = f32 (0.5 -. Float.abs ui) in
      let accept1 = us >= 0.07 && vi <= v_r.(i) in
      let cand =
        Float.floor (f32 ((((2.0 *. a.(i) /. us) +. b.(i)) *. ui) +. c.(i)))
      in
      let reject = cand < 0.0 || cand > count in
      let logv =
        f32 (log (vi *. alpha.(i) /. ((a.(i) /. (us *. us)) +. b.(i))))
      in
      let ub =
        ((mi +. 0.5) *. log ((mi +. 1.0) /. (ri *. (count -. mi +. 1.0))))
        +. (count +. 1.0)
           *. log ((count -. mi +. 1.0) /. (count -. cand +. 1.0))
        +. ((cand +. 0.5) *. log (ri *. (count -. cand +. 1.0) /. (cand +. 1.0)))
        +. stirling_tail mi
        +. stirling_tail (count -. mi)
        -. stirling_tail cand
        -. stirling_tail (count -. cand)
      in
      let accept = accept1 || ((not reject) && logv <= f32 ub) in
      if accept then out.(i) <- cand;
      if accept then accepted.(i) <- true
    done
  done;
  out

let poisson_knuth k ~shape ~lams =
  let n = prod shape in
  let k_arr = Array.make n 0 in
  let logp = Array.make n 0.0 in
  let neg_lam = Array.map (fun l -> f32 (-.l)) lams in
  let key = ref k in
  let active () =
    let any = ref false in
    Array.iteri (fun i lp -> if lp > neg_lam.(i) then any := true) logp;
    !any
  in
  while active () do
    let sub = split_n !key 2 in
    key := sub.(0);
    let u = concrete (unif01 sub.(1) ~shape) in
    for i = 0 to n - 1 do
      if logp.(i) > neg_lam.(i) then k_arr.(i) <- k_arr.(i) + 1;
      let ui = Nd.get_f u (unravel shape i) in
      logp.(i) <- f32 (logp.(i) +. f32 (log ui))
    done
  done;
  Array.map (fun c -> float_of_int (c - 1)) k_arr

let poisson_arr k ~shape ~lams =
  with_dtypes ~float_dtype:D.F32 (fun () ->
      let use_knuth = Array.map (fun l -> l < 10.0) lams in
      let any_knuth = Array.exists Fun.id use_knuth in
      let any_rejection = Array.exists not use_knuth in
      let knuth =
        if any_knuth then
          poisson_knuth k ~shape
            ~lams:
              (Array.mapi (fun i l -> if use_knuth.(i) then l else 0.0) lams)
        else [||]
      in
      let rejection =
        if any_rejection then
          poisson_rejection k ~shape
            ~lams:
              (Array.mapi
                 (fun i l -> if use_knuth.(i) then 100000.0 else l)
                 lams)
        else [||]
      in
      let out =
        Array.mapi
          (fun i l ->
            if l = 0.0 then 0.0
            else if use_knuth.(i) then knuth.(i)
            else rejection.(i))
          lams
      in
      Concrete (Nd.of_floats (default_int ()) shape out))

let poisson k ~shape ~lam =
  poisson_arr k ~shape ~lams:(Array.make (prod shape) lam)

let binomial_arr key ~shape ~counts ~probs =
  let n = prod shape in
  let q =
    Array.init n (fun i ->
        f32 (if probs.(i) < 0.5 then probs.(i) else 1.0 -. probs.(i)))
  in
  let cnt = Array.map (fun c -> f32 (Float.floor c)) counts in
  let use_inv = Array.mapi (fun i c -> c *. q.(i) <= 10.0) cnt in
  let btrs_samples =
    if Array.exists not use_inv then
      btrs key ~shape
        ~counts:(Array.mapi (fun i c -> if use_inv.(i) then 10000.0 else c) cnt)
        ~probs:(Array.mapi (fun i p -> if use_inv.(i) then 0.5 else p) q)
    else [||]
  in
  let cnt_inv = Array.mapi (fun i c -> if use_inv.(i) then c else 0.0) cnt in
  let cnt_all = cnt in
  let cnt = cnt_inv in
  let num_geom = Array.make n 0 in
  let geom_sum = Array.make n 0.0 in
  let log1mp = Array.map (fun qi -> f32 (log1p (-.qi))) q in
  let key = ref key in
  let active () =
    let r = ref false in
    for i = 0 to n - 1 do
      if geom_sum.(i) <= cnt.(i) then r := true
    done;
    !r
  in
  while active () do
    let sub = split_n !key 2 in
    key := sub.(1);
    let u = concrete (unif01 sub.(0) ~shape) in
    for i = 0 to n - 1 do
      if geom_sum.(i) <= cnt.(i) then num_geom.(i) <- num_geom.(i) + 1;
      let ui = Nd.get_f u (unravel shape i) in
      let geom = Float.ceil (f32 (f32 (log ui) /. log1mp.(i))) in
      geom_sum.(i) <- f32 (geom_sum.(i) +. geom)
    done
  done;
  Array.init n (fun i ->
      let samp =
        if use_inv.(i) then float_of_int (num_geom.(i) - 1)
        else btrs_samples.(i)
      in
      if probs.(i) < 0.5 then samp else cnt_all.(i) -. samp)

let binomial k ~shape ~count ~prob =
  let n = prod shape in
  let out =
    binomial_arr k ~shape ~counts:(Array.make n count)
      ~probs:(Array.make n prob)
  in
  Concrete (Nd.of_floats (fdt ()) shape out)

let multinomial k ~p ~n_trials =
  let p_nd = concrete p in
  let psh = Nd.shape p_nd in
  let kk = psh.(Array.length psh - 1) in
  let remaining =
    concrete (C.bind1 (Cumsum { axis = 0; reverse = true }) [ p ])
  in
  let ratios =
    Array.init kk (fun j ->
        let rem = Nd.get_f remaining [| j |] in
        let pj = Nd.get_f p_nd [| j |] in
        f32 (pj /. if rem = 0.0 then 1.0 else rem))
  in
  let keys = split_n k kk in
  let counts = Array.make kk 0.0 in
  let remainder = ref (f32 n_trials) in
  for j = 0 to kk - 1 do
    let ratio = ratios.(j) in
    let ratio_c =
      if ratio < 0.0 then 0.0 else if ratio > 1.0 then 1.0 else ratio
    in
    let cj =
      (binomial_arr keys.(j) ~shape:[||] ~counts:[| !remainder |]
         ~probs:[| ratio_c |]).(0)
    in
    counts.(j) <- cj;
    remainder := f32 (!remainder -. cj)
  done;
  Concrete (Nd.of_floats (fdt ()) [| kk |] counts)

let is_tracer = function Tracer _ -> true | Concrete _ | Device _ -> false
let sampler_key_aval key = C.get_aval key

let host_key_of_aval (a : aval) =
  let n = prod a.shape in
  Concrete (Nd.of_floats a.dtype a.shape (Array.make n 0.0))

let via_host f key =
  try f key
  with Failure msg when String.starts_with ~prefix:"random/core" msg ->
    if not (is_tracer key) then C.with_eval_dynamic (fun () -> f key)
    else
      let probe =
        C.with_eval_dynamic (fun () ->
            f (host_key_of_aval (sampler_key_aval key)))
      in
      let result_avals = [ C.get_aval probe ] in
      let host = function
        | [ k ] -> [ f k ]
        | _ -> failwith "random/core: callback arity"
      in
      List.hd (Custom_derivatives.pure_callback host ~result_avals [ key ])

let randint k ~shape ~minval ~maxval =
  via_host (fun k -> randint k ~shape ~minval ~maxval) k

let uniform k ~shape ~minval ~maxval =
  via_host (fun k -> uniform k ~shape ~minval ~maxval) k

let normal k ~shape = via_host (fun k -> normal k ~shape) k

let truncated_normal k ~lower ~upper ~shape =
  via_host (fun k -> truncated_normal k ~lower ~upper ~shape) k

let permutation k n = via_host (fun k -> permutation k n) k

let choice k ~n ~shape ~replace =
  via_host (fun k -> choice k ~n ~shape ~replace) k

let exponential k ~shape = via_host (fun k -> exponential k ~shape) k
let cauchy k ~shape = via_host (fun k -> cauchy k ~shape) k
let laplace k ~shape = via_host (fun k -> laplace k ~shape) k
let logistic k ~shape = via_host (fun k -> logistic k ~shape) k
let gumbel k ~shape = via_host (fun k -> gumbel k ~shape) k
let bernoulli k ~shape ~p = via_host (fun k -> bernoulli k ~shape ~p) k
let rademacher k ~shape = via_host (fun k -> rademacher k ~shape) k

let categorical k ~logits ~axis =
  via_host (fun k -> categorical k ~logits ~axis) k

let gamma k ~shape ~a = via_host (fun k -> gamma k ~shape ~a) k
let loggamma k ~shape ~a = via_host (fun k -> loggamma k ~shape ~a) k
let beta k ~shape ~a ~b = via_host (fun k -> beta k ~shape ~a ~b) k
let chisquare k ~shape ~df = via_host (fun k -> chisquare k ~shape ~df) k
let t k ~shape ~df = via_host (fun k -> t k ~shape ~df) k
let f k ~shape ~dfnum ~dfden = via_host (fun k -> f k ~shape ~dfnum ~dfden) k

let generalized_normal k ~shape ~p =
  via_host (fun k -> generalized_normal k ~shape ~p) k

let dirichlet k ~alpha ~shape = via_host (fun k -> dirichlet k ~alpha ~shape) k
let poisson k ~shape ~lam = via_host (fun k -> poisson k ~shape ~lam) k
let pareto k ~shape ~b = via_host (fun k -> pareto k ~shape ~b) k
let rayleigh k ~shape ~scale = via_host (fun k -> rayleigh k ~shape ~scale) k

let weibull_min k ~shape ~scale ~concentration =
  via_host (fun k -> weibull_min k ~shape ~scale ~concentration) k

let lognormal k ~shape ~sigma = via_host (fun k -> lognormal k ~shape ~sigma) k

let triangular k ~shape ~left ~mode ~right =
  via_host (fun k -> triangular k ~shape ~left ~mode ~right) k

let wald k ~shape ~mean = via_host (fun k -> wald k ~shape ~mean) k
let geometric k ~shape ~p = via_host (fun k -> geometric k ~shape ~p) k

let binomial k ~shape ~count ~prob =
  via_host (fun k -> binomial k ~shape ~count ~prob) k

let multinomial k ~p ~n_trials =
  via_host (fun k -> multinomial k ~p ~n_trials) k

let bernoulli_prob k ~shape ~p =
  let float_dtype = (C.get_aval p).dtype in
  with_dtypes ~float_dtype (fun () ->
      e_lt (via_host (fun k -> unif01 k ~shape) k) p)

let uniform_range k ~shape ~minval ~maxval =
  let dt = C.get_aval minval |> fun a -> a.dtype in
  with_dtypes ~float_dtype:dt (fun () ->
      let u = uniform k ~shape ~minval:0.0 ~maxval:1.0 in
      let y = e_add (e_mul u (e_sub maxval minval)) minval in
      e_max minval y)

let value_floats v =
  let nd = concrete v in
  Array.of_list (List.rev (Nd.fold (fun acc x -> x :: acc) [] nd))

let host_gamma ~log_space key alpha =
  let a = C.get_aval alpha in
  with_dtypes ~float_dtype:a.dtype (fun () ->
      gamma_arr key ~shape:a.shape ~alphas:(value_floats alpha) ~log_space)

let host_gamma_call ~log_space args =
  match args with
  | [ key; alpha ] ->
      if is_tracer key || is_tracer alpha then
        Custom_derivatives.pure_callback
          (function
            | [ k; al ] -> [ host_gamma ~log_space k al ]
            | _ -> failwith "random/core: gamma callback arity")
          ~result_avals:[ C.get_aval alpha ]
          args
      else [ C.with_eval_dynamic (fun () -> host_gamma ~log_space key alpha) ]
  | _ -> failwith "random/core: gamma arity"

let random_gamma_grad alpha sample =
  let log_pdf =
    e_sub
      (e_sub
         (e_mul (e_sub alpha (const_like_value alpha 1.0)) (e_log sample))
         sample)
      (C.bind1 Lgamma [ alpha ])
  in
  e_neg (e_div (C.bind1 Igamma_grad_a [ alpha; sample ]) (e_exp log_pdf))

let gamma_value_rule =
  Custom_derivatives.custom_jvp (host_gamma_call ~log_space:false)
    ~jvp:(fun primals tangents ->
      match (primals, tangents) with
      | [ _; alpha ], [ _; ta ] ->
          let out = host_gamma_call ~log_space:false primals in
          let z = List.hd out in
          (out, [ e_mul (random_gamma_grad alpha z) ta ])
      | _ -> failwith "random/core: gamma jvp arity")

let loggamma_value_rule =
  Custom_derivatives.custom_jvp (host_gamma_call ~log_space:true)
    ~jvp:(fun primals tangents ->
      match (primals, tangents) with
      | [ _; alpha ], [ _; ta ] ->
          let out = host_gamma_call ~log_space:true primals in
          let lz = List.hd out in
          let z = e_exp lz in
          (out, [ e_mul (e_div (random_gamma_grad alpha z) z) ta ])
      | _ -> failwith "random/core: loggamma jvp arity")

let broadcast_param v shape =
  let a = C.get_aval v in
  if a.shape = shape then v
  else
    let rank = Array.length shape and r = Array.length a.shape in
    C.bind1
      (Broadcast_in_dim { shape; dims = Array.init r (fun i -> rank - r + i) })
      [ v ]

let gamma_value k ~shape ~a =
  List.hd (gamma_value_rule [ k; broadcast_param a shape ])

let loggamma_value k ~shape ~a =
  List.hd (loggamma_value_rule [ k; broadcast_param a shape ])

let beta_value k ~shape ~a ~b =
  let keys = split k 2 in
  let pick i =
    C.bind1 (Reshape [| 2 |])
      [
        C.bind1
          (Slice
             {
               start_indices = [| i; 0 |];
               limit_indices = [| i + 1; 2 |];
               strides = None;
             })
          [ keys ];
      ]
  in
  let la = loggamma_value (pick 0) ~shape ~a in
  let lb = loggamma_value (pick 1) ~shape ~a:b in
  let top = e_max la lb in
  let ga = e_exp (e_sub la top) and gb = e_exp (e_sub lb top) in
  e_div ga (e_add ga gb)

let poisson_value k ~shape ~lam =
  let host = function
    | [ key; l ] -> [ poisson_arr key ~shape ~lams:(value_floats l) ]
    | _ -> failwith "random/core: poisson arity"
  in
  let lam = broadcast_param lam shape in
  let args = [ k; lam ] in
  if is_tracer k || is_tracer lam then
    List.hd
      (Custom_derivatives.pure_callback host
         ~result_avals:[ { shape; dtype = default_int (); weak_type = false } ]
         args)
  else List.hd (C.with_eval_dynamic (fun () -> host args))

let bits_as k ~shape ~dtype =
  let dtype = Dtypes.canonicalize_dtype dtype in
  let host key =
    match dtype with
    | D.Uint64 ->
        Concrete (Nd.of_int64s D.Uint64 shape (TF.bits64 (concrete key) shape))
    | D.Uint32 -> PR.random_bits (wrap key) ~bit_width:32 ~shape
    | D.Uint8 | D.Uint16 ->
        let wide = concrete (PR.random_bits (wrap key) ~bit_width:32 ~shape) in
        let modulus = if dtype = D.Uint8 then 256 else 65536 in
        Concrete
          (Nd.of_floats dtype shape
             (Array.init (prod shape) (fun i ->
                  float_of_int (geti wide i mod modulus))))
    | _ -> invalid_arg "random.bits requires an unsigned integer dtype"
  in
  via_host host k

let is_key = Key_reuse.is_typed

let legacy_message =
  "Legacy uint32 key array passed as key to jax.random function. Please create \
   keys using jax.random.key(). If use of a raw key array was intended, set \
   jax_legacy_prng_key=\"allow\"."

let check_prng_key k =
  if Key_reuse.known_raw k then
    match Config.value Config.legacy_prng_key with
    | "error" -> invalid_arg legacy_message
    | "warn" -> Config.log legacy_message
    | _ ->
        if Config.value Config.enable_custom_prng then
          Config.log
            "Raw arrays as random keys to jax.random functions are deprecated. \
             Assuming valid threefry2x32 key for now."

let sampler ?(name = "jit-compiled function") k thunk =
  check_prng_key k;
  Key_reuse.consume name k;
  Key_reuse.suspended thunk

let upgrade_keys out =
  if Config.value Config.enable_custom_prng && not (Key_reuse.is_typed out) then
    Key_reuse.mark_typed out
  else out

let key_impl k =
  check_prng_key k;
  key_impl k

let key_data k =
  check_prng_key k;
  key_data k

let clone k =
  check_prng_key k;
  clone k

let split k num =
  check_prng_key k;
  upgrade_keys (split k num)

let fold_in k data =
  check_prng_key k;
  upgrade_keys (fold_in k data)

let prng_key seed =
  let raw = PR.random_seed seed in
  if Config.value Config.enable_custom_prng then Key_reuse.mark_typed raw
  else raw

let bits k ~shape = sampler ~name:"random_bits" k (fun () -> bits k ~shape)

let randint k ~shape ~minval ~maxval =
  sampler k (fun () -> randint k ~shape ~minval ~maxval)

let uniform k ~shape ~minval ~maxval =
  sampler k (fun () -> uniform k ~shape ~minval ~maxval)

let normal k ~shape = sampler k (fun () -> normal k ~shape)

let truncated_normal k ~lower ~upper ~shape =
  sampler k (fun () -> truncated_normal k ~lower ~upper ~shape)

let permutation k x0 = sampler k (fun () -> permutation k x0)

let choice k ~n ~shape ~replace =
  sampler k (fun () -> choice k ~n ~shape ~replace)

let exponential k ~shape = sampler k (fun () -> exponential k ~shape)
let cauchy k ~shape = sampler k (fun () -> cauchy k ~shape)
let laplace k ~shape = sampler k (fun () -> laplace k ~shape)
let logistic k ~shape = sampler k (fun () -> logistic k ~shape)
let gumbel k ~shape = sampler k (fun () -> gumbel k ~shape)
let pareto k ~shape ~b = sampler k (fun () -> pareto k ~shape ~b)
let rayleigh k ~shape ~scale = sampler k (fun () -> rayleigh k ~shape ~scale)

let weibull_min k ~shape ~scale ~concentration =
  sampler k (fun () -> weibull_min k ~shape ~scale ~concentration)

let lognormal k ~shape ~sigma = sampler k (fun () -> lognormal k ~shape ~sigma)

let triangular k ~shape ~left ~mode ~right =
  sampler k (fun () -> triangular k ~shape ~left ~mode ~right)

let wald k ~shape ~mean = sampler k (fun () -> wald k ~shape ~mean)
let geometric k ~shape ~p = sampler k (fun () -> geometric k ~shape ~p)
let bernoulli k ~shape ~p = sampler k (fun () -> bernoulli k ~shape ~p)
let rademacher k ~shape = sampler k (fun () -> rademacher k ~shape)

let categorical k ~logits ~axis =
  sampler k (fun () -> categorical k ~logits ~axis)

let gamma k ~shape ~a = sampler k (fun () -> gamma k ~shape ~a)
let loggamma k ~shape ~a = sampler k (fun () -> loggamma k ~shape ~a)
let beta k ~shape ~a ~b = sampler k (fun () -> beta k ~shape ~a ~b)
let chisquare k ~shape ~df = sampler k (fun () -> chisquare k ~shape ~df)
let t k ~shape ~df = sampler k (fun () -> t k ~shape ~df)
let f k ~shape ~dfnum ~dfden = sampler k (fun () -> f k ~shape ~dfnum ~dfden)

let generalized_normal k ~shape ~p =
  sampler k (fun () -> generalized_normal k ~shape ~p)

let dirichlet k ~alpha ~shape = sampler k (fun () -> dirichlet k ~alpha ~shape)
let poisson k ~shape ~lam = sampler k (fun () -> poisson k ~shape ~lam)

let binomial k ~shape ~count ~prob =
  sampler k (fun () -> binomial k ~shape ~count ~prob)

let multinomial k ~p ~n_trials =
  sampler k (fun () -> multinomial k ~p ~n_trials)

let bernoulli_prob k ~shape ~p =
  sampler k (fun () -> bernoulli_prob k ~shape ~p)

let uniform_range k ~shape ~minval ~maxval =
  sampler k (fun () -> uniform_range k ~shape ~minval ~maxval)

let gamma_value k ~shape ~a = sampler k (fun () -> gamma_value k ~shape ~a)

let loggamma_value k ~shape ~a =
  sampler k (fun () -> loggamma_value k ~shape ~a)

let beta_value k ~shape ~a ~b = sampler k (fun () -> beta_value k ~shape ~a ~b)

let poisson_value k ~shape ~lam =
  sampler k (fun () -> poisson_value k ~shape ~lam)

let bits_as k ~shape ~dtype = sampler k (fun () -> bits_as k ~shape ~dtype)
