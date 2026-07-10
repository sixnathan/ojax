open Types
module Nd = Ndarray
module C = Ojax__Core
module D = Dtype
module PR = Prng
module TF = Threefry

let prod a = Array.fold_left ( * ) 1 a
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let concrete = function
  | Concrete nd -> nd
  | Tracer _ -> failwith "random/core: sampler on a tracer not supported in M3"
  | Device _ -> failwith "random/core: sampler on a tracer not supported in M3"

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
let key seed = PR.random_seed seed
let key_data keys = unwrap keys
let wrap_key_data data = wrap data
let clone k = k

let fold_in k data =
  unwrap
    (PR.random_fold_in (wrap k)
       (C.bind1 (Convert_element_type D.Uint32) [ data ]))

let split k num = unwrap (PR.random_split (wrap k) [| num |])
let bits k ~shape = PR.random_bits (wrap k) ~bit_width:32 ~shape

let uniform k ~shape ~minval ~maxval =
  let dt = D.F32 in
  let raw = concrete (PR.random_bits (wrap k) ~bit_width:32 ~shape) in
  let n = prod shape in
  let floats =
    Array.init n (fun i ->
        let b = geti raw i in
        let fb = (b lsr 9) lor 0x3F800000 in
        Int32.float_of_bits (Int32.of_int fb))
  in
  let floats_v = Concrete (Nd.of_floats dt shape floats) in
  let minv = f32 minval and maxv = f32 maxval in
  let span = f32 (maxv -. minv) in
  let u = C.bind1 Sub [ floats_v; const dt shape 1.0 ] in
  let scaled = C.bind1 Mul [ u; const dt shape span ] in
  let y = C.bind1 Add [ scaled; const dt shape minv ] in
  C.bind1 Max [ const dt shape minv; y ]

let scalar_f v = Nd.get_f (concrete v) [||]

let normal k ~shape =
  let dt = D.F32 in
  let lo =
    scalar_f (C.bind1 Nextafter [ const dt [||] (-1.0); const dt [||] 0.0 ])
  in
  let u = uniform k ~shape ~minval:lo ~maxval:1.0 in
  let sqrt2 = f32 (sqrt 2.0) in
  C.bind1 Mul [ const dt shape sqrt2; C.bind1 Erf_inv [ u ] ]

let truncated_normal k ~lower ~upper ~shape =
  let dt = D.F32 in
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

let randint k ~shape ~minval ~maxval =
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

let s x = const D.F32 [||] x
let cf shape x = const D.F32 shape x
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

let f32_eps = f32 (2.0 ** -23.0)
let f32_epsneg = f32 (2.0 ** -24.0)
let f32_tiny = f32 (2.0 ** -126.0)
let pi_f32 = f32 Float.pi
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
  let u = uniform k ~shape ~minval:f32_eps ~maxval:1.0 in
  e_tan (e_mul (cf shape pi_f32) (e_sub u (cf shape 0.5)))

let laplace k ~shape =
  let u = uniform k ~shape ~minval:(-1.0 +. f32_epsneg) ~maxval:1.0 in
  e_mul (e_sign u) (e_log1p (e_neg (e_abs u)))

let logistic k ~shape =
  let x = uniform k ~shape ~minval:f32_tiny ~maxval:1.0 in
  e_sub (e_log x) (e_log1p (e_neg x))

let gumbel k ~shape =
  let u = uniform k ~shape ~minval:f32_tiny ~maxval:1.0 in
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
  let b = C.bind1 (Convert_element_type D.F32) [ e_lt u (cf shape 0.5) ] in
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
  Concrete (Nd.of_floats D.F32 shape out)

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

let poisson k ~shape ~lam =
  let n = prod shape in
  if lam >= 10.0 then
    failwith "random/core: poisson rejection path (lam>=10) unimplemented in M3";
  let dt = default_int () in
  if lam = 0.0 then Concrete (Nd.of_floats dt shape (Array.make n 0.0))
  else begin
    let k_arr = Array.make n 0 in
    let logp = Array.make n 0.0 in
    let neg_lam = f32 (-.lam) in
    let key = ref k in
    let active () = Array.exists (fun lp -> lp > neg_lam) logp in
    while active () do
      let sub = split_n !key 2 in
      key := sub.(0);
      let u = concrete (unif01 sub.(1) ~shape) in
      for i = 0 to n - 1 do
        if logp.(i) > neg_lam then k_arr.(i) <- k_arr.(i) + 1;
        let ui = Nd.get_f u (unravel shape i) in
        logp.(i) <- f32 (logp.(i) +. f32 (log ui))
      done
    done;
    let out = Array.init n (fun i -> float_of_int (k_arr.(i) - 1)) in
    Concrete (Nd.of_floats dt shape out)
  end

let binomial_arr key ~shape ~counts ~probs =
  let n = prod shape in
  let q =
    Array.init n (fun i ->
        f32 (if probs.(i) < 0.5 then probs.(i) else 1.0 -. probs.(i)))
  in
  let cnt = Array.map (fun c -> f32 (Float.floor c)) counts in
  Array.iteri
    (fun i c ->
      if c *. q.(i) > 10.0 then
        failwith "random/core: binomial btrs path (n*q>10) unimplemented in M3")
    cnt;
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
      let samp = float_of_int (num_geom.(i) - 1) in
      if probs.(i) < 0.5 then samp else cnt.(i) -. samp)

let binomial k ~shape ~count ~prob =
  let n = prod shape in
  let out =
    binomial_arr k ~shape ~counts:(Array.make n count)
      ~probs:(Array.make n prob)
  in
  Concrete (Nd.of_floats D.F32 shape out)

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
  Concrete (Nd.of_floats D.F32 [| kk |] counts)
