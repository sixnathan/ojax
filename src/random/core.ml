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
let const dt shape x = C.splat dt shape x
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

module L = Lax_api
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module JU = Numpy.Jnp_util
module RED = Numpy.Reductions
module AC = Numpy.Array_creation

let statics_key statics =
  Digest.to_hex (Digest.string (Marshal.to_string statics []))

let jit_region name statics args f =
  Eager_jit.one ("random." ^ name) statics [] args (fun _ vals ->
      JU.jit1 name ~key:(statics_key statics) f vals)

let dtype_of v = (C.get_aval v).dtype
let shape_of v = (C.get_aval v).shape
let npf dt x = Concrete (Nd.of_floats dt [||] [| x |])
let like v x = npf (dtype_of v) x
let pyf = JU.py_float
let conv v dt = if dtype_of v = dt then v else L.convert_element_type v dt
let ( +! ) = UF.add
let ( -! ) = UF.subtract
let ( *! ) = UF.multiply
let ( /! ) = UF.true_divide
let pick keys i = L.index_in_dim ~keepdims:false keys i
let split_keys k n = unwrap (PR.random_split (wrap k) [| n |])

let key2 k =
  let keys = split_keys k 2 in
  let a = pick keys 0 in
  let b = pick keys 1 in
  (a, b)

let float_info dt =
  match dt with
  | D.F32 -> (32, 23, 0x3F800000L, D.Uint32)
  | D.F64 -> (64, 52, 0x3FF0000000000000L, D.Uint64)
  | D.F16 -> (16, 10, 0x3C00L, D.Uint16)
  | D.Bf16 -> (16, 7, 0x3F80L, D.Uint16)
  | _ ->
      raise
        (Errors.jax_type_error "uniform only accepts floating point dtypes.")

let tiny_of dt = (Dtypes.finfo dt).tiny
let eps_of dt = (Dtypes.finfo dt).eps
let epsneg_of dt = (Dtypes.finfo dt).epsneg

let round_to dt x =
  match dt with D.F32 -> Int32.float_of_bits (Int32.bits_of_float x) | _ -> x

let randint_v key ~minval ~maxval ~shape ~dt =
  let nbits = Dtypes.itemsize_bits dt in
  let uint_dt =
    match nbits with
    | 8 -> D.Uint8
    | 16 -> D.Uint16
    | 32 -> D.Uint32
    | _ -> D.Uint64
  in
  jit_region "_randint" (shape, dt) [ key; minval; maxval ] (function
    | [ key; minval; maxval ] ->
        let int_const d v =
          if d = D.I64 || d = D.Uint64 then
            Concrete (Nd.of_int64s d [||] [| v |])
          else Concrete (Nd.of_floats d [||] [| Int64.to_float v |])
        in
        let hi =
          if nbits >= 64 then Int64.max_int
          else Int64.sub (Int64.shift_left 1L (nbits - 1)) 1L
        in
        let maxval_out_of_range =
          let md = dtype_of maxval in
          let clipped =
            if Dtypes.itemsize_bits md >= nbits then hi
            else
              Int64.sub (Int64.shift_left 1L (Dtypes.itemsize_bits md - 1)) 1L
          in
          L.gt maxval (int_const md clipped)
        in
        let minval = conv minval dt in
        let maxval = conv maxval dt in
        let rank = Array.length shape in
        let minval = L.broadcast_to_rank minval rank in
        let maxval = L.broadcast_to_rank maxval rank in
        let k1, k2 = key2 key in
        let higher = PR.random_bits (wrap k1) ~bit_width:nbits ~shape in
        let lower = PR.random_bits (wrap k2) ~bit_width:nbits ~shape in
        let span = conv (maxval -! minval) uint_dt in
        let span =
          L.select (UF.less_equal maxval minval) (L.full_like span 1.0) span
        in
        let span =
          L.select
            (UF.bitwise_and maxval_out_of_range (UF.greater maxval minval))
            (L.add span (int_const uint_dt 1L))
            span
        in
        let half = int_const uint_dt (Int64.shift_left 1L (nbits / 2)) in
        let multiplier = L.rem half span in
        let multiplier = L.rem (L.mul multiplier multiplier) span in
        let offset =
          L.add (L.mul (L.rem higher span) multiplier) (L.rem lower span)
        in
        let offset = L.rem offset span in
        L.add minval (conv offset dt)
    | _ -> assert false)

let randint k ~shape ~minval ~maxval =
  randint_v k ~minval:(JU.py_int minval) ~maxval:(JU.py_int maxval) ~shape
    ~dt:(idt ())

let shuffle key x =
  let sh = shape_of x in
  let size = prod sh in
  let rounds =
    int_of_float
      (ceil (3.0 *. log (float_of_int (max 1 size)) /. log 4294967295.0))
  in
  jit_region "_shuffle" (sh, rounds) [ key; x ] (function
    | [ key; x ] ->
        let rec loop key x r =
          if r = 0 then x
          else
            let k, sub = key2 key in
            let sort_keys = PR.random_bits (wrap sub) ~bit_width:32 ~shape:sh in
            let outs =
              C.bind
                (Sort { dimension = 0; is_stable = true; num_keys = 1 })
                [ sort_keys; x ]
            in
            loop k (List.nth outs 1) (r - 1)
        in
        loop key x rounds
    | _ -> assert false)

let permutation k n =
  let dt = Dtypes.default_int_dtype () in
  shuffle k (C.bind1 (Iota { dtype = dt; shape = [| n |]; dimension = 0 }) [])

let choice k ~n ~shape ~replace =
  let dt = Dtypes.default_int_dtype () in
  let n_draws = prod shape in
  if n_draws = 0 then Concrete (Nd.of_floats dt shape [||])
  else if replace then
    randint_v k ~minval:(JU.py_int 0) ~maxval:(JU.py_int n) ~shape ~dt
  else
    let perm = permutation k n in
    L.reshape (L.slice_in_dim perm 0 n_draws) shape

let uniform_v key ~minval ~maxval ~shape ~dt =
  jit_region "_uniform" (shape, dt) [ key; minval; maxval ] (function
    | [ key; minval; maxval ] ->
        let minval = conv minval dt in
        let maxval = conv maxval dt in
        let rank = Array.length shape in
        let minval = L.broadcast_to_rank minval rank in
        let maxval = L.broadcast_to_rank maxval rank in
        let nbits, nmant, one_bits, uint_dt = float_info dt in
        let rng_bits = if nmant < 8 then 8 else nbits in
        let bits = PR.random_bits (wrap key) ~bit_width:rng_bits ~shape in
        let bits = if rng_bits <> nbits then conv bits uint_dt else bits in
        let uint_const v =
          if uint_dt = D.Uint64 then
            Concrete (Nd.of_int64s uint_dt [||] [| v |])
          else Concrete (Nd.of_floats uint_dt [||] [| Int64.to_float v |])
        in
        let shift = uint_const (Int64.of_int (rng_bits - nmant)) in
        let float_bits = L.shift_right_logical bits shift in
        let float_bits = L.bitwise_or float_bits (uint_const one_bits) in
        let as_float = L.bitcast_convert_type float_bits dt in
        let floats = as_float -! npf dt 1.0 in
        let scaled = floats *! (maxval -! minval) in
        let value = scaled +! minval in
        let reshaped =
          if shape_of value = shape then value else L.reshape value shape
        in
        L.max minval reshaped
    | _ -> assert false)

let uniform_py key ~shape ~dt =
  uniform_v key ~minval:(pyf 0.0) ~maxval:(pyf 1.0) ~shape ~dt

let uniform k ~shape ~minval ~maxval =
  uniform_v k ~minval:(pyf minval) ~maxval:(pyf maxval) ~shape ~dt:(fdt ())

let uniform_range k ~shape ~minval ~maxval =
  uniform_v k ~minval ~maxval ~shape ~dt:(dtype_of minval)

let unif01 k ~shape = uniform_py k ~shape ~dt:(fdt ())

let normal_real key ~shape ~dt =
  jit_region "_normal_real" (shape, dt) [ key ] (function
    | [ key ] ->
        let lo = Float.succ (-1.0) in
        let lo =
          match dt with
          | D.F32 -> -1.0 +. (2.0 ** -24.0)
          | D.F64 -> lo
          | D.F16 -> -1.0 +. (2.0 ** -11.0)
          | D.Bf16 -> -1.0 +. (2.0 ** -8.0)
          | _ -> lo
        in
        let u =
          uniform_v key ~minval:(npf dt lo) ~maxval:(npf dt 1.0) ~shape ~dt
        in
        L.mul (npf dt (sqrt 2.0)) (C.bind1 Erf_inv [ u ])
    | _ -> assert false)

let normal_dt key ~shape ~dt =
  jit_region "_normal" (shape, dt) [ key ] (function
    | [ key ] -> normal_real key ~shape ~dt
    | _ -> assert false)

let normal k ~shape = normal_dt k ~shape ~dt:(fdt ())

let clip_arr v lo hi =
  JU.jit1 "clip"
    (function
      | [ v; lo; hi ] ->
          let v = UF.maximum lo v in
          UF.minimum hi v
      | _ -> assert false)
    [ v; lo; hi ]

let truncated_normal_v key ~lower ~upper ~shape ~dt =
  jit_region "_truncated_normal" (shape, dt) [ key; lower; upper ] (function
    | [ key; lower; upper ] ->
        let sqrt2 = npf dt (sqrt 2.0) in
        let lower = conv lower dt in
        let upper = conv upper dt in
        let a = C.bind1 Erf [ lower /! sqrt2 ] in
        let b = C.bind1 Erf [ upper /! sqrt2 ] in
        let u = uniform_v key ~minval:a ~maxval:b ~shape ~dt in
        let out = sqrt2 *! C.bind1 Erf_inv [ u ] in
        let lo = L.nextafter (L.stop_gradient lower) (npf dt Float.infinity) in
        let hi =
          L.nextafter (L.stop_gradient upper) (npf dt Float.neg_infinity)
        in
        clip_arr out lo hi
    | _ -> assert false)

let truncated_normal k ~lower ~upper ~shape =
  truncated_normal_v k ~lower:(pyf lower) ~upper:(pyf upper) ~shape ~dt:(fdt ())

let bernoulli_v key ~p ~shape =
  let pdt = dtype_of p in
  Eager_jit.one "random._bernoulli" (shape, pdt) [] [ key; p ] (fun _ vals ->
      JU.jit1 "_bernoulli"
        ~key:(statics_key (shape, pdt))
        (function
          | [ key; p ] ->
              let u = uniform_py key ~shape ~dt:pdt in
              UF.less u p
          | _ -> assert false)
        vals)

let bernoulli k ~shape ~p = bernoulli_v k ~p:(npf (fdt ()) p) ~shape

let bernoulli_prob k ~shape ~p =
  let dt = dtype_of p in
  let shaped = if shape_of p = shape then p else NL.broadcast_to p shape in
  ignore dt;
  bernoulli_v k ~p:shaped ~shape

let rademacher_dt key ~shape ~dt =
  jit_region "_rademacher" (shape, dt) [ key ] (function
    | [ key ] ->
        let b = conv (bernoulli_v key ~p:(pyf 0.5) ~shape) dt in
        let doubled = npf dt 2.0 *! b in
        conv (doubled -! npf dt 1.0) dt
    | _ -> assert false)

let float_dtype = fdt
let rademacher_as = rademacher_dt

let rademacher k ~shape =
  rademacher_dt k ~shape ~dt:(Dtypes.default_int_dtype ())

let exponential_dt key ~shape ~dt =
  jit_region "_exponential" (shape, dt) [ key ] (function
    | [ key ] ->
        let u = uniform_py key ~shape ~dt in
        L.neg (L.log1p (L.neg u))
    | _ -> assert false)

let exponential k ~shape = exponential_dt k ~shape ~dt:(fdt ())

let cauchy k ~shape =
  let dt = fdt () in
  jit_region "_cauchy" (shape, dt) [ k ] (function
    | [ key ] ->
        let u =
          uniform_v key
            ~minval:(npf dt (eps_of dt))
            ~maxval:(pyf 1.0) ~shape ~dt
        in
        let pi = like u Float.pi in
        L.tan (L.mul pi (L.sub u (like u 0.5)))
    | _ -> assert false)

let laplace k ~shape =
  let dt = fdt () in
  jit_region "_laplace" (shape, dt) [ k ] (function
    | [ key ] ->
        let u =
          uniform_v key
            ~minval:(npf dt (round_to dt (-1.0 +. epsneg_of dt)))
            ~maxval:(pyf 1.0) ~shape ~dt
        in
        let sign = L.sign u in
        L.mul sign (L.log1p (L.neg (L.abs u)))
    | _ -> assert false)

let logistic k ~shape =
  let dt = fdt () in
  jit_region "_logistic" (shape, dt) [ k ] (function
    | [ key ] ->
        let x =
          uniform_v key
            ~minval:(npf dt (tiny_of dt))
            ~maxval:(pyf 1.0) ~shape ~dt
        in
        let lx = L.log x in
        L.sub lx (L.log1p (L.neg x))
    | _ -> assert false)

let gumbel_dt key ~shape ~dt =
  jit_region "_gumbel" (shape, dt) [ key ] (function
    | [ key ] ->
        let u =
          uniform_v key
            ~minval:(npf dt (tiny_of dt))
            ~maxval:(pyf 1.0) ~shape ~dt
        in
        UF.negative (UF.log (UF.negative (UF.log u)))
    | _ -> assert false)

let gumbel k ~shape = gumbel_dt k ~shape ~dt:(fdt ())

let gamma_one dt log_space key alpha =
  let zero = like alpha 0.0
  and one = like alpha 1.0
  and two = like alpha 2.0
  and minus_one = like alpha (-1.0)
  and one_over_two = like alpha 0.5
  and one_over_three = like alpha (1.0 /. 3.0)
  and squeeze_const = like alpha 0.0331 in
  let boost_mask = L.ge alpha one in
  let alpha_orig = alpha in
  let alpha = L.select boost_mask alpha (L.add alpha one) in
  let d = L.sub alpha one_over_three in
  let c = L.div one_over_three (L.sqrt d) in
  let keys = split_keys key 2 in
  let key = pick keys 0 in
  let subkey = pick keys 1 in
  let cond = function
    | [ _; x_big; v_big; u_big ] ->
        let c1 =
          L.ge u_big (L.sub one (L.mul squeeze_const (L.mul x_big x_big)))
        in
        let right =
          L.add (L.mul x_big one_over_two)
            (L.mul d (L.add (L.sub one v_big) (L.log v_big)))
        in
        L.bitwise_and c1 (L.ge (L.log u_big) right)
    | _ -> assert false
  in
  let body = function
    | key :: _ ->
        let keys = split_keys key 3 in
        let key = pick keys 0 in
        let x_key = pick keys 1 in
        let u_key = pick keys 2 in
        let out =
          L.while_loop
            (function [ _; _; v ] -> L.le v zero | _ -> assert false)
            (function
              | [ key; _; _ ] ->
                  let k2 = split_keys key 2 in
                  let key = pick k2 0 in
                  let sub = pick k2 1 in
                  let x = normal_dt sub ~shape:[||] ~dt in
                  let v = L.add one (L.mul x c) in
                  [ key; x; v ]
              | _ -> assert false)
            [ x_key; zero; minus_one ]
        in
        let x = List.nth out 1 and v = List.nth out 2 in
        let x_big = L.mul x x in
        let v_big = L.mul (L.mul v v) v in
        let u_big = uniform_py u_key ~shape:[||] ~dt in
        [ key; x_big; v_big; u_big ]
    | _ -> assert false
  in
  let v_big = List.nth (L.while_loop cond body [ key; zero; one; two ]) 2 in
  if log_space then begin
    let log_samples = L.neg (exponential_dt subkey ~shape:[||] ~dt) in
    let log_boost =
      L.select
        (UF.bitwise_or boost_mask (UF.equal log_samples (like log_samples 0.0)))
        zero
        (L.mul log_samples (L.div one alpha_orig))
    in
    L.add (L.add (L.log d) (L.log v_big)) log_boost
  end
  else begin
    let samples = like one 1.0 -! uniform_py subkey ~shape:[||] ~dt in
    let boost =
      L.select boost_mask one (L.pow samples (L.div one alpha_orig))
    in
    L.mul (L.mul d v_big) boost
  end

let gamma_impl dt log_space key a =
  let a_shape = shape_of a in
  let split_count = prod a_shape in
  let keys = unwrap (PR.random_split (wrap key) [| split_count |]) in
  let alphas = L.reshape a [| split_count |] in
  let samples =
    Interpreters.Batching.vmap
      (function
        | [ k; alpha ] -> [ gamma_one dt log_space k alpha ] | _ -> assert false)
      [ Some 0; Some 0 ] [ keys; alphas ]
  in
  L.reshape (List.hd samples) a_shape

let random_gamma_grad_prim alpha sample = L.random_gamma_grad alpha sample

let gamma_grad sample a ~log_space =
  let sh = shape_of a in
  let n = prod sh in
  let samples = L.reshape sample [| n |] in
  let alphas = L.reshape a [| n |] in
  let samples, grad_fn =
    if log_space then
      let exped = L.exp samples in
      let tiny = L.full_like exped (tiny_of (dtype_of exped)) in
      let safe = L.select (L.eq exped (like exped 0.0)) tiny exped in
      (safe, fun alpha s -> random_gamma_grad_prim alpha s /! s)
    else (samples, random_gamma_grad_prim)
  in
  let grads =
    L.map
      (function [ alpha; s ] -> [ grad_fn alpha s ] | _ -> assert false)
      [ alphas; samples ]
  in
  L.reshape (List.hd grads) sh

let random_gamma dt ~log_space key a =
  List.hd
    (Custom_derivatives.custom_jvp
       (function
         | [ key; a ] -> [ gamma_impl dt log_space key a ] | _ -> assert false)
       ~jvp:(fun primals tangents ->
         match (primals, tangents) with
         | [ key; a ], [ _; ta ] ->
             let out = gamma_impl dt log_space key a in
             ([ out ], [ ta *! gamma_grad out a ~log_space ])
         | _ -> assert false)
       [ key; a ])

let gamma_v key ~a ~shape ~dt ~log_space =
  jit_region "_gamma" (shape, dt, log_space) [ key; a ] (function
    | [ key; a ] ->
        let a = conv a dt in
        let a = if shape_of a <> shape then NL.broadcast_to a shape else a in
        random_gamma dt ~log_space key a
    | _ -> assert false)

let gamma k ~shape ~a =
  gamma_v k ~a:(pyf a) ~shape ~dt:(fdt ()) ~log_space:false

let loggamma k ~shape ~a =
  gamma_v k ~a:(pyf a) ~shape ~dt:(fdt ()) ~log_space:true

let gamma_value k ~shape ~a =
  gamma_v k ~a ~shape ~dt:(dtype_of a) ~log_space:false

let loggamma_value k ~shape ~a =
  gamma_v k ~a ~shape ~dt:(dtype_of a) ~log_space:true

let beta_v key ~a ~b ~shape ~dt =
  jit_region "_beta" (shape, dt) [ key; a; b ] (function
    | [ key; a; b ] ->
        let a = conv a dt in
        let b = conv b dt in
        let key_a, key_b = key2 key in
        let a = NL.broadcast_to a shape in
        let b = NL.broadcast_to b shape in
        let log_gamma_a = gamma_v key_a ~a ~shape ~dt ~log_space:true in
        let log_gamma_b = gamma_v key_b ~a:b ~shape ~dt ~log_space:true in
        let log_max = L.max log_gamma_a log_gamma_b in
        let gamma_a_scaled = UF.exp (log_gamma_a -! log_max) in
        let gamma_b_scaled = UF.exp (log_gamma_b -! log_max) in
        gamma_a_scaled /! (gamma_a_scaled +! gamma_b_scaled)
    | _ -> assert false)

let beta k ~shape ~a ~b = beta_v k ~a:(pyf a) ~b:(pyf b) ~shape ~dt:(fdt ())
let beta_value k ~shape ~a ~b = beta_v k ~a ~b ~shape ~dt:(dtype_of a)
let default_float_for () = fdt ()

let chisquare_v key ~df ~shape ~dt =
  jit_region "_chisquare" (shape, dt) [ key; df ] (function
    | [ key; df ] ->
        let df = conv df dt in
        let two = like df 2.0 in
        let half_df = L.div df two in
        let log_g = gamma_v key ~a:half_df ~shape ~dt ~log_space:true in
        L.mul (UF.exp log_g) two
    | _ -> assert false)

let chisquare k ~shape ~df = chisquare_v k ~df:(pyf df) ~shape ~dt:(fdt ())

let t k ~shape ~df =
  let dt = fdt () in
  jit_region "_t" (shape, dt)
    [ k; pyf df ]
    (function
      | [ key; df ] ->
          let df = conv df dt in
          let key_n, key_g = key2 key in
          let n = normal_dt key_n ~shape ~dt in
          let two = like n 2.0 in
          let half_df = L.div df two in
          let g = gamma_v key_g ~a:half_df ~shape ~dt ~log_space:false in
          n *! UF.sqrt (half_df /! g)
      | _ -> assert false)

let f k ~shape ~dfnum ~dfden =
  let dt = fdt () in
  jit_region "_f" (shape, dt)
    [ k; pyf dfnum; pyf dfden ]
    (function
      | [ key; dfnum; dfden ] ->
          let dfden = conv dfden dt in
          let dfnum = conv dfnum dt in
          let key_dfd, key_dfn = key2 key in
          let chi2_dfn = chisquare_v key_dfn ~df:dfnum ~shape ~dt in
          let chi2_dfd = chisquare_v key_dfd ~df:dfden ~shape ~dt in
          let num = L.div chi2_dfn dfnum in
          let den = L.div chi2_dfd dfden in
          L.div num den
      | _ -> assert false)

let pareto k ~shape ~b =
  let dt = fdt () in
  jit_region "_pareto" (shape, dt)
    [ k; pyf b ]
    (function
      | [ key; b ] ->
          let b = conv b dt in
          let e = exponential_dt key ~shape ~dt in
          L.exp (e /! b)
      | _ -> assert false)

let rayleigh k ~shape ~scale =
  let dt = fdt () in
  jit_region "_rayleigh" (shape, dt)
    [ k; pyf scale ]
    (function
      | [ key; scale ] ->
          let u = uniform_py key ~shape ~dt in
          let scale = conv scale dt in
          let scale = NL.broadcast_to scale shape in
          let log_u = L.log u in
          let n_two = like scale (-2.0) in
          let sqrt_u = L.sqrt (L.mul log_u n_two) in
          L.mul scale sqrt_u
      | _ -> assert false)

let wald k ~shape ~mean =
  let dt = fdt () in
  jit_region "_wald" (shape, dt)
    [ k; pyf mean ]
    (function
      | [ key; mean ] ->
          let k1, k2 = key2 key in
          let mean = conv mean dt in
          let mean = NL.broadcast_to mean shape in
          let v = normal_dt k1 ~shape ~dt in
          let z = uniform_py k2 ~shape ~dt in
          let y = L.integer_pow v 2 in
          let y_sq = L.integer_pow y 2 in
          let mean_sq = L.integer_pow mean 2 in
          let four = like mean 4.0 in
          let left = four *! mean *! y in
          let right = mean_sq *! y_sq in
          let sqrt_term = L.sqrt (left +! right) in
          let two = like mean 2.0 in
          let term1 = mean_sq *! y /! two in
          let term2 = mean /! two *! sqrt_term in
          let x = mean +! term1 -! term2 in
          let threshold = mean /! (mean +! x) in
          let pred = L.le z threshold in
          let alt = mean_sq /! x in
          L.select pred x alt
      | _ -> assert false)

let geometric k ~shape ~p =
  let idt_ = Dtypes.default_int_dtype () in
  let pdt = fdt () in
  jit_region "_geometric" (shape, idt_)
    [ k; pyf p ]
    (function
      | [ key; p ] ->
          let u = uniform_py key ~shape ~dt:pdt in
          let u = NL.where_ (UF.equal u (like u 0.0)) (like u 1.0) u in
          let log_u = L.log u in
          let log_one_minus_p = L.log1p (UF.negative (conv p pdt)) in
          let log_one_minus_p = NL.broadcast_to log_one_minus_p shape in
          let g = L.floor (L.div log_u log_one_minus_p) +! like u 1.0 in
          conv g idt_
      | _ -> assert false)

let weibull_min k ~shape ~scale ~concentration =
  let dt = fdt () in
  jit_region "_weibull_min" (shape, dt)
    [ k; pyf scale; pyf concentration ]
    (function
      | [ key; scale; concentration ] ->
          let u = uniform_py key ~shape ~dt in
          let base = UF.negative (UF.log1p (UF.negative u)) in
          let power =
            conv (pyf 1.0) (dtype_of concentration) /! concentration
          in
          conv (UF.power base (conv power dt)) dt *! conv scale dt
      | _ -> assert false)

let triangular k ~shape ~left ~mode ~right =
  let dt = fdt () in
  jit_region "_triangular" (shape, dt)
    [ k; pyf left; pyf mode; pyf right ]
    (function
      | [ key; left; mode; right ] ->
          let left = NL.broadcast_to (conv left dt) shape in
          let right = NL.broadcast_to (conv right dt) shape in
          let mode = NL.broadcast_to (conv mode dt) shape in
          let fc = (mode -! left) /! (right -! left) in
          let u = uniform_py key ~shape ~dt in
          let out1 = left +! L.sqrt (u *! (right -! left) *! (mode -! left)) in
          let out2 =
            right
            -! L.sqrt ((like u 1.0 -! u) *! (right -! left) *! (right -! mode))
          in
          L.select (UF.less u fc) out1 out2
      | _ -> assert false)

let lognormal k ~shape ~sigma =
  let dt = fdt () in
  jit_region "_lognormal" (shape, dt)
    [ k; pyf sigma ]
    (function
      | [ key; sigma ] ->
          let sigma = conv sigma dt in
          let scaled = normal_dt key ~shape ~dt *! sigma in
          L.exp scaled
      | _ -> assert false)

let poisson_knuth key lam ~shape ~dt ~max_iters =
  let idx v =
    Concrete (Nd.of_floats (Dtypes.default_int_dtype ()) [||] [| v |])
  in
  let body = function
    | [ i; k; rng; log_prod ] ->
        let keys = split_keys rng 2 in
        let rng = pick keys 0 and subkey = pick keys 1 in
        let k =
          L.select
            (UF.greater log_prod (UF.negative lam))
            (k +! Concrete (Nd.of_floats dt [||] [| 1.0 |]))
            k
        in
        let u = uniform_py subkey ~shape ~dt:D.F32 in
        [ i +! idx 1.0; k; rng; log_prod +! UF.log u ]
    | _ -> assert false
  in
  let cond = function
    | i :: _ :: _ :: log_prod :: _ ->
        UF.bitwise_and
          (RED.any (UF.greater log_prod (UF.negative lam)))
          (UF.less i
             (Concrete
                (Nd.of_floats (dtype_of i) [||] [| Int64.to_float max_iters |])))
    | _ -> assert false
  in
  let k_init = L.full shape 0.0 dt in
  let log_rate_init = L.full shape 0.0 D.F32 in
  let out = L.while_loop cond body [ idx 0.0; k_init; key; log_rate_init ] in
  conv (List.nth out 1 -! Concrete (Nd.of_floats dt [||] [| 1.0 |])) dt

let poisson_rejection key lam ~shape ~dt ~max_iters =
  let c v = like lam v in
  let log_lam = L.log lam in
  let b = c 0.931 +! (c 2.53 *! L.sqrt lam) in
  let a = c (-0.059) +! (c 0.02483 *! b) in
  let inv_alpha = c 1.1239 +! (c 1.1328 /! (b -! c 3.4)) in
  let v_r = c 0.9277 -! (c 3.6224 /! (b -! c 2.0)) in
  let idx v =
    Concrete (Nd.of_floats (Dtypes.default_int_dtype ()) [||] [| v |])
  in
  let ldt = dtype_of lam in
  let body = function
    | [ i; k_out; accepted; key ] ->
        let keys = split_keys key 3 in
        let key = pick keys 0
        and subkey_0 = pick keys 1
        and subkey_1 = pick keys 2 in
        let u = uniform_py subkey_0 ~shape ~dt:ldt -! c 0.5 in
        let v = uniform_py subkey_1 ~shape ~dt:ldt in
        let u_shifted = c 0.5 -! UF.abs u in
        let k =
          L.floor ((((c 2.0 *! a /! u_shifted) +! b) *! u) +! lam +! c 0.43)
        in
        let s =
          L.log (v *! inv_alpha /! ((a /! (u_shifted *! u_shifted)) +! b))
        in
        let t =
          UF.negative lam +! (k *! log_lam) -! C.bind1 Lgamma [ k +! c 1.0 ]
        in
        let accept1 =
          UF.bitwise_and
            (UF.greater_equal u_shifted (c 0.07))
            (UF.less_equal v v_r)
        in
        let reject =
          UF.bitwise_or
            (UF.less k (c 0.0))
            (UF.bitwise_and
               (UF.less u_shifted (c 0.013))
               (UF.greater v u_shifted))
        in
        let accept2 = UF.less_equal s t in
        let accept =
          UF.bitwise_or accept1 (UF.bitwise_and (UF.bitwise_not reject) accept2)
        in
        let k_out = L.select accept k k_out in
        let accepted = UF.bitwise_or accepted accept in
        [ i +! idx 1.0; k_out; accepted; key ]
    | _ -> assert false
  in
  let cond = function
    | i :: _ :: accepted :: _ ->
        UF.bitwise_and
          (RED.any (UF.bitwise_not accepted))
          (UF.less i
             (Concrete
                (Nd.of_floats (dtype_of i) [||] [| Int64.to_float max_iters |])))
    | _ -> assert false
  in
  let k_init = L.full shape (-1.0) ldt in
  let accepted = L.full shape 0.0 D.Bool in
  let out = L.while_loop cond body [ idx 0.0; k_init; accepted; key ] in
  conv (List.nth out 1) dt

let poisson_impl key lam ~shape ~dt =
  jit_region "_poisson" (shape, dt) [ key; lam ] (function
    | [ key; lam ] ->
        let use_knuth =
          UF.bitwise_or (UF.not_equal lam lam) (UF.less lam (like lam 10.0))
        in
        let lam_knuth = L.select use_knuth lam (L.full_like lam 0.0) in
        let lam_rejection = L.select use_knuth (L.full_like lam 1e5) lam in
        let max_iters =
          match dt with D.I64 -> Int64.max_int | _ -> 2147483647L
        in
        let knuth = poisson_knuth key lam_knuth ~shape ~dt ~max_iters in
        let rej = poisson_rejection key lam_rejection ~shape ~dt ~max_iters in
        let result = L.select use_knuth knuth rej in
        L.select (UF.equal lam (like lam 0.0)) (AC.zeros_like result) result
    | _ -> assert false)

let poisson_value k ~shape ~lam =
  let lam = NL.broadcast_to lam shape in
  let lam = conv lam D.F32 in
  poisson_impl k lam ~shape ~dt:(Dtypes.default_int_dtype ())

let poisson k ~shape ~lam = poisson_value k ~shape ~lam:(pyf lam)

let stirling_vals =
  [|
    0.0810614667953272;
    0.0413406959554092;
    0.0276779256849983;
    0.02079067210376509;
    0.0166446911898211;
    0.0138761288230707;
    0.0118967099458917;
    0.010411265261972;
    0.00925546218271273;
    0.00833056343336287;
  |]

let stirling_tail k =
  let dt = dtype_of k in
  let vals = Concrete (Nd.of_floats dt [| 10 |] stirling_vals) in
  let use_tail = UF.less_equal k (like k 9.0) in
  let k = L.clamp (like k 0.0) k (like k 9.0) in
  let kp1sq = (k +! like k 1.0) *! (k +! like k 1.0) in
  let approx =
    (like k (1.0 /. 12.0)
    -! ((like k (1.0 /. 360.0) -! (like k (1.0 /. 1260.0) /! kp1sq)) /! kp1sq))
    /! (k +! like k 1.0)
  in
  let k = UF.floor k in
  let idx = conv k D.I32 in
  L.select use_tail (Numpy.Indexing.take vals idx) approx

let btrs key count prob ~shape ~dt =
  let c v = like prob v in
  let stddev = UF.sqrt (count *! prob *! (c 1.0 -! prob)) in
  let b = c 1.15 +! (c 2.53 *! stddev) in
  let a = c (-0.0873) +! (c 0.0248 *! b) +! (c 0.01 *! prob) in
  let cc = (count *! prob) +! c 0.5 in
  let v_r = c 0.92 -! (c 4.2 /! b) in
  let r = prob /! (c 1.0 -! prob) in
  let alpha = (c 2.83 +! (c 5.1 /! b)) *! stddev in
  let m = UF.floor ((count +! c 1.0) *! prob) in
  let idx v =
    Concrete (Nd.of_floats (Dtypes.default_int_dtype ()) [||] [| v |])
  in
  let pdt = dtype_of prob in
  let body = function
    | [ i; k_out; accepted; key ] ->
        let keys = split_keys key 3 in
        let key = pick keys 0
        and subkey_0 = pick keys 1
        and subkey_1 = pick keys 2 in
        let u = uniform_py subkey_0 ~shape ~dt:pdt in
        let v = uniform_py subkey_1 ~shape ~dt:pdt in
        let u = u -! c 0.5 in
        let us = c 0.5 -! UF.abs u in
        let accept1 =
          UF.bitwise_and (UF.greater_equal us (c 0.07)) (UF.less_equal v v_r)
        in
        let k = UF.floor ((((c 2.0 *! a /! us) +! b) *! u) +! cc) in
        let reject = UF.bitwise_or (UF.less k (c 0.0)) (UF.greater k count) in
        let v = UF.log (v *! alpha /! ((a /! (us *! us)) +! b)) in
        let ub =
          ((m +! c 0.5) *! UF.log ((m +! c 1.0) /! (r *! (count -! m +! c 1.0))))
          +! (count +! c 1.0)
             *! UF.log ((count -! m +! c 1.0) /! (count -! k +! c 1.0))
          +! (k +! c 0.5)
             *! UF.log (r *! (count -! k +! c 1.0) /! (k +! c 1.0))
          +! stirling_tail m
          +! stirling_tail (count -! m)
          -! stirling_tail k
          -! stirling_tail (count -! k)
        in
        let accept2 = UF.less_equal v ub in
        let accept =
          UF.bitwise_or accept1 (UF.bitwise_and (UF.bitwise_not reject) accept2)
        in
        let k_out = L.select accept k k_out in
        let accepted = UF.bitwise_or accepted accept in
        [ i +! idx 1.0; k_out; accepted; key ]
    | _ -> assert false
  in
  let cond = function
    | i :: _ :: accepted :: _ ->
        UF.bitwise_and
          (RED.any (UF.bitwise_not accepted))
          (UF.less (conv i pdt) (like prob (Dtypes.finfo dt).fmax))
    | _ -> assert false
  in
  let k_init = L.full shape (-1.0) pdt in
  let accepted = L.full shape 0.0 D.Bool in
  let out = L.while_loop cond body [ idx 0.0; k_init; accepted; key ] in
  conv (List.nth out 1) dt

let binomial_inversion key count prob ~shape ~dt =
  let c v = like prob v in
  let log1minusprob = UF.log1p (UF.negative prob) in
  let idx v =
    Concrete (Nd.of_floats (Dtypes.default_int_dtype ()) [||] [| v |])
  in
  let pdt = dtype_of prob in
  let body = function
    | [ i; num_geom; geom_sum; key ] ->
        let keys = split_keys key 2 in
        let subkey = pick keys 0 and key = pick keys 1 in
        let num_geom_out =
          L.select (UF.less_equal geom_sum count) (num_geom +! c 1.0) num_geom
        in
        let u = uniform_py subkey ~shape ~dt:pdt in
        let geom = UF.ceil (UF.log u /! log1minusprob) in
        let geom_sum = geom_sum +! geom in
        [ i +! idx 1.0; num_geom_out; geom_sum; key ]
    | _ -> assert false
  in
  let cond = function
    | i :: _ :: geom_sum :: _ ->
        UF.bitwise_and
          (RED.any (UF.less_equal geom_sum count))
          (UF.less (conv i pdt) (like prob (Dtypes.finfo dt).fmax))
    | _ -> assert false
  in
  let init = L.full shape 0.0 pdt in
  let out = L.while_loop cond body [ idx 0.0; init; init; key ] in
  conv (List.nth out 1 -! c 1.0) dt

let binomial_v key ~count ~prob ~shape ~dt =
  jit_region "_binomial" (shape, dt) [ key; count; prob ] (function
    | [ key; count; prob ] ->
        let pdt = dtype_of prob in
        let count = conv count pdt in
        let count = NL.broadcast_to count shape in
        let prob = NL.broadcast_to prob shape in
        let c v = like prob v in
        let p_lt_half = UF.less prob (c 0.5) in
        let q = L.select p_lt_half prob (c 1.0 -! prob) in
        let count_nan_or_neg =
          UF.bitwise_or (UF.not_equal count count) (UF.less count (c 0.0))
        in
        let count_inf = UF.isinf count in
        let q_is_nan = UF.not_equal q q in
        let q_l_0 = UF.less q (c 0.0) in
        let q =
          L.select (UF.bitwise_or q_is_nan q_l_0) (L.full_like q 0.01) q
        in
        let use_inversion =
          UF.bitwise_or count_nan_or_neg (UF.less_equal (count *! q) (c 10.0))
        in
        let count = UF.floor count in
        let count_inv = L.select use_inversion count (L.full_like count 0.0) in
        let count_btrs =
          L.select use_inversion (L.full_like count 10000.0) count
        in
        let q_btrs = L.select use_inversion (L.full_like q 0.5) q in
        let inv = binomial_inversion key count_inv q ~shape ~dt in
        let btr = btrs key count_btrs q_btrs ~shape ~dt in
        let samples = L.select use_inversion inv btr in
        let invalid =
          UF.bitwise_or (UF.bitwise_or q_l_0 q_is_nan) count_nan_or_neg
        in
        let samples =
          L.select invalid (AC.full_like samples Float.nan) samples
        in
        let samples =
          L.select
            (UF.bitwise_and count_inf (UF.bitwise_not invalid))
            (AC.full_like samples Float.infinity)
            samples
        in
        L.select
          (UF.bitwise_or
             (UF.bitwise_or (UF.bitwise_or p_lt_half count_nan_or_neg) q_is_nan)
             count_inf)
          samples
          (conv count dt -! samples)
    | _ -> assert false)

let binomial k ~shape ~count ~prob =
  let dt = fdt () in
  binomial_v k ~count:(pyf count) ~prob:(npf dt prob) ~shape ~dt

let dirichlet k ~alpha ~shape =
  let dt = fdt () in
  jit_region "_dirichlet" (shape, dt) [ k; alpha ] (function
    | [ key; alpha ] ->
        let alpha = conv alpha dt in
        let ash = shape_of alpha in
        let event = ash.(Array.length ash - 1) in
        let log_gamma =
          gamma_v key ~a:alpha
            ~shape:(Array.append shape [| event |])
            ~dt ~log_space:true
        in
        let top = RED.max ~axis:[| -1 |] ~keepdims:true log_gamma in
        let unnormalized = UF.exp (log_gamma -! L.stop_gradient top) in
        unnormalized /! RED.sum ~axis:[| -1 |] ~keepdims:true unnormalized
    | _ -> assert false)

let categorical k ~logits ~axis =
  let dt = dtype_of logits in
  let g = gumbel_dt k ~shape:(shape_of logits) ~dt in
  NL.argmax ~axis (g +! logits)

let multinomial k ~p ~n_trials =
  let dt = fdt () in
  let n = pyf n_trials in
  let pdt = dtype_of p in
  let n = conv n pdt in
  let ones_like v = like v 1.0 in
  let remaining = L.cumsum ~axis:0 ~reverse:true p in
  let ratios =
    p
    /! NL.where_
         (UF.equal remaining (like remaining 0.0))
         (ones_like remaining) remaining
  in
  let count_n = (shape_of ratios).(0) in
  let keys = split_keys k count_n in
  let remainder = ref n in
  let counts =
    List.init count_n (fun i ->
        let ratio = L.index_in_dim ~keepdims:false ratios i in
        let key = pick keys i in
        let clipped = clip_arr ratio (like ratio 0.0) (like ratio 1.0) in
        let count =
          binomial_v key ~count:!remainder ~prob:clipped ~shape:[||]
            ~dt:(dtype_of !remainder)
        in
        remainder := !remainder -! count;
        count)
  in
  conv (L.stack counts 0) dt

let generalized_normal k ~shape ~p =
  let dt = fdt () in
  jit_region "_generalized_normal" (shape, dt)
    [ k; pyf p ]
    (function
      | [ key; p ] ->
          let k0, k1 = key2 key in
          let inv = conv (like p 1.0 /! p) dt in
          let g = gamma_v k0 ~a:inv ~shape ~dt ~log_space:false in
          let r = conv (rademacher_dt k1 ~shape ~dt) dt in
          r *! UF.power g inv
      | _ -> assert false)

let random_gamma_grad alpha sample = L.random_gamma_grad alpha sample
let is_tracer = function Tracer _ -> true | Concrete _ | Device _ -> false

let via_host f key =
  if is_tracer key then failwith "random/core: bits_as on tracer" else f key

let bits_as k ~shape ~dtype =
  let dtype = Dtypes.canonicalize_dtype dtype in
  let host key =
    match dtype with
    | D.Uint64 ->
        Concrete (Nd.of_int64s D.Uint64 shape (TF.bits64 (concrete key) shape))
    | D.Uint32 -> PR.random_bits (wrap key) ~bit_width:32 ~shape
    | D.Uint8 | D.Uint16 | D.Uint2 | D.Uint4 ->
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
