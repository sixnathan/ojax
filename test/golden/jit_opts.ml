module U = Yojson.Safe.Util
module T = Ojax.Types
module L = Ojax.Lax_api
module F = Ojax.Numpy.Ufuncs

let goldens_root =
  match Sys.getenv_opt "OJAX_GOLDENS" with
  | Some d -> d
  | None -> Filename.concat (Filename.concat ".." "..") "goldens"

type result = {
  kind : string;
  cls : string;
  message : string;
  outs : (string * int list * string) list;
  warnings : (string * string) list;
  notices : string list;
}

let operand shape dtype offset =
  let n = Array.fold_left ( * ) 1 shape in
  let vals = Array.init n (fun i -> float_of_int ((i mod 5) + 1 + offset)) in
  T.Concrete (Ojax.Ndarray.of_floats dtype shape vals)

let host = function
  | T.Concrete nd -> nd
  | T.Device b -> Ojax.Pjrt.Buffer.to_host b
  | T.Tracer _ -> failwith "jit_opts: expected a concrete result"

let encode v =
  let nd = host v in
  ( Ojax.Dtype.numpy_name (Ojax.Ndarray.dtype nd),
    Array.to_list (Ojax.Ndarray.shape nd),
    Parity_cases.hex_of_nd nd )

let run_case thunk =
  let notices = ref [] in
  let saved = !Ojax.Debug_special.notice_sink in
  (Ojax.Debug_special.notice_sink := fun s -> notices := s :: !notices);
  let outcome, warnings =
    Fun.protect
      ~finally:(fun () -> Ojax.Debug_special.notice_sink := saved)
      (fun () ->
        Ojax.Py_warnings.catch (fun () ->
            match thunk () with
            | outs -> Ok (List.map encode outs)
            | exception e -> Error e))
  in
  let base =
    {
      kind = "ok";
      cls = "";
      message = "";
      outs = [];
      warnings;
      notices = List.rev !notices;
    }
  in
  match outcome with
  | Ok outs -> { base with outs }
  | Error e -> (
      match (Ojax.Py_exn.class_name e, Ojax.Py_exn.message_of e) with
      | Some cls, Some message -> { base with kind = "raise"; cls; message }
      | _ ->
          {
            base with
            kind = "raise";
            cls = "OCaml";
            message = Printexc.to_string e;
          })

let rank_fn = function
  | "add" -> F.add
  | "subtract" -> F.subtract
  | "multiply" -> F.multiply
  | "maximum" -> F.maximum
  | "minimum" -> F.minimum
  | "arctan2" -> F.arctan2
  | "float_power" -> F.float_power
  | "nextafter" -> F.nextafter
  | "equal" -> F.equal
  | "not_equal" -> F.not_equal
  | "greater" -> F.greater
  | "greater_equal" -> F.greater_equal
  | "less" -> F.less
  | "less_equal" -> F.less_equal
  | "bitwise_and" -> F.bitwise_and
  | "bitwise_or" -> F.bitwise_or
  | "bitwise_xor" -> F.bitwise_xor
  | "left_shift" -> F.left_shift
  | "right_shift" -> F.right_shift
  | "logical_and" -> F.logical_and
  | "logical_or" -> F.logical_or
  | "logical_xor" -> F.logical_xor
  | "true_divide" -> F.true_divide
  | "power" -> F.power
  | "floor_divide" -> F.floor_divide
  | "remainder" -> F.remainder
  | "fmod" -> F.fmod
  | "copysign" -> F.copysign
  | "hypot" -> F.hypot
  | "logaddexp" -> F.logaddexp
  | "logaddexp2" -> F.logaddexp2
  | "heaviside" -> F.heaviside
  | name -> failwith ("jit_opts: unknown function " ^ name)

let f32 vals =
  T.Concrete
    (Ojax.Ndarray.of_floats Ojax.Dtype.F32 [| Array.length vals |] vals)

let jit1 f args =
  match
    Ojax.Api.jit
      (fun xs ->
        match xs with
        | [ Ojax.Tree_util.Leaf a ] -> f a
        | _ -> failwith "jit_opts: arity")
      (List.map (fun a -> Ojax.Tree_util.Leaf a) args)
  with
  | t -> Ojax.Tree_util.tree_leaves t

let leaf v = Ojax.Tree_util.Leaf v

let py_float x =
  Ojax.Core.py_scalar
    (if Ojax.Config.x64_enabled () then Ojax.Dtype.F64 else Ojax.Dtype.F32)
    x

let unary f = function
  | [ Ojax.Tree_util.Leaf a ] -> f a
  | _ -> failwith "jit_opts: arity"

let special_thunk name =
  let x = f32 [| -1.0; 2.0 |]
  and z = f32 [| 0.0 |]
  and n = f32 [| Float.nan |]
  and one = f32 [| 1.0 |] in
  match name with
  | "eager_log" -> fun () -> [ L.log x ]
  | "eager_sqrt" -> fun () -> [ L.sqrt x ]
  | "eager_div_zero_zero" -> fun () -> [ L.div z z ]
  | "eager_div_one_zero" -> fun () -> [ L.div one z ]
  | "eager_add_nan_input" -> fun () -> [ L.add n one ]
  | "jit_log" -> fun () -> jit1 (fun a -> leaf (L.mul (L.log a) a)) [ x ]
  | "jit_sqrt_sum" ->
      fun () -> jit1 (fun a -> leaf (L.reduce_sum (L.sqrt a) [| 0 |])) [ x ]
  | "jit_identity_nan" -> fun () -> jit1 (fun a -> leaf a) [ n ]
  | "jit_pair_nan" ->
      fun () -> jit1 (fun a -> Ojax.Tree_util.Tuple [ leaf a; leaf a ]) [ n ]
  | "jit_nested_log" ->
      fun () ->
        jit1
          (fun a ->
            match jit1 (fun b -> leaf (L.log b)) [ a ] with
            | [ r ] -> leaf r
            | _ -> failwith "jit_opts: arity")
          [ x ]
  | "jit_div_one_zero" -> fun () -> jit1 (fun a -> leaf (L.div one a)) [ z ]
  | "jit_clean" -> fun () -> jit1 (fun a -> leaf (L.exp a)) [ x ]
  | "eager_int_div" ->
      fun () ->
        let i v =
          T.Concrete (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [| 1 |] [| v |])
        in
        [ L.div (i 1.0) (i 0.0) ]
  | "jnp_log" -> fun () -> [ F.log x ]
  | "jnp_sqrt" -> fun () -> [ F.sqrt x ]
  | "jnp_div" -> fun () -> [ F.true_divide z z ]
  | "jnp_div1" -> fun () -> [ F.true_divide (py_float 1.0) z ]
  | "jnp_arccos" -> fun () -> [ F.arccos (F.multiply x (py_float 2.0)) ]
  | "jnp_mean_jit" ->
      fun () ->
        jit1 (fun a -> leaf (Ojax.Numpy.Reductions.mean (F.log a))) [ x ]
  | "vmap_log" ->
      fun () ->
        Ojax.Tree_util.tree_leaves
          (Ojax.Api.vmap
             (unary (fun a -> leaf (L.log a)))
             [ Some 0 ]
             [ leaf x ])
  | "grad_sqrt" | "grad_log" ->
      let op = if name = "grad_sqrt" then L.sqrt else L.log in
      fun () ->
        let z0 =
          T.Concrete (Ojax.Ndarray.of_floats Ojax.Dtype.F32 [||] [| 0.0 |])
        in
        Ojax.Tree_util.tree_leaves
          (Ojax.Api.grad (unary (fun a -> leaf (op a))) [ leaf z0 ])
  | "cond_log" ->
      fun () ->
        Ojax.Lax.cond
          (T.Concrete (Ojax.Ndarray.of_floats Ojax.Dtype.Bool [||] [| 1.0 |]))
          (List.map L.log) Fun.id [ x ]
  | "fori_log" ->
      fun () ->
        let i v = Ojax.Core.py_scalar Ojax.Dtype.I32 v in
        L.fori_loop (i 0.0) (i 2.0) (fun _ vs -> List.map L.log vs) [ x ]
  | "softmax_inf" ->
      fun () -> [ Ojax.Nn.Functions.softmax (f32 [| Float.infinity; 1.0 |]) ]
  | other -> failwith ("jit_opts: unknown case " ^ other)

let leaves_of = function
  | Ojax.Tree_util.Tuple l -> List.map (fun t -> Ojax.Tree_util.tree_leaves t) l
  | t -> [ Ojax.Tree_util.tree_leaves t ]

let two f = function
  | [ Ojax.Tree_util.Leaf a; Ojax.Tree_util.Leaf b ] -> f a b
  | _ -> failwith "jit_opts: arity"

let one f = function
  | [ Ojax.Tree_util.Leaf a ] -> f a
  | _ -> failwith "jit_opts: arity"

let ones shape =
  T.Concrete
    (Ojax.Ndarray.of_floats Ojax.Dtype.F32 shape
       (Array.make (Array.fold_left ( * ) 1 shape) 1.0))

let pair a b = Ojax.Tree_util.Tuple [ leaf a; leaf b ]

let lowered_text ?donate_argnums ?keep_unused f args =
  let wrapped = Ojax.Stages.jit ?donate_argnums ?keep_unused f in
  Ojax.Stages.as_text (wrapped.Ojax.Stages.wrapped_lower (List.map leaf args))

let inner_sin ?inline () = Ojax.Api.jit ?inline (one (fun b -> leaf (F.sin b)))

let text_thunk name =
  ignore leaves_of;
  let x3 = ones [| 3 |] and y2 = ones [| 2 |] and m23 = ones [| 2; 3 |] in
  let two_f = two and mul2 a = F.multiply a (py_float 2.0) in
  match name with
  | "don_mul" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0 ]
          (two_f (fun a _ -> leaf (mul2 a)))
          [ x3; y2 ]
  | "keep_unused" ->
      fun () ->
        lowered_text ~keep_unused:true
          (two_f (fun a _ -> leaf (mul2 a)))
          [ x3; y2 ]
  | "don_pair" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0; 1 ]
          (two_f (fun a b -> pair (mul2 a) (F.add b (py_float 1.0))))
          [ x3; y2 ]
  | "don_sum" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0; 1 ]
          (two_f (fun a _ -> leaf (Ojax.Numpy.Reductions.sum a)))
          [ x3; y2 ]
  | "don_reshape" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0 ]
          (one (fun a -> leaf (Ojax.Numpy.Lax_numpy.reshape a [| 6 |])))
          [ m23 ]
  | "don_astype" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0 ]
          (one (fun a -> leaf (Ojax.Numpy.Lax_numpy.astype a Ojax.Dtype.I32)))
          [ m23 ]
  | "don_weak" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0; 1 ]
          (two_f (fun a s -> leaf (F.multiply a s)))
          [ y2; py_float 2.0 ]
  | "don_swap" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0 ]
          (two_f (fun a b -> pair b a))
          [ y2; y2 ]
  | "don_dup" ->
      fun () ->
        lowered_text ~donate_argnums:[ 0; 1 ]
          (two_f (fun a _ -> pair a a))
          [ y2; y2 ]
  | "keep_don" ->
      fun () ->
        lowered_text ~donate_argnums:[ 1 ] ~keep_unused:true
          (two_f (fun a _ -> leaf a))
          [ y2; y2 ]
  | "keep_unused_none" ->
      fun () ->
        lowered_text ~keep_unused:false
          (two_f (fun a _ -> leaf (F.add a (py_float 1.0))))
          [ y2; x3 ]
  | "don_none" ->
      fun () ->
        lowered_text ~donate_argnums:[]
          (two_f (fun a b -> leaf (F.add a b)))
          [ y2; y2 ]
  | "inline_lowered" | "noinline_lowered" ->
      let inline = name = "inline_lowered" in
      fun () ->
        lowered_text
          (one (fun a ->
               match
                 Ojax.Tree_util.tree_leaves (inner_sin ~inline () [ leaf a ])
               with
               | [ r ] -> leaf (F.add r (py_float 1.0))
               | _ -> failwith "jit_opts: arity"))
          [ x3 ]
  | "inline_jaxpr" | "noinline_jaxpr" ->
      let inline = name = "inline_jaxpr" in
      fun () ->
        Ojax.Pretty_printer.closed_jaxpr_to_jax_string
          (Ojax.Api.make_jaxpr
             (one (fun a ->
                  match
                    Ojax.Tree_util.tree_leaves (inner_sin ~inline () [ leaf a ])
                  with
                  | [ r ] -> leaf (F.add r (py_float 1.0))
                  | _ -> failwith "jit_opts: arity"))
             [ leaf x3 ])
  | other -> failwith ("jit_opts: unknown text case " ^ other)

let int_seed v =
  let dt =
    if Ojax.Config.x64_enabled () then Ojax.Dtype.I64 else Ojax.Dtype.I32
  in
  T.Concrete (Ojax.Ndarray.of_floats dt [||] [| v |])

let random_thunk ?(seed = 0.0) name =
  let module R = Ojax.Random.Core in
  let int_seed v = int_seed (v +. seed) in
  match name with
  | "key_data" -> fun () -> [ R.key_data (R.key (int_seed 11.0)) ]
  | "prng_key" -> fun () -> [ R.prng_key (int_seed 11.0) ]
  | "uniform" ->
      fun () ->
        [
          R.uniform
            (R.key (int_seed 3.0))
            ~shape:[| 4 |] ~minval:0.0 ~maxval:1.0;
        ]
  | "jit_key_data" ->
      fun () ->
        jit1
          (fun s -> leaf (R.key_data (R.key s)))
          [
            T.Concrete
              (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [||] [| 11.0 +. seed |]);
          ]
  | "gumbel" -> fun () -> [ R.gumbel (R.key (int_seed 1.0)) ~shape:[| 5 |] ]
  | "gumbel_f64" ->
      fun () ->
        [ R.gumbel ~dtype:Ojax.Dtype.F64 (R.key (int_seed 1.0)) ~shape:[| 5 |] ]
  | "categorical" ->
      fun () ->
        let logits =
          T.Concrete
            (Ojax.Ndarray.of_floats Ojax.Dtype.F32 [| 2; 4 |]
               [| 0.1; -2.0; 3.0; 0.5; 1.0; 1.0; -30.0; 2.0 |])
        in
        [ R.categorical_full ~shape:[| 6; 2 |] (R.key (int_seed 2.0)) logits ]
  | other -> failwith ("jit_opts: unknown random case " ^ other)

let deleted v =
  T.Concrete
    (Ojax.Ndarray.of_floats Ojax.Dtype.Bool [||]
       [| (if Ojax.Api.is_deleted v then 1.0 else 0.0) |])

let donate_run_thunk name () =
  let x = ones [| 3 |] in
  let mul2 a = F.multiply a (py_float 2.0) in
  let call ?static_argnums ?keep_unused ~donate_argnums f args =
    Ojax.Tree_util.tree_leaves
      (Ojax.Api.jit ?static_argnums ?keep_unused ~donate_argnums f
         (List.map leaf args))
  in
  let f args = call ~donate_argnums:[ 0 ] (one (fun a -> leaf (mul2 a))) args in
  match name with
  | "run_ok" ->
      let y = f [ x ] in
      y @ [ deleted x ]
  | "run_use" ->
      ignore (f [ x ]);
      [ F.add x (py_float 1.0) ]
  | "run_again" ->
      ignore (f [ x ]);
      f [ x ]
  | "run_unusable" ->
      let y =
        call ~donate_argnums:[ 0 ]
          (one (fun a -> leaf (Ojax.Numpy.Reductions.sum a)))
          [ x ]
      in
      y @ [ deleted x ]
  | "run_static" ->
      let y =
        call ~static_argnums:[ 0 ] ~donate_argnums:[ 1 ]
          (two (fun n a -> leaf (F.multiply a n)))
          [ py_float 3.0; x ]
      in
      y @ [ deleted x ]
  | "run_keep" ->
      let w = ones [| 2 |] in
      let y =
        call ~keep_unused:true ~donate_argnums:[ 0; 1 ]
          (two (fun a _ -> leaf (mul2 a)))
          [ x; w ]
      in
      y @ [ deleted x; deleted w ]
  | other -> failwith ("jit_opts: unknown donate case " ^ other)

let copts_thunk name () =
  let module E = Ojax.Pjrt.Executable in
  let options =
    match name with
    | "none" -> None
    | "fast_min_max_on" ->
        Some [ ("xla_cpu_enable_fast_min_max", E.Option_bool true) ]
    | "fast_min_max_off" ->
        Some [ ("xla_cpu_enable_fast_min_max", E.Option_bool false) ]
    | "fast_math_on" ->
        Some [ ("xla_cpu_enable_fast_math", E.Option_bool true) ]
    | "bogus" -> Some [ ("xla_bogus_flag", E.Option_int 1) ]
    | other -> failwith ("jit_opts: unknown copts case " ^ other)
  in
  let x = f32 [| Float.nan; 1.0; 2.0 |] in
  Ojax.Tree_util.tree_leaves
    (Ojax.Api.jit ?compiler_options:options
       (unary (fun a -> leaf (Ojax.Numpy.Reductions.max a)))
       [ leaf x ])

let special_fn name a b =
  let module S = Ojax.Scipy.Special in
  match name with
  | "betaln" -> S.betaln a b
  | "beta" -> S.beta a b
  | "betainc" -> S.betainc a b (F.multiply a (py_float 0.1))
  | "gammainc" -> S.gammainc a b
  | "gammaincc" -> S.gammaincc a b
  | "xlogy" -> S.xlogy a b
  | "xlog1py" -> S.xlog1py a b
  | "boxcox" -> S.boxcox a b
  | "boxcox1p" -> S.boxcox1p a b
  | "rel_entr" -> S.rel_entr a b
  | "kl_div" -> S.kl_div a b
  | "polygamma" -> S.polygamma a b
  | "zeta" -> S.zeta ~q:b a
  | "poch" -> S.poch a b
  | "owens_t" -> S.owens_t a b
  | "expn" -> S.expn a b
  | other -> failwith ("jit_opts: unknown special " ^ other)

let stats_fn dist fn x loc =
  let module S = Ojax.Scipy.Stats in
  let pick = function
    | "logpdf" -> `Logpdf
    | "pdf" -> `Pdf
    | "cdf" -> `Cdf
    | "logcdf" -> `Logcdf
    | "ppf" -> `Ppf
    | "sf" -> `Sf
    | "logsf" -> `Logsf
    | "isf" -> `Isf
    | other -> failwith ("jit_opts: unknown stats function " ^ other)
  in
  match (dist, pick fn) with
  | "norm", `Logpdf -> S.Norm.logpdf ~loc x
  | "norm", `Pdf -> S.Norm.pdf ~loc x
  | "norm", `Cdf -> S.Norm.cdf ~loc x
  | "norm", `Logcdf -> S.Norm.logcdf ~loc x
  | "norm", `Ppf -> S.Norm.ppf ~loc x
  | "norm", `Sf -> S.Norm.sf ~loc x
  | "norm", `Logsf -> S.Norm.logsf ~loc x
  | "norm", `Isf -> S.Norm.isf ~loc x
  | "cauchy", `Logpdf -> S.Cauchy.logpdf ~loc x
  | "cauchy", `Pdf -> S.Cauchy.pdf ~loc x
  | "cauchy", `Cdf -> S.Cauchy.cdf ~loc x
  | "cauchy", `Logcdf -> S.Cauchy.logcdf ~loc x
  | "cauchy", `Sf -> S.Cauchy.sf ~loc x
  | "cauchy", `Logsf -> S.Cauchy.logsf ~loc x
  | "cauchy", `Isf -> S.Cauchy.isf ~loc x
  | "cauchy", `Ppf -> S.Cauchy.ppf ~loc x
  | "logistic", `Logpdf -> S.Logistic.logpdf ~loc x
  | "logistic", `Pdf -> S.Logistic.pdf ~loc x
  | "logistic", `Cdf -> S.Logistic.cdf ~loc x
  | "logistic", `Sf -> S.Logistic.sf ~loc x
  | "logistic", `Ppf -> S.Logistic.ppf ~loc x
  | "logistic", `Isf -> S.Logistic.isf ~loc x
  | "expon", `Logpdf -> S.Expon.logpdf ~loc x
  | "expon", `Pdf -> S.Expon.pdf ~loc x
  | "expon", `Cdf -> S.Expon.cdf ~loc x
  | "expon", `Logsf -> S.Expon.logsf ~loc x
  | "expon", `Sf -> S.Expon.sf ~loc x
  | "expon", `Logcdf -> S.Expon.logcdf ~loc x
  | "expon", `Ppf -> S.Expon.ppf ~loc x
  | "uniform", `Logpdf -> S.Uniform.logpdf ~loc x
  | "uniform", `Pdf -> S.Uniform.pdf ~loc x
  | "uniform", `Cdf -> S.Uniform.cdf ~loc x
  | "uniform", `Ppf -> S.Uniform.ppf ~loc x
  | _ -> failwith ("jit_opts: unknown stats case " ^ dist ^ "." ^ fn)

let scaled shape offset factor =
  let f32 v = Int32.float_of_bits (Int32.bits_of_float v) in
  let factor = f32 factor in
  let n = Array.fold_left ( * ) 1 shape in
  T.Concrete
    (Ojax.Ndarray.of_floats Ojax.Dtype.F32 shape
       (Array.init n (fun i ->
            f32 (float_of_int ((i mod 5) + 1 + offset) *. factor))))

let user_dtype_thunk name dtype seed () =
  let module R = Ojax.Random.Core in
  let module X = Ojax.Random_extras in
  let k = R.key (int_seed seed) in
  let s = [| 3 |] in
  [
    (match name with
    | "bits" -> R.bits ~dtype k ~shape:s
    | "randint" -> R.randint ~dtype k ~shape:s ~minval:0 ~maxval:10
    | "uniform" -> R.uniform ~dtype k ~shape:s ~minval:0.0 ~maxval:1.0
    | "normal" -> R.normal ~dtype k ~shape:s
    | "truncated_normal" ->
        R.truncated_normal ~dtype k ~lower:(-1.0) ~upper:1.0 ~shape:s
    | "exponential" -> R.exponential ~dtype k ~shape:s
    | "cauchy" -> R.cauchy ~dtype k ~shape:s
    | "laplace" -> R.laplace ~dtype k ~shape:s
    | "logistic" -> R.logistic ~dtype k ~shape:s
    | "gumbel" -> R.gumbel ~dtype k ~shape:s
    | "pareto" -> R.pareto ~dtype k ~shape:s ~b:2.0
    | "rayleigh" -> R.rayleigh ~dtype k ~shape:s ~scale:1.5
    | "weibull_min" ->
        R.weibull_min ~dtype k ~shape:s ~scale:1.5 ~concentration:2.0
    | "lognormal" -> R.lognormal ~dtype k ~shape:s ~sigma:0.5
    | "triangular" ->
        R.triangular ~dtype k ~shape:s ~left:0.0 ~mode:0.5 ~right:1.0
    | "wald" -> R.wald ~dtype k ~shape:s ~mean:1.5
    | "geometric" -> R.geometric ~dtype k ~shape:s ~p:0.5
    | "rademacher" -> R.rademacher ~dtype k ~shape:s
    | "gamma" -> R.gamma ~dtype k ~shape:s ~a:2.0
    | "loggamma" -> R.loggamma ~dtype k ~shape:s ~a:2.0
    | "beta" -> R.beta ~dtype k ~shape:s ~a:2.0 ~b:3.0
    | "chisquare" -> R.chisquare ~dtype k ~shape:s ~df:3.0
    | "t" -> R.t ~dtype k ~shape:s ~df:3.0
    | "f" -> R.f ~dtype k ~shape:s ~dfnum:2.0 ~dfden:3.0
    | "generalized_normal" -> R.generalized_normal ~dtype k ~shape:s ~p:2.0
    | "poisson" -> R.poisson ~dtype k ~shape:s ~lam:2.0
    | "binomial" -> R.binomial ~dtype k ~shape:s ~n:5.0 ~p:0.5
    | "maxwell" -> X.maxwell ~dtype k ~shape:s
    | "double_sided_maxwell" ->
        X.double_sided_maxwell ~dtype k ~loc:0.0 ~scale:1.0 ~shape:s
    | "ball" -> X.ball ~dtype ~p:2.0 k ~d:2 ~shape:s
    | "orthogonal" -> X.orthogonal ~dtype k ~n:3 ~shape:[||]
    | other -> failwith ("jit_opts: unknown user_dtype function " ^ other));
  ]

let linalg_half_thunk name dtype mode () =
  let module L = Ojax.Lax.Linalg in
  let a =
    T.Concrete
      (Ojax.Ndarray.of_floats dtype [| 3; 3 |]
         [| 1.1; 0.1; 0.1; 0.1; 1.1; 0.1; 0.1; 0.1; 1.1 |])
  in
  let row i a = Ojax.Lax_api.index_in_dim ~keepdims:false a i in
  let f a =
    match name with
    | "cholesky" -> L.cholesky a
    | "lu" ->
        let x, _, _ = L.lu a in
        x
    | "qr" -> fst (L.qr a)
    | "eigh" -> fst (L.eigh a)
    | "eig" -> List.hd (L.eig a)
    | "svd" -> List.nth (L.svd a) 1
    | "schur" -> List.hd (L.schur a)
    | "hessenberg" -> fst (L.hessenberg a)
    | "tridiagonal" ->
        let x, _, _, _ = L.tridiagonal a in
        x
    | "tridiagonal_solve" -> L.tridiagonal_solve (row 0 a) (row 1 a) (row 2 a) a
    | "triangular_solve" -> L.triangular_solve ~left_side:true ~lower:true a a
    | other -> failwith ("jit_opts: unknown linalg_half function " ^ other)
  in
  match mode with
  | "jit" -> Ojax.Api.jit_flat (fun l -> [ f (List.hd l) ]) [ a ]
  | _ -> [ f a ]

let strings j = List.map U.to_string (U.to_list j)

let expected_of row =
  let kind = U.to_string (U.member "kind" row) in
  let str k = match U.member k row with `String s -> s | _ -> "" in
  {
    kind;
    cls = str "class";
    message =
      (if kind = "ok" && str "group" = "text" then str "text" else str "message");
    outs =
      (match U.member "outs" row with
      | `List l ->
          List.map
            (fun o ->
              ( U.to_string (U.member "dtype" o),
                List.map U.to_int (U.to_list (U.member "shape" o)),
                U.to_string (U.member "hex" o) ))
            l
      | _ -> []);
    warnings =
      List.map
        (fun w ->
          match strings w with
          | [ c; m ] -> (c, m)
          | _ -> failwith "jit_opts: warning entry")
        (U.to_list (U.member "warnings" row));
    notices = strings (U.member "notices" row);
  }

let actual_of row =
  match U.to_string (U.member "group" row) with
  | "rank" ->
      let fn = U.to_string (U.member "fn" row) in
      let mode = U.to_string (U.member "mode" row) in
      let dtype =
        Option.get
          (Ojax.Dtype.of_numpy_name (U.to_string (U.member "dtype" row)))
      in
      let shapes =
        List.map
          (fun s -> Array.of_list (List.map U.to_int (U.to_list s)))
          (U.to_list (U.member "shapes" row))
      in
      let a, b =
        match shapes with
        | [ sa; sb ] -> (operand sa dtype 0, operand sb dtype 1)
        | _ -> failwith "jit_opts: shapes"
      in
      Ojax.Config.with_value Ojax.Config.numpy_rank_promotion mode (fun () ->
          run_case (fun () -> [ rank_fn fn a b ]))
  | "special" ->
      let nans, infs =
        match U.to_string (U.member "flag" row) with
        | "nans" -> (true, false)
        | "infs" -> (false, true)
        | _ -> (true, true)
      in
      let thunk = special_thunk (U.to_string (U.member "case" row)) in
      Ojax.Config.with_value Ojax.Config.debug_nans nans (fun () ->
          Ojax.Config.with_value Ojax.Config.debug_infs infs (fun () ->
              run_case thunk))
  | "random" ->
      let arg = U.to_string (U.member "arg" row) in
      let offset = Option.value (int_of_string_opt arg) ~default:0 in
      let seed =
        match U.member "seed" row with
        | `Null -> 0.0
        | v -> float_of_int (U.to_int v)
      in
      let thunk = random_thunk ~seed (U.to_string (U.member "case" row)) in
      Ojax.Config.with_value Ojax.Config.random_seed_offset offset (fun () ->
          Ojax.Config.with_value Ojax.Config.high_dynamic_range_gumbel
            (arg = "high") (fun () -> run_case thunk))
  | "donate_run" ->
      run_case (donate_run_thunk (U.to_string (U.member "case" row)))
  | "rank_special" ->
      let fn = U.to_string (U.member "fn" row) in
      let mode = U.to_string (U.member "mode" row) in
      let shapes =
        List.map
          (fun s -> Array.of_list (List.map U.to_int (U.to_list s)))
          (U.to_list (U.member "shapes" row))
      in
      let a, b =
        match shapes with
        | [ sa; sb ] ->
            (operand sa Ojax.Dtype.F32 0, operand sb Ojax.Dtype.F32 1)
        | _ -> failwith "jit_opts: shapes"
      in
      Ojax.Config.with_value Ojax.Config.numpy_rank_promotion mode (fun () ->
          run_case (fun () -> [ special_fn fn a b ]))
  | "rank_stats" ->
      let dist = U.to_string (U.member "dist" row) in
      let fn = U.to_string (U.member "fn" row) in
      let mode = U.to_string (U.member "mode" row) in
      let shapes =
        List.map
          (fun s -> Array.of_list (List.map U.to_int (U.to_list s)))
          (U.to_list (U.member "shapes" row))
      in
      let a, b =
        match shapes with
        | [ sa; sb ] -> (scaled sa 0 0.1, scaled sb 1 0.05)
        | _ -> failwith "jit_opts: shapes"
      in
      Ojax.Config.with_value Ojax.Config.numpy_rank_promotion mode (fun () ->
          run_case (fun () -> [ stats_fn dist fn a b ]))
  | "copts" -> run_case (copts_thunk (U.to_string (U.member "case" row)))
  | "linalg_half" ->
      let dtype =
        Option.get
          (Ojax.Dtype.of_numpy_name (U.to_string (U.member "dtype" row)))
      in
      run_case
        (linalg_half_thunk
           (U.to_string (U.member "fn" row))
           dtype
           (U.to_string (U.member "mode" row)))
  | "user_dtype" ->
      let dtype =
        Option.get
          (Ojax.Dtype.of_numpy_name (U.to_string (U.member "dtype" row)))
      in
      let seed =
        match U.member "seed" row with
        | `Null -> 1.0
        | v -> float_of_int (U.to_int v)
      in
      run_case (user_dtype_thunk (U.to_string (U.member "fn" row)) dtype seed)
  | "text" ->
      let thunk = text_thunk (U.to_string (U.member "case" row)) in
      let text = ref "" in
      let r =
        run_case (fun () ->
            text := thunk ();
            [])
      in
      { r with message = (if r.kind = "ok" then !text else r.message) }
  | g -> failwith ("jit_opts: unknown group " ^ g)

let same_main e a =
  e.kind = a.kind && e.cls = a.cls && e.message = a.message && e.outs = a.outs
  && e.warnings = a.warnings

let record suite name same =
  Compare.bump suite same;
  Compare.bump (Compare.raw_prefix ^ suite) same;
  Compare.note_row suite name ~bit:same ~raw:same;
  match Lazy.force Compare.exact_log with
  | None -> ()
  | Some oc -> (
      try
        Printf.fprintf oc "%s\t%d\t%d\t%g\t%s\t%s\n%!" name 1
          (if same then 0 else 1)
          0.0 suite
          (if same then Compare.raw_identical else "differ")
      with Sys_error _ -> ())

let show r =
  Printf.sprintf "%s %s %S outs=%d warnings=[%s] notices=%d" r.kind r.cls
    r.message (List.length r.outs)
    (String.concat "; " (List.map (fun (c, m) -> c ^ ": " ^ m) r.warnings))
    (List.length r.notices)

let check_row ?(set = "jit_opts") ~x64 row () =
  let id = U.to_string (U.member "id" row) in
  let expected = expected_of row in
  let actual =
    Ojax.Config.with_value Ojax.Config.enable_x64 x64 (fun () -> actual_of row)
  in
  let suite = set ^ ":" ^ if x64 then "x64_on" else "x64_off" in
  let main = same_main expected actual in
  let notices = expected.notices = actual.notices in
  record suite id main;
  record (set ^ "_notices:" ^ if x64 then "x64_on" else "x64_off") id notices;
  if not main then
    Printf.eprintf "JIT_OPTS DIFFER %s %s\n  jax:  %s\n  ojax: %s\n%!" suite id
      (show expected) (show actual)

let suite_for ?(set = "jit_opts") x64 =
  let tag = if x64 then "x64_on" else "x64_off" in
  let path =
    Filename.concat (Filename.concat goldens_root set) (tag ^ ".json")
  in
  let rows =
    if Sys.file_exists path then
      U.to_list (U.member "rows" (Yojson.Safe.from_file path))
    else []
  in
  ( set ^ "_" ^ tag,
    List.map
      (fun row ->
        Alcotest.test_case
          (U.to_string (U.member "id" row))
          `Quick (check_row ~set ~x64 row))
      rows )
