module U = Yojson.Safe.Util
module T = Ojax.Types
module D = Ojax.Dtype
module PC = Parity_cases
module RC = Ojax.Random.Core
module NL = Ojax.Numpy.Lax_numpy
module IDX = Ojax.Numpy.Indexing

let dtype_named = function
  | "float16" -> D.F16
  | "bfloat16" -> D.Bf16
  | "float32" -> D.F32
  | "float64" -> D.F64
  | "complex64" -> D.Complex64
  | "complex128" -> D.Complex128
  | "int8" -> D.I8
  | "int16" -> D.I16
  | "int32" -> D.I32
  | "int64" -> D.I64
  | "uint8" -> D.Uint8
  | "uint16" -> D.Uint16
  | "uint32" -> D.Uint32
  | "uint64" -> D.Uint64
  | "bool" -> D.Bool
  | other -> failwith ("coverage: dtype " ^ other)

let opt_dtype p =
  match U.member "dtype" p with
  | `Null -> None
  | v -> Some (dtype_named (U.to_string v))

let opt_string = PC.opt_string
let opt_bool = PC.opt_bool
let opt_int = PC.opt_int
let ints = PC.ints

let opt_shape p =
  match U.member "shape" p with `Null -> None | _ -> Some (ints "shape" p)

let float_ name p =
  match U.member name p with
  | `Float f -> f
  | `Int i -> float_of_int i
  | _ -> failwith ("coverage: float " ^ name)

let random_run op p a b =
  let k () = PC.seed_key p in
  let shape () = ints "shape" p in
  let dtype = opt_dtype p in
  let sampler f = Some [ f ?dtype (k ()) ~shape:(shape ()) ] in
  match op with
  | "cv_r_normal" -> sampler RC.normal
  | "cv_r_exponential" -> sampler RC.exponential
  | "cv_r_cauchy" -> sampler RC.cauchy
  | "cv_r_laplace" -> sampler RC.laplace
  | "cv_r_logistic" -> sampler RC.logistic
  | "cv_r_rademacher" -> sampler RC.rademacher
  | "cv_r_uniform" ->
      Some
        [
          RC.uniform ?dtype (k ()) ~shape:(shape ()) ~minval:(-1.0) ~maxval:3.0;
        ]
  | "cv_r_truncated_normal" ->
      Some
        [
          RC.truncated_normal ?dtype (k ()) ~lower:(-1.0) ~upper:2.0
            ~shape:(shape ());
        ]
  | "cv_r_randint" ->
      Some
        [ RC.randint ?dtype (k ()) ~shape:(shape ()) ~minval:(-3) ~maxval:50 ]
  | "cv_r_gumbel" ->
      Some
        [
          RC.gumbel ?dtype ?mode:(opt_string "mode" p) (k ()) ~shape:(shape ());
        ]
  | "cv_r_bernoulli" ->
      Some
        [
          RC.bernoulli ?mode:(opt_string "mode" p) (k ()) ~shape:(shape ())
            ~p:0.4;
        ]
  | "cv_r_gamma" -> Some [ RC.gamma ?dtype (k ()) ~shape:(shape ()) ~a:2.5 ]
  | "cv_r_loggamma" ->
      Some [ RC.loggamma ?dtype (k ()) ~shape:(shape ()) ~a:2.5 ]
  | "cv_r_beta" ->
      Some [ RC.beta ?dtype (k ()) ~shape:(shape ()) ~a:2.0 ~b:3.0 ]
  | "cv_r_chisquare" ->
      Some [ RC.chisquare ?dtype (k ()) ~shape:(shape ()) ~df:3.0 ]
  | "cv_r_t" -> Some [ RC.t ?dtype (k ()) ~shape:(shape ()) ~df:4.0 ]
  | "cv_r_f" ->
      Some [ RC.f ?dtype (k ()) ~shape:(shape ()) ~dfnum:3.0 ~dfden:5.0 ]
  | "cv_r_pareto" -> Some [ RC.pareto ?dtype (k ()) ~shape:(shape ()) ~b:2.0 ]
  | "cv_r_rayleigh" ->
      Some [ RC.rayleigh ?dtype (k ()) ~shape:(shape ()) ~scale:1.5 ]
  | "cv_r_wald" -> Some [ RC.wald ?dtype (k ()) ~shape:(shape ()) ~mean:1.5 ]
  | "cv_r_lognormal" ->
      Some [ RC.lognormal ?dtype (k ()) ~shape:(shape ()) ~sigma:0.5 ]
  | "cv_r_geometric" ->
      Some [ RC.geometric ?dtype (k ()) ~shape:(shape ()) ~p:0.3 ]
  | "cv_r_poisson" ->
      Some [ RC.poisson ?dtype (k ()) ~shape:(shape ()) ~lam:3.0 ]
  | "cv_r_binomial" ->
      Some [ RC.binomial ?dtype (k ()) ~shape:(shape ()) ~count:10.0 ~prob:0.3 ]
  | "cv_r_dirichlet" ->
      let dt = Option.get dtype in
      let alpha =
        T.Concrete (Ojax.Ndarray.of_floats dt [| 3 |] [| 1.0; 2.0; 3.0 |])
      in
      Some [ RC.dirichlet ?dtype (k ()) ~alpha ~shape:(shape ()) ]
  | "cv_r_permutation" ->
      Some
        [
          RC.permutation_array ?axis:(opt_int "axis" p)
            ?independent:(opt_bool "independent" p) (k ()) (a ());
        ]
  | "cv_r_choice" ->
      Some
        [
          RC.choice_array ~shape:(shape ()) ?replace:(opt_bool "replace" p)
            ?axis:(opt_int "axis" p) (k ()) (a ());
        ]
  | "cv_r_choice_p" ->
      let key = k () in
      let probs = Ojax.Numpy.Ufuncs.abs (b ()) in
      Some
        [
          RC.choice_array ~shape:(shape ()) ?replace:(opt_bool "replace" p)
            ?axis:(opt_int "axis" p) ?mode:(opt_string "mode" p) ~p:probs key
            (a ());
        ]
  | "cv_r_choice_int" ->
      Some
        [
          RC.choice_array ~shape:(shape ()) ?replace:(opt_bool "replace" p)
            (k ())
            (Ojax.Numpy.Jnp_util.py_int (PC.int "n" p));
        ]
  | "cv_r_categorical" ->
      Some
        [
          RC.categorical_full ?axis:(opt_int "axis" p) ?shape:(opt_shape p)
            ?replace:(opt_bool "replace" p) ?mode:(opt_string "mode" p) (k ())
            (a ());
        ]
  | "cv_r_multinomial" ->
      let key = k () in
      let x = Ojax.Numpy.Ufuncs.abs (a ()) in
      let y = Ojax.Numpy.Ufuncs.abs (a ()) in
      let probs =
        Ojax.Numpy.Ufuncs.true_divide x
          (Ojax.Numpy.Reductions.sum ~axis:[| -1 |] ~keepdims:true y)
      in
      Some
        [
          RC.multinomial_full ?shape:(opt_shape p) ?dtype
            ?unroll:(opt_int "unroll" p) key
            ~n:(Ojax.Numpy.Jnp_util.py_float (float_ "n" p))
            ~p:probs;
        ]
  | _ -> None

let int_list name p =
  let values = ints name p in
  T.Concrete
    (Ojax.Ndarray.of_floats
       (Ojax.Dtypes.default_int_dtype ())
       [| Array.length values |]
       (Array.map float_of_int values))

let gather_run op p a =
  match op with
  | "cv_take_flags" ->
      Some
        [
          IDX.take ?axis:(opt_int "axis" p) ?mode:(opt_string "mode" p)
            ?unique_indices:(opt_bool "unique" p)
            ?indices_are_sorted:(opt_bool "sorted" p) (a ()) (int_list "idx" p);
        ]
  | _ -> None

let run op params get =
  let a () = get "a" in
  let b () = get "b" in
  let runners =
    [ (fun () -> random_run op params a b); (fun () -> gather_run op params a) ]
  in
  match List.find_map (fun r -> r ()) runners with
  | Some out -> out
  | None -> failwith ("coverage: unknown op " ^ op)
