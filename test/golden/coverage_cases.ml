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

module A = Ojax.Api
module TR = Ojax.Tree_util
module UF = Ojax.Numpy.Ufuncs
module RED = Ojax.Numpy.Reductions
module AC = Ojax.Numpy.Array_creation
module LX = Ojax.Lax_api

let leaf1 = function
  | [ TR.Leaf v ] -> v
  | _ -> failwith "coverage: expected one leaf"

let f_scalar args =
  let x = leaf1 args in
  let product = UF.multiply (UF.sin x) x in
  let s = RED.sum product in
  TR.Tuple
    [ TR.Leaf s; TR.Leaf (UF.multiply x (Ojax.Numpy.Jnp_util.py_float 2.0)) ]

let f_vec args =
  let x = leaf1 args in
  let product = UF.multiply (UF.sin x) x in
  TR.Tuple
    [
      TR.Leaf product;
      TR.Leaf (UF.multiply x (Ojax.Numpy.Jnp_util.py_float 2.0));
    ]

let f_holo args =
  let x = leaf1 args in
  TR.Leaf (RED.sum (UF.multiply (UF.multiply x x) x))

let leaves t = TR.tree_leaves t
let doubled v = UF.multiply v (Ojax.Numpy.Jnp_util.py_float 2.0)

let f_plain args =
  let x = leaf1 args in
  TR.Leaf (UF.multiply (UF.sin x) x)

let f_plain_sum args =
  let x = leaf1 args in
  TR.Leaf (RED.sum (UF.multiply (UF.sin x) x))

let f_pair = function
  | [ TR.Tuple [ TR.Leaf x; TR.Leaf y ] ] ->
      let first = UF.multiply x y in
      TR.Tuple [ TR.Leaf first; TR.Leaf (UF.sin y) ]
  | _ -> failwith "coverage: pair"

let transform_run op a =
  let x () = TR.Leaf (a ()) in
  match op with
  | "cv_t_grad_aux" ->
      let g, aux = A.grad_aux f_scalar [ x () ] in
      Some (leaves g @ leaves aux)
  | "cv_t_vg_aux" ->
      let y, g, aux = A.value_and_grad_aux f_scalar [ x () ] in
      Some (leaves y @ leaves aux @ leaves g)
  | "cv_t_jvp_aux" ->
      let tangents = TR.Leaf (AC.ones_like (a ())) in
      let y, t, aux = A.jvp_aux f_vec [ x () ] [ tangents ] in
      Some (leaves y @ leaves t @ leaves aux)
  | "cv_t_vjp_aux" ->
      let y, back, aux = A.vjp_aux f_vec [ x () ] in
      let g = back (TR.Leaf (AC.ones_like (a ()))) in
      Some (leaves y @ List.concat_map leaves g @ leaves aux)
  | "cv_t_linearize_aux" ->
      let y, lin, aux = A.linearize_aux f_vec [ x () ] in
      let t = lin [ TR.Leaf (AC.ones_like (a ())) ] in
      Some (leaves y @ leaves t @ leaves aux)
  | "cv_t_jacfwd_aux" ->
      let j, aux = A.jacfwd_aux f_vec [ x () ] in
      Some (leaves j @ leaves aux)
  | "cv_t_jacrev_aux" ->
      let j, aux = A.jacrev_aux f_vec [ x () ] in
      Some (leaves j @ leaves aux)
  | "cv_t_hessian_aux" ->
      let j, aux = A.hessian_aux f_scalar [ x () ] in
      Some (leaves j @ leaves aux)
  | "cv_t_jacfwd" | "cv_t_jacfwd_scalar" ->
      Some (leaves (A.jacfwd f_plain [ x () ]))
  | "cv_t_jacrev" | "cv_t_jacrev_scalar" ->
      Some (leaves (A.jacrev f_plain [ x () ]))
  | "cv_t_hessian" -> Some (leaves (A.hessian f_plain_sum [ x () ]))
  | "cv_t_jacfwd_pair" ->
      let v = a () in
      Some
        (leaves
           (A.jacfwd f_pair [ TR.Tuple [ TR.Leaf v; TR.Leaf (doubled v) ] ]))
  | "cv_t_jacrev_pair" ->
      let v = a () in
      Some
        (leaves
           (A.jacrev f_pair [ TR.Tuple [ TR.Leaf v; TR.Leaf (doubled v) ] ]))
  | "cv_t_grad_holo" -> Some (leaves (A.grad_holomorphic f_holo [ x () ]))
  | "cv_t_vg_holo" ->
      let y, g = A.value_and_grad_holomorphic f_holo [ x () ] in
      Some (leaves y @ leaves g)
  | _ -> None

let idx_column p = NL.reshape (int_list "idx" p) [| -1; 1 |]

let scatter_dims () =
  {
    T.update_window_dims = [||];
    inserted_window_dims = [| 0 |];
    scatter_dims_to_operand_dims = [| 0 |];
    s_operand_batching_dims = [||];
    s_scatter_indices_batching_dims = [||];
    s_sorted_unique = false;
    s_clip = false;
  }

let gather_dims () =
  {
    T.offset_dims = [||];
    collapsed_slice_dims = [| 0 |];
    start_index_map = [| 0 |];
    g_operand_batching_dims = [||];
    g_start_indices_batching_dims = [||];
    g_sorted_unique = false;
  }

let shifted_argmax x = UF.subtract (NL.argmax x) (Ojax.Numpy.Jnp_util.py_int 1)

let prefix x n =
  let sh = (Ojax.Core.get_aval x).T.shape in
  let start = Array.make (Array.length sh) 0 in
  let limit = Array.copy sh in
  limit.(0) <- n;
  Ojax.Core.bind1
    (T.Slice { start_indices = start; limit_indices = limit; strides = None })
    [ x ]

let head_two x = prefix x 2

let lax_run op p a =
  let scatter
      (f :
        ?indices_are_sorted:bool ->
        ?unique_indices:bool ->
        T.value ->
        T.value ->
        T.value ->
        T.scatter_dims ->
        T.value) =
    let x = a () in
    let idx = idx_column p in
    let n = Array.length (ints "idx" p) in
    let updates = UF.multiply (prefix x n) (Ojax.Numpy.Jnp_util.py_float 2.0) in
    Some
      [
        f ?indices_are_sorted:(opt_bool "sorted" p)
          ?unique_indices:(opt_bool "unique" p) x idx updates (scatter_dims ());
      ]
  in
  let allow = opt_bool "allow" p in
  match op with
  | "cv_l_scatter" ->
      scatter (fun ?indices_are_sorted ?unique_indices x i u d ->
          LX.scatter ?indices_are_sorted ?unique_indices x i u d)
  | "cv_l_scatter_add" ->
      scatter (fun ?indices_are_sorted ?unique_indices x i u d ->
          LX.scatter_add ?indices_are_sorted ?unique_indices x i u d)
  | "cv_l_scatter_sub" ->
      scatter (fun ?indices_are_sorted ?unique_indices x i u d ->
          LX.scatter_sub ?indices_are_sorted ?unique_indices x i u d)
  | "cv_l_scatter_mul" ->
      scatter (fun ?indices_are_sorted ?unique_indices x i u d ->
          LX.scatter_mul ?indices_are_sorted ?unique_indices x i u d)
  | "cv_l_scatter_min" ->
      scatter (fun ?indices_are_sorted ?unique_indices x i u d ->
          LX.scatter_min ?indices_are_sorted ?unique_indices x i u d)
  | "cv_l_scatter_max" ->
      scatter (fun ?indices_are_sorted ?unique_indices x i u d ->
          LX.scatter_max ?indices_are_sorted ?unique_indices x i u d)
  | "cv_l_gather" ->
      let mode =
        match U.to_string (U.member "mode" p) with
        | "clip" -> Ojax.Lax_config.Clip
        | "fill" -> Ojax.Lax_config.Fill_or_drop
        | _ -> Ojax.Lax_config.Promise_in_bounds
      in
      let idx = idx_column p in
      Some
        [
          LX.gather ~mode ?indices_are_sorted:(opt_bool "sorted" p)
            ?unique_indices:(opt_bool "unique" p) (a ()) idx (gather_dims ())
            [| 1 |];
        ]
  | "cv_l_dynamic_slice" ->
      let x = a () in
      Some
        [
          LX.dynamic_slice ?allow_negative_indices:allow x
            [ shifted_argmax x ]
            [| 2 |];
        ]
  | "cv_l_dynamic_slice_in_dim" ->
      let x = a () in
      Some
        [
          LX.dynamic_slice_in_dim ?allow_negative_indices:allow x
            (shifted_argmax x) 2;
        ]
  | "cv_l_dynamic_update_slice" ->
      let x = a () in
      let update = head_two x in
      Some
        [
          LX.dynamic_update_slice ?allow_negative_indices:allow x update
            [ shifted_argmax x ];
        ]
  | "cv_l_dynamic_update_slice_in_dim" ->
      let x = a () in
      let update = head_two x in
      Some
        [
          LX.dynamic_update_slice_in_dim ?allow_negative_indices:allow x update
            (shifted_argmax x);
        ]
  | "cv_l_fori_unroll" ->
      let py i = Ojax.Numpy.Jnp_util.py_int i in
      Some
        (LX.fori_loop ~unroll:(PC.int "unroll" p) (py 0)
           (py (PC.int "n" p))
           (fun _ rest ->
             List.map
               (fun v -> UF.multiply v (Ojax.Numpy.Jnp_util.py_float 2.0))
               rest)
           [ a () ])
  | "cv_l_collapse_from" -> Some [ LX.collapse_from (a ()) (PC.int "start" p) ]
  | _ -> None

module SG = Ojax.Ops.Segment
module STC = Ojax.Scipy.Stats.Stats_core
module RX = Ojax.Random_extras
module INI = Ojax.Nn.Initializers

let segment_run op p a =
  let call f =
    let ids = int_list "ids" p in
    Some
      [
        f ?num_segments:(opt_int "n" p)
          ?indices_are_sorted:(opt_bool "sorted" p)
          ?unique_indices:(opt_bool "unique" p)
          ?bucket_size:(opt_int "bucket" p) ?mode:(opt_string "mode" p) (a ())
          ids;
      ]
  in
  match op with
  | "cv_s_sum" -> call SG.segment_sum
  | "cv_s_prod" -> call SG.segment_prod
  | "cv_s_max" -> call SG.segment_max
  | "cv_s_min" -> call SG.segment_min
  | _ -> None

let misc_run op p a =
  let k () = PC.seed_key p in
  let shape () = ints "shape" p in
  let dtype = opt_dtype p in
  let init_of = function
    | "cv_i_normal" -> Some (INI.normal ())
    | "cv_i_uniform" -> Some (INI.uniform ())
    | "cv_i_truncated_normal" -> Some (INI.truncated_normal ())
    | "cv_i_glorot_normal" -> Some (INI.glorot_normal ())
    | "cv_i_glorot_uniform" -> Some (INI.glorot_uniform ())
    | "cv_i_he_normal" -> Some (INI.he_normal ())
    | "cv_i_he_uniform" -> Some (INI.he_uniform ())
    | "cv_i_lecun_normal" -> Some (INI.lecun_normal ())
    | "cv_i_lecun_uniform" -> Some (INI.lecun_uniform ())
    | "cv_i_orthogonal" -> Some (INI.orthogonal ())
    | "cv_i_delta_orthogonal" -> Some (INI.delta_orthogonal ())
    | _ -> None
  in
  match init_of op with
  | Some init ->
      Some [ INI.with_dtype (Option.get dtype) init (k ()) ~shape:(shape ()) ]
  | None -> (
      match op with
      | "cv_sem" ->
          Some
            [
              STC.sem ?axis:(opt_int "axis" p)
                ~ddof:(Option.value (opt_int "ddof" p) ~default:1)
                ?nan_policy:(opt_string "nan_policy" p)
                ?keepdims:(opt_bool "keepdims" p) (a ());
            ]
      | "cv_take_along" ->
          let x = a () in
          let sorted = Ojax.Numpy.Sorting.argsort ~axis:(Some 0) x in
          let idx =
            UF.subtract sorted (Ojax.Numpy.Jnp_util.py_int (PC.int "shift" p))
          in
          Some
            [
              IDX.take_along_axis ~axis:0 ?mode:(opt_string "mode" p)
                ?wrap_negative_indices:(opt_bool "wrap" p) x idx;
            ]
      | "cv_tree_map_leaf" ->
          let x = a () in
          let tree =
            TR.Dict
              [
                ( "p",
                  TR.Tuple
                    [
                      TR.Leaf x;
                      TR.Leaf (UF.multiply x (Ojax.Numpy.Jnp_util.py_float 3.0));
                    ] );
                ("q", TR.Leaf x);
              ]
          in
          let mapped =
            TR.tree_map_is_leaf
              ~is_leaf:(function TR.Tuple _ -> true | _ -> false)
              (function
                | TR.Tuple [ TR.Leaf l; TR.Leaf r ] -> TR.Leaf (UF.add l r)
                | TR.Leaf v -> TR.Leaf (doubled v)
                | other -> other)
              tree
          in
          Some (TR.tree_leaves mapped)
      | "cv_x_maxwell" -> Some [ RX.maxwell ?dtype (k ()) ~shape:(shape ()) ]
      | "cv_x_double_sided_maxwell" ->
          Some
            [
              RX.double_sided_maxwell ?dtype (k ()) ~loc:0.5 ~scale:2.0
                ~shape:(shape ());
            ]
      | "cv_x_ball" ->
          Some [ RX.ball ?dtype (k ()) ~d:(PC.int "d" p) ~shape:(shape ()) ]
      | "cv_x_orthogonal" ->
          Some
            [ RX.orthogonal ?dtype (k ()) ~n:(PC.int "n" p) ~shape:(shape ()) ]
      | _ -> None)

let run op params get =
  let a () = get "a" in
  let b () = get "b" in
  let runners =
    [
      (fun () -> random_run op params a b);
      (fun () -> gather_run op params a);
      (fun () -> transform_run op a);
      (fun () -> lax_run op params a);
      (fun () -> segment_run op params a);
      (fun () -> misc_run op params a);
    ]
  in
  match List.find_map (fun r -> r ()) runners with
  | Some out -> out
  | None -> failwith ("coverage: unknown op " ^ op)
