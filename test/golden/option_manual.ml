module A = Option_args
module R = Ojax.Numpy.Reductions
module X = Ojax.Numpy.Extras
module AC = Ojax.Numpy.Array_creation
module NN = Ojax.Nn.Functions
module SS = Ojax.Scipy.Special

let flag options name =
  match A.member options name with Some (`Bool b) -> b | _ -> false

let pair (a, b) = [ a; b ]
let axis_array options = A.opt options "axis" A.int_array

let flags =
  [
    "returned";
    "retstep";
    "return_indices";
    "full";
    "cov";
    "return_sign";
    "return_residual";
    "sparse";
  ]

let without_false_flags options =
  match options with
  | `Assoc fields ->
      `Assoc
        (List.filter
           (fun (name, value) ->
             not (List.mem name flags && value = `Bool false))
           fields)
  | other -> other

let signal_option op (name, value) =
  let signal = String.length op > 10 && String.sub op 0 10 = "sp_signal." in
  match (name, value) with
  | "window", `String "hann" when signal -> None
  | "detrend", `Bool false when signal -> Some (name, `Null)
  | "detrend", `Bool true when signal -> Some (name, `String "True")
  | _ -> Some (name, value)

let normalize op options =
  match options with
  | `Assoc fields -> `Assoc (List.filter_map (signal_option op) fields)
  | other -> other

let float_pair j =
  match A.elements j with
  | [ a; b ] -> (A.float_ a, A.float_ b)
  | _ -> A.no "type:pair"

let scalar_dtype : A.json -> Ojax.Dtype.t = function
  | `Bool _ -> Ojax.Dtype.Bool
  | `Int _ -> Ojax.Dtypes.default_int_dtype ()
  | `Float _ -> Ojax.Dtypes.default_float_dtype ()
  | _ -> A.no "type:dtype"

let operands arr =
  List.concat_map (function
    | `Assoc [ (("$list" | "$star"), _) ] as j -> A.value_list arr j
    | j -> [ A.value arr j ])

let run op positional options arr =
  let v j = A.value arr j in
  match op with
  | "lax.collapse" -> (
      let operand = v positional.(0) and start = A.int_ positional.(1) in
      match A.get options "stop_dimension" `Null with
      | `Null -> Some [ Ojax.Lax_api.collapse_from operand start ]
      | j -> Some [ Ojax.Lax_api.collapse operand start (A.int_ j) ])
  | "lax.full" ->
      let dtype =
        match A.get options "dtype" `Null with
        | `Null -> scalar_dtype positional.(1)
        | j -> A.dtype j
      in
      Some
        [
          Ojax.Lax_api.full
            (A.int_array positional.(0))
            (A.float_ positional.(1))
            dtype;
        ]
  | "jnp.einsum" ->
      let operands = operands arr (List.tl (Array.to_list positional)) in
      Some
        [
          Ojax.Numpy.Einsum.einsum
            ?optimize:(A.opt options "optimize" A.einsum_optimize)
            ?precision:(A.opt options "precision" A.precision)
            ?preferred_element_type:
              (A.opt options "preferred_element_type" A.dtype)
            (A.string_ positional.(0))
            operands;
        ]
  | "jnp.einsum_path" ->
      let operands = operands arr (List.tl (Array.to_list positional)) in
      ignore
        (Ojax.Numpy.Einsum.einsum_path
           ?optimize:(A.opt options "optimize" A.einsum_optimize)
           (A.string_ positional.(0))
           operands);
      A.no "result:einsum_path"
  | "jnp.histogramdd" ->
      let range =
        A.opt options "range" (fun j ->
            Array.of_list (List.map float_pair (A.elements j)))
      in
      let weights = A.opt options "weights" v in
      let density = A.opt options "density" A.bool_ in
      let sample = v positional.(0) in
      let hist, edges =
        match A.get options "bins" `Null with
        | `Null -> X.histogramdd_uniform ?range ?weights ?density sample
        | `Int n ->
            X.histogramdd_uniform ~bins:n ?range ?weights ?density sample
        | j ->
            X.histogramdd ~bins:(A.int_array j) ?range ?weights ?density sample
      in
      Some (hist :: edges)
  | "nn.dot_product_attention" when flag options "return_residual" ->
      Some
        (pair
           (NN.dot_product_attention_residual ?bias:(A.opt options "bias" v)
              ?mask:(A.opt options "mask" v)
              ?scale:(A.opt options "scale" A.float_)
              ?is_causal:(A.opt options "is_causal" A.bool_)
              ?query_seq_lengths:(A.opt options "query_seq_lengths" v)
              ?key_value_seq_lengths:(A.opt options "key_value_seq_lengths" v)
              ?local_window_size:
                (A.opt options "local_window_size" (function
                  | `Int n -> (n, n)
                  | j -> A.int_pair j))
              ~query:(v positional.(0))
              ~key:(v positional.(1))
              ~value:(v positional.(2))
              ()))
  | "lax_linalg.qr" ->
      Some
        (Ojax.Lax.Linalg.qr_full
           ?pivoting:(A.opt options "pivoting" A.bool_)
           ?full_matrices:(A.opt options "full_matrices" A.bool_)
           (v positional.(0)))
  | "jnp.average" when flag options "returned" ->
      Some
        (pair
           (R.average_returned ?axis:(axis_array options)
              ?keepdims:(A.opt options "keepdims" A.bool_)
              ?weights:(A.opt options "weights" v)
              (v positional.(0))))
  | "jnp.linspace" when flag options "retstep" ->
      Some
        (pair
           (AC.linspace_retstep
              ?num:(A.opt options "num" A.int_)
              ?endpoint:(A.opt options "endpoint" A.bool_)
              ?axis:(A.opt options "axis" A.int_)
              ?dtype:(A.opt options "dtype" A.dtype)
              (v positional.(0))
              (v positional.(1))))
  | "jnp.intersect1d" when flag options "return_indices" ->
      let a, b, c =
        A.refuse options [ "size"; "fill_value" ];
        X.intersect1d_indices
          ?assume_unique:(A.opt options "assume_unique" A.bool_)
          (v positional.(0))
          (v positional.(1))
      in
      Some [ a; b; c ]
  | "jnp.polyfit" when flag options "full" ->
      let a, b, c, d, e =
        X.polyfit_full
          ?rcond:(A.opt options "rcond" A.float_)
          ?w:(A.opt options "w" v)
          (v positional.(0))
          (v positional.(1))
          (A.int_ positional.(2))
      in
      Some [ a; b; c; d; e ]
  | "jnp.polyfit" when flag options "cov" ->
      Some
        (pair
           (X.polyfit_cov
              ?rcond:(A.opt options "rcond" A.float_)
              ?w:(A.opt options "w" v)
              (v positional.(0))
              (v positional.(1))
              (A.int_ positional.(2))))
  | "nn.logsumexp" when flag options "return_sign" ->
      Some
        (pair
           (NN.logsumexp_sign ?axis:(axis_array options)
              ?b:(A.opt options "b" v)
              ?keepdims:(A.opt options "keepdims" A.bool_)
              ?where:(A.opt options "where" v)
              (v positional.(0))))
  | "sp_special.logsumexp" when flag options "return_sign" ->
      Some
        (pair
           (SS.logsumexp_sign ?axis:(axis_array options)
              ?b:(A.opt options "b" v)
              ?keepdims:(A.opt options "keepdims" A.bool_)
              ?where:(A.opt options "where" v)
              (v positional.(0))))
  | "jnp.unique_all" ->
      let r =
        X.unique_all
          ?size:(A.opt options "size" A.int_)
          ?fill_value:(A.opt options "fill_value" A.float_)
          (v positional.(0))
      in
      Some [ r.X.values; r.X.indices; r.X.inverse; r.X.counts ]
  | "jnp.indices" when flag options "sparse" ->
      Some
        (Ojax.Numpy.Lax_numpy.indices_sparse
           ~dtype:(A.default_int (A.get options "dtype" `Null))
           (A.int_array positional.(0)))
  | "jnp.rollaxis" ->
      Some
        [
          Ojax.Numpy.Lax_numpy.rollaxis
            ?start:(A.opt options "start" A.int_)
            (A.int_ positional.(1))
            (v positional.(0));
        ]
  | "sp_linalg.cho_factor" ->
      let c, lower =
        Ojax.Scipy.Linalg.cho_factor
          ?lower:(A.opt options "lower" A.bool_)
          (v positional.(0))
      in
      Some [ c; v (`Bool lower) ]
  | "jnp.size" ->
      let n =
        X.size ?axis:(A.opt options "axis" A.int_array) (v positional.(0))
      in
      Some [ Ojax.Numpy.Jnp_util.py_int n ]
  | "jnp.arange" ->
      let start = A.float_ positional.(0) in
      let is_int = match positional.(0) with `Int _ -> true | _ -> false in
      let stop_given = A.member options "stop" in
      let dtype =
        match A.member options "dtype" with
        | Some (`String _ as j) -> A.dtype j
        | _ ->
            if
              is_int
              && (match stop_given with Some (`Float _) -> false | _ -> true)
              &&
              match A.member options "step" with
              | Some (`Float _) -> false
              | _ -> true
            then Ojax.Dtypes.default_int_dtype ()
            else Ojax.Dtypes.default_float_dtype ()
      in
      let step = A.opt options "step" A.float_ in
      let r =
        match stop_given with
        | Some `Null | None -> Ojax.Numpy.Lax_numpy.arange ?step ~dtype start
        | Some j -> Ojax.Numpy.Lax_numpy.arange ~start ?step ~dtype (A.float_ j)
      in
      Some [ r ]
  | _ -> None
