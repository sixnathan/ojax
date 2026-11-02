module A = Option_args
module R = Ojax.Numpy.Reductions
module X = Ojax.Numpy.Extras
module AC = Ojax.Numpy.Array_creation
module NN = Ojax.Nn.Functions
module SS = Ojax.Scipy.Special

let flag options name =
  match A.member options name with Some (`Bool b) -> b | _ -> false

let pair (a, b) = [ a; b ]

let axis_array options =
  A.opt options "axis" A.int_array

let run op positional options arr =
  let v j = A.value arr j in
  match op with
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
              (v positional.(0)) (v positional.(1))))
  | "jnp.intersect1d" when flag options "return_indices" ->
      let a, b, c =
        A.refuse options [ "size"; "fill_value" ];
        X.intersect1d_indices
          ?assume_unique:(A.opt options "assume_unique" A.bool_)
          (v positional.(0)) (v positional.(1))
      in
      Some [ a; b; c ]
  | "jnp.polyfit" when flag options "full" ->
      let a, b, c, d, e =
        X.polyfit_full
          ?rcond:(A.opt options "rcond" A.float_)
          ?w:(A.opt options "w" v)
          (v positional.(0)) (v positional.(1)) (A.int_ positional.(2))
      in
      Some [ a; b; c; d; e ]
  | "jnp.polyfit" when flag options "cov" ->
      Some
        (pair
           (X.polyfit_cov
              ?rcond:(A.opt options "rcond" A.float_)
              ?w:(A.opt options "w" v)
              (v positional.(0)) (v positional.(1)) (A.int_ positional.(2))))
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
  | "jnp.size" ->
      let n = X.size ?axis:(A.opt options "axis" A.int_) (v positional.(0)) in
      Some [ Ojax.Numpy.Jnp_util.py_int n ]
  | "jnp.arange" ->
      let start = A.float_ positional.(0) in
      let is_int =
        match positional.(0) with `Int _ -> true | _ -> false
      in
      let stop_given = A.member options "stop" in
      let dtype =
        match A.member options "dtype" with
        | Some (`String _ as j) -> A.dtype j
        | _ ->
            if
              is_int
              && (match stop_given with
                 | Some (`Float _) -> false
                 | _ -> true)
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
        | Some (`Null) | None ->
            Ojax.Numpy.Lax_numpy.arange ?step ~dtype start
        | Some j ->
            Ojax.Numpy.Lax_numpy.arange ~start ?step ~dtype (A.float_ j)
      in
      Some [ r ]
  | _ -> None
