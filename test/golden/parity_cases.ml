module U = Yojson.Safe.Util
module F = Ojax.Numpy.Fft

let opt_int name params =
  match U.member name params with `Null -> None | v -> Some (U.to_int v)

let opt_string name params =
  match U.member name params with `Null -> None | v -> Some (U.to_string v)

let opt_ints name params =
  match U.member name params with
  | `Null -> None
  | v -> Some (List.map U.to_int (U.to_list v))

let fft1 op a p =
  let n = opt_int "n" p in
  let axis = opt_int "axis" p in
  let norm = opt_string "norm" p in
  [ op ?n ?axis ?norm a ]

let fftnd op a p =
  [
    op ?s:(opt_ints "s" p) ?axes:(opt_ints "axes" p) ?norm:(opt_string "norm" p)
      a;
  ]

module L = Ojax.Lax_api
module T = Ojax.Types

let ints name p =
  Array.of_list (List.map U.to_int (U.to_list (U.member name p)))

let int name p = U.to_int (U.member name p)

let opt_bool name p =
  match U.member name p with `Null -> None | v -> Some (U.to_bool v)

let like x v = L.full [||] v (Ojax.Core.get_aval x).dtype
let dt_of_input x = (Ojax.Core.get_aval x).dtype

let scalar_int n =
  Ojax.Types.Concrete
    (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [||] [| float_of_int n |])

let padding_of p =
  match U.to_string (U.member "padding" p) with
  | "SAME" -> L.Same
  | "VALID" -> L.Valid
  | other -> failwith ("parity: padding " ^ other)

let dn_of p =
  match List.map U.to_string (U.to_list (U.member "dn" p)) with
  | [ l; r; o ] -> L.conv_dimension_numbers (l, r, o)
  | _ -> failwith "parity: dn"

let opt_int_array name p =
  match U.member name p with `Null -> None | _ -> Some (ints name p)

let newton f x0 =
  let open Ojax.Interpreters in
  let ones v = L.full_like v 1.0 in
  let step x =
    let fx = f x in
    let _, slope = Ad.jvp f x (List.map ones x) in
    List.map2
      (fun xi (fi, si) -> L.sub xi (L.div fi si))
      x (List.combine fx slope)
  in
  let rec go x n = if n = 0 then x else go (step x) (n - 1) in
  go x0 40

let lax_run op p a b =
  match op with
  | "lax_clamp" ->
      Some [ L.clamp (like (a ()) (-0.5)) (a ()) (like (a ()) 0.5) ]
  | "lax_add_scalar" -> Some [ L.add (a ()) (like (a ()) 2.0) ]
  | "lax_expand_dims" -> Some [ L.expand_dims (a ()) (ints "dims" p) ]
  | "lax_broadcast" -> Some [ L.broadcast (a ()) (ints "sizes" p) ]
  | "lax_collapse" -> Some [ L.collapse (a ()) (int "start" p) (int "stop" p) ]
  | "lax_slice_in_dim" ->
      Some
        [
          L.slice_in_dim ?stride:(opt_int "stride" p) ?axis:(opt_int "axis" p)
            (a ()) (int "start" p) (int "stop" p);
        ]
  | "lax_index_in_dim" ->
      Some
        [
          L.index_in_dim ?axis:(opt_int "axis" p)
            ?keepdims:(opt_bool "keepdims" p) (a ()) (int "index" p);
        ]
  | "lax_dynamic_slice_in_dim" ->
      Some
        [
          L.dynamic_slice_in_dim ?axis:(opt_int "axis" p) (a ())
            (Ojax.Numpy.Jnp_util.py_int (int "start" p))
            (int "size" p);
        ]
  | "lax_dynamic_index_in_dim" ->
      Some
        [
          L.dynamic_index_in_dim ?axis:(opt_int "axis" p)
            ?keepdims:(opt_bool "keepdims" p) (a ())
            (Ojax.Numpy.Jnp_util.py_int (int "index" p));
        ]
  | "lax_reduce_custom" ->
      let zero =
        T.Concrete (Ojax.Ndarray.of_floats (dt_of_input (a ())) [||] [| 0.0 |])
      in
      Some [ L.reduce (a ()) zero L.add (ints "dims" p) ]
  | "lax_reduce_window" ->
      let pad =
        Array.of_list
          (List.map
             (fun pair ->
               match U.to_list pair with
               | [ lo; hi ] -> (U.to_int lo, U.to_int hi)
               | _ -> failwith "parity: pad")
             (U.to_list (U.member "pad" p)))
      in
      let init =
        T.Concrete (Ojax.Ndarray.of_floats (dt_of_input (a ())) [||] [| 0.0 |])
      in
      let init =
        if Ojax.Core.staging () then
          Ojax.Core.full_raise (Ojax.Core.find_top_trace [ a () ]) init
        else init
      in
      Some
        [
          L.reduce_window (a ()) init L.add ~window_dimensions:(ints "win" p)
            ~window_strides:(ints "str" p) pad;
        ]
  | "lax_sort" -> Some (L.sort ?dimension:(opt_int "dimension" p) [ a () ])
  | "lax_sort_key_val" ->
      let k, v = L.sort_key_val (a ()) (b ()) in
      Some [ k; v ]
  | "lax_top_k" ->
      let v, i = L.top_k (a ()) (int "k" p) in
      Some [ v; i ]
  | "lax_dot" -> Some [ L.dot (a ()) (b ()) ]
  | "lax_batch_matmul" -> Some [ L.batch_matmul (a ()) (b ()) ]
  | "lax_conv_same" ->
      Some [ L.conv (a ()) (b ()) ~window_strides:(ints "str" p) L.Same ]
  | "lax_conv_valid" ->
      Some [ L.conv (a ()) (b ()) ~window_strides:(ints "str" p) L.Valid ]
  | "lax_conv_general" ->
      Some
        [
          L.conv_general_dilated ?lhs_dilation:(opt_int_array "ld" p)
            ?rhs_dilation:(opt_int_array "rd" p) ~dimension_numbers:(dn_of p)
            (a ()) (b ()) ~window_strides:(ints "str" p) (padding_of p);
        ]
  | "lax_conv_transpose" ->
      Some
        [
          L.conv_transpose ?transpose_kernel:(opt_bool "tk" p) (a ()) (b ())
            ~strides:(ints "str" p) (padding_of p);
        ]
  | "lax_map" ->
      Some
        (L.map
           (fun xs ->
             let x = List.hd xs in
             [ L.add (L.mul x x) (like x 1.0) ])
           [ a () ])
  | "lax_switch" ->
      let i =
        T.Concrete
          (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [||]
             [| float_of_int (int "i" p) |])
      in
      Some
        (L.switch i
           [
             (fun xs -> [ L.add (List.hd xs) (like (List.hd xs) 1.0) ]);
             (fun xs -> [ L.mul (List.hd xs) (like (List.hd xs) 2.0) ]);
             (fun xs -> [ L.neg (List.hd xs) ]);
           ]
           [ a () ])
  | "lax_fori_loop" ->
      let scalar_i = Ojax.Numpy.Jnp_util.py_int in
      Some
        (L.fori_loop (scalar_i 0)
           (scalar_i (int "n" p))
           (fun _ xs -> [ L.mul (List.hd xs) (like (List.hd xs) 2.0) ])
           [ a () ])
  | "lax_associative_scan" ->
      Some [ L.associative_scan ?axis:(opt_int "axis" p) L.add (a ()) ]
  | "lax_custom_root" ->
      let target = a () in
      Some
        (L.custom_root
           (fun xs ->
             let x = List.hd xs in
             [ L.sub (L.mul x x) target ])
           [ L.full_like target 1.0 ]
           newton
           (fun g ys ->
             let y = List.hd ys in
             [ L.div y (List.hd (g [ L.full_like y 1.0 ])) ]))
  | "lax_reciprocal" -> Some [ L.reciprocal (a ()) ]
  | "lax_full_like" ->
      Some [ L.full_like (a ()) (U.to_number (U.member "v" p)) ]
  | "lax_stop_gradient" ->
      let f xs =
        let x = List.hd xs in
        L.reduce_sum
          (L.mul (L.stop_gradient x) x)
          (Array.init (Array.length (Ojax.Core.get_aval x).shape) Fun.id)
      in
      Some (Ojax.Interpreters.Ad.grad f [ a () ])
  | _ -> None

module E = Ojax.Numpy.Extras
module NL = Ojax.Numpy.Lax_numpy
module UF = Ojax.Numpy.Ufuncs
module RD = Ojax.Numpy.Reductions

let dt_of v = (Ojax.Core.get_aval v).T.dtype
let shp v = (Ojax.Core.get_aval v).T.shape
let binarize a = UF.greater a (like a 0.0)

let rounded a =
  let two = like a 2.0 in
  UF.divide (NL.round (UF.multiply a two)) two

let nonneg a = NL.astype (UF.absolute a) Ojax.Dtype.I32

let opt_number name p =
  match U.member name p with `Null -> None | v -> Some (U.to_number v)

let int_list_value l =
  let xs = Array.of_list (List.map U.to_int (U.to_list l)) in
  T.Concrete
    (Ojax.Ndarray.of_floats
       (Ojax.Dtypes.default_int_dtype ())
       [| Array.length xs |]
       (Array.map float_of_int xs))

let pair (x, y) = [ x; y ]

let numpy_run op p a b c =
  match op with
  | "np_nonzero" -> Some (E.nonzero (binarize (a ())))
  | "np_nonzero_size" ->
      Some
        (E.nonzero ~size:(int "size" p)
           ~fill_value:(Option.value (opt_int "fill" p) ~default:0)
           (binarize (a ())))
  | "np_flatnonzero" -> Some [ E.flatnonzero (binarize (a ())) ]
  | "np_argwhere" -> Some [ E.argwhere (binarize (a ())) ]
  | "np_unique_values" -> Some [ E.unique_values (rounded (a ())) ]
  | "np_unique_counts" -> Some (pair (E.unique_counts (rounded (a ()))))
  | "np_unique_inverse" -> Some (pair (E.unique_inverse (rounded (a ()))))
  | "np_unique_all" ->
      let r = E.unique_all (rounded (a ())) in
      Some [ r.E.values; r.E.indices; r.E.inverse; r.E.counts ]
  | "np_unique_size" ->
      Some
        (E.unique ~size:(int "size" p) ?fill_value:(opt_number "fill" p)
           (rounded (a ())))
  | "np_unique_axis" -> Some (E.unique ~axis:(int "axis" p) (rounded (a ())))
  | "np_union1d" -> Some [ E.union1d (rounded (a ())) (rounded (b ())) ]
  | "np_intersect1d" -> Some [ E.intersect1d (rounded (a ())) (rounded (b ())) ]
  | "np_setdiff1d" -> Some [ E.setdiff1d (rounded (a ())) (rounded (b ())) ]
  | "np_setxor1d" -> Some [ E.setxor1d (rounded (a ())) (rounded (b ())) ]
  | "np_bincount" -> Some [ E.bincount (nonneg (a ())) ]
  | "np_bincount_length" ->
      Some
        [ E.bincount ~weights:(b ()) ~length:(int "length" p) (nonneg (a ())) ]
  | "np_histogram" ->
      Some
        (pair
           (E.histogram ~bins:(int "bins" p) ?density:(opt_bool "density" p)
              (a ())))
  | "np_histogram_range" ->
      let range =
        match List.map U.to_number (U.to_list (U.member "range" p)) with
        | [ lo; hi ] -> (lo, hi)
        | _ -> failwith "parity: range"
      in
      Some (pair (E.histogram ~bins:(int "bins" p) ~range (a ())))
  | "np_histogram_bin_edges" ->
      Some [ E.histogram_bin_edges ~bins:(int "bins" p) (a ()) ]
  | "np_histogram2d" ->
      let h, x, y =
        E.histogram2d ~bins:(ints "bins" p) ?density:(opt_bool "density" p)
          (a ()) (b ())
      in
      Some [ h; x; y ]
  | "np_histogramdd" ->
      let h, edges = E.histogramdd ~bins:(ints "bins" p) (a ()) in
      Some (h :: edges)
  | "np_interp" ->
      Some [ E.interp (a ()) (Ojax.Numpy.Sorting.sort (b ())) (c ()) ]
  | "np_interp_fill" ->
      Some
        [
          E.interp ?left:(opt_number "left" p) ?right:(opt_number "right" p)
            (a ())
            (Ojax.Numpy.Sorting.sort (b ()))
            (c ());
        ]
  | "np_interp_period" ->
      Some
        [
          E.interp ?period:(opt_number "period" p) (a ())
            (Ojax.Numpy.Sorting.sort (b ()))
            (c ());
        ]
  | "np_gradient" ->
      let spacing = Option.value (opt_number "spacing" p) ~default:1.0 in
      Some (E.gradient ~spacing (a ()))
  | "np_gradient_axis" -> Some (E.gradient ?axis:(opt_ints "axis" p) (a ()))
  | "np_choose" ->
      let x = nonneg (a ()) in
      let picks = UF.remainder x (Ojax.Numpy.Jnp_util.py_int 2) in
      let y = b () in
      Some [ E.choose ~mode:"clip" picks [ y; UF.negative y ] ]
  | "np_compress" ->
      let bools =
        Array.of_list
          (List.map
             (fun v -> if U.to_bool v then 1.0 else 0.0)
             (U.to_list (U.member "cond" p)))
      in
      let cond =
        Ojax.Jaxpr.stage_constant
          (T.Concrete
             (Ojax.Ndarray.of_floats Ojax.Dtype.Bool
                [| Array.length bools |]
                bools))
      in
      Some [ E.compress ~axis:(int "axis" p) cond (a ()) ]
  | "np_extract" -> Some [ E.extract (binarize (a ())) (a ()) ]
  | "np_delete" ->
      let obj = ints "obj" p in
      let dt = Ojax.Dtypes.default_int_dtype () in
      let obj =
        Ojax.Jaxpr.stage_constant
          (T.Concrete
             (Ojax.Ndarray.of_floats dt
                [| Array.length obj |]
                (Array.map float_of_int obj)))
      in
      Some [ E.delete_array ?axis:(opt_int "axis" p) (a ()) obj ]
  | "np_insert" ->
      let v = L.full [||] (U.to_number (U.member "val" p)) (dt_of (a ())) in
      Some [ E.insert ?axis:(opt_int "axis" p) (a ()) (int "pos" p) v ]
  | "np_place" ->
      let v = L.full [||] (U.to_number (U.member "val" p)) (dt_of (a ())) in
      Some [ E.place (a ()) (binarize (a ())) v ]
  | "np_piecewise" ->
      let x = a () in
      Some
        [
          E.piecewise x
            [ UF.less x (like x 0.0); UF.greater_equal x (like x 0.0) ]
            [ L.neg; (fun v -> L.mul v (like v 2.0)) ];
        ]
  | "np_fill_diagonal" ->
      let v = L.full [||] (U.to_number (U.member "val" p)) (dt_of (a ())) in
      Some [ E.fill_diagonal (a ()) v ]
  | "np_tril_indices" ->
      Some
        (pair
           (E.tril_indices ?k:(opt_int "k" p) ?m:(opt_int "m" p) (int "n" p)))
  | "np_triu_indices" ->
      Some
        (pair
           (E.triu_indices ?k:(opt_int "k" p) ?m:(opt_int "m" p) (int "n" p)))
  | "np_tril_indices_from" ->
      Some (pair (E.tril_indices_from ?k:(opt_int "k" p) (a ())))
  | "np_triu_indices_from" ->
      Some (pair (E.triu_indices_from ?k:(opt_int "k" p) (a ())))
  | "np_ravel_multi_index" ->
      let coords = List.map int_list_value (U.to_list (U.member "coords" p)) in
      Some
        [
          E.ravel_multi_index
            ~mode:(U.to_string (U.member "mode" p))
            coords (ints "dims" p);
        ]
  | "np_trim_zeros" ->
      let x = a () in
      let z n = L.full [| n |] 0.0 (dt_of x) in
      Some
        [
          E.trim_zeros
            ~trim:(U.to_string (U.member "trim" p))
            (NL.concatenate [ z 2; x; z 1 ]);
        ]
  | "np_frexp" -> Some (pair (E.frexp (a ())))
  | "np_ldexp" ->
      let e =
        T.Concrete
          (Ojax.Ndarray.of_floats Ojax.Dtype.I32
             [| Array.length (ints "e" p) |]
             (Array.map float_of_int (ints "e" p)))
      in
      Some [ E.ldexp (a ()) e ]
  | "np_apply_along_axis" ->
      Some [ E.apply_along_axis (fun v -> RD.sum v) (int "axis" p) (a ()) ]
  | "np_apply_along_axis_keep" ->
      Some
        [
          E.apply_along_axis
            (fun v -> L.mul v (like v 2.0))
            (int "axis" p) (a ());
        ]
  | "np_apply_over_axes" ->
      Some
        [
          E.apply_over_axes
            (fun v ax -> RD.sum ~axis:[| ax |] v)
            (a ())
            (Array.to_list (ints "axes" p));
        ]
  | "np_polydiv" -> Some (pair (E.polydiv (a ()) (b ())))
  | "np_polyfit" -> Some [ E.polyfit (a ()) (b ()) (int "deg" p) ]
  | "np_roots" -> Some [ E.sort_complex (E.roots (a ())) ]
  | "np_block" ->
      Some
        [
          E.block
            (E.Group
               [
                 E.Group [ E.Item (a ()); E.Item (b ()) ];
                 E.Group [ E.Item (b ()); E.Item (a ()) ];
               ]);
        ]
  | _ -> None

module SF = Ojax.Scipy.Fft
module SG = Ojax.Scipy.Signal
module SP = Ojax.Scipy.Linalg_special
module RX = Ojax.Random_extras
module NN = Ojax.Nn.Functions

let seed_key p =
  Ojax.Random.Core.key
    (T.Concrete
       (Ojax.Ndarray.of_floats
          (Ojax.Dtypes.default_int_dtype ())
          [||]
          [| float_of_int (int "seed" p) |]))

let required_string name p = U.to_string (U.member name p)

let scipy_run op p a b c =
  let mode () = required_string "mode" p in
  match op with
  | "sp_dct" ->
      Some
        [
          SF.dct ?n:(opt_int "n" p) ?axis:(opt_int "axis" p)
            ?norm:(opt_string "norm" p) (a ());
        ]
  | "sp_idct" ->
      Some
        [
          SF.idct ?n:(opt_int "n" p) ?axis:(opt_int "axis" p)
            ?norm:(opt_string "norm" p) (a ());
        ]
  | "sp_dctn" -> Some [ SF.dctn ?norm:(opt_string "norm" p) (a ()) ]
  | "sp_idctn" -> Some [ SF.idctn ?norm:(opt_string "norm" p) (a ()) ]
  | "sp_convolve" ->
      Some
        [
          SG.convolve ~mode:(mode ()) ?method_:(opt_string "method" p) (a ())
            (b ());
        ]
  | "sp_correlate" -> Some [ SG.correlate ~mode:(mode ()) (a ()) (b ()) ]
  | "sp_convolve2d" -> Some [ SG.convolve2d ~mode:(mode ()) (a ()) (b ()) ]
  | "sp_correlate2d" -> Some [ SG.correlate2d ~mode:(mode ()) (a ()) (b ()) ]
  | "sp_fftconvolve" -> Some [ SG.fftconvolve ~mode:(mode ()) (a ()) (b ()) ]
  | "sp_detrend" ->
      Some
        [
          SG.detrend ?axis:(opt_int "axis" p) ~type_:(required_string "type" p)
            (a ());
        ]
  | "sl_hilbert" -> Some [ SP.hilbert (int "n" p) ]
  | "sl_circulant" -> Some [ SP.circulant (a ()) ]
  | "sl_hankel" -> Some [ SP.hankel (a ()) ]
  | "sl_hankel_r" -> Some [ SP.hankel ~r:(b ()) (a ()) ]
  | "sl_fiedler" -> Some [ SP.fiedler (a ()) ]
  | "sl_companion" -> Some [ SP.companion (a ()) ]
  | "sl_fiedler_companion" -> Some [ SP.fiedler_companion (a ()) ]
  | "sl_leslie" -> Some [ SP.leslie (a ()) (b ()) ]
  | "sl_dft" -> Some [ SP.dft ?scale:(opt_string "scale" p) (int "n" p) ]
  | "sl_pascal" ->
      Some [ SP.pascal ~kind:(required_string "kind" p) (int "n" p) ]
  | "sl_invpascal" ->
      Some [ SP.invpascal ~kind:(required_string "kind" p) (int "n" p) ]
  | "sl_helmert" -> Some [ SP.helmert ?full:(opt_bool "full" p) (int "n" p) ]
  | "sl_hadamard" -> Some [ SP.hadamard (int "n" p) ]
  | "sl_convolution_matrix" ->
      Some [ SP.convolution_matrix ~mode:(mode ()) (a ()) (int "n" p) ]
  | "sl_invhilbert" -> Some [ SP.invhilbert (int "n" p) ]
  | "sl_solve_sylvester" ->
      let shift v =
        let n = (shp v).(0) in
        let eye = NL.identity ~dtype:(dt_of v) n in
        L.add v (L.mul (like eye 4.0) eye)
      in
      Some [ SP.solve_sylvester (shift (a ())) (shift (b ())) (c ()) ]
  | "rnd_maxwell" -> Some [ RX.maxwell (seed_key p) ~shape:(ints "shape" p) ]
  | "rnd_double_sided_maxwell" ->
      Some
        [
          RX.double_sided_maxwell (seed_key p)
            ~loc:(U.to_number (U.member "loc" p))
            ~scale:(U.to_number (U.member "scale" p))
            ~shape:(ints "shape" p);
        ]
  | "rnd_ball" ->
      Some [ RX.ball (seed_key p) ~d:(int "d" p) ~shape:(ints "shape" p) ]
  | "rnd_multivariate_normal" ->
      let key = seed_key p in
      let m = b () in
      let n = (shp m).(0) in
      let product = L.dot_typed m (L.transpose m [| 1; 0 |]) in
      let eye = NL.identity ~dtype:(dt_of m) n in
      let cov = L.add product (L.mul (like eye 2.0) eye) in
      Some
        [
          RX.multivariate_normal ~shape:(ints "shape" p)
            ~method_:(required_string "method" p)
            key ~mean:(a ()) ~cov;
        ]
  | "nn_swish" -> Some [ NN.swish (a ()) ]
  | "nn_hard_swish" -> Some [ NN.hard_swish (a ()) ]
  | "nn_logsumexp" ->
      Some [ Ojax.Ops.Special.logsumexp ~axis:[| int "axis" p |] (a ()) ]
  | "nn_attention" ->
      Some
        [ NN.dot_product_attention ~query:(a ()) ~key:(b ()) ~value:(c ()) () ]
  | _ -> None

module OP = Ojax.Example_libraries.Optimizers

let optimizer_of name =
  let lr = OP.constant 0.1 in
  match name with
  | "opt_sgd" -> Some (OP.sgd lr)
  | "opt_momentum" -> Some (OP.momentum lr ~mass:0.9)
  | "opt_nesterov" -> Some (OP.nesterov lr ~mass:0.9)
  | "opt_adagrad" -> Some (OP.adagrad lr)
  | "opt_rmsprop" -> Some (OP.rmsprop lr)
  | "opt_rmsprop_momentum" -> Some (OP.rmsprop_momentum lr)
  | "opt_adam" -> Some (OP.adam lr)
  | "opt_adamax" -> Some (OP.adamax lr)
  | "opt_sm3" -> Some (OP.sm3 lr)
  | _ -> None

let dtype_named = function
  | "int8" -> Ojax.Dtype.I8
  | "int16" -> Ojax.Dtype.I16
  | "int32" -> Ojax.Dtype.I32
  | "uint8" -> Ojax.Dtype.Uint8
  | "uint16" -> Ojax.Dtype.Uint16
  | "uint32" -> Ojax.Dtype.Uint32
  | "float16" -> Ojax.Dtype.F16
  | "bfloat16" -> Ojax.Dtype.Bf16
  | "float32" -> Ojax.Dtype.F32
  | other -> failwith ("parity: dtype " ^ other)

let scaled v x = L.mul v (like v x)

let in_range v name =
  if String.length name >= 4 && String.sub name 0 4 = "uint" then L.abs v else v

let narrow_run op p a b =
  let named name = dtype_named (required_string name p) in
  match op with
  | "dt_cast" ->
      Some
        [
          NL.astype
            (scaled (in_range (a ()) (required_string "to" p)) 40.0)
            (named "to");
        ]
  | "dt_wrap_add" ->
      let dt = named "to" in
      let c name =
        T.Concrete
          (Ojax.Ndarray.of_floats dt [||] [| float_of_int (int name p) |])
      in
      Some [ UF.add (c "x") (c "y") ]
  | "dt_binary" ->
      Some
        [
          UF.add
            (NL.astype
               (scaled (in_range (a ()) (required_string "x" p)) 5.0)
               (named "x"))
            (NL.astype
               (scaled (in_range (b ()) (required_string "y" p)) 5.0)
               (named "y"));
        ]
  | "dt_float_ops" ->
      let v = UF.sin (NL.astype (a ()) (named "to")) in
      Some [ L.mul v (like v 2.0) ]
  | "dt_sum" ->
      Some
        [
          RD.sum
            (NL.astype
               (scaled (in_range (a ()) (required_string "to" p)) 5.0)
               (named "to"));
        ]
  | _ -> None

module LA = Ojax.Numpy.Linalg

let spd v =
  let sh = shp v in
  let n = sh.(Array.length sh - 1) in
  ignore sh;
  let product = L.batch_matmul_typed v (NL.swapaxes (-1) (-2) v) in
  let eye = NL.eye ~dtype:(dt_of v) n in
  UF.add product (UF.multiply (like eye 3.0) eye)

let batched_run op p a b =
  match op with
  | "bl_inv" -> Some [ LA.inv (spd (a ())) ]
  | "bl_det" -> Some [ LA.det (spd (a ())) ]
  | "bl_slogdet" -> Some (pair (LA.slogdet (spd (a ()))))
  | "bl_solve" -> Some [ LA.solve (spd (a ())) (b ()) ]
  | "bl_cholesky" -> Some [ LA.cholesky (spd (a ())) ]
  | "bl_eigvalsh" -> Some [ LA.eigvalsh (spd (a ())) ]
  | "bl_svdvals" -> Some [ LA.svdvals (a ()) ]
  | "bl_matrix_power" -> Some [ LA.matrix_power (a ()) (int "n" p) ]
  | "bl_norm" ->
      let rank = Array.length (shp (a ())) in
      Some [ LA.norm ~axis:(LA.Apair (rank - 2, rank - 1)) (a ()) ]
  | "bl_grad_det" ->
      let x = a () in
      let f args =
        let d = LA.det (spd (List.hd args)) in
        L.reduce_sum d (Array.init (Array.length (shp d)) Fun.id)
      in
      Some (Ojax.Interpreters.Ad.grad f [ x ])
  | "bl_vmap_inv" ->
      Some
        (Ojax.Interpreters.Batching.vmap
           (fun args -> [ LA.inv (spd (List.hd args)) ])
           [ Some 0 ]
           [ a () ])
  | "np_packbits" ->
      Some
        [
          E.packbits ?axis:(opt_int "axis" p) ?bitorder:(opt_string "order" p)
            (binarize (a ()));
        ]
  | "np_unpackbits" ->
      Some
        [
          E.unpackbits ?count:(opt_int "count" p)
            ?bitorder:(opt_string "order" p)
            (E.packbits (binarize (a ())));
        ]
  | _ -> None

module IP = Ojax.Scipy.Interpolate
module KD = Ojax.Scipy.Stats.Kde

let grid_of v =
  let sh = shp v in
  let dt = dt_of v in
  let axis size = L.iota dt size in
  [ axis sh.(0); axis sh.(1) ]

let triple (x, y, z) = [ x; y; z ]

let misc_scipy_run op p a b =
  let kde v = KD.gaussian_kde ?bw_method:(opt_string "bw" p) v in
  let boundary_of () =
    match U.member "boundary" p with
    | `Null when List.mem_assoc "boundary" (U.to_assoc p) -> Some None
    | `Null -> None
    | v -> Some (Some (U.to_string v))
  in
  match op with
  | "sg_stft" ->
      Some
        (triple
           (SG.stft ~nperseg:(int "nperseg" p) ?noverlap:(opt_int "noverlap" p)
              ?boundary:(boundary_of ()) ?padded:(opt_bool "padded" p) (a ())))
  | "sg_csd" ->
      Some
        (pair
           (SG.csd ~nperseg:(int "nperseg" p) ?average:(opt_string "average" p)
              (a ())
              (Some (b ()))))
  | "sg_welch" ->
      Some
        (pair
           (SG.welch ~nperseg:(int "nperseg" p)
              ?detrend:(Option.map Option.some (opt_string "detrend" p))
              ?scaling:(opt_string "scaling" p)
              ?average:(opt_string "average" p) (a ())))
  | "sg_istft" -> Some (pair (SG.istft ~nperseg:(int "nperseg" p) (a ())))
  | "ip_linear" ->
      Some
        [
          IP.evaluate
            (IP.regular_grid_interpolator (grid_of (a ())) (a ()))
            (b ());
        ]
  | "ip_nearest" ->
      Some
        [
          IP.evaluate
            (IP.regular_grid_interpolator ~method_:"nearest"
               (grid_of (a ()))
               (a ()))
            (b ());
        ]
  | "kde_evaluate" -> Some [ KD.evaluate (kde (a ())) (b ()) ]
  | "kde_logpdf" -> Some [ KD.logpdf (kde (a ())) (b ()) ]
  | "kde_box" ->
      Some
        [
          KD.integrate_box_1d
            (kde (a ()))
            (U.to_number (U.member "lo" p))
            (U.to_number (U.member "hi" p));
        ]
  | "sp_fresnel" ->
      let x = a () in
      let factor = Option.value (opt_number "scale" p) ~default:1.0 in
      Some (pair (Ojax.Scipy.Special.fresnel (L.mul x (like x factor))))
  | _ -> None

let lax_extra_run op p a =
  match op with
  | "lax_patches" ->
      Some
        [
          L.conv_general_dilated_patches ~window_strides:(ints "str" p)
            ~padding:(padding_of p) (a ()) ~filter_shape:(ints "filter" p);
        ]
  | _ -> None

let optimizer_run op a b =
  match optimizer_of op with
  | None -> None
  | Some o ->
      let module Tr = Ojax.Tree_util in
      let grads = Tr.Leaf (b ()) in
      let state = ref (o.OP.init (Tr.Leaf (a ()))) in
      for i = 0 to 2 do
        state := o.OP.update i grads !state
      done;
      Some (Tr.tree_leaves (o.OP.get_params !state))

module RC = Ojax.Random.Core

let default_random_run op p =
  let k () = seed_key p in
  let shape =
    if List.mem_assoc "shape" (U.to_assoc p) then ints "shape" p else [||]
  in
  match op with
  | "rd_uniform" -> Some [ RC.uniform (k ()) ~shape ~minval:(-1.0) ~maxval:3.0 ]
  | "rd_normal" -> Some [ RC.normal (k ()) ~shape ]
  | "rd_randint" -> Some [ RC.randint (k ()) ~shape ~minval:(-3) ~maxval:50 ]
  | "rd_truncated_normal" ->
      Some [ RC.truncated_normal (k ()) ~lower:(-1.0) ~upper:2.0 ~shape ]
  | "rd_exponential" -> Some [ RC.exponential (k ()) ~shape ]
  | "rd_cauchy" -> Some [ RC.cauchy (k ()) ~shape ]
  | "rd_laplace" -> Some [ RC.laplace (k ()) ~shape ]
  | "rd_logistic" -> Some [ RC.logistic (k ()) ~shape ]
  | "rd_gumbel" -> Some [ RC.gumbel (k ()) ~shape ]
  | "rd_bernoulli" -> Some [ RC.bernoulli (k ()) ~shape ~p:0.4 ]
  | "rd_rademacher" -> Some [ RC.rademacher (k ()) ~shape ]
  | "rd_gamma" -> Some [ RC.gamma (k ()) ~shape ~a:2.5 ]
  | "rd_beta" -> Some [ RC.beta (k ()) ~shape ~a:2.0 ~b:3.0 ]
  | "rd_chisquare" -> Some [ RC.chisquare (k ()) ~shape ~df:3.0 ]
  | "rd_t" -> Some [ RC.t (k ()) ~shape ~df:4.0 ]
  | "rd_f" -> Some [ RC.f (k ()) ~shape ~dfnum:3.0 ~dfden:5.0 ]
  | "rd_lognormal" -> Some [ RC.lognormal (k ()) ~shape ~sigma:0.5 ]
  | "rd_pareto" -> Some [ RC.pareto (k ()) ~shape ~b:2.0 ]
  | "rd_rayleigh" -> Some [ RC.rayleigh (k ()) ~shape ~scale:1.5 ]
  | "rd_wald" -> Some [ RC.wald (k ()) ~shape ~mean:1.5 ]
  | "rd_geometric" -> Some [ RC.geometric (k ()) ~shape ~p:0.3 ]
  | "rd_poisson" -> Some [ RC.poisson (k ()) ~shape ~lam:3.0 ]
  | "rd_binomial" -> Some [ RC.binomial (k ()) ~shape ~count:10.0 ~prob:0.3 ]
  | "rd_poisson_large" -> Some [ RC.poisson (k ()) ~shape ~lam:30.0 ]
  | "rd_binomial_large" ->
      Some [ RC.binomial (k ()) ~shape ~count:100.0 ~prob:0.3 ]
  | "rd_binomial_flip" ->
      Some [ RC.binomial (k ()) ~shape ~count:100.0 ~prob:0.7 ]
  | "rd_permutation" -> Some [ RC.permutation (k ()) 6 ]
  | "rd_choice" -> Some [ RC.choice (k ()) ~n:6 ~shape ~replace:true ]
  | _ -> None

let array_random_run op p a b =
  let k () = seed_key p in
  match op with
  | "ra_uniform" ->
      let key = k () in
      let lo = a () in
      let hi = L.add (L.add lo (L.abs (b ()))) (like lo 1.0) in
      Some [ RC.uniform_range key ~shape:(shp lo) ~minval:lo ~maxval:hi ]
  | "ra_bernoulli" ->
      let key = k () in
      let pr = L.abs (L.sin (a ())) in
      Some [ RC.bernoulli_prob key ~shape:(shp pr) ~p:pr ]
  | "ra_gamma" -> Some [ RC.gamma_value (k ()) ~shape:(shp (a ())) ~a:(a ()) ]
  | "ra_loggamma" ->
      Some [ RC.loggamma_value (k ()) ~shape:(shp (a ())) ~a:(a ()) ]
  | "ra_beta" ->
      Some [ RC.beta_value (k ()) ~shape:(shp (a ())) ~a:(a ()) ~b:(b ()) ]
  | "ra_poisson" ->
      let key = k () in
      let lam = L.mul (a ()) (like (a ()) 8.0) in
      Some [ RC.poisson_value key ~shape:(shp lam) ~lam ]
  | "ra_bits" ->
      Some
        [
          RC.bits_as (k ()) ~shape:(ints "shape" p)
            ~dtype:
              (match required_string "dtype" p with
              | "uint8" -> Ojax.Dtype.Uint8
              | "uint16" -> Ojax.Dtype.Uint16
              | "uint32" -> Ojax.Dtype.Uint32
              | _ -> Ojax.Dtype.Uint64);
        ]
  | "ra_gamma_grad" ->
      let x = a () in
      let z = L.add (L.mul x (like x 0.7)) (like x 0.2) in
      Some [ L.random_gamma_grad x z ]
  | "ra_grad_gamma" ->
      let x = a () in
      let f args =
        let v = RC.gamma_value (k ()) ~shape:(shp x) ~a:(List.hd args) in
        L.reduce_sum v [| 0 |]
      in
      Some (Ojax.Interpreters.Ad.grad f [ x ])
  | "sp_fresnel_jit" ->
      let f args =
        let s, c = Ojax.Scipy.Special.fresnel (List.hd args) in
        [ s; c ]
      in
      Some (Ojax.Api.jit_flat f [ a () ])
  | "sp_fresnel_grad" ->
      let f args =
        let s = fst (Ojax.Scipy.Special.fresnel (List.hd args)) in
        L.reduce_sum s (Array.init (Array.length (shp s)) Fun.id)
      in
      Some (Ojax.Interpreters.Ad.grad f [ a () ])
  | _ -> None

module ST = Ojax.Scipy.Stats.Stats_core
module INI = Ojax.Nn.Initializers

let gap_run op p a b =
  let k () = seed_key p in
  match op with
  | "gp_qr_full" -> Some (LA.qr ~mode:"complete" (a ()))
  | "gp_svd_hermitian" ->
      let x = a () in
      let sym = L.add x (L.transpose x [| 1; 0 |]) in
      Some (LA.svd ~hermitian:true ~compute_uv:false sym)
  | "gp_pinv_hermitian" ->
      let x = a () in
      Some [ LA.pinv ~hermitian:true (L.add x (L.transpose x [| 1; 0 |])) ]
  | "gp_init_orthogonal" ->
      Some
        [
          RC.with_dtypes ~float_dtype:Ojax.Dtype.F32 (fun () ->
              INI.orthogonal () (k ()) ~shape:(ints "shape" p));
        ]
  | "gp_init_delta" ->
      Some
        [
          RC.with_dtypes ~float_dtype:Ojax.Dtype.F32 (fun () ->
              INI.delta_orthogonal () (k ()) ~shape:(ints "shape" p));
        ]
  | "gp_mode" ->
      let x = NL.round (a ()) in
      Some (pair (ST.mode ?keepdims:(opt_bool "keepdims" p) x))
  | "gp_rankdata" ->
      let x = a () in
      let scaled = NL.round (L.mul x (like x 2.0)) in
      Some [ ST.rankdata ~method_:(required_string "method" p) scaled ]
  | "gp_rankdata_axis" ->
      let x = a () in
      let scaled = NL.round (L.mul x (like x 2.0)) in
      Some
        [
          ST.rankdata
            ~method_:(required_string "method" p)
            ~axis:(int "axis" p) scaled;
        ]
  | "gp_wofz" ->
      let z =
        UF.add (a ())
          (UF.multiply
             (Ojax.Numpy.Jnp_util.py_complex { Complex.re = 0.0; im = 1.0 })
             (b ()))
      in
      Some [ Ojax.Scipy.Special.wofz z ]
  | "gp_bernoulli" -> Some [ Ojax.Scipy.Special.bernoulli (int "n" p) ]
  | "gp_expm_batched" -> Some [ Ojax.Scipy.Linalg.expm (a ()) ]
  | "gp_reduce_precision" ->
      let x = a () in
      let scaled = L.mul x (like x (U.to_number (U.member "scale" p))) in
      Some
        [
          L.reduce_precision scaled ~exponent_bits:(int "e" p)
            ~mantissa_bits:(int "m" p);
        ]
  | "gp_clamp_jit" ->
      Some
        (Ojax.Api.jit_flat
           (fun args ->
             let x = List.hd args in
             [ L.clamp (like x (-0.5)) x (like x 0.5) ])
           [ a () ])
  | "gp_int_pow_neg" ->
      Some
        (Ojax.Api.jit_flat
           (fun args -> [ L.integer_pow (List.hd args) (-2) ])
           [ a () ])
  | _ -> None

let herm v =
  let sh = shp v in
  let n = sh.(Array.length sh - 1) in
  ignore sh;
  let product = L.batch_matmul_typed v (UF.conj (NL.swapaxes (-1) (-2) v)) in
  let eye = NL.eye ~dtype:(dt_of v) n in
  UF.add product (UF.multiply (like eye 3.0) eye)

let sum_all v = L.reduce_sum v (Array.init (Array.length (shp v)) Fun.id)

let complex_linalg_run op a b =
  let grad f = Some (Ojax.Interpreters.Ad.grad f [ a () ]) in
  match op with
  | "cl_inv" -> Some [ LA.inv (herm (a ())) ]
  | "cl_det" -> Some [ LA.det (herm (a ())) ]
  | "cl_solve" -> Some [ LA.solve (herm (a ())) (b ()) ]
  | "cl_cholesky" -> Some [ LA.cholesky (herm (a ())) ]
  | "cl_eigvalsh" -> Some [ LA.eigvalsh (herm (a ())) ]
  | "cl_svdvals" -> Some [ LA.svdvals (a ()) ]
  | "cl_qr_r" -> (
      match LA.qr (a ()) with
      | [ _; r ] -> Some [ UF.absolute r ]
      | _ -> failwith "parity: qr arity")
  | "cl_slogdet" -> Some (pair (LA.slogdet (herm (a ()))))
  | "cl_grad_eigvalsh" ->
      grad (fun args -> sum_all (LA.eigvalsh (herm (List.hd args))))
  | "cl_grad_logdet" ->
      grad (fun args -> sum_all (snd (LA.slogdet (herm (List.hd args)))))
  | "cl_grad_eigvals" ->
      grad (fun args -> sum_all (UF.absolute (LA.eigvals (List.hd args))))
  | "cl_jvp_eigvals" ->
      let x = a () in
      let dt = dt_of x in
      let tangent = UF.add (L.mul x (like x 0.5)) (like x 0.1) in
      ignore dt;
      let _, t =
        Ojax.Interpreters.Ad.jvp
          (fun args -> [ LA.eigvals (List.hd args) ])
          [ x ] [ tangent ]
      in
      Some [ sum_all (List.hd t) ]
  | "cl_grad_svdvals" -> grad (fun args -> sum_all (LA.svdvals (List.hd args)))
  | _ -> None

let complex_grad_run op a =
  let on f =
    Some
      (Ojax.Interpreters.Ad.grad
         (fun args -> sum_all (f (List.hd args)))
         [ a () ])
  in
  let abs_of g z = UF.absolute (g z) in
  match op with
  | "cg_abs" -> on UF.absolute
  | "cg_sin" -> on (abs_of UF.sin)
  | "cg_cos" -> on (abs_of UF.cos)
  | "cg_tan" -> on (abs_of UF.tan)
  | "cg_exp" -> on (abs_of UF.exp)
  | "cg_log" -> on (abs_of UF.log)
  | "cg_sqrt" -> on (abs_of UF.sqrt)
  | "cg_tanh" -> on (abs_of UF.tanh)
  | "cg_sinh" -> on (abs_of UF.sinh)
  | "cg_cosh" -> on (abs_of UF.cosh)
  | "cg_square" -> on (abs_of (fun z -> L.mul z z))
  | "cg_reciprocal" -> on (abs_of (fun z -> UF.divide (like z 1.0) z))
  | "cg_conj" -> on (abs_of (fun z -> L.mul (L.conj z) z))
  | "cg_real" -> on (fun z -> UF.power (L.real z) (scalar_int 2))
  | "cg_imag" -> on (fun z -> UF.power (L.imag z) (scalar_int 2))
  | "cg_matmul" ->
      on (abs_of (fun z -> Ojax.Numpy.Tensor_contractions.matmul z (L.conj z)))
  | "cg_fft" -> on (abs_of (fun z -> Ojax.Numpy.Fft.fft z))
  | "cg_angle" -> on (fun z -> NL.angle z)
  | _ -> None

let linalg_batch_run op params a b =
  let module X = Ojax.Linalg_extras in
  let module SL = Ojax.Scipy.Linalg in
  let module LL = Ojax.Lax.Linalg in
  let h_of v = UF.conj (NL.swapaxes (-1) (-2) v) in
  let recon z t = L.batch_matmul_typed (L.batch_matmul_typed z t) (h_of z) in
  let diag_mag t =
    Ojax.Numpy.Sorting.sort ~axis:(Some (-1))
      (UF.absolute (NL.diagonal ~axis1:(-2) ~axis2:(-1) t))
  in
  let bool_p k = U.to_bool (U.member k params) in
  let str_p k = U.to_string (U.member k params) in
  match op with
  | "lx_cholesky_update" ->
      Some
        [
          X.cholesky_update
            (NL.swapaxes (-1) (-2) (LA.cholesky (spd (a ()))))
            (b ());
        ]
  | "lx_symmetric_product" ->
      Some
        [
          X.symmetric_product ~alpha:2.0 ~beta:0.5
            ~symmetrize_output:(bool_p "sym") (a ())
            (spd (b ()));
        ]
  | "lx_ormqr" ->
      let h, t = X.geqrf (a ()) in
      Some
        [
          X.ormqr ~left:(bool_p "left") ~transpose:(bool_p "transpose") h t
            (b ());
        ]
  | "lx_qdwh" ->
      let u, h, _, ok = X.qdwh (a ()) in
      Some [ u; h; ok ]
  | "lx_qr_pivot" -> (
      match SL.qr ~mode:(str_p "mode") ~pivoting:true (a ()) with
      | [ r; p ] -> Some [ UF.absolute r; p ]
      | [ q; r; p ] -> Some [ UF.absolute q; UF.absolute r; p ]
      | _ -> failwith "parity: qr arity")
  | "lx_qr_multiply" ->
      Some
        (SL.qr_multiply ~mode:(str_p "mode") ~conjugate:(bool_p "conjugate")
           ~pivoting:(bool_p "pivoting") (a ()) (b ()))
  | "lx_rsf2csf" -> (
      match SL.schur (a ()) with
      | [ t; z ] ->
          let tc, zc = SL.rsf2csf t z in
          Some [ recon zc tc; diag_mag tc ]
      | _ -> failwith "parity: schur arity")
  | "lx_schur_complex" -> (
      match SL.schur ~output:"complex" (a ()) with
      | [ t; z ] -> Some [ recon z t; diag_mag t ]
      | _ -> failwith "parity: schur arity")
  | "lx_hessenberg" -> (
      match SL.hessenberg ~calc_q:true (a ()) with
      | [ h; q ] -> Some [ recon q h; UF.absolute h ]
      | _ -> failwith "parity: hessenberg arity")
  | "lx_sqrtm" ->
      let r = SL.sqrtm (a ()) in
      Some [ L.dot_typed r r; SL.sqrtm (herm (a ())) ]
  | "lx_expm_frechet" -> Some (SL.expm_frechet (a ()) (b ()))
  | "lx_polar" ->
      Some (SL.polar ~side:(str_p "side") ~method_:(str_p "method") (a ()))
  | "lx_eigh_tri" -> (
      match SL.eigh_tridiagonal (a ()) (b ()) with
      | [ w; v ] -> Some [ w; UF.absolute v ]
      | _ -> failwith "parity: eigh_tridiagonal arity")
  | "lx_tridiagonal" ->
      let _, d, e, _ = LL.tridiagonal (herm (a ())) in
      Some [ d; UF.absolute e ]
  | _ -> None

let surface_batch_run op params a b =
  let module X = Ojax.Lax_extras in
  let module P = Ojax.Parallel in
  let module G = Ojax.Numpy.Grids in
  let module UFN = Ojax.Numpy.Ufunc in
  let over f =
    List.hd
      (Ojax.Interpreters.Batching.vmap ~axis_name:"i"
         (fun args -> [ f (List.hd args) ])
         [ Some 0 ]
         [ a () ])
  in
  let int_p k = U.to_int (U.member k params) in
  let bool_p k = U.to_bool (U.member k params) in
  let int_list k = List.map U.to_int (U.to_list (U.member k params)) in
  let uf =
    UFN.frompyfunc
      (fun args ->
        match args with
        | [ x; y ] -> [ UF.add (UF.multiply x y) (like x 1.0) ]
        | _ -> failwith "parity: ufunc arity")
      ~nin:2 ~nout:1
  in
  match op with
  | "sb_approx_max" ->
      let v, i = X.approx_max_k (a ()) (int_p "k") in
      Some [ v; i ]
  | "sb_approx_min" ->
      let v, i = X.approx_min_k (a ()) (int_p "k") in
      Some [ v; i ]
  | "sb_broadcast_like" -> Some [ X.broadcast_like (a ()) (b ()) ]
  | "sb_ragged_dot" ->
      let groups = int_list "groups" in
      let sizes =
        T.Concrete
          (Ojax.Ndarray.of_floats
             (if Ojax.Config.x64_enabled () then Ojax.Dtype.I64
              else Ojax.Dtype.I32)
             [| List.length groups |]
             (Array.of_list (List.map float_of_int groups)))
      in
      Some [ X.ragged_dot (a ()) (b ()) sizes ]
  | "sb_conv_local" ->
      Some
        [
          X.conv_general_dilated_local (a ()) (b ())
            ~filter_shape:(Array.of_list (int_list "filter"));
        ]
  | "sb_rng" ->
      let key =
        T.Concrete
          (Ojax.Ndarray.of_floats Ojax.Dtype.Uint32 [| 4 |] [| 1.; 2.; 3.; 4. |])
      in
      let dtype =
        match U.to_string (U.member "dtype" params) with
        | "uint8" -> Ojax.Dtype.Uint8
        | "uint16" -> Ojax.Dtype.Uint16
        | _ -> Ojax.Dtype.Uint32
      in
      let algorithm =
        match U.to_string (U.member "algorithm" params) with
        | "RNG_THREE_FRY" -> X.Three_fry
        | "RNG_PHILOX" -> X.Philox
        | _ -> X.Rng_default
      in
      let k, v =
        X.rng_bit_generator ~dtype ~algorithm key
          (Array.of_list (int_list "shape"))
      in
      Some [ k; v ]
  | "sb_all_to_all" ->
      Some
        [
          over (fun x ->
              P.all_to_all ~tiled:(bool_p "tiled") x ~axis_name:"i"
                ~split_axis:(int_p "split") ~concat_axis:(int_p "concat"));
        ]
  | "sb_pswapaxes" ->
      Some [ over (fun x -> P.pswapaxes x ~axis_name:"i" ~axis:(int_p "axis")) ]
  | "sb_psum_scatter" ->
      Some
        [
          over (fun x ->
              P.psum_scatter ~scatter_dimension:(int_p "dim")
                ~tiled:(bool_p "tiled") x ~axis_name:"i");
        ]
  | "sb_pshuffle" ->
      Some
        [ over (fun x -> P.pshuffle x ~axis_name:"i" ~perm:(int_list "perm")) ]
  | "sb_mgrid_int" ->
      Some [ G.mgrid [ G.Step (0., 4., 1.); G.Step (1., 7., 2.) ] ]
  | "sb_mgrid_count" ->
      Some [ G.mgrid [ G.Count (0., 1., 5); G.Count (0., 2., 3) ] ]
  | "sb_mgrid_one" -> Some [ G.mgrid [ G.Step (1., 5., 1.) ] ]
  | "sb_ogrid" -> Some (G.ogrid [ G.Step (0., 3., 1.); G.Count (0., 1., 4) ])
  | "sb_r" -> Some [ G.r_ [ a (); b () ] ]
  | "sb_c" -> Some [ G.c_ [ a (); b () ] ]
  | "sb_ufunc_call" -> Some (UFN.call uf [ a (); b () ])
  | "sb_ufunc_reduce" -> Some [ UFN.reduce ~axis:(int_p "axis") uf (a ()) ]
  | "sb_ufunc_accumulate" ->
      Some [ UFN.accumulate ~axis:(int_p "axis") uf (a ()) ]
  | "sb_ufunc_outer" -> Some [ UFN.outer uf (a ()) (b ()) ]
  | _ -> None

let tracer_batch_run op params a b =
  let int_p k = U.to_int (U.member k params) in
  let opt_int k =
    match U.member k params with `Null -> None | v -> Some (U.to_int v)
  in
  match op with
  | "tb_unique" ->
      let fill_value = Option.map float_of_int (opt_int "fill") in
      Some
        (E.unique ~return_index:true ~return_inverse:true ~return_counts:true
           ?axis:(opt_int "axis") ~size:(int_p "size") ?fill_value (a ()))
  | "tb_lexsort" ->
      let v = a () in
      let column j =
        Ojax.Numpy.Nd_index.take v
          [
            Ojax.Numpy.Nd_index.Range { start = None; stop = None };
            Ojax.Numpy.Nd_index.Int j;
          ]
      in
      let first = column 0 in
      let second = column 1 in
      Some [ Ojax.Numpy.Sorting.lexsort [ first; second ] ]
  | "tb_poly_matrix" -> Some [ Ojax.Numpy.Polynomial.poly (a ()) ]
  | "tb_qr_raw" -> Some (LA.qr ~mode:"raw" (a ()))
  | "tb_vectorize" ->
      Some
        [
          Ojax.Numpy.Vectorize.vectorize
            (fun args ->
              match args with
              | [ x; y ] -> UF.add (UF.multiply x y) (like x 1.0)
              | _ -> failwith "parity: vectorize arity")
            [ a (); b () ];
        ]
  | "tb_vectorize_core" ->
      Some
        [
          Ojax.Numpy.Vectorize.vectorize ~signature:"(n)->()"
            (fun args -> RD.sum (List.hd args))
            [ a () ];
        ]
  | "tb_pad_vmap" ->
      Some
        (Ojax.Interpreters.Batching.vmap
           (fun args ->
             match args with
             | [ x; v ] -> [ L.pad x v [| (1, 1, 0) |] ]
             | _ -> failwith "parity: pad arity")
           [ Some 0; Some 0 ]
           [ a (); b () ])
  | "tb_convolve_c" ->
      Some
        [
          NL.convolve ~mode:(U.to_string (U.member "mode" params)) (a ()) (b ());
        ]
  | "tb_correlate_c" ->
      Some
        [
          NL.correlate
            ~mode:(U.to_string (U.member "mode" params))
            (a ()) (b ());
        ]
  | "tb_lax_conv_c" ->
      let x = a () and y = b () in
      let widen v =
        match dt_of v with
        | Ojax.Dtype.Complex64 | Ojax.Dtype.Complex128 -> v
        | _ -> NL.astype v (dt_of y)
      in
      ignore widen;
      Some [ L.conv_general_dilated x y ~window_strides:[| 1; 1 |] L.Same ]
  | "tb_grad_conv_c" ->
      let kernel = b () in
      Some
        [
          List.hd
            (Ojax.Interpreters.Ad.grad
               (fun args ->
                 RD.sum (UF.absolute (NL.convolve (List.hd args) kernel)))
               [ a () ]);
        ]
  | "tb_seed" ->
      let seed = NL.astype (RD.sum (a ())) Ojax.Dtype.I32 in
      Some [ Ojax.Random.Core.key_data (Ojax.Random.Core.prng_key seed) ]
  | _ -> None

let slice_prefix v stop =
  let sh = (Ojax.Core.get_aval v).Ojax.Types.shape in
  let limit = Array.copy sh in
  limit.(0) <- min stop sh.(0);
  if limit = sh then v
  else
    Ojax.Core.bind1
      (Ojax.Types.Slice
         {
           start_indices = Array.make (Array.length sh) 0;
           limit_indices = limit;
           strides = None;
         })
      [ v ]

let library_sweep op v =
  let module NF = Ojax.Nn.Functions in
  let module SP = Ojax.Scipy.Special in
  let module St = Ojax.Scipy.Stats in
  let lit x reference = L.full [||] x (dt_of reference) in
  let py = Ojax.Numpy.Jnp_util.py_float in
  let u () = NF.sigmoid v in
  let q () =
    let s = NF.softplus v in
    UF.add s (py 0.1)
  in
  let q1 () =
    let p = q () in
    UF.add p (py 1.0)
  in
  match op with
  | "sw_nn_relu" -> Some (NF.relu v)
  | "sw_nn_relu6" -> Some (NF.relu6 v)
  | "sw_nn_softplus" -> Some (NF.softplus v)
  | "sw_nn_sparse_plus" -> Some (NF.sparse_plus v)
  | "sw_nn_soft_sign" -> Some (NF.soft_sign v)
  | "sw_nn_sigmoid" -> Some (NF.sigmoid v)
  | "sw_nn_sparse_sigmoid" -> Some (NF.sparse_sigmoid v)
  | "sw_nn_silu" -> Some (NF.silu v)
  | "sw_nn_log_sigmoid" -> Some (NF.log_sigmoid v)
  | "sw_nn_hard_tanh" -> Some (NF.hard_tanh v)
  | "sw_nn_hard_sigmoid" -> Some (NF.hard_sigmoid v)
  | "sw_nn_hard_silu" -> Some (NF.hard_silu v)
  | "sw_nn_selu" -> Some (NF.selu v)
  | "sw_nn_elu" -> Some (NF.elu v)
  | "sw_nn_celu" -> Some (NF.celu v)
  | "sw_nn_leaky_relu" -> Some (NF.leaky_relu v)
  | "sw_nn_squareplus" -> Some (NF.squareplus v)
  | "sw_nn_gelu" -> Some (NF.gelu v)
  | "sw_nn_gelu_exact" -> Some (NF.gelu ~approximate:false v)
  | "sw_nn_glu" -> Some (NF.glu ~axis:0 (slice_prefix v 2))
  | "sw_nn_softmax" -> Some (NF.softmax v)
  | "sw_nn_log_softmax" -> Some (NF.log_softmax v)
  | "sw_nn_standardize" -> Some (NF.standardize v)
  | "sw_nn_logmeanexp" -> Some (NF.logmeanexp ~axis:[| 0 |] v)
  | "sw_nn_log1mexp" -> Some (NF.log1mexp (q ()))
  | "sw_sp_gammaln" -> Some (SP.gammaln v)
  | "sw_sp_gammasgn" -> Some (SP.gammasgn v)
  | "sw_sp_digamma" -> Some (SP.digamma v)
  | "sw_sp_gamma" -> Some (SP.gamma v)
  | "sw_sp_betaln" -> Some (SP.betaln (q ()) (q1 ()))
  | "sw_sp_erf" -> Some (SP.erf v)
  | "sw_sp_erfc" -> Some (SP.erfc v)
  | "sw_sp_erfinv" ->
      let p = u () in
      Some (SP.erfinv (UF.subtract (UF.multiply (lit 2.0 p) p) (lit 1.0 p)))
  | "sw_sp_erfcx" -> Some (SP.erfcx v)
  | "sw_sp_dawsn" -> Some (SP.dawsn v)
  | "sw_sp_expit" -> Some (SP.expit v)
  | "sw_sp_logit" -> Some (SP.logit (u ()))
  | "sw_sp_xlogy" -> Some (SP.xlogy v (q ()))
  | "sw_sp_xlog1py" -> Some (SP.xlog1py v (q ()))
  | "sw_sp_entr" -> Some (SP.entr (q ()))
  | "sw_sp_boxcox" -> Some (SP.boxcox (q ()) (py 0.5))
  | "sw_sp_boxcox1p" -> Some (SP.boxcox1p (q ()) (py 0.5))
  | "sw_sp_rel_entr" -> Some (SP.rel_entr (q ()) (q1 ()))
  | "sw_sp_kl_div" -> Some (SP.kl_div (q ()) (q1 ()))
  | "sw_sp_i0" -> Some (SP.i0 v)
  | "sw_sp_i0e" -> Some (SP.i0e v)
  | "sw_sp_i1" -> Some (SP.i1 v)
  | "sw_sp_i1e" -> Some (SP.i1e v)
  | "sw_sp_ndtr" -> Some (SP.ndtr v)
  | "sw_sp_ndtri" -> Some (SP.ndtri (u ()))
  | "sw_sp_log_ndtr" -> Some (SP.log_ndtr v)
  | "sw_sp_polygamma" ->
      Some
        (SP.polygamma
           (T.Concrete (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [||] [| 1.0 |]))
           (q ()))
  | "sw_sp_zeta" ->
      let x = q1 () in
      Some (SP.zeta ~q:(q ()) x)
  | "sw_sp_poch" -> Some (SP.poch (q ()) (py 2.0))
  | "sw_sp_spence" -> Some (SP.spence (q ()))
  | "sw_sp_owens_t" -> Some (SP.owens_t v (q ()))
  | "sw_sp_betainc" -> Some (SP.betainc (q ()) (q1 ()) (u ()))
  | "sw_sp_gammainc" -> Some (SP.gammainc (q ()) (q ()))
  | "sw_sp_gammaincc" -> Some (SP.gammaincc (q ()) (q ()))
  | "sw_sp_hyp1f1" -> Some (SP.hyp1f1 (q ()) (q1 ()) (u ()))
  | "sw_st_norm_logpdf" -> Some (St.Norm.logpdf v)
  | "sw_st_norm_pdf" -> Some (St.Norm.pdf v)
  | "sw_st_norm_cdf" -> Some (St.Norm.cdf v)
  | "sw_st_norm_logcdf" -> Some (St.Norm.logcdf v)
  | "sw_st_norm_ppf" -> Some (St.Norm.ppf (u ()))
  | "sw_st_norm_sf" -> Some (St.Norm.sf v)
  | "sw_st_norm_logsf" -> Some (St.Norm.logsf v)
  | "sw_st_norm_isf" -> Some (St.Norm.isf (u ()))
  | "sw_st_expon_logpdf" -> Some (St.Expon.logpdf (q ()))
  | "sw_st_expon_cdf" -> Some (St.Expon.cdf (q ()))
  | "sw_st_gamma_logpdf" -> Some (St.Gamma.logpdf (q ()) (py 2.0))
  | "sw_st_gamma_cdf" -> Some (St.Gamma.cdf (q ()) (py 2.0))
  | "sw_st_beta_logpdf" -> Some (St.Beta.logpdf (u ()) (py 2.0) (py 3.0))
  | "sw_st_beta_cdf" -> Some (St.Beta.cdf (u ()) (py 2.0) (py 3.0))
  | "sw_st_laplace_logpdf" -> Some (St.Laplace.logpdf v)
  | "sw_st_laplace_cdf" -> Some (St.Laplace.cdf v)
  | "sw_st_logistic_logpdf" -> Some (St.Logistic.logpdf v)
  | "sw_st_logistic_cdf" -> Some (St.Logistic.cdf v)
  | "sw_st_cauchy_logpdf" -> Some (St.Cauchy.logpdf v)
  | "sw_st_cauchy_cdf" -> Some (St.Cauchy.cdf v)
  | "sw_st_chi2_logpdf" -> Some (St.Chi2.logpdf (q ()) (py 3.0))
  | "sw_st_pareto_logpdf" -> Some (St.Pareto.logpdf (q1 ()) (py 2.0))
  | _ -> None

let bytebuf_of_hex hex =
  let n = String.length hex / 2 in
  let buf = Bigarray.Array1.create Bigarray.int8_unsigned Bigarray.c_layout n in
  for i = 0 to n - 1 do
    Bigarray.Array1.set buf i (int_of_string ("0x" ^ String.sub hex (2 * i) 2))
  done;
  buf

let nd_of_hex dtype shape hex =
  Ojax.Pjrt.Buffer.host_of_bytes
    (Option.get (Ojax.Dtype.of_numpy_name dtype))
    shape (bytebuf_of_hex hex)

let hex_of_nd nd =
  let raw = (Npz.raw_of_nd nd).Npz.bytes in
  String.concat ""
    (List.map
       (fun c -> Printf.sprintf "%02x" (Char.code c))
       (List.of_seq (String.to_seq raw)))

let numpy_product (input : T.value) =
  match input with
  | T.Concrete nd when Ojax.Ndarray.shape nd = [| 3; 3 |] ->
      let dtype = Ojax.Dtype.numpy_name (Ojax.Ndarray.dtype nd) in
      let hex = hex_of_nd nd in
      List.find_map
        (fun (_, _, d, input_hex, product, _) ->
          if d = dtype && input_hex = hex then
            Some (T.Concrete (nd_of_hex dtype [| 3; 3 |] product))
          else None)
        Numpy_products.rows
  | _ -> None

let jnp_matmul_self v =
  Ojax.Numpy.Tensor_contractions.matmul v
    (Ojax.Numpy.Lax_numpy.matrix_transpose v)

let dtype_sweep_run op a =
  let module AC = Ojax.Numpy.Array_creation in
  let module PO = Ojax.Numpy.Polynomial in
  let module TC = Ojax.Numpy.Tensor_contractions in
  let module S = Ojax.Numpy.Sorting in
  let module FFT = Ojax.Numpy.Fft in
  let row v i = NL.index_at v i in
  let cell v i = NL.index_at (row v 0) i in
  let t = NL.matrix_transpose in
  let one f = Some [ f (a ()) ] in
  let ax0 = [| 0 |] in
  match op with
  | "sw_mean" -> one (fun v -> RD.mean v)
  | "sw_mean_axis" -> one (RD.mean ~axis:ax0)
  | "sw_var" -> one (RD.var ~axis:ax0)
  | "sw_std" -> one (RD.std ~axis:ax0)
  | "sw_sum" -> one (RD.sum ~axis:ax0)
  | "sw_prod" -> one (RD.prod ~axis:ax0)
  | "sw_cumsum" -> one (RD.cumsum ~axis:0)
  | "sw_cumprod" -> one (RD.cumprod ~axis:0)
  | "sw_diff" -> one (NL.diff ~axis:0)
  | "sw_gradient" -> one (fun v -> List.hd (E.gradient ~axis:[ 0 ] v))
  | "sw_trapezoid" -> one (NL.trapezoid ~axis:0)
  | "sw_average" -> one (RD.average ~axis:ax0)
  | "sw_cov" -> one (fun v -> NL.cov v)
  | "sw_corrcoef" -> one (fun v -> NL.corrcoef v)
  | "sw_sort" -> one (S.sort ~axis:(Some 0))
  | "sw_argsort" -> one (S.argsort ~axis:(Some 0))
  | "sw_flip" -> one (NL.flip ~axis:[| 0 |])
  | "sw_roll" -> one (fun v -> NL.roll ~axis:[| 0 |] v [| 1 |])
  | "sw_nanmean" -> one (RD.nanmean ~axis:ax0)
  | "sw_nansum" -> one (RD.nansum ~axis:ax0)
  | "sw_nanvar" -> one (RD.nanvar ~axis:ax0)
  | "sw_sinc" -> one UF.sinc
  | "sw_sign" -> one UF.sign
  | "sw_square" -> one UF.square
  | "sw_sqrt" -> one UF.sqrt
  | "sw_exp" -> one UF.exp
  | "sw_log" -> one UF.log
  | "sw_log1p" -> one UF.log1p
  | "sw_expm1" -> one UF.expm1
  | "sw_sin" -> one UF.sin
  | "sw_cos" -> one UF.cos
  | "sw_tan" -> one UF.tan
  | "sw_arcsin" -> one UF.arcsin
  | "sw_arccos" -> one UF.arccos
  | "sw_arctan" -> one UF.arctan
  | "sw_sinh" -> one UF.sinh
  | "sw_cosh" -> one UF.cosh
  | "sw_tanh" -> one UF.tanh
  | "sw_arcsinh" -> one UF.arcsinh
  | "sw_arccosh" -> one UF.arccosh
  | "sw_arctanh" -> one UF.arctanh
  | "sw_real" -> one UF.real
  | "sw_imag" -> one UF.imag
  | "sw_conj" -> one UF.conj
  | "sw_angle" -> one NL.angle
  | "sw_abs" -> one UF.absolute
  | "sw_isnan" -> one UF.isnan
  | "sw_isfinite" -> one UF.isfinite
  | "sw_isinf" -> one UF.isinf
  | "sw_round" -> one (NL.round ~decimals:1)
  | "sw_bitwise_and" -> one (fun v -> UF.bitwise_and v (L.rev v [| 0 |]))
  | "sw_bitwise_or" -> one (fun v -> UF.bitwise_or v (L.rev v [| 0 |]))
  | "sw_bitwise_xor" -> one (fun v -> UF.bitwise_xor v (L.rev v [| 0 |]))
  | "sw_bitwise_not" -> one UF.bitwise_not
  | "sw_left_shift" -> one (fun v -> UF.left_shift v (AC.ones_like v))
  | "sw_right_shift" -> one (fun v -> UF.right_shift v (AC.ones_like v))
  | "sw_remainder" ->
      one (fun v -> UF.remainder v (UF.add (L.rev v [| 0 |]) (AC.ones_like v)))
  | "sw_floor_divide" ->
      one (fun v ->
          UF.floor_divide v (UF.add (L.rev v [| 0 |]) (AC.ones_like v)))
  | "sw_power_self" ->
      one (fun v -> UF.power v (UF.add (AC.ones_like v) (AC.ones_like v)))
  | "sw_gcd" -> one (fun v -> NL.gcd v (L.rev v [| 0 |]))
  | "sw_lcm" -> one (fun v -> NL.lcm v (L.rev v [| 0 |]))
  | "sw_floor_div_self" ->
      one (fun v ->
          let y = UF.add v (Ojax.Numpy.Jnp_util.py_float 2.0) in
          UF.divide v y)
  | "sw_linspace_end" -> one (fun v -> AC.linspace_v ~num:4 (row v 0) (row v 1))
  | "sw_geomspace" ->
      one (fun v ->
          let py = Ojax.Numpy.Jnp_util.py_float in
          let lower = UF.add (row v 0) (py 2.0) in
          let upper = UF.add (row v 1) (py 3.0) in
          AC.geomspace_v ~num:3 lower upper)
  | "sw_logspace" -> one (fun v -> AC.logspace_v ~num:3 (row v 0) (row v 1))
  | "sw_polyval" -> one (fun v -> PO.polyval (slice_prefix v 2) (row v 0))
  | "sw_polyder" -> one (fun v -> PO.polyder (row v 0))
  | "sw_polyint" -> one (fun v -> PO.polyint (row v 0))
  | "sw_polyadd" -> one (fun v -> PO.polyadd (row v 0) (row v 1))
  | "sw_polymul" -> one (fun v -> PO.polymul (row v 0) (row v 1))
  | "sw_roots" ->
      one (fun v -> S.sort ~axis:(Some 0) (UF.absolute (E.roots (row v 0))))
  | "sw_kron" -> one (fun v -> NL.kron v v)
  | "sw_dot" -> one (fun v -> TC.dot v (NL.transpose v))
  | "sw_vdot" -> one (fun v -> TC.vdot v v)
  | "sw_inner" -> one (fun v -> TC.inner v v)
  | "sw_outer" -> one (fun v -> TC.outer (row v 0) (row v 1))
  | "sw_matmul" ->
      one (fun v ->
          match numpy_product v with
          | Some product -> product
          | None -> TC.matmul v (t v))
  | "sw_trace" ->
      one (fun v ->
          match numpy_product v with
          | Some product -> NL.trace product
          | None -> NL.trace (TC.matmul v (t v)))
  | "sw_cross" -> one (fun v -> NL.cross (row v 0) (row v 1))
  | "sw_tensordot" -> one (fun v -> TC.tensordot ~axes:(TC.Ax_int 1) v (t v))
  | "sw_einsum" -> one (fun v -> Ojax.Numpy.Einsum.einsum "ij,kj->ik" [ v; v ])
  | "sw_clip" -> one (NL.clip ~min:(-0.5) ~max:0.5)
  | "sw_minimum" -> one (fun v -> UF.minimum v (L.rev v [| 0 |]))
  | "sw_maximum" -> one (fun v -> UF.maximum v (L.rev v [| 0 |]))
  | "sw_max" -> one (RD.amax ~axis:ax0)
  | "sw_min" -> one (RD.amin ~axis:ax0)
  | "sw_argmax" -> one (NL.argmax ~axis:0)
  | "sw_take" ->
      one (fun v ->
          Ojax.Numpy.Indexing.take ~axis:0 v
            (T.Concrete
               (Ojax.Ndarray.of_floats
                  (Ojax.Dtypes.default_int_dtype ())
                  [| 2 |] [| 0.; 1. |])))
  | "sw_where" ->
      one (fun v ->
          let c = UF.greater (UF.real v) (Ojax.Numpy.Jnp_util.py_int 0) in
          let y = UF.multiply v (Ojax.Numpy.Jnp_util.py_float 2.0) in
          NL.where_ c v y)
  | "sw_concatenate" -> one (fun v -> NL.concatenate ~axis:0 [ v; v ])
  | "sw_pad" ->
      one (fun v -> NL.pad v (Array.map (fun _ -> (1, 1)) (shp v)) 0.0)
  | "sw_repeat" -> one (fun v -> Ojax.Numpy.Array_methods.repeat ~axis:0 v 2)
  | "sw_tile" -> one (fun v -> NL.tile v [| 2; 1 |])
  | "sw_interp" ->
      one (fun v ->
          let dt = Ojax.Dtypes.default_float_dtype () in
          let arr sh xs = T.Concrete (Ojax.Ndarray.of_floats dt sh xs) in
          let xs = arr [| 1 |] [| 0.5 |] in
          let xp = arr [| 3 |] [| 0.; 1.; 2. |] in
          E.interp xs xp (row v 0))
  | "sw_fft" -> one (FFT.fft ~axis:0)
  | "sw_ifft" -> one (FFT.ifft ~axis:0)
  | "sw_tril" -> one (fun v -> NL.tril v)
  | "sw_diag" -> one (fun v -> NL.diag v)
  | _ -> (
      ignore cell;
      match library_sweep op (a ()) with
      | Some out -> Some [ out ]
      | None -> None)

let fft_run op params a =
  match op with
  | "fft" -> fft1 F.fft (a ()) params
  | "ifft" -> fft1 F.ifft (a ()) params
  | "rfft" -> fft1 F.rfft (a ()) params
  | "irfft" -> fft1 F.irfft (a ()) params
  | "hfft" -> fft1 F.hfft (a ()) params
  | "ihfft" -> fft1 F.ihfft (a ()) params
  | "fftn" -> fftnd F.fftn (a ()) params
  | "ifftn" -> fftnd F.ifftn (a ()) params
  | "rfftn" -> fftnd F.rfftn (a ()) params
  | "irfftn" -> fftnd F.irfftn (a ()) params
  | "fft2" -> [ F.fft2 ?norm:(opt_string "norm" params) (a ()) ]
  | "ifft2" -> [ F.ifft2 ?norm:(opt_string "norm" params) (a ()) ]
  | "rfft2" -> [ F.rfft2 ?norm:(opt_string "norm" params) (a ()) ]
  | "irfft2" -> [ F.irfft2 ?norm:(opt_string "norm" params) (a ()) ]
  | "fftshift" -> [ F.fftshift ?axes:(opt_ints "axes" params) (a ()) ]
  | "ifftshift" -> [ F.ifftshift ?axes:(opt_ints "axes" params) (a ()) ]
  | other -> failwith ("parity: unknown op " ^ other)

let special_batch_run op params a =
  let module SP = Ojax.Scipy.Special in
  let float_p k = U.to_float (U.member k params) in
  let int_p k = U.to_int (U.member k params) in
  let abc () =
    Array.of_list (List.map U.to_float (U.to_list (U.member "abc" params)))
  in
  let ints k =
    let values = List.map U.to_int (U.to_list (U.member k params)) in
    T.Concrete
      (Ojax.Ndarray.of_floats
         (if Ojax.Config.x64_enabled () then Ojax.Dtype.I64 else Ojax.Dtype.I32)
         [| List.length values |]
         (Array.of_list (List.map float_of_int values)))
  in
  let scaled v k = UF.multiply v (like v (float_p k)) in
  match op with
  | "sx_expi" -> Some [ SP.expi (scaled (a ()) "scale") ]
  | "sx_exp1" -> Some [ SP.exp1 (scaled (UF.absolute (a ())) "scale") ]
  | "sx_expn" ->
      let x = scaled (UF.absolute (a ())) "scale" in
      Some [ SP.expn (Ojax.Numpy.Jnp_util.py_float (float_p "n")) x ]
  | "sx_expn_grad" ->
      let x = scaled (UF.absolute (a ())) "scale" in
      let x = UF.add x (like x 0.1) in
      let order = Ojax.Numpy.Jnp_util.py_float (float_p "n") in
      Some
        (Ojax.Interpreters.Batching.vmap
           (fun args ->
             Ojax.Interpreters.Ad.grad
               (fun inner -> SP.expn order (List.hd inner))
               args)
           [ Some 0 ] [ x ])
  | "sx_expi_grad" ->
      let x = scaled (a ()) "scale" in
      let x = UF.add x (like x 0.5) in
      Some
        (Ojax.Interpreters.Batching.vmap
           (fun args ->
             Ojax.Interpreters.Ad.grad
               (fun inner -> SP.expi (List.hd inner))
               args)
           [ Some 0 ] [ x ])
  | "sx_hyp2f1" ->
      let v = a () in
      let t = abc () in
      let full x =
        if Ojax.Core.top_is_staging [] then
          NL.broadcast_to (Ojax.Numpy.Jnp_util.py_float x) (shp v)
        else
          let dt = Ojax.Dtypes.default_float_dtype () in
          let n = Array.fold_left ( * ) 1 (shp v) in
          T.Concrete
            (Ojax.Ndarray.as_weak
               (Ojax.Ndarray.of_floats dt (shp v) (Array.make n x)))
      in
      let first = full t.(0) in
      let second = full t.(1) in
      let third = full t.(2) in
      let z =
        UF.multiply (Ojax.Numpy.Jnp_util.py_float (float_p "scale")) (UF.tanh v)
      in
      Some [ SP.hyp2f1 first second third z ]
  | "sx_hyp2f1_grad" ->
      let v = a () in
      let t = abc () in
      let py = Ojax.Numpy.Jnp_util.py_float in
      let z = UF.multiply (py 0.5) (UF.tanh v) in
      let lit x = py x in
      Some
        (Ojax.Interpreters.Batching.vmap
           (fun args ->
             Ojax.Interpreters.Ad.grad
               (fun inner ->
                 let p0 = lit t.(0) in
                 let p1 = lit t.(1) in
                 let p2 = lit t.(2) in
                 SP.hyp2f1 p0 p1 p2 (List.hd inner))
               args)
           [ Some 0 ] [ z ])
  | "sx_lpmn" ->
      let l = int_p "l" in
      Some (SP.lpmn l l (UF.tanh (a ())))
  | "sx_lpmn_values" ->
      let l = int_p "l" in
      Some
        [
          SP.lpmn_values l l
            (UF.tanh (a ()))
            (U.to_bool (U.member "norm" params));
        ]
  | "sx_sph" ->
      let v = a () in
      let n = ints "n" in
      let m = ints "m" in
      let theta = UF.absolute v in
      let phi = UF.add (UF.multiply v (like v 0.5)) (like v 0.2) in
      let out = SP.sph_harm_y ~n_max:(int_p "n_max") n m theta phi in
      Some
        [
          (if dt_of v = Ojax.Dtype.F32 then NL.astype out Ojax.Dtype.Complex64
           else out);
        ]
  | _ -> None

let transform_sweep_run op a =
  let prefixed p =
    String.length op > String.length p && String.sub op 0 (String.length p) = p
  in
  let base () = "sw_" ^ String.sub op 4 (String.length op - 4) in
  let apply v =
    match dtype_sweep_run (base ()) (fun () -> v) with
    | Some [ out ] -> out
    | Some _ | None -> failwith "parity: transform sweep expects one output"
  in
  if prefixed "swv_" then
    Some
      (Ojax.Interpreters.Batching.vmap
         (fun args -> [ apply (List.hd args) ])
         [ Some 0 ]
         [ a () ])
  else if prefixed "swg_" then
    Some
      (Ojax.Interpreters.Ad.grad
         (fun args ->
           let m = UF.absolute (apply (List.hd args)) in
           RD.sum (UF.power m (scalar_int 2)))
         [ a () ])
  else if prefixed "swj_" then begin
    let x = a () in
    let half = UF.multiply x (like x 0.5) in
    let tangent = UF.add half (like x 0.25) in
    let _, t =
      Ojax.Interpreters.Ad.jvp_instantiated
        (fun args -> [ apply (List.hd args) ])
        [ x ] [ tangent ]
    in
    Some t
  end
  else None

let compares_at_single_precision op =
  let name =
    match String.index_opt op '_' with
    | Some i -> String.sub op (i + 1) (String.length op - i - 1)
    | None -> op
  in
  List.mem name [ "sp_owens_t"; "st_norm_ppf"; "st_norm_isf" ]

let rec run op params get =
  if compares_at_single_precision op && dt_of (get "a") <> Ojax.Dtype.F64 then
    let narrow =
      if dt_of (get "a") = Ojax.Dtype.F16 then Ojax.Dtype.F16
      else Ojax.Dtype.F32
    in
    List.map
      (fun v ->
        match dt_of v with
        | Ojax.Dtype.F64 | Ojax.Dtype.F32 -> NL.astype v narrow
        | _ -> v)
      (run_cases op params get)
  else run_cases op params get

and run_cases op params get =
  let a () = get "a" in
  let b () = get "b" in
  let c () = get "c" in
  let runners =
    [
      (fun () -> special_batch_run op params a);
      (fun () -> transform_sweep_run op a);
      (fun () -> dtype_sweep_run op a);
      (fun () -> tracer_batch_run op params a b);
      (fun () -> surface_batch_run op params a b);
      (fun () -> linalg_batch_run op params a b);
      (fun () -> complex_grad_run op a);
      (fun () -> complex_linalg_run op a b);
      (fun () -> gap_run op params a b);
      (fun () -> array_random_run op params a b);
      (fun () -> default_random_run op params);
      (fun () -> lax_extra_run op params a);
      (fun () -> misc_scipy_run op params a b);
      (fun () -> batched_run op params a b);
      (fun () -> narrow_run op params a b);
      (fun () -> optimizer_run op a b);
      (fun () -> scipy_run op params a b c);
      (fun () -> numpy_run op params a b c);
      (fun () -> lax_run op params a b);
    ]
  in
  match List.find_map (fun runner -> runner ()) runners with
  | Some out -> out
  | None -> fft_run op params a

let needs_concrete op =
  List.mem op
    [
      "sw_roots";
      "swg_roots";
      "swj_roots";
      "np_nonzero";
      "np_flatnonzero";
      "np_argwhere";
      "np_unique_values";
      "np_unique_counts";
      "np_unique_inverse";
      "np_unique_all";
      "np_unique_axis";
      "np_union1d";
      "np_intersect1d";
      "np_setdiff1d";
      "np_setxor1d";
      "np_bincount";
      "np_compress";
      "np_extract";
      "np_delete";
      "np_trim_zeros";
      "np_polydiv";
      "np_roots";
    ]

let result_names = function
  | "gp_qr_full" ->
      Some (Ojax.Stages.result_names_of_tree Ojax.Numpy.Linalg.qr_result_def)
  | _ -> None
