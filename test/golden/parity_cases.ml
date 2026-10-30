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
            (T.Concrete
               (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [||]
                  [| float_of_int (int "start" p) |]))
            (int "size" p);
        ]
  | "lax_dynamic_index_in_dim" ->
      Some
        [
          L.dynamic_index_in_dim ?axis:(opt_int "axis" p)
            ?keepdims:(opt_bool "keepdims" p) (a ())
            (T.Concrete
               (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [||]
                  [| float_of_int (int "index" p) |]));
        ]
  | "lax_reduce_custom" ->
      Some [ L.reduce (a ()) (like (a ()) 0.0) L.add (ints "dims" p) ]
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
      Some
        [
          L.reduce_window (a ())
            (like (a ()) 0.0)
            L.add ~window_dimensions:(ints "win" p)
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
      let scalar_i v =
        T.Concrete
          (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [||] [| float_of_int v |])
      in
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
let binarize a = UF.greater a (L.full_like a 0.0)

let rounded a =
  let two = L.full_like a 2.0 in
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
        (E.nonzero ~size:(int "size" p) ?fill_value:(opt_int "fill" p)
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
        [
          E.unique_values ~size:(int "size" p) ?fill_value:(opt_number "fill" p)
            (rounded (a ()));
        ]
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
  | "np_gradient" -> Some (E.gradient ?spacing:(opt_number "spacing" p) (a ()))
  | "np_gradient_axis" -> Some (E.gradient ?axis:(opt_ints "axis" p) (a ()))
  | "np_choose" ->
      let two = L.full_like (nonneg (a ())) 2.0 in
      Some
        [
          E.choose ~mode:"clip"
            (UF.remainder (nonneg (a ())) two)
            [ b (); L.neg (b ()) ];
        ]
  | "np_compress" ->
      let bools =
        Array.of_list
          (List.map
             (fun v -> if U.to_bool v then 1.0 else 0.0)
             (U.to_list (U.member "cond" p)))
      in
      let cond =
        T.Concrete
          (Ojax.Ndarray.of_floats Ojax.Dtype.Bool
             [| Array.length bools |]
             bools)
      in
      Some [ E.compress ~axis:(int "axis" p) cond (a ()) ]
  | "np_extract" -> Some [ E.extract (binarize (a ())) (a ()) ]
  | "np_delete" ->
      Some
        [
          E.delete ?axis:(opt_int "axis" p) (a ())
            (Array.to_list (ints "obj" p));
        ]
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
            [
              UF.less x (L.full_like x 0.0);
              UF.greater_equal x (L.full_like x 0.0);
            ]
            [ L.neg; (fun v -> L.mul v (L.full_like v 2.0)) ];
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
             (shp (a ()))
             (Array.map float_of_int (ints "e" p)))
      in
      Some [ E.ldexp (a ()) e ]
  | "np_apply_along_axis" ->
      Some [ E.apply_along_axis (fun v -> RD.sum v) (int "axis" p) (a ()) ]
  | "np_apply_along_axis_keep" ->
      Some
        [
          E.apply_along_axis
            (fun v -> L.mul v (L.full_like v 2.0))
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
       (Ojax.Ndarray.of_floats Ojax.Dtype.I32 [||]
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
        L.add v (L.mul eye (L.full_like eye 4.0))
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
      let m = b () in
      let n = (shp m).(0) in
      let product = L.dot m (L.transpose m [| 1; 0 |]) in
      let eye = NL.identity ~dtype:(dt_of m) n in
      let cov = L.add product (L.mul eye (L.full_like eye 2.0)) in
      Some
        [
          RX.multivariate_normal ~shape:(ints "shape" p)
            ~method_:(required_string "method" p)
            (seed_key p) ~mean:(a ()) ~cov;
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

let scaled v x = L.mul v (L.full_like v x)

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
      Some [ L.mul v (L.full_like v 2.0) ]
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
  let product = L.batch_matmul v (NL.swapaxes (-1) (-2) v) in
  let eye = NL.broadcast_to (NL.identity ~dtype:(dt_of v) n) sh in
  UF.add product (L.mul eye (L.full_like eye 3.0))

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
  let axis size =
    T.Concrete
      (Ojax.Ndarray.of_floats dt [| size |] (Array.init size float_of_int))
  in
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
      Some (pair (Ojax.Scipy.Special.fresnel (L.mul x (L.full_like x factor))))
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
      let lo = a () in
      let hi = L.add (L.add lo (L.abs (b ()))) (L.full_like lo 1.0) in
      Some [ RC.uniform_range (k ()) ~shape:(shp lo) ~minval:lo ~maxval:hi ]
  | "ra_bernoulli" ->
      let pr = L.abs (L.sin (a ())) in
      Some [ RC.bernoulli_prob (k ()) ~shape:(shp pr) ~p:pr ]
  | "ra_gamma" -> Some [ RC.gamma_value (k ()) ~shape:(shp (a ())) ~a:(a ()) ]
  | "ra_loggamma" ->
      Some [ RC.loggamma_value (k ()) ~shape:(shp (a ())) ~a:(a ()) ]
  | "ra_beta" ->
      Some [ RC.beta_value (k ()) ~shape:(shp (a ())) ~a:(a ()) ~b:(b ()) ]
  | "ra_poisson" ->
      let lam = L.mul (a ()) (L.full_like (a ()) 8.0) in
      Some [ RC.poisson_value (k ()) ~shape:(shp lam) ~lam ]
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
      let z = L.add (L.mul x (L.full_like x 0.7)) (L.full_like x 0.2) in
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
        L.reduce_sum (fst (Ojax.Scipy.Special.fresnel (List.hd args))) [| 0 |]
      in
      Some (Ojax.Interpreters.Ad.grad f [ a () ])
  | _ -> None

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

let run op params get =
  let a () = get "a" in
  let b () = get "b" in
  let c () = get "c" in
  let runners =
    [
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
      "np_nonzero";
      "np_flatnonzero";
      "np_argwhere";
      "np_unique_values";
      "np_unique_counts";
      "np_unique_inverse";
      "np_unique_all";
      "np_unique_size";
      "np_unique_axis";
      "np_union1d";
      "np_intersect1d";
      "np_setdiff1d";
      "np_setxor1d";
      "np_bincount";
      "np_compress";
      "np_extract";
      "np_delete";
      "np_place";
      "np_trim_zeros";
      "np_frexp";
      "np_polydiv";
      "np_roots";
      "np_packbits";
      "np_unpackbits";
      "np_histogramdd";
      "sp_fresnel";
    ]
