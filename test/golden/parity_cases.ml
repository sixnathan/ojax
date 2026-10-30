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

let run op params get =
  let a () = get "a" in
  let b () = get "b" in
  let c () = get "c" in
  match numpy_run op params a b c with
  | Some out -> out
  | None -> (
      match lax_run op params a b with
      | Some out -> out
      | None -> (
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
          | other -> failwith ("parity: unknown op " ^ other)))
