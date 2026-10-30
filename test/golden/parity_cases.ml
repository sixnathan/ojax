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

let run op params get =
  let a () = get "a" in
  let b () = get "b" in
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
      | other -> failwith ("parity: unknown op " ^ other))
