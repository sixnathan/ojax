open Types

let b1 = Core.bind1
let shape_of v = (Core.get_aval v).shape
let rank_of v = Array.length (shape_of v)
let canon_axis ndim a = if a < 0 then a + ndim else a

let as_complex v =
  match (Core.get_aval v).dtype with
  | Dtype.Complex64 | Dtype.Complex128 -> v
  | Dtype.F64 | Dtype.I64 -> b1 (Convert_element_type Dtype.Complex128) [ v ]
  | _ -> b1 (Convert_element_type Dtype.Complex64) [ v ]

let as_real v =
  match (Core.get_aval v).dtype with
  | Dtype.F32 | Dtype.F64 -> v
  | Dtype.I64 -> b1 (Convert_element_type Dtype.F64) [ v ]
  | _ -> b1 (Convert_element_type Dtype.F32) [ v ]

let shape_array dt sizes =
  let n = [| List.length sizes |] in
  let values = Array.of_list (List.map float_of_int sizes) in
  match dt with
  | Dtype.Complex64 | Dtype.Complex128 ->
      Ndarray.of_complex dt n
        (Array.map (fun re -> { Complex.re; im = 0.0 }) values)
  | _ -> Ndarray.of_floats dt n values

let fft_norm ~inverse sizes dt norm =
  let product () = Reductions.prod (Concrete (shape_array dt sizes)) in
  let reciprocal v = Ufuncs.divide (Jnp_util.py_int 1) v in
  match norm with
  | "backward" ->
      Concrete (Ndarray.of_floats (Dtypes.default_int_dtype ()) [||] [| 1.0 |])
  | "ortho" ->
      let root = Ufuncs.sqrt (product ()) in
      if inverse then root else reciprocal root
  | "forward" -> if inverse then product () else reciprocal (product ())
  | other ->
      invalid_arg
        (Printf.sprintf
           "Invalid norm value %s; should be \"backward\",\"ortho\" or \
            \"forward\"."
           other)

let apply_norm ~inverse sizes norm v =
  match norm with
  | None -> v
  | Some norm ->
      Ufuncs.multiply v (fft_norm ~inverse sizes (Core.get_aval v).dtype norm)

let scale_const factor v =
  if factor = 1.0 then v
  else
    let aval = Core.get_aval v in
    b1 Mul [ v; Core.splat aval.dtype [||] factor ]

let moved_perm ndim source destination =
  let perm =
    ref (List.filter (fun d -> not (List.mem d source)) (List.init ndim Fun.id))
  in
  List.iter
    (fun (dest, src) ->
      let rec insert k l =
        if k = 0 then src :: l
        else match l with x :: r -> x :: insert (k - 1) r | [] -> [ src ]
      in
      perm := insert dest !perm)
    (List.sort compare (List.combine destination source));
  Array.of_list !perm

let lax_fft kind sizes v =
  let key =
    (match kind with
      | Fft_forward -> "f"
      | Fft_inverse -> "i"
      | Fft_real -> "r"
      | Fft_inverse_real -> "ir")
    ^ String.concat "," (List.map string_of_int sizes)
  in
  Jnp_util.jit1 "fft" ~key
    (function
      | [ v ] ->
          let prepared =
            match kind with
            | Fft_real -> as_real v
            | Fft_forward | Fft_inverse | Fft_inverse_real -> as_complex v
          in
          b1
            (Fft { fft_type = kind; fft_lengths = Array.of_list sizes })
            [ prepared ]
      | _ -> invalid_arg "fft: arity")
    [ v ]

let pad_zero v widths =
  let key =
    String.concat ";"
      (List.map
         (fun (lo, hi) -> string_of_int lo ^ "," ^ string_of_int hi)
         widths)
  in
  let default_int =
    if Config.value Config.enable_x64 then Dtype.I64 else Dtype.I32
  in
  Jnp_util.jit1 "_pad" ~key
    (function
      | [ v; cval ] ->
          let dt = (Core.get_aval v).dtype in
          let cval =
            if (Core.get_aval cval).dtype = dt then cval
            else b1 (Convert_element_type dt) [ cval ]
          in
          b1
            (Np_ints.mark_prim
               (Pad (Array.of_list (List.map (fun (l, h) -> (l, h, 0)) widths))))
            [ v; cval ]
      | _ -> invalid_arg "pad: arity")
    [ v; Concrete (Ndarray.of_floats default_int [||] [| 0.0 |]) ]

let fft_core kind ~inverse ?s ?axes ?norm a =
  let ndim = rank_of a in
  let orig_axes = Option.map (List.map (canon_axis ndim)) axes in
  let axes =
    match (orig_axes, s) with
    | Some ax, _ -> ax
    | None, Some sizes ->
        let k = List.length sizes in
        List.init k (fun i -> ndim - k + i)
    | None, None -> List.init ndim Fun.id
  in
  if List.length (List.sort_uniq compare axes) <> List.length axes then
    invalid_arg "numpy.fft: repeated axis";
  (match s with
  | Some sizes when List.length sizes <> List.length axes ->
      invalid_arg "Shape and axes have different lengths."
  | _ -> ());
  let k = List.length axes in
  let last_axes = List.init k (fun i -> ndim - k + i) in
  let arr =
    match orig_axes with
    | Some ax -> b1 (Transpose (moved_perm ndim ax last_axes)) [ a ]
    | None -> a
  in
  let shape = shape_of arr in
  let arr, sizes =
    match s with
    | Some sizes ->
        let in_s = Array.copy shape in
        List.iter2 (fun ax n -> in_s.(ax) <- n) last_axes sizes;
        (match kind with
        | Fft_inverse_real when ndim > 0 ->
            in_s.(ndim - 1) <- (in_s.(ndim - 1) / 2) + 1
        | _ -> ());
        let limit = Array.mapi (fun i n -> min n shape.(i)) in_s in
        let cropped =
          if limit = shape then arr
          else
            b1
              (Slice
                 {
                   start_indices = Array.make ndim 0;
                   limit_indices = limit;
                   strides = None;
                 })
              [ arr ]
        in
        let widths = List.init ndim (fun i -> (0, in_s.(i) - limit.(i))) in
        (pad_zero cropped widths, sizes)
    | None -> (
        match kind with
        | Fft_inverse_real ->
            let n = List.length last_axes in
            ( arr,
              List.mapi
                (fun i ax ->
                  if i = n - 1 then max 0 (2 * (shape.(ax) - 1)) else shape.(ax))
                last_axes )
        | Fft_forward | Fft_inverse | Fft_real ->
            (arr, List.map (fun ax -> shape.(ax)) last_axes))
  in
  let transformed = lax_fft kind sizes arr in
  let scaled = apply_norm ~inverse sizes norm transformed in
  match orig_axes with
  | Some ax -> b1 (Transpose (moved_perm ndim last_axes ax)) [ scaled ]
  | None -> scaled

let single_axis axis n = (Some [ axis ], Option.map (fun x -> [ x ]) n)

let fft ?n ?(axis = -1) ?norm a =
  let axes, s = single_axis axis n in
  fft_core Fft_forward ~inverse:false ?s ?axes ?norm a

let ifft ?n ?(axis = -1) ?norm a =
  let axes, s = single_axis axis n in
  fft_core Fft_inverse ~inverse:true ?s ?axes ?norm a

let rfft ?n ?(axis = -1) ?norm a =
  let axes, s = single_axis axis n in
  fft_core Fft_real ~inverse:false ?s ?axes ?norm a

let irfft ?n ?(axis = -1) ?norm a =
  let axes, s = single_axis axis n in
  fft_core Fft_inverse_real ~inverse:true ?s ?axes ?norm a

let hfft ?n ?(axis = -1) ?norm a =
  let conj_a = b1 Conj [ as_complex a ] in
  let ax = canon_axis (rank_of a) axis in
  let nn =
    match n with Some x -> x | None -> ((shape_of conj_a).(ax) * 2) - 2
  in
  let axes, s = single_axis axis n in
  scale_const (float_of_int nn)
    (fft_core Fft_inverse_real ~inverse:false ?s ?axes ?norm conj_a)

let ihfft ?n ?(axis = -1) ?norm a =
  let ax = canon_axis (rank_of a) axis in
  let nn = match n with Some x -> x | None -> (shape_of a).(ax) in
  let axes, s = single_axis axis n in
  scale_const
    (1.0 /. float_of_int nn)
    (b1 Conj [ fft_core Fft_real ~inverse:true ?s ?axes ?norm a ])

let fftn ?s ?axes ?norm a = fft_core Fft_forward ~inverse:false ?s ?axes ?norm a
let ifftn ?s ?axes ?norm a = fft_core Fft_inverse ~inverse:true ?s ?axes ?norm a
let rfftn ?s ?axes ?norm a = fft_core Fft_real ~inverse:false ?s ?axes ?norm a

let irfftn ?s ?axes ?norm a =
  fft_core Fft_inverse_real ~inverse:true ?s ?axes ?norm a

let two_d = [ -2; -1 ]
let fft2 ?s ?(axes = two_d) ?norm a = fftn ?s ~axes ?norm a
let ifft2 ?s ?(axes = two_d) ?norm a = ifftn ?s ~axes ?norm a
let rfft2 ?s ?(axes = two_d) ?norm a = rfftn ?s ~axes ?norm a
let irfft2 ?s ?(axes = two_d) ?norm a = irfftn ?s ~axes ?norm a
let default_real () = if Config.x64_enabled () then Dtype.F64 else Dtype.F32

let fftfreq ?(d = 1.0) ?dtype n =
  let dt = Option.value dtype ~default:(default_real ()) in
  let half = n / 2 in
  Concrete
    (Ndarray.of_floats dt [| n |]
       (Array.init n (fun i ->
            let k = ((i + half) mod n) - half in
            float_of_int k /. (d *. float_of_int n))))

let rfftfreq ?(d = 1.0) ?dtype n =
  let dt = Option.value dtype ~default:(default_real ()) in
  let count = (n / 2) + 1 in
  Concrete
    (Ndarray.of_floats dt [| count |]
       (Array.init count (fun i -> float_of_int i /. (d *. float_of_int n))))

let roll_axis v axis shift =
  let shape = shape_of v in
  let n = shape.(axis) in
  if n = 0 then v
  else
    let shift = ((shift mod n) + n) mod n in
    if shift = 0 then v
    else
      let slice lo hi =
        let start = Array.make (Array.length shape) 0 in
        let limit = Array.copy shape in
        start.(axis) <- lo;
        limit.(axis) <- hi;
        b1
          (Slice
             { start_indices = start; limit_indices = limit; strides = None })
          [ v ]
      in
      b1 (Concatenate axis) [ slice (n - shift) n; slice 0 (n - shift) ]

let shift_with amount ?axes x =
  let ndim = rank_of x in
  let axes =
    match axes with
    | Some a -> List.map (canon_axis ndim) a
    | None -> List.init ndim Fun.id
  in
  let shifts = List.map (fun ax -> amount (shape_of x).(ax)) axes in
  Lax_numpy.roll ~axis:(Array.of_list axes) x (Array.of_list shifts)

let fftshift ?axes x = shift_with (fun n -> n / 2) ?axes x
let ifftshift ?axes x = shift_with (fun n -> -(n / 2)) ?axes x
