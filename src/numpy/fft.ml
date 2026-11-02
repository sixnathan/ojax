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
      Py_exn.value_error
        (Printf.sprintf
           "Invalid norm value %s; should be \"backward\",\"ortho\" or \
            \"forward\"."
           other)

let apply_norm ~inverse sizes norm v =
  match norm with
  | None -> v
  | Some norm ->
      Ufuncs.multiply v (fft_norm ~inverse sizes (Core.get_aval v).dtype norm)

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

let moveaxis source destination v =
  let ndim = rank_of v in
  let source = List.map (fun i -> Validate.canonicalize_axis i ndim) source in
  let destination =
    List.map (fun i -> Validate.canonicalize_axis i ndim) destination
  in
  b1 (Transpose (moved_perm ndim source destination)) [ v ]

let python_list items =
  "[" ^ String.concat ", " (List.map string_of_int items) ^ "]"

let range lo hi = List.init (max 0 (hi - lo)) (fun i -> lo + i)

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
            | Fft_real ->
                (match (Core.get_aval v).dtype with
                | Dtype.Complex64 | Dtype.Complex128 ->
                    Py_exn.value_error
                      "only real valued inputs supported for rfft"
                | _ -> ());
                as_real v
            | Fft_forward | Fft_inverse | Fft_inverse_real -> as_complex v
          in
          if sizes = [] then prepared
          else
            b1
              (Fft { fft_type = kind; fft_lengths = Array.of_list sizes })
              [ prepared ]
      | _ -> assert false)
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
            (Pad (Lax_numpy.pad_widths_config (Array.of_list widths)))
            [ v; cval ]
      | _ -> assert false)
    [ v; Concrete (Ndarray.of_floats default_int [||] [| 0.0 |]) ]

let rec fft_core_nd arr kind sizes =
  let count = List.length sizes in
  if count <= 3 then lax_fft kind sizes arr
  else
    let n = if count = 4 then 2 else 3 in
    let ndim = rank_of arr in
    let src = range (ndim - count) (ndim - n) in
    let dst = range (ndim - count + n) ndim in
    let head = List.filteri (fun i _ -> i < count - n) sizes in
    let tail = List.filteri (fun i _ -> i >= count - n) sizes in
    match kind with
    | Fft_real | Fft_forward ->
        let arr = lax_fft kind tail arr in
        let arr = moveaxis src dst arr in
        let arr = fft_core_nd arr Fft_forward head in
        moveaxis dst src arr
    | Fft_inverse | Fft_inverse_real ->
        let arr = moveaxis src dst arr in
        let arr = fft_core_nd arr Fft_inverse head in
        let arr = moveaxis dst src arr in
        lax_fft kind tail arr

let fft_core name kind ~inverse ?s ?axes ?norm a =
  let ndim = rank_of a in
  (match s with
  | Some sizes when List.exists (fun x -> x <= 0) sizes ->
      Py_exn.value_error "Shape should be positive."
  | _ -> ());
  (match (s, axes) with
  | Some sizes, Some ax when List.length sizes <> List.length ax ->
      Py_exn.value_error "Shape and axes have different lengths."
  | _ -> ());
  let given_axes =
    match (axes, s) with
    | Some ax, _ -> ax
    | None, Some sizes -> range (ndim - List.length sizes) ndim
    | None, None -> range 0 ndim
  in
  if List.length (List.sort_uniq compare given_axes) <> List.length given_axes
  then
    Py_exn.value_error
      (Printf.sprintf
         "jax.numpy.fft.%s does not support repeated axes. Got axes %s." name
         (python_list given_axes));
  let inner_axes =
    match axes with
    | Some ax -> range (ndim - List.length ax) ndim
    | None -> given_axes
  in
  let arr = match axes with Some ax -> moveaxis ax inner_axes a | None -> a in
  let shape = shape_of arr in
  let arr, sizes =
    match s with
    | Some sizes ->
        if ndim = 0 && sizes = [] then
          Py_exn.value_error
            "jnp.pad: pad_width with nd=0 has unsupported shape (0,). Valid \
             shapes are (0, 2), (1, 2), (2,), (1,), or ().";
        let in_s = Array.copy shape in
        List.iter2
          (fun ax n -> in_s.(Validate.list_assign_index ndim ax) <- n)
          inner_axes sizes;
        (match kind with
        | Fft_inverse_real ->
            if ndim = 0 then Py_exn.index_error "list index out of range";
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
            let n = List.length inner_axes in
            ( arr,
              List.mapi
                (fun i ax ->
                  if i = n - 1 then max 0 (2 * (shape.(ax) - 1)) else shape.(ax))
                inner_axes )
        | Fft_forward | Fft_inverse | Fft_real ->
            (arr, List.map (fun ax -> shape.(ax)) inner_axes))
  in
  let transformed = fft_core_nd arr kind sizes in
  let scaled = apply_norm ~inverse sizes norm transformed in
  match axes with Some ax -> moveaxis inner_axes ax scaled | None -> scaled

let single_axis axis n = (Some [ axis ], Option.map (fun x -> [ x ]) n)

let fft ?n ?(axis = -1) ?norm a =
  let axes, s = single_axis axis n in
  fft_core "fft" Fft_forward ~inverse:false ?s ?axes ?norm a

let ifft ?n ?(axis = -1) ?norm a =
  let axes, s = single_axis axis n in
  fft_core "ifft" Fft_inverse ~inverse:true ?s ?axes ?norm a

let check_real_input a =
  match (Core.get_aval a).dtype with
  | Dtype.Complex64 | Dtype.Complex128 ->
      raise (Py_exn.Value_error "only real valued inputs supported for rfft")
  | _ -> ()

let rfft ?n ?(axis = -1) ?norm a =
  check_real_input a;
  let axes, s = single_axis axis n in
  fft_core "rfft" Fft_real ~inverse:false ?s ?axes ?norm a

let irfft ?n ?(axis = -1) ?norm a =
  let axes, s = single_axis axis n in
  fft_core "irfft" Fft_inverse_real ~inverse:true ?s ?axes ?norm a

let hfft ?n ?(axis = -1) ?norm a =
  let conj_a = b1 Conj [ as_complex a ] in
  let nn =
    match n with
    | Some x -> x
    | None -> (Validate.tuple_index (shape_of conj_a) axis * 2) - 2
  in
  let axes, s = single_axis axis n in
  Ufuncs.multiply
    (fft_core "hfft" Fft_inverse_real ~inverse:false ?s ?axes ?norm conj_a)
    (Jnp_util.py_int nn)

let ihfft ?n ?(axis = -1) ?norm a =
  check_real_input a;
  let nn =
    match n with Some x -> x | None -> Validate.tuple_index (shape_of a) axis
  in
  let axes, s = single_axis axis n in
  let output =
    b1 Conj [ fft_core "ihfft" Fft_real ~inverse:true ?s ?axes ?norm a ]
  in
  if nn = 0 then raise (Py_exn.Zero_division_error "division by zero");
  Ufuncs.multiply output (Jnp_util.py_float (1.0 /. float_of_int nn))

let fftn ?s ?axes ?norm a =
  fft_core "fftn" Fft_forward ~inverse:false ?s ?axes ?norm a

let ifftn ?s ?axes ?norm a =
  fft_core "ifftn" Fft_inverse ~inverse:true ?s ?axes ?norm a

let rfftn ?s ?axes ?norm a =
  fft_core "rfftn" Fft_real ~inverse:false ?s ?axes ?norm a

let irfftn ?s ?axes ?norm a =
  fft_core "irfftn" Fft_inverse_real ~inverse:true ?s ?axes ?norm a

let two_d = [ -2; -1 ]

let fft_core_2d name kind ~inverse ?s ~axes ?norm a =
  if List.length axes <> 2 then
    Py_exn.value_error
      (Printf.sprintf "jax.numpy.fft.%s only supports 2 axes. Got axes = %s."
         name (python_list axes));
  fft_core name kind ~inverse ?s ~axes ?norm a

let fft2 ?s ?(axes = two_d) ?norm a =
  fft_core_2d "fft2" Fft_forward ~inverse:false ?s ~axes ?norm a

let ifft2 ?s ?(axes = two_d) ?norm a =
  fft_core_2d "ifft2" Fft_inverse ~inverse:true ?s ~axes ?norm a

let rfft2 ?s ?(axes = two_d) ?norm a =
  fft_core_2d "rfft2" Fft_real ~inverse:false ?s ~axes ?norm a

let irfft2 ?s ?(axes = two_d) ?norm a =
  fft_core_2d "irfft2" Fft_inverse_real ~inverse:true ?s ~axes ?norm a

let default_real () = if Config.x64_enabled () then Dtype.F64 else Dtype.F32

let freq_dtypes dtype =
  let out = Option.value dtype ~default:(default_real ()) in
  let inexact = Dtypes.to_inexact_dtype (Dtypes.canonicalize_dtype out) in
  let real =
    match inexact with
    | Dtype.Complex64 -> Dtype.F32
    | Dtype.Complex128 -> Dtype.F64
    | d -> d
  in
  (Dtypes.canonicalize_dtype out, real)

let fftfreq ?(d = 1.0) ?dtype n =
  let out, dt = freq_dtypes dtype in
  let i = Lax_numpy.arange ~dtype:dt (float_of_int n) in
  let half = Jnp_util.py_int (n / 2) in
  let k =
    Ufuncs.subtract
      (Ufuncs.remainder (Ufuncs.add i half) (Jnp_util.py_int n))
      half
  in
  Lax_numpy.astype
    (Ufuncs.divide (Lax_numpy.astype k dt)
       (Concrete (Ndarray.of_floats dt [||] [| d *. float_of_int n |])))
    out

let scalar_of_dtype dt x =
  let dt = Dtypes.canonicalize_dtype dt in
  let x =
    if dt = Dtype.Bool then if x <> 0.0 then 1.0 else 0.0
    else if Dtypes.issubdtype dt `Integer then Float.trunc x
    else x
  in
  Concrete (Ndarray.of_floats dt [||] [| x |])

let rfftfreq ?(d = 1.0) ?dtype n =
  let dt = Option.value dtype ~default:(default_real ()) in
  let count = if n mod 2 = 0 then (n / 2) + 1 else ((n - 1) / 2) + 1 in
  Ufuncs.divide
    (Lax_numpy.arange ~dtype:dt (float_of_int count))
    (scalar_of_dtype dt (d *. float_of_int n))

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
  let shape = shape_of x in
  let ndim = Array.length shape in
  let shifts, axes =
    match axes with
    | Some a ->
        let shifts =
          List.map (fun ax -> amount (Validate.tuple_index shape ax)) a
        in
        (shifts, List.map (canon_axis ndim) a)
    | None ->
        let a = List.init ndim Fun.id in
        (List.map (fun ax -> amount shape.(ax)) a, a)
  in
  Lax_numpy.roll ~axis:(Array.of_list axes) x (Array.of_list shifts)

let fftshift ?axes x = shift_with (fun n -> n / 2) ?axes x
let ifftshift ?axes x = shift_with (fun n -> -(n / 2)) ?axes x
