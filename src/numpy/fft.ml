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

let resize_axis v axis n =
  let shape = shape_of v in
  let cur = shape.(axis) in
  if cur = n then v
  else if cur > n then begin
    let limit = Array.copy shape in
    limit.(axis) <- n;
    b1
      (Slice
         {
           start_indices = Array.make (Array.length shape) 0;
           limit_indices = limit;
           strides = None;
         })
      [ v ]
  end
  else
    let cfg = Array.make (Array.length shape) (0, 0, 0) in
    cfg.(axis) <- (0, n - cur, 0);
    let dt = (Core.get_aval v).dtype in
    let zero =
      match dt with
      | Dtype.Complex64 | Dtype.Complex128 ->
          Concrete (Ndarray.of_complex dt [||] [| Complex.zero |])
      | _ -> Concrete (Ndarray.of_floats dt [||] [| 0.0 |])
    in
    b1 (Pad cfg) [ v; zero ]

let move_axes_last v axes =
  let ndim = rank_of v in
  let others =
    List.filter (fun d -> not (List.mem d axes)) (List.init ndim Fun.id)
  in
  let perm = Array.of_list (others @ axes) in
  let inverse = Array.make ndim 0 in
  Array.iteri (fun i p -> inverse.(p) <- i) perm;
  (b1 (Transpose perm) [ v ], inverse)

let norm_factor norm ~inverse lengths =
  let product = float_of_int (List.fold_left ( * ) 1 lengths) in
  match norm with
  | None | Some "backward" -> 1.0
  | Some "ortho" -> if inverse then sqrt product else 1.0 /. sqrt product
  | Some "forward" -> if inverse then product else 1.0 /. product
  | Some other ->
      invalid_arg
        (Printf.sprintf
           "Invalid norm value %s; should be \"backward\",\"ortho\" or \
            \"forward\"."
           other)

let scale_by factor v =
  if factor = 1.0 then v
  else
    let aval = Core.get_aval v in
    let c = Core.splat aval.dtype aval.shape factor in
    b1 Mul [ v; c ]

let fft_core kind ~inverse ?s ?axes ?norm a =
  let ndim = rank_of a in
  let axes =
    match (axes, s) with
    | Some ax, _ -> List.map (canon_axis ndim) ax
    | None, Some sizes ->
        let k = List.length sizes in
        List.init k (fun i -> ndim - k + i)
    | None, None -> List.init ndim Fun.id
  in
  if List.length (List.sort_uniq compare axes) <> List.length axes then
    invalid_arg "numpy.fft: repeated axis";
  let shape = shape_of a in
  let sizes =
    match s with
    | Some sizes ->
        if List.length sizes <> List.length axes then
          invalid_arg "Shape and axes have different lengths.";
        sizes
    | None -> (
        match kind with
        | Fft_inverse_real ->
            let n = List.length axes in
            List.mapi
              (fun i ax ->
                if i = n - 1 then 2 * (shape.(ax) - 1) else shape.(ax))
              axes
        | Fft_forward | Fft_inverse | Fft_real ->
            List.map (fun ax -> shape.(ax)) axes)
  in
  let n_axes = List.length axes in
  let resized =
    List.fold_left2
      (fun v ax n ->
        let is_last =
          match List.rev axes with last :: _ -> last = ax | [] -> false
        in
        match kind with
        | Fft_inverse_real when is_last -> resize_axis v ax ((n / 2) + 1)
        | _ -> resize_axis v ax n)
      a axes sizes
  in
  let prepared =
    match kind with
    | Fft_real -> as_real resized
    | Fft_forward | Fft_inverse | Fft_inverse_real -> as_complex resized
  in
  let moved, inverse_perm = move_axes_last prepared axes in
  ignore n_axes;
  let out =
    b1 (Fft { fft_type = kind; fft_lengths = Array.of_list sizes }) [ moved ]
  in
  let back = b1 (Transpose inverse_perm) [ out ] in
  scale_by (norm_factor norm ~inverse sizes) back

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
  scale_by (float_of_int nn)
    (fft_core Fft_inverse_real ~inverse:false ?s ?axes ?norm conj_a)

let ihfft ?n ?(axis = -1) ?norm a =
  let ax = canon_axis (rank_of a) axis in
  let nn = match n with Some x -> x | None -> (shape_of a).(ax) in
  let axes, s = single_axis axis n in
  scale_by
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
  List.fold_left (fun v ax -> roll_axis v ax (amount (shape_of x).(ax))) x axes

let fftshift ?axes x = shift_with (fun n -> n / 2) ?axes x
let ifftshift ?axes x = shift_with (fun n -> -(n / 2)) ?axes x
