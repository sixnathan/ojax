module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Numpy.Lax_numpy
module NF = Numpy.Fft
module UF = Numpy.Ufuncs
module R = Numpy.Reductions
module AC = Numpy.Array_creation

let pi = 4.0 *. atan 1.0
let aval v = C.get_aval v
let shape v = (aval v).T.shape
let dtype v = (aval v).T.dtype
let ndim v = Array.length (shape v)
let size v = Array.fold_left ( * ) 1 (shape v)
let is_complex = function D.Complex64 | D.Complex128 -> true | _ -> false

let complexify = function
  | D.F64 | D.Complex128 -> D.Complex128
  | _ -> D.Complex64

let complex_basetype = function
  | D.Complex128 -> D.F64
  | D.Complex64 -> D.F32
  | d -> d

let to_inexact = function
  | (D.F32 | D.F64 | D.Complex64 | D.Complex128) as d -> d
  | D.I64 -> D.F64
  | _ -> D.F32

let scalar dt x = T.Concrete (Nd.of_floats dt [||] [| x |])

let canonicalize_axis axis nd =
  let axis = if axis < 0 then axis + nd else axis in
  if axis < 0 || axis >= nd then
    invalid_arg
      (Printf.sprintf "axis %d out of bounds for array of ndim %d" axis nd);
  axis

let canonicalize_axes axes nd =
  let seen = Hashtbl.create (Array.length axes) in
  Array.map
    (fun axis ->
      let axis = canonicalize_axis axis nd in
      if Hashtbl.mem seen axis then invalid_arg "axes must be unique";
      Hashtbl.add seen axis ();
      axis)
    axes

let slice_axis x axis start limit stride =
  let sh = shape x in
  let nd = Array.length sh in
  let starts = Array.make nd 0 in
  let limits = Array.copy sh in
  let strides = Array.make nd 1 in
  starts.(axis) <- start;
  limits.(axis) <- limit;
  strides.(axis) <- stride;
  C.bind1
    (T.Slice
       {
         start_indices = starts;
         limit_indices = limits;
         strides = Some strides;
       })
    [ x ]

let broadcast_shapes a b =
  let na = Array.length a and nb = Array.length b in
  let n = max na nb in
  Array.init n (fun i ->
      let da = if i < n - na then 1 else a.(i - (n - na)) in
      let db = if i < n - nb then 1 else b.(i - (n - nb)) in
      if da = db then da
      else if da = 1 then db
      else if db = 1 then da
      else invalid_arg "x and y cannot be broadcast together")

let promote_inexact2 x y =
  let dt, _ =
    Dtypes.result_type
      [ (dtype x, (aval x).T.weak_type); (dtype y, (aval y).T.weak_type) ]
  in
  let dt = to_inexact dt in
  (NL.astype x dt, NL.astype y dt)

let fftconvolve ?(mode = "full") ?axes in1 in2 =
  if ndim in1 <> ndim in2 then
    invalid_arg "in1 and in2 should have the same dimensionality";
  if mode <> "same" && mode <> "full" && mode <> "valid" then
    invalid_arg "mode must be one of ['same', 'full', 'valid']";
  let in1, in2 = promote_inexact2 in1 in2 in
  let nd = ndim in1 in
  let axes =
    match axes with
    | None -> Array.init nd Fun.id
    | Some axes -> canonicalize_axes axes nd
  in
  let transformed = Array.make nd false in
  Array.iter (fun axis -> transformed.(axis) <- true) axes;
  for axis = 0 to nd - 1 do
    if (not transformed.(axis)) && (shape in1).(axis) <> (shape in2).(axis) then
      invalid_arg "mapped axes must have the same shape"
  done;
  let no_swap =
    Array.for_all (fun axis -> (shape in1).(axis) >= (shape in2).(axis)) axes
  in
  let swap =
    Array.for_all (fun axis -> (shape in1).(axis) <= (shape in2).(axis)) axes
  in
  if mode = "valid" && not (no_swap || swap) then
    invalid_arg
      "For 'valid' mode, one input must be at least as large as the other in \
       every dimension.";
  let in1, in2 = if mode = "valid" && swap then (in2, in1) else (in1, in2) in
  let full_shape = Array.copy (shape in1) in
  Array.iter
    (fun axis ->
      full_shape.(axis) <- (shape in1).(axis) + (shape in2).(axis) - 1)
    axes;
  let fft_shape = Array.map (fun axis -> full_shape.(axis)) axes in
  let singleton =
    Array.for_all
      (fun axis -> (shape in1).(axis) = 1 || (shape in2).(axis) = 1)
      axes
  in
  let conv =
    if singleton then UF.multiply in1 in2
    else if is_complex (dtype in1) then
      let sp1 = NF.fftn ~s:fft_shape ~axes in1 in
      let sp2 = NF.fftn ~s:fft_shape ~axes in2 in
      NF.ifftn ~s:fft_shape ~axes (UF.multiply sp1 sp2)
    else
      let sp1 = NF.rfftn ~s:fft_shape ~axes in1 in
      let sp2 = NF.rfftn ~s:fft_shape ~axes in2 in
      NF.irfftn ~s:fft_shape ~axes (UF.multiply sp1 sp2)
  in
  let out_shape = Array.copy full_shape in
  (match mode with
  | "full" -> ()
  | "same" ->
      Array.iter (fun axis -> out_shape.(axis) <- (shape in1).(axis)) axes
  | "valid" ->
      Array.iter
        (fun axis ->
          out_shape.(axis) <- (shape in1).(axis) - (shape in2).(axis) + 1)
        axes
  | _ -> assert false);
  let starts = Array.make nd 0 in
  Array.iter
    (fun axis -> starts.(axis) <- (full_shape.(axis) - out_shape.(axis)) / 2)
    axes;
  let limits = Array.mapi (fun i start -> start + out_shape.(i)) starts in
  C.bind1
    (T.Slice
       {
         start_indices = starts;
         limit_indices = limits;
         strides = Some (Array.make nd 1);
       })
    [ conv ]

let hann dtype n =
  if n = 1 then AC.ones ~dtype [| 1 |]
  else
    AC.linspace ~num:n ~endpoint:false ~dtype:(complex_basetype dtype) 0.0 pi
    |> UF.sin |> UF.square
    |> fun win -> NL.astype win dtype

let make_window dtype name n =
  match name with
  | "hann" -> hann dtype n
  | other -> invalid_arg ("unsupported window: " ^ other)

let extend_boundary x n boundary =
  if n = 0 then x
  else
    let axis = ndim x - 1 in
    let len = (shape x).(axis) in
    match boundary with
    | "zeros" ->
        let pads = Array.make (ndim x) (0, 0) in
        pads.(axis) <- (n, n);
        NL.pad x pads 0.0
    | "constant" ->
        let left = slice_axis x axis 0 1 1 |> fun v -> NL.repeat ~axis v n in
        let right =
          slice_axis x axis (len - 1) len 1 |> fun v -> NL.repeat ~axis v n
        in
        NL.concatenate ~axis [ left; x; right ]
    | "even" | "odd" ->
        if n > len - 1 then invalid_arg "extension length is too large";
        let left = slice_axis x axis 1 (n + 1) 1 |> NL.flip ~axis:[| axis |] in
        let right =
          slice_axis x axis (len - n - 1) (len - 1) 1
          |> NL.flip ~axis:[| axis |]
        in
        if boundary = "even" then NL.concatenate ~axis [ left; x; right ]
        else
          let two = scalar (dtype x) 2.0 in
          let left_end = slice_axis x axis 0 1 1 in
          let right_end = slice_axis x axis (len - 1) len 1 in
          NL.concatenate ~axis
            [
              UF.subtract (UF.multiply two left_end) left;
              x;
              UF.subtract (UF.multiply two right_end) right;
            ]
    | other -> invalid_arg ("unknown boundary option: " ^ other)

let segment x nperseg noverlap =
  let axis = ndim x - 1 in
  let len = (shape x).(axis) in
  let step = nperseg - noverlap in
  let frames = 1 + ((len - nperseg) / step) in
  List.init frames (fun i ->
      slice_axis x axis (i * step) ((i * step) + nperseg) 1)
  |> NL.stack ~axis

let detrend_segments kind x =
  let axis = ndim x - 1 in
  match kind with
  | "none" -> x
  | "constant" -> UF.subtract x (R.mean ~axis:[| axis |] ~keepdims:true x)
  | "linear" ->
      let n = (shape x).(axis) in
      let dt = complex_basetype (dtype x) in
      let t =
        T.Concrete
          (Nd.of_floats dt [| n |]
             (Array.init n (fun i -> float_of_int (i + 1) /. float_of_int n)))
      in
      let mt = R.mean t in
      let tc = UF.subtract t mt in
      let mx = R.mean ~axis:[| axis |] ~keepdims:true x in
      let slope =
        UF.divide
          (R.sum ~axis:[| axis |] ~keepdims:true (UF.multiply x tc))
          (R.sum (UF.square tc))
      in
      UF.subtract x (UF.add mx (UF.multiply slope tc))
  | other -> invalid_arg ("unsupported detrend type: " ^ other)

let fft_helper x win detrend_type nperseg noverlap nfft sides =
  let result = segment x nperseg noverlap |> detrend_segments detrend_type in
  let result = UF.multiply result win in
  if sides = "twosided" then NF.fft ~n:nfft result
  else NF.rfft ~n:nfft (UF.real result)

let pad_shorter x target =
  let axis = ndim x - 1 in
  let current = (shape x).(axis) in
  if current = target then x
  else
    let pads = Array.make (ndim x) (0, 0) in
    pads.(axis) <- (0, target - current);
    NL.pad x pads 0.0

let spectral_helper x y fs window_name nperseg noverlap nfft detrend_type
    return_onesided scaling axis mode boundary padded =
  if mode <> "psd" && mode <> "stft" then invalid_arg "unknown spectral mode";
  let axis = canonicalize_axis axis (ndim x) in
  let x = NL.astype x (to_inexact (dtype x)) in
  let y =
    match y with
    | None -> None
    | Some y ->
        if mode <> "psd" then
          invalid_arg "two-argument mode is available only for psd";
        if ndim x <> ndim y then
          invalid_arg "two arguments must have the same rank";
        let x2, y2 = promote_inexact2 x y in
        Some (x2, y2)
  in
  let x, y = match y with None -> (x, None) | Some (x, y) -> (x, Some y) in
  let x = NL.moveaxis [| axis |] [| ndim x - 1 |] x in
  let y =
    match y with
    | None -> None
    | Some y -> Some (NL.moveaxis [| axis |] [| ndim y - 1 |] y)
  in
  let x, y =
    match y with
    | None -> (x, None)
    | Some y ->
        let outer =
          broadcast_shapes
            (Array.sub (shape x) 0 (ndim x - 1))
            (Array.sub (shape y) 0 (ndim y - 1))
        in
        let lx = (shape x).(ndim x - 1) and ly = (shape y).(ndim y - 1) in
        let target = max lx ly in
        let x = pad_shorter x target and y = pad_shorter y target in
        let xshape = Array.append outer [| target |] in
        (NL.broadcast_to x xshape, Some (NL.broadcast_to y xshape))
  in
  let signal_length = (shape x).(ndim x - 1) in
  let nperseg =
    match nperseg with None -> signal_length | Some n -> min n signal_length
  in
  if nperseg < 1 then invalid_arg "nperseg must be a positive integer";
  let noverlap = match noverlap with None -> nperseg / 2 | Some n -> n in
  let nfft = match nfft with None -> nperseg | Some n -> n in
  if nfft < nperseg then
    invalid_arg "nfft must be greater than or equal to nperseg";
  if noverlap >= nperseg then invalid_arg "noverlap must be less than nperseg";
  let step = nperseg - noverlap in
  let win = make_window (dtype x) window_name nperseg in
  let x, y =
    match boundary with
    | None -> (x, y)
    | Some boundary ->
        ( extend_boundary x (nperseg / 2) boundary,
          Option.map (fun v -> extend_boundary v (nperseg / 2) boundary) y )
  in
  let x, y =
    if not padded then (x, y)
    else
      let len = (shape x).(ndim x - 1) in
      let nadd = ((-(len - nperseg) mod step) + step) mod step in
      ( pad_shorter x (len + nadd),
        Option.map (fun v -> pad_shorter v (len + nadd)) y )
  in
  let sides =
    if
      return_onesided
      && (not (is_complex (dtype x)))
      && Option.fold ~none:true ~some:(fun v -> not (is_complex (dtype v))) y
    then "onesided"
    else "twosided"
  in
  let freq_dtype = complex_basetype (complexify (dtype x)) in
  let freqs =
    if sides = "onesided" then NF.rfftfreq ~d:(1.0 /. fs) ~dtype:freq_dtype nfft
    else NF.fftfreq ~d:(1.0 /. fs) ~dtype:freq_dtype nfft
  in
  let result = fft_helper x win detrend_type nperseg noverlap nfft sides in
  let result =
    match y with
    | Some y ->
        UF.multiply (UF.conjugate result)
          (fft_helper y win detrend_type nperseg noverlap nfft sides)
    | None when mode = "psd" -> UF.multiply (UF.conjugate result) result
    | None -> result
  in
  let win_sum = R.sum win in
  let scale =
    if scaling = "density" then
      let denom = UF.multiply (scalar (dtype win) fs) (R.sum (UF.square win)) in
      UF.divide (scalar (dtype win) 1.0) denom
    else if scaling = "spectrum" then
      UF.divide (scalar (dtype win) 1.0) (UF.square win_sum)
    else invalid_arg ("unknown scaling: " ^ scaling)
  in
  let scale = if mode = "stft" then UF.sqrt scale else scale in
  let result = UF.multiply result scale in
  let result =
    if sides = "onesided" && mode = "psd" then
      let nf = (shape result).(ndim result - 1) in
      let factors =
        T.Concrete
          (Nd.of_floats freq_dtype [| nf |]
             (Array.init nf (fun i ->
                  if i = 0 || (nfft land 1 = 0 && i = nf - 1) then 1.0 else 2.0)))
      in
      UF.multiply result factors
    else result
  in
  let result = NL.astype result (complexify (dtype x)) in
  let result =
    match (y, mode) with None, "psd" -> UF.real result | _ -> result
  in
  let frames = (shape result).(ndim result - 2) in
  let times =
    T.Concrete
      (Nd.of_floats freq_dtype [| frames |]
         (Array.init frames (fun i ->
              let t =
                (float_of_int nperseg /. 2.0) +. float_of_int (i * step)
              in
              let t = t /. fs in
              match boundary with
              | None -> t
              | Some _ -> t -. (float_of_int nperseg /. 2.0 /. fs))))
  in
  let result = NL.moveaxis [| ndim result - 1 |] [| axis |] result in
  (freqs, times, result)

let stft ?(fs = 1.0) ?(window = "hann") ?(nperseg = 256) ?noverlap ?nfft
    ?(detrend = "none") ?(return_onesided = true) ?(boundary = Some "zeros")
    ?(padded = true) ?(axis = -1) x =
  spectral_helper x None fs window (Some nperseg) noverlap nfft detrend
    return_onesided "spectrum" axis "stft" boundary padded

let median_bias dtype n =
  let bias = ref 1.0 in
  let i = ref 2 in
  while !i < n do
    bias := !bias +. (1.0 /. float_of_int (!i + 1)) -. (1.0 /. float_of_int !i);
    i := !i + 2
  done;
  scalar dtype !bias

let csd ?(fs = 1.0) ?(window = "hann") ?nperseg ?noverlap ?nfft
    ?(detrend = "constant") ?(return_onesided = true) ?(scaling = "density")
    ?(axis = -1) ?(average = "mean") x y =
  let freqs, _, pxy =
    spectral_helper x y fs window nperseg noverlap nfft detrend return_onesided
      scaling axis "psd" None false
  in
  let pxy =
    let frames = (shape pxy).(ndim pxy - 1) in
    if ndim pxy < 2 || size pxy = 0 then pxy
    else if frames = 1 then
      let sh = Array.sub (shape pxy) 0 (ndim pxy - 1) in
      NL.reshape pxy sh
    else if average = "mean" then R.mean ~axis:[| ndim pxy - 1 |] pxy
    else if average = "median" then
      let axis = ndim pxy - 1 in
      let median =
        if is_complex (dtype pxy) then
          C.bind1 T.Complex
            [ R.median ~axis (UF.real pxy); R.median ~axis (UF.imag pxy) ]
        else R.median ~axis pxy
      in
      UF.divide median (median_bias (dtype median) frames)
    else
      invalid_arg
        (Printf.sprintf "average must be mean or median, got %s" average)
  in
  (freqs, pxy)

let welch ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided
    ?scaling ?axis ?average x =
  let freqs, pxx =
    csd ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided ?scaling
      ?axis ?average x None
  in
  (freqs, UF.real pxx)

let transpose_last_two x =
  let nd = ndim x in
  let axes = Array.init nd Fun.id in
  axes.(nd - 2) <- nd - 1;
  axes.(nd - 1) <- nd - 2;
  NL.transpose ~axes x

let overlap_and_add x step =
  if ndim x < 2 then invalid_arg "input must have frames and frame_length axes";
  let sh = shape x in
  let frame_axis = ndim x - 2 in
  let frames = sh.(frame_axis) in
  let segment_len = sh.(ndim x - 1) in
  let output_len = (step * (frames - 1)) + segment_len in
  let out_shape = Array.sub sh 0 (ndim x - 1) in
  out_shape.(Array.length out_shape - 1) <- output_len;
  let acc = ref (AC.zeros ~dtype:(dtype x) out_shape) in
  for frame = 0 to frames - 1 do
    let part =
      slice_axis x frame_axis frame (frame + 1) 1
      |> NL.squeeze ~axis:[| frame_axis |]
    in
    let pads = Array.make (ndim part) (0, 0) in
    let left = frame * step in
    pads.(ndim part - 1) <- (left, output_len - left - segment_len);
    acc := UF.add !acc (NL.pad part pads 0.0)
  done;
  !acc

let istft ?(fs = 1.0) ?(window = "hann") ?nperseg ?noverlap ?nfft
    ?(input_onesided = true) ?(boundary = true) ?(time_axis = -1)
    ?(freq_axis = -2) zxx =
  if ndim zxx < 2 then invalid_arg "input stft must be at least 2d";
  let original_nd = ndim zxx in
  let freq_axis = canonicalize_axis freq_axis original_nd in
  let time_axis = canonicalize_axis time_axis original_nd in
  if freq_axis = time_axis then
    invalid_arg "time and frequency axes must differ";
  let zxx = NL.astype zxx (complexify (dtype zxx)) in
  let n_default =
    if input_onesided then 2 * ((shape zxx).(freq_axis) - 1)
    else (shape zxx).(freq_axis)
  in
  let nperseg = match nperseg with None -> n_default | Some n -> n in
  if nperseg < 1 then invalid_arg "nperseg must be a positive integer";
  let nfft =
    match nfft with
    | Some n -> n
    | None ->
        if input_onesided && nperseg = n_default + 1 then n_default + 1
        else n_default
  in
  if nfft < nperseg then invalid_arg "fft length must be longer than nperseg";
  let noverlap = match noverlap with None -> nperseg / 2 | Some n -> n in
  if noverlap >= nperseg then invalid_arg "noverlap must be less than nperseg";
  let step = nperseg - noverlap in
  let zxx =
    if time_axis = original_nd - 1 && freq_axis = original_nd - 2 then zxx
    else
      let outer =
        Array.to_list (Array.init original_nd Fun.id)
        |> List.filter (fun i -> i <> time_axis && i <> freq_axis)
        |> Array.of_list
      in
      NL.transpose ~axes:(Array.append outer [| freq_axis; time_axis |]) zxx
  in
  let xsubs =
    if input_onesided then NF.irfft ~n:nfft ~axis:(ndim zxx - 2) zxx
    else NF.ifft ~n:nfft ~axis:(ndim zxx - 2) zxx
  in
  let xsubs = slice_axis xsubs (ndim xsubs - 2) 0 nperseg 1 in
  let win = make_window (complex_basetype (dtype xsubs)) window nperseg in
  let xsubs = UF.multiply xsubs (R.sum win) in
  let wshape = Array.make (ndim xsubs) 1 in
  wshape.(ndim xsubs - 2) <- nperseg;
  let win_b = NL.reshape win wshape in
  let xframes = transpose_last_two (UF.multiply xsubs win_b) in
  let x = overlap_and_add xframes step in
  let norm_frames =
    UF.square win_b |> fun v ->
    NL.broadcast_to v (shape xsubs) |> transpose_last_two
  in
  let norm = overlap_and_add norm_frames step |> UF.real in
  let x, norm =
    if not boundary then (x, norm)
    else
      let cut = nperseg / 2 in
      let last = ndim x - 1 in
      ( slice_axis x last cut ((shape x).(last) - cut) 1,
        slice_axis norm last cut ((shape norm).(last) - cut) 1 )
  in
  let denom =
    NL.where_
      (UF.greater norm (scalar (dtype norm) 1e-10))
      norm
      (scalar (dtype norm) 1.0)
  in
  let x = UF.divide x denom in
  let x =
    if ndim x <= 1 || time_axis = original_nd - 1 then x
    else
      let target = if freq_axis < time_axis then time_axis - 1 else time_axis in
      NL.moveaxis [| ndim x - 1 |] [| target |] x
  in
  let time_len = (shape x).(0) in
  let tdt = complex_basetype (dtype x) in
  let time =
    T.Concrete
      (Nd.of_floats tdt [| time_len |]
         (Array.init time_len (fun i -> float_of_int i /. fs)))
  in
  (time, x)
