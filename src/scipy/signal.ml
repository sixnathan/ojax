open Types
module L = Lax_api
module NL = Numpy.Lax_numpy
module NF = Numpy.Fft
module R = Numpy.Reductions

let b1 = Core.bind1
let shape v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype
let rank v = Array.length (shape v)

let inexact x =
  match dtype_of x with
  | Dtype.F32 | Dtype.F64 | Dtype.Complex64 | Dtype.Complex128 -> x
  | Dtype.I64 -> b1 (Convert_element_type Dtype.F64) [ x ]
  | _ -> b1 (Convert_element_type Dtype.F32) [ x ]

let promote a b =
  let a = inexact a and b = inexact b in
  let dt = NL.result_type [ a; b ] in
  (NL.astype a dt, NL.astype b dt)

let all_pairs test a b = Array.for_all (fun x -> x) (Array.map2 test a b)
let flip_all v = L.rev v (Array.init (rank v) Fun.id)
let conj_of v = NL.astype (b1 Conj [ v ]) (dtype_of v)

let conj_if_complex v =
  match dtype_of v with
  | Dtype.Complex64 | Dtype.Complex128 -> conj_of v
  | _ -> v

let convolve_nd in1 in2 mode =
  if rank in1 <> rank in2 then
    invalid_arg "in1 and in2 must have the same number of dimensions";
  let in1, in2 = promote in1 in2 in
  let s1 = shape in1 and s2 = shape in2 in
  let no_swap = all_pairs ( >= ) s1 s2 and swap = all_pairs ( <= ) s1 s2 in
  if not (no_swap || swap) then
    invalid_arg "One input must be smaller than the other in every dimension.";
  let shape_o = s2 in
  let big, small = if swap then (in2, in1) else (in1, in2) in
  let small_shape = shape small in
  let kernel = flip_all small in
  let padding =
    match mode with
    | "valid" -> Array.map (fun _ -> (0, 0)) small_shape
    | "same" ->
        Array.mapi
          (fun i s ->
            let so = shape_o.(i) in
            (s - 1 - ((so - 1) / 2), s - so + ((so - 1) / 2)))
          small_shape
    | "full" -> Array.map (fun s -> (s - 1, s - 1)) small_shape
    | other -> invalid_arg ("mode must be one of full, same, valid: " ^ other)
  in
  let lift v = L.expand_dims v [| 0; 1 |] in
  let out =
    L.conv_general_dilated (lift big) (lift kernel)
      ~window_strides:(Array.map (fun _ -> 1) small_shape)
      (L.Explicit padding)
  in
  L.squeeze out [| 0; 1 |]

let fftconvolve_unbatched in1 in2 mode =
  let s1 = shape in1 and s2 = shape in2 in
  let full_shape = Array.map2 (fun a b -> a + b - 1) s1 s2 in
  let in1, in2 =
    if mode = "valid" then begin
      let no_swap = all_pairs ( >= ) s1 s2 and swap = all_pairs ( <= ) s1 s2 in
      if not (no_swap || swap) then
        invalid_arg
          "For 'valid' mode, One input must be at least as large as the other \
           in every dimension";
      if swap then (in2, in1) else (in1, in2)
    end
    else (in1, in2)
  in
  let s1 = shape in1 and s2 = shape in2 in
  let conv =
    if all_pairs (fun a b -> a = 1 || b = 1) s1 s2 then
      NL.broadcast_to in1 full_shape |> fun big ->
      b1 Mul [ big; NL.broadcast_to in2 full_shape ]
    else
      let sizes = Array.to_list full_shape in
      let is_complex =
        match dtype_of in1 with
        | Dtype.Complex64 | Dtype.Complex128 -> true
        | _ -> false
      in
      if is_complex then
        NF.ifftn ~s:sizes
          (b1 Mul [ NF.fftn ~s:sizes in1; NF.fftn ~s:sizes in2 ])
      else
        NF.irfftn ~s:sizes
          (b1 Mul [ NF.rfftn ~s:sizes in1; NF.rfftn ~s:sizes in2 ])
  in
  let out_shape =
    match mode with
    | "full" -> full_shape
    | "same" -> s1
    | "valid" -> Array.map2 (fun a b -> a - b + 1) s1 s2
    | other -> invalid_arg ("Unrecognized mode=" ^ other)
  in
  let start = Array.map2 (fun f o -> (f - o) / 2) full_shape out_shape in
  let index i =
    Concrete
      (Ndarray.of_floats Dtype.I32 [||] [| float_of_int i |])
  in
  b1
    (Dynamic_slice { slice_sizes = out_shape })
    (conv :: List.map index (Array.to_list start))

let fftconvolve ?(mode = "full") ?axes in1 in2 =
  if rank in1 <> rank in2 then
    invalid_arg "in1 and in2 should have the same dimensionality";
  if not (List.mem mode [ "same"; "full"; "valid" ]) then
    invalid_arg "mode must be one of ['same', 'full', 'valid']";
  let in1, in2 = promote in1 in2 in
  match axes with
  | None -> fftconvolve_unbatched in1 in2 mode
  | Some axes ->
      let n = rank in1 in
      let axes = List.map (fun a -> if a < 0 then a + n else a) axes in
      let mapped =
        List.filter (fun i -> not (List.mem i axes)) (List.init n Fun.id)
      in
      List.iter
        (fun i ->
          if (shape in1).(i) <> (shape in2).(i) then
            invalid_arg "mapped axes must have same shape")
        mapped;
      let rec apply mapped in1 in2 =
        match mapped with
        | [] -> fftconvolve_unbatched in1 in2 mode
        | ax :: rest ->
            let later = List.map (fun i -> if i > ax then i - 1 else i) rest in
            let results =
              Interpreters.Batching.vmap
                (fun args ->
                  match args with
                  | [ a; b ] -> [ apply later a b ]
                  | _ -> failwith "fftconvolve: arity")
                [ Some ax; Some ax ] [ in1; in2 ]
            in
            NL.moveaxis [| 0 |] [| ax |] (List.hd results)
      in
      apply mapped in1 in2

let convolve ?(mode = "full") ?(method_ = "auto") in1 in2 =
  match method_ with
  | "fft" -> fftconvolve ~mode in1 in2
  | "direct" | "auto" -> convolve_nd in1 in2 mode
  | other ->
      invalid_arg ("Got method=" ^ other ^ "; expected auto, fft, or direct.")

let check_boundary name boundary fillvalue =
  if boundary <> "fill" || fillvalue <> 0.0 then
    failwith (name ^ "() only supports boundary='fill', fillvalue=0")

let convolve2d ?(mode = "full") ?(boundary = "fill") ?(fillvalue = 0.0) in1 in2
    =
  check_boundary "convolve2d" boundary fillvalue;
  if rank in1 <> 2 || rank in2 <> 2 then
    invalid_arg "convolve2d() only supports 2-dimensional inputs.";
  convolve_nd in1 in2 mode

let correlate ?(mode = "full") ?method_ in1 in2 =
  convolve ~mode ?method_ in1 (flip_all (conj_if_complex in2))

let correlate2d ?(mode = "full") ?(boundary = "fill") ?(fillvalue = 0.0) in1 in2
    =
  check_boundary "correlate2d" boundary fillvalue;
  if rank in1 <> 2 || rank in2 <> 2 then
    invalid_arg "correlate2d() only supports 2-dimensional inputs.";
  let s1 = shape in1 and s2 = shape in2 in
  let swap = all_pairs ( <= ) s1 s2 and same_shape = s1 = s2 in
  let flipped_first () =
    flip_all (convolve_nd (flip_all in1) (conj_if_complex in2) mode)
  in
  match mode with
  | "same" -> flipped_first ()
  | "valid" ->
      if swap && not same_shape then
        convolve_nd (flip_all in2) (conj_if_complex in1) mode
      else flipped_first ()
  | _ ->
      if swap then
        conj_if_complex (convolve_nd (flip_all in2) (conj_if_complex in1) mode)
      else flipped_first ()

let detrend ?(axis = -1) ?(type_ = "linear") ?(bp = []) data =
  let data = inexact data in
  match type_ with
  | "constant" ->
      b1 Sub
        [
          data;
          NL.broadcast_to
            (R.mean ~axis:[| axis |] ~keepdims:true data)
            (shape data);
        ]
  | "linear" ->
      let moved = NL.moveaxis [| axis |] [| 0 |] data in
      let moved_shape = shape moved in
      let n = moved_shape.(0) in
      let rows = NL.reshape moved [| n; -1 |] in
      let dt = dtype_of data in
      let bounds = List.sort_uniq compare ((0 :: bp) @ [ n ]) in
      if List.hd bounds < 0 || List.nth bounds (List.length bounds - 1) > n then
        invalid_arg
          "Breakpoints must be non-negative and less than length of data along \
           given axis.";
      let rec segments = function
        | lo :: (hi :: _ as rest) -> (lo, hi) :: segments rest
        | _ -> []
      in
      let cleaned =
        List.map
          (fun (lo, hi) ->
            let npts = hi - lo in
            let design =
              Concrete
                (Ndarray.of_floats dt [| npts; 2 |]
                   (Array.init (2 * npts) (fun i ->
                        if i mod 2 = 0 then 1.0
                        else float_of_int ((i / 2) + 1) /. float_of_int npts)))
            in
            let part = L.slice_in_dim rows lo hi in
            let coef, _, _, _ = Numpy.Linalg.lstsq design part in
            b1 Sub [ part; L.dot design coef ])
          (segments bounds)
      in
      let out = NL.reshape (NL.concatenate ~axis:0 cleaned) moved_shape in
      NL.moveaxis [| 0 |] [| axis |] out
  | _ -> invalid_arg "Trend type must be 'linear' or 'constant'."

let pi = 4.0 *. Float.atan 1.0

let hann dt n =
  if n = 1 then L.full [| 1 |] 1.0 dt
  else
    let wide = Dtypes.default_float_dtype () in
    let points =
      Numpy.Array_creation.linspace_v ~num:n ~endpoint:false
        (L.full [||] 0.0 wide) (L.full [||] pi wide)
    in
    let sine = Numpy.Ufuncs.sin points in
    NL.astype (b1 (Integer_pow 2) [ sine ]) dt

let real_part_dtype = function
  | Dtype.Complex128 | Dtype.F64 -> Dtype.F64
  | _ -> Dtype.F32

let complex_dtype_of = function
  | Dtype.F64 | Dtype.Complex128 -> Dtype.Complex128
  | _ -> Dtype.Complex64

let is_complex v =
  match dtype_of v with
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let pad_edges mode x n =
  let last = rank x - 1 in
  let size = (shape x).(last) in
  let zero () = L.full [||] 0.0 (dtype_of x) in
  let cfg = Array.make (rank x) (0, 0, 0) in
  cfg.(last) <- (n, n, 0);
  match mode with
  | "zeros" -> L.pad x (zero ()) cfg
  | "constant" ->
      let first = L.slice_in_dim ~axis:last x 0 1 in
      let final = L.slice_in_dim ~axis:last x (size - 1) size in
      let repeat v = L.concatenate (List.init n (fun _ -> v)) last in
      if n = 0 then x else L.concatenate [ repeat first; x; repeat final ] last
  | "even" | "odd" ->
      if n = 0 then x
      else
        let left = L.rev (L.slice_in_dim ~axis:last x 1 (n + 1)) [| last |] in
        let right =
          L.rev
            (L.slice_in_dim ~axis:last x (size - n - 1) (size - 1))
            [| last |]
        in
        if mode = "even" then L.concatenate [ left; x; right ] last
        else
          let edge lo =
            let v = L.slice_in_dim ~axis:last x lo (lo + 1) in
            let twice = L.mul v (L.full_like v 2.0) in
            NL.broadcast_to twice
              (Array.mapi (fun d s -> if d = last then n else s) (shape x))
          in
          L.concatenate
            [ b1 Sub [ edge 0; left ]; x; b1 Sub [ edge (size - 1); right ] ]
            last
  | other -> invalid_arg ("Unknown boundary option '" ^ other ^ "'")

let segments x ~nperseg ~nstep =
  let last = rank x - 1 in
  let size = (shape x).(last) in
  let count = if size < nperseg then 0 else ((size - nperseg) / nstep) + 1 in
  L.stack
    (List.init count (fun k ->
         L.slice_in_dim ~axis:last x (k * nstep) ((k * nstep) + nperseg)))
    last

let window_on x win =
  let sh = shape x in
  let n = Array.length sh in
  L.mul x (L.broadcast_in_dim win sh [| n - 1 |])

let spectral_helper ?y ~fs ~window ~nperseg ~noverlap ~nfft ~detrend_type
    ~return_onesided ~scaling ~axis ~mode ~boundary ~padded x =
  let x = inexact x in
  let axis = if axis < 0 then axis + rank x else axis in
  let x = NL.moveaxis [| axis |] [| rank x - 1 |] x in
  let y =
    Option.map
      (fun v ->
        let v = inexact v in
        NL.moveaxis [| axis |] [| rank v - 1 |] v)
      y
  in
  let dt = dtype_of x in
  let rdt = real_part_dtype dt in
  let input_len = (shape x).(rank x - 1) in
  let nperseg = Stdlib.min (Option.value nperseg ~default:256) input_len in
  let win =
    match window with Some w -> NL.astype w rdt | None -> hann rdt nperseg
  in
  let noverlap = Option.value noverlap ~default:(nperseg / 2) in
  let nfft = Option.value nfft ~default:nperseg in
  if nfft < nperseg then
    invalid_arg "nfft must be greater than or equal to nperseg.";
  if noverlap >= nperseg then invalid_arg "noverlap must be less than nperseg.";
  let nstep = nperseg - noverlap in
  let extend v =
    match boundary with
    | Some mode -> pad_edges mode v (nperseg / 2)
    | None -> v
  in
  let pad_tail v =
    if not padded then v
    else
      let size = (shape v).(rank v - 1) in
      let extra =
        ((-(size - nperseg) mod nstep) + nstep) mod nstep mod nperseg
      in
      if extra = 0 then v
      else
        let cfg = Array.make (rank v) (0, 0, 0) in
        cfg.(rank v - 1) <- (0, extra, 0);
        L.pad v (L.full [||] 0.0 (dtype_of v)) cfg
  in
  let x = pad_tail (extend x) in
  let y = Option.map (fun v -> pad_tail (extend v)) y in
  let fs_value = L.full [||] fs (dtype_of win) in
  let one = L.full [||] 1.0 (dtype_of win) in
  let scale =
    match scaling with
    | "density" ->
        let total = Numpy.Reductions.sum (b1 Mul [ win; win ]) in
        b1 Div [ one; b1 Mul [ fs_value; total ] ]
    | "spectrum" ->
        let total = Numpy.Reductions.sum win in
        b1 Div [ one; b1 (Integer_pow 2) [ total ] ]
    | other -> invalid_arg ("Unknown scaling: " ^ other)
  in
  let scale = if mode = "stft" then b1 Sqrt [ scale ] else scale in
  let scale = NL.astype scale (complex_dtype_of (dtype_of scale)) in
  let onesided =
    return_onesided
    && (not (is_complex x))
    && not (match y with Some v -> is_complex v | None -> false)
  in
  let transform v =
    let segs = segments v ~nperseg ~nstep in
    let segs =
      match detrend_type with
      | None -> segs
      | Some t -> detrend ~axis:(-1) ~type_:t segs
    in
    let windowed = window_on segs win in
    if onesided then Numpy.Fft.rfft ~n:nfft windowed
    else Numpy.Fft.fft ~n:nfft (NL.astype windowed (complex_dtype_of dt))
  in
  let result =
    match y with
    | Some v -> b1 Mul [ b1 Conj [ transform x ]; transform v ]
    | None ->
        let r = transform x in
        if mode = "psd" then b1 Mul [ b1 Conj [ r ]; r ] else r
  in
  let result = b1 Mul [ result; scale ] in
  let result =
    if onesided && mode = "psd" then begin
      let sh = shape result in
      let n_freq = sh.(Array.length sh - 1) in
      let cdt = complex_dtype_of dt in
      let lanes = Array.fold_left ( * ) 1 sh / Stdlib.max n_freq 1 in
      let mask =
        Concrete
          (Ndarray.of_complex cdt sh
             (Array.init (lanes * n_freq) (fun k ->
                  let f = k mod n_freq in
                  let doubled = f > 0 && (nfft mod 2 = 1 || f < n_freq - 1) in
                  { Complex.re = (if doubled then 2.0 else 1.0); im = 0.0 })))
      in
      b1 Mul [ result; mask ]
    end
    else result
  in
  let freqs =
    if onesided then Numpy.Fft.rfftfreq ~d:(1.0 /. fs) ~dtype:rdt nfft
    else Numpy.Fft.fftfreq ~d:(1.0 /. fs) ~dtype:rdt nfft
  in
  let processed_len = (shape x).(rank x - 1) in
  let n_times = ((processed_len - nperseg) / nstep) + 1 in
  let time =
    let steps =
      b1 (Iota { dtype = rdt; shape = [| n_times |]; dimension = 0 }) []
    in
    let scaled = b1 Mul [ steps; L.full [||] (float_of_int nstep) rdt ] in
    let moved =
      b1 Add [ scaled; L.full [||] (float_of_int nperseg /. 2.0) rdt ]
    in
    let over = b1 Div [ moved; L.full [||] fs rdt ] in
    if boundary <> None then
      b1 Sub [ over; L.full [||] (float_of_int nperseg /. 2.0 /. fs) rdt ]
    else over
  in
  let result =
    if y = None && mode <> "stft" then b1 Real [ result ] else result
  in
  (freqs, time, result)

let stft ?(fs = 1.0) ?window ?(nperseg = 256) ?noverlap ?nfft
    ?detrend:detrend_type ?(return_onesided = true) ?(boundary = Some "zeros")
    ?(padded = true) ?(axis = -1) x =
  let f, t, z =
    spectral_helper ~fs ~window ~nperseg:(Some nperseg) ~noverlap ~nfft
      ~detrend_type ~return_onesided ~scaling:"spectrum" ~axis ~mode:"stft"
      ~boundary ~padded x
  in
  (f, t, NL.swapaxes (-1) (-2) z)

let csd ?(fs = 1.0) ?window ?nperseg ?noverlap ?nfft
    ?(detrend = Some "constant") ?(return_onesided = true)
    ?(scaling = "density") ?(axis = -1) ?(average = "mean") x y =
  let freqs, _, pxy =
    spectral_helper ?y ~fs ~window ~nperseg ~noverlap ~nfft
      ~detrend_type:detrend ~return_onesided ~scaling ~axis ~mode:"psd"
      ~boundary:None ~padded:false x
  in
  let swapped = NL.swapaxes (-1) (-2) pxy in
  let last = rank swapped - 1 in
  let n = (shape swapped).(last) in
  let averaged =
    if n <= 1 then L.squeeze swapped [| last |]
    else
      match average with
      | "mean" -> Numpy.Reductions.mean ~axis:[| last |] swapped
      | "median" ->
          let bias =
            let total = ref 1.0 in
            let k = ref 2 in
            while !k < n do
              total :=
                !total
                +. (1.0 /. float_of_int (!k + 1))
                -. (1.0 /. float_of_int !k);
              k := !k + 2
            done;
            !total
          in
          let med =
            if is_complex swapped then
              b1 Complex
                [
                  Numpy.Reductions.median ~axis:last (b1 Real [ swapped ]);
                  Numpy.Reductions.median ~axis:last (b1 Imag [ swapped ]);
                ]
            else Numpy.Reductions.median ~axis:last swapped
          in
          b1 Div
            [
              med;
              NL.astype
                (L.full (shape med) bias (real_part_dtype (dtype_of med)))
                (dtype_of med);
            ]
      | other -> invalid_arg ("average must be 'median' or 'mean', got " ^ other)
  in
  (freqs, averaged)

let welch ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided
    ?scaling ?axis ?average x =
  let freqs, pxx =
    csd ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided ?scaling
      ?axis ?average x None
  in
  (freqs, if is_complex pxx then b1 Real [ pxx ] else pxx)

let overlap_and_add x step =
  let sh = shape x in
  let n = Array.length sh in
  let frames = sh.(n - 2) and seg = sh.(n - 1) in
  let out_len = (step * (frames - 1)) + seg in
  let zero = L.full [||] 0.0 (dtype_of x) in
  let shifted k piece =
    let cfg = Array.make (n - 1) (0, 0, 0) in
    cfg.(n - 2) <- (k * step, out_len - seg - (k * step), 0);
    L.pad piece zero cfg
  in
  match List.mapi shifted (L.unstack x (n - 2)) with
  | first :: rest -> List.fold_left L.add first rest
  | [] -> invalid_arg "overlap_and_add: no frames"

let istft ?(fs = 1.0) ?window ?nperseg ?noverlap ?nfft ?(input_onesided = true)
    ?(boundary = true) ?(time_axis = -1) ?(freq_axis = -2) zxx =
  let rankz = rank zxx in
  if rankz < 2 then invalid_arg "Input stft must be at least 2d!";
  let canon a = if a < 0 then a + rankz else a in
  if canon freq_axis = canon time_axis then
    invalid_arg "Must specify differing time and frequency axes!";
  let zxx =
    NL.moveaxis
      [| canon freq_axis; canon time_axis |]
      [| rankz - 2; rankz - 1 |]
      zxx
  in
  let freq_len = (shape zxx).(rankz - 2) in
  let default_seg = if input_onesided then 2 * (freq_len - 1) else freq_len in
  let nperseg = Option.value nperseg ~default:default_seg in
  let nfft =
    Option.value nfft
      ~default:
        (if input_onesided && nperseg = default_seg + 1 then default_seg + 1
         else nperseg)
  in
  let noverlap = Option.value noverlap ~default:(nperseg / 2) in
  let nstep = nperseg - noverlap in
  let rdt = real_part_dtype (dtype_of zxx) in
  let win =
    match window with Some w -> NL.astype w rdt | None -> hann rdt nperseg
  in
  let frames = NL.swapaxes (-1) (-2) (NL.astype zxx (complex_dtype_of rdt)) in
  let time_domain =
    if input_onesided then Numpy.Fft.irfft ~n:nfft frames
    else Numpy.Fft.ifft ~n:nfft frames
  in
  let last = rank time_domain - 1 in
  let cut = L.slice_in_dim ~axis:last time_domain 0 nperseg in
  let scale = Numpy.Reductions.sum win in
  let scaled = b1 Mul [ cut; NL.astype scale (dtype_of cut) ] in
  let windowed =
    b1 Mul
      [
        scaled;
        NL.astype
          (L.broadcast_in_dim win (shape scaled) [| rank scaled - 1 |])
          (dtype_of scaled);
      ]
  in
  let x = overlap_and_add windowed nstep in
  let n_frames = (shape cut).(last - 1) in
  let win_sq = L.mul win win in
  let stacked = L.stack (List.init n_frames (fun _ -> win_sq)) 0 in
  let norm = overlap_and_add stacked nstep in
  let trim v =
    if boundary then
      let d = rank v - 1 in
      let size = (shape v).(d) in
      L.slice_in_dim ~axis:d v (nperseg / 2) (size - (nperseg / 2))
    else v
  in
  let x = trim x and norm = trim norm in
  let safe =
    L.select (L.gt norm (L.full_like norm 1e-10)) norm (L.full_like norm 1.0)
  in
  let broadcast_norm = L.broadcast_in_dim safe (shape x) [| rank x - 1 |] in
  let out =
    if is_complex x then b1 Div [ x; NL.astype broadcast_norm (dtype_of x) ]
    else L.div x broadcast_norm
  in
  let count = (shape out).(rank out - 1) in
  let time =
    Concrete
      (Ndarray.of_floats rdt [| count |]
         (Array.init count (fun i -> float_of_int i /. fs)))
  in
  (time, out)
