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
let flip_all v = NL.flip v
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
      (Ndarray.of_floats
         (Dtypes.default_int_dtype ())
         [||]
         [| float_of_int i |])
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

let index_dtype () = Dtypes.default_int_dtype ()
let int_literal dt v = Concrete (Ndarray.of_floats dt [||] [| float_of_int v |])

let slice_rows x lo hi =
  if lo = 0 && hi = (shape x).(0) then x
  else
    let n = rank x in
    let start = Array.make n 0 and limit = Array.copy (shape x) in
    start.(0) <- lo;
    limit.(0) <- hi;
    L.slice x start limit

let add_rows x lo updates =
  let n = rank x in
  let dims =
    Np_ints.mark_sorted_unique
      {
        update_window_dims = Array.init n Fun.id;
        inserted_window_dims = [||];
        scatter_dims_to_operand_dims = (if lo = 0 then [||] else [| 0 |]);
        s_operand_batching_dims = [||];
        s_scatter_indices_batching_dims = [||];
      }
  in
  let indices =
    if lo = 0 then Concrete (Ndarray.of_floats (index_dtype ()) [| 0 |] [||])
    else L.broadcast_in_dim (int_literal (index_dtype ()) lo) [| 1 |] [||]
  in
  L.scatter_add x indices updates dims

let detrend ?(axis = -1) ?(type_ = "linear") ?(bp = []) data =
  if type_ <> "constant" && type_ <> "linear" then
    invalid_arg "Trend type must be 'linear' or 'constant'.";
  let data = inexact data in
  if type_ = "constant" then
    Numpy.Ufuncs.subtract data (R.mean ~axis:[| axis |] ~keepdims:true data)
  else begin
    let axis_c = if axis < 0 then axis + rank data else axis in
    let n = (shape data).(axis_c) in
    let bounds = List.sort_uniq compare ((0 :: bp) @ [ n ]) in
    if List.hd bounds < 0 || List.nth bounds (List.length bounds - 1) > n then
      invalid_arg
        "Breakpoints must be non-negative and less than length of data along \
         given axis.";
    let moved = NL.moveaxis [| axis |] [| 0 |] data in
    let moved_shape = shape moved in
    let rows = NL.reshape moved [| n; -1 |] in
    let dt = dtype_of rows in
    let rec pairs = function
      | lo :: (hi :: _ as rest) -> (lo, hi) :: pairs rest
      | _ -> []
    in
    let highest = Lax_config.precision_of_level Lax_config.Precision_highest in
    let out =
      List.fold_left
        (fun rows (lo, hi) ->
          let npts = hi - lo in
          let ones = Numpy.Array_creation.ones ~dtype:dt [| npts |] in
          let column =
            NL.arange ~start:1.0 ~dtype:dt (float_of_int (npts + 1))
          in
          let design =
            NL.transpose
              (NL.vstack
                 [
                   ones;
                   Numpy.Ufuncs.divide column
                     (Concrete
                        (Ndarray.of_floats dt [||] [| float_of_int npts |]));
                 ])
          in
          let coef, _, _, _ =
            Numpy.Linalg.lstsq design (slice_rows rows lo hi)
          in
          add_rows rows lo
            (Numpy.Ufuncs.negative
               (Numpy.Tensor_contractions.matmul ~precision:highest design coef)))
        rows (pairs bounds)
    in
    NL.moveaxis [| 0 |] [| axis |] (NL.reshape out moved_shape)
  end

let pi = 4.0 *. Float.atan 1.0

let hann n =
  if n = 1 then
    Concrete
      (Ndarray.of_floats (Dtypes.default_float_dtype ()) [| 1 |] [| 1.0 |])
  else
    let points =
      Numpy.Array_creation.linspace_v ~num:n ~endpoint:false
        (Numpy.Jnp_util.py_int 0)
        (Numpy.Jnp_util.py_float_arg pi)
    in
    b1 (Integer_pow 2) [ Numpy.Ufuncs.sin points ]

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

let odd_ext x n =
  if n < 1 then x
  else begin
    let last = rank x - 1 in
    let size = (shape x).(last) in
    if n > size - 1 then
      invalid_arg
        (Printf.sprintf
           "The extension length n (%d) is too big. It must not exceed \
            x.shape[axis]-1, which is %d."
           n (size - 1));
    let piece lo hi = L.slice_in_dim ~axis:last x lo hi in
    let left_end = piece 0 1 in
    let left_ext = NL.flip ~axis:[| last |] (piece 1 (n + 1)) in
    let right_end = piece (size - 1) size in
    let right_ext =
      NL.flip ~axis:[| last |] (piece (size - n - 1) (size - 1))
    in
    let twice v = Numpy.Ufuncs.multiply (Numpy.Jnp_util.py_int 2) v in
    NL.concatenate ~axis:last
      [
        Numpy.Ufuncs.subtract (twice left_end) left_ext;
        x;
        Numpy.Ufuncs.subtract (twice right_end) right_ext;
      ]
  end

let pad_edges mode x n =
  let last = rank x - 1 in
  let widths = Array.make (rank x) (0, 0) in
  widths.(last) <- (n, n);
  match mode with
  | "zeros" -> NL.pad ~float_literal:true x widths 0.0
  | "constant" -> NL.pad ~mode:"edge" x widths 0.0
  | "even" -> NL.pad ~mode:"reflect" x widths 0.0
  | "odd" -> odd_ext x n
  | other ->
      invalid_arg
        (Printf.sprintf
           "Unknown boundary option '%s', must be one of: ['even', 'odd', \
            'constant', 'zeros', None]"
           other)

let fft_helper x win detrend_func ~nperseg ~noverlap ~nfft ~twosided =
  let sh = shape x in
  let r = Array.length sh in
  let signal_length = sh.(r - 1) in
  let result =
    if nperseg = 1 && noverlap = 0 then NL.expand_dims x [| r |]
    else
      let step = nperseg - noverlap in
      let starts =
        NL.arange ~step:(float_of_int step)
          ~dtype:(Dtypes.default_int_dtype ())
          (float_of_int (signal_length - nperseg + 1))
      in
      match
        Interpreters.Batching.vmap_out
          (function
            | [ start ] -> [ L.dynamic_slice_in_dim ~axis:(-1) x start nperseg ]
            | _ -> invalid_arg "fft_helper: arity")
          [ Some 0 ] [ starts ] (-2)
      with
      | [ v ] -> v
      | _ -> invalid_arg "fft_helper: arity"
  in
  let result = detrend_func result in
  let result =
    if is_complex win && not (is_complex result) then
      NL.astype result (complex_dtype_of (dtype_of result))
    else result
  in
  let reshaped =
    NL.reshape win (Array.append (Array.make (r - 1) 1) [| 1; nperseg |])
  in
  let result = Numpy.Ufuncs.multiply reshaped result in
  if twosided then Numpy.Fft.fft ~n:nfft result
  else Numpy.Fft.rfft ~n:nfft (Numpy.Ufuncs.real result)

let triage_segments window nperseg input_length dt =
  match window with
  | None ->
      let n = Option.value nperseg ~default:input_length in
      let n = Stdlib.min n input_length in
      let win = hann n in
      let win =
        if dtype_of win = dt then NL.copy win else NL.astype (NL.copy win) dt
      in
      (win, n)
  | Some w ->
      let win = NL.astype w dt in
      if rank win <> 1 then invalid_arg "window must be 1-D";
      let size = (shape win).(0) in
      let n = Option.value nperseg ~default:size in
      if input_length < size then
        invalid_arg "window is longer than input signal";
      if n <> size then
        invalid_arg
          "value specified for nperseg is different from length of window";
      (win, n)

let tail_zeros v like count =
  let sh = Array.copy (shape v) in
  sh.(Array.length sh - 1) <- count;
  Numpy.Array_creation.zeros_like ~shape:sh like

let py_mod a b = ((a mod b) + b) mod b

let spectral_helper ?y ~fs ~window ~nperseg ~noverlap ~nfft ~detrend_type
    ~return_onesided ~scaling ~axis ~mode ~boundary ~padded x =
  if mode <> "psd" && mode <> "stft" then
    invalid_arg
      (Printf.sprintf
         "Unknown value for mode %s, must be one of: ('psd', 'stft')" mode);
  (match boundary with
  | None | Some ("even" | "odd" | "constant" | "zeros") -> ()
  | Some other ->
      invalid_arg
        (Printf.sprintf
           "Unknown boundary option '%s', must be one of: ['even', 'odd', \
            'constant', 'zeros', None]"
           other));
  let axis = if axis < 0 then axis + rank x else axis in
  let x, y =
    match y with
    | None -> (inexact x, None)
    | Some v ->
        let a, b = promote x v in
        if rank a <> rank b then
          invalid_arg "two-arguments must have the same rank";
        (a, Some b)
  in
  if y <> None && mode <> "psd" then
    invalid_arg "two-argument mode is available only when mode=='psd'";
  let dt = dtype_of x in
  let result_dt = complex_dtype_of dt in
  let freq_dt = real_part_dtype dt in
  (match nperseg with
  | Some n when n < 1 -> invalid_arg "nperseg must be a positive integer"
  | _ -> ());
  let win, nperseg = triage_segments window nperseg (shape x).(axis) dt in
  let noverlap = Option.value noverlap ~default:(nperseg / 2) in
  let nfft = Option.value nfft ~default:nperseg in
  let size_of v = Array.fold_left ( * ) 1 (shape v) in
  let empty =
    match y with
    | None -> size_of x = 0
    | Some v -> size_of x = 0 || size_of v = 0
  in
  if empty then begin
    let sh =
      match y with
      | None -> shape x
      | Some v ->
          Array.mapi
            (fun i d -> if i = axis then Stdlib.min d (shape v).(i) else d)
            (shape x)
    in
    let zeros d = Numpy.Array_creation.zeros ~dtype:d sh in
    (zeros freq_dt, zeros freq_dt, zeros result_dt)
  end
  else begin
    let x = NL.moveaxis [| axis |] [| -1 |] x in
    let y =
      Option.map
        (fun v -> if rank v > 1 then NL.moveaxis [| axis |] [| -1 |] v else v)
        y
    in
    let last v = (shape v).(rank v - 1) in
    let x, y =
      match y with
      | Some v when last x <> last v ->
          if last x < last v then
            ( NL.concatenate ~axis:(-1) [ x; tail_zeros x x (last v - last x) ],
              Some v )
          else
            ( x,
              Some
                (NL.concatenate ~axis:(-1)
                   [ v; tail_zeros v x (last x - last v) ]) )
      | _ -> (x, y)
    in
    if nfft < nperseg then
      invalid_arg "nfft must be greater than or equal to nperseg.";
    if noverlap >= nperseg then
      invalid_arg "noverlap must be less than nperseg.";
    let nstep = nperseg - noverlap in
    let extend v =
      match boundary with Some m -> pad_edges m v (nperseg / 2) | None -> v
    in
    let x = extend x in
    let y = Option.map extend y in
    let x, y =
      if padded then begin
        let nadd = py_mod (py_mod (-(last x - nperseg)) nstep) nperseg in
        let pad_tail v = NL.concatenate ~axis:(-1) [ v; tail_zeros v x nadd ] in
        let x' = pad_tail x in
        (x', Option.map pad_tail y)
      end
      else (x, y)
    in
    let detrend_func v =
      match detrend_type with
      | None -> v
      | Some t -> detrend ~axis:(-1) ~type_:t v
    in
    let py_float = Numpy.Jnp_util.py_float in
    let scale =
      match scaling with
      | "density" ->
          Numpy.Ufuncs.divide (py_float 1.0)
            (Numpy.Ufuncs.multiply (py_float fs)
               (R.sum (Numpy.Ufuncs.multiply win win)))
      | "spectrum" ->
          Numpy.Ufuncs.divide (py_float 1.0) (b1 (Integer_pow 2) [ R.sum win ])
      | other -> invalid_arg ("Unknown scaling: " ^ other)
    in
    let scale = if mode = "stft" then Numpy.Ufuncs.sqrt scale else scale in
    let scale = NL.astype scale (complex_dtype_of (dtype_of scale)) in
    let onesided =
      return_onesided
      && (not (is_complex x))
      && not (match y with Some v -> is_complex v | None -> false)
    in
    let freqs =
      if onesided then Numpy.Fft.rfftfreq ~d:(1.0 /. fs) ~dtype:freq_dt nfft
      else Numpy.Fft.fftfreq ~d:(1.0 /. fs) ~dtype:freq_dt nfft
    in
    let helper v =
      fft_helper v win detrend_func ~nperseg ~noverlap ~nfft
        ~twosided:(not onesided)
    in
    let first = helper x in
    let result =
      match y with
      | Some v ->
          let other = helper v in
          Numpy.Ufuncs.multiply (Numpy.Ufuncs.conjugate first) other
      | None when mode = "psd" ->
          Numpy.Ufuncs.multiply (Numpy.Ufuncs.conjugate first) first
      | None -> first
    in
    let result = Numpy.Ufuncs.multiply result scale in
    let result =
      if onesided && mode = "psd" then begin
        let sh = shape result in
        let n = Array.length sh in
        let n_freq = sh.(n - 1) in
        let stop = if nfft mod 2 = 1 then n_freq else n_freq - 1 in
        let update_shape =
          Array.mapi (fun i d -> if i = n - 1 then stop - 1 else d) sh
        in
        L.scatter_mul ~unique_indices:true result
          (L.broadcast_in_dim (int_literal (index_dtype ()) 1) [| 1 |] [||])
          (Numpy.Array_creation.full ~dtype:(dtype_of result) update_shape 2.0)
          (Np_ints.mark_sorted_unique
             {
               update_window_dims = Array.init n Fun.id;
               inserted_window_dims = [||];
               scatter_dims_to_operand_dims = [| n - 1 |];
               s_operand_batching_dims = [||];
               s_scatter_indices_batching_dims = [||];
             })
      end
      else result
    in
    let half = float_of_int nperseg /. 2.0 in
    let time =
      Numpy.Ufuncs.divide
        (NL.arange ~start:half
           ~step:(float_of_int (nperseg - noverlap))
           ~dtype:freq_dt
           (float_of_int (last x) -. half +. 1.0))
        (py_float fs)
    in
    let time =
      if boundary <> None then
        Numpy.Ufuncs.subtract time (py_float (half /. fs))
      else time
    in
    let result = NL.astype result result_dt in
    let result =
      if y = None && mode <> "stft" then Numpy.Ufuncs.real result else result
    in
    let result = NL.moveaxis [| -1 |] [| axis |] result in
    (freqs, time, result)
  end

let stft ?(fs = 1.0) ?window ?(nperseg = 256) ?noverlap ?nfft
    ?detrend:detrend_type ?(return_onesided = true) ?(boundary = Some "zeros")
    ?(padded = true) ?(axis = -1) x =
  spectral_helper ~fs ~window ~nperseg:(Some nperseg) ~noverlap ~nfft
    ~detrend_type ~return_onesided ~scaling:"spectrum" ~axis ~mode:"stft"
    ~boundary ~padded x

let median_bias n =
  let py_float = Numpy.Jnp_util.py_float in
  let ii_2 =
    NL.arange ~start:2.0 ~step:2.0
      ~dtype:(Dtypes.default_float_dtype ())
      (float_of_int n)
  in
  Numpy.Ufuncs.add (Numpy.Jnp_util.py_int 1)
    (R.sum
       (Numpy.Ufuncs.subtract
          (Numpy.Ufuncs.divide (py_float 1.0)
             (Numpy.Ufuncs.add ii_2 (Numpy.Jnp_util.py_int 1)))
          (Numpy.Ufuncs.divide (py_float 1.0) ii_2)))

let py_complex re im = Numpy.Jnp_util.py_complex { Complex.re; im }

let csd ?(fs = 1.0) ?window ?nperseg ?noverlap ?nfft
    ?(detrend = Some "constant") ?(return_onesided = true)
    ?(scaling = "density") ?(axis = -1) ?(average = "mean") x y =
  let freqs, _, pxy =
    spectral_helper ?y ~fs ~window ~nperseg ~noverlap ~nfft
      ~detrend_type:detrend ~return_onesided ~scaling ~axis ~mode:"psd"
      ~boundary:None ~padded:false x
  in
  let pxy =
    if y <> None then Numpy.Ufuncs.add pxy (py_complex 0.0 0.0) else pxy
  in
  let size = Array.fold_left ( * ) 1 (shape pxy) in
  let out =
    if rank pxy >= 2 && size > 0 then
      let sh = shape pxy in
      let r = Array.length sh in
      if sh.(r - 1) > 1 then
        match average with
        | "median" ->
            let bias = NL.astype (median_bias sh.(r - 1)) (dtype_of pxy) in
            let med =
              if is_complex pxy then
                let re = R.median ~axis:(-1) (Numpy.Ufuncs.real pxy) in
                let im = R.median ~axis:(-1) (Numpy.Ufuncs.imag pxy) in
                Numpy.Ufuncs.add re
                  (Numpy.Ufuncs.multiply (py_complex 0.0 1.0) im)
              else R.median ~axis:(-1) pxy
            in
            Numpy.Ufuncs.divide med bias
        | "mean" -> R.mean ~axis:[| -1 |] pxy
        | other ->
            invalid_arg
              (Printf.sprintf "average must be \"median\" or \"mean\", got %s"
                 other)
      else NL.reshape pxy (Array.sub sh 0 (r - 1))
    else pxy
  in
  (freqs, out)

let welch ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided
    ?scaling ?axis ?average x =
  let freqs, pxx =
    csd ?fs ?window ?nperseg ?noverlap ?nfft ?detrend ?return_onesided ?scaling
      ?axis ?average x None
  in
  (freqs, Numpy.Ufuncs.real pxx)

let slice_to x axis stop =
  if stop = (shape x).(axis) then x
  else
    let n = rank x in
    let limit = Array.copy (shape x) in
    limit.(axis) <- stop;
    L.slice x (Array.make n 0) limit

let overlap_and_add x step =
  let sh = shape x in
  let n = Array.length sh in
  if n < 2 then invalid_arg "Input must have (..., frames, frame_length) shape.";
  let batch = Array.sub sh 0 (n - 2) in
  let nframes = sh.(n - 2) and segment_len = sh.(n - 1) in
  let flat = Array.fold_left ( * ) 1 batch in
  let x = NL.reshape x [| flat; nframes; segment_len |] in
  let output_size = (step * (nframes - 1)) + segment_len in
  let nstep_per_segment = 1 + ((segment_len - 1) / step) in
  let padded_segment_len = nstep_per_segment * step in
  let x =
    NL.pad x [| (0, 0); (0, 0); (0, padded_segment_len - segment_len) |] 0.0
  in
  let x = NL.reshape x [| flat; nframes; nstep_per_segment; step |] in
  let x = NL.transpose ~axes:[| 0; 2; 1; 3 |] x in
  let x = NL.pad x [| (0, 0); (0, 0); (0, nframes); (0, 0) |] 0.0 in
  let shrunken = (shape x).(2) - 1 in
  let x = NL.reshape x [| flat; -1 |] in
  let x = slice_to x 1 (nstep_per_segment * shrunken * step) in
  let x = NL.reshape x [| flat; nstep_per_segment; shrunken * step |] in
  let x = slice_to (R.sum ~axis:[| 1 |] x) 1 output_size in
  NL.reshape x (Array.append batch [| -1 |])

let trim_edges v k =
  let n = rank v in
  let size = (shape v).(n - 1) in
  let start = Array.make n 0 and limit = Array.copy (shape v) in
  start.(n - 1) <- k;
  limit.(n - 1) <- (if k = 0 then 0 else size - k);
  L.slice v start limit

let istft ?(fs = 1.0) ?window ?nperseg ?noverlap ?nfft ?(input_onesided = true)
    ?(boundary = true) ?(time_axis = -1) ?(freq_axis = -2) zxx =
  let rankz = rank zxx in
  if rankz < 2 then invalid_arg "Input stft must be at least 2d!";
  let canon a = if a < 0 then a + rankz else a in
  let freq_axis = canon freq_axis and time_axis = canon time_axis in
  if freq_axis = time_axis then
    invalid_arg "Must specify differing time and frequency axes!";
  let zxx = NL.astype zxx (complex_dtype_of (dtype_of zxx)) in
  let n_default =
    if input_onesided then 2 * ((shape zxx).(freq_axis) - 1)
    else (shape zxx).(freq_axis)
  in
  let nperseg_int =
    match nperseg with Some 0 | None -> n_default | Some n -> n
  in
  if nperseg_int < 1 then invalid_arg "nperseg must be a positive integer";
  let nfft_int =
    match nfft with
    | None ->
        if input_onesided && nperseg_int = n_default + 1 then n_default + 1
        else n_default
    | Some n -> n
  in
  if nfft_int < nperseg_int then
    invalid_arg
      (Printf.sprintf "FFT length (%d) must be longer than nperseg (%d)."
         nfft_int nperseg_int);
  let noverlap_int =
    match noverlap with Some 0 | None -> nperseg_int / 2 | Some n -> n
  in
  if noverlap_int >= nperseg_int then
    invalid_arg "noverlap must be less than nperseg.";
  let nstep = nperseg_int - noverlap_int in
  let zxx =
    if time_axis <> rankz - 1 || freq_axis <> rankz - 2 then
      let outer =
        List.filter
          (fun i -> i <> time_axis && i <> freq_axis)
          (List.init rankz Fun.id)
      in
      NL.transpose ~axes:(Array.of_list (outer @ [ freq_axis; time_axis ])) zxx
    else zxx
  in
  let transformed =
    if input_onesided then Numpy.Fft.irfft ?n:nfft ~axis:(-2) zxx
    else Numpy.Fft.ifft ?n:nfft ~axis:(-2) zxx
  in
  let xsubs = slice_to transformed (rankz - 2) nperseg_int in
  let win =
    match window with
    | None -> NL.astype (hann nperseg_int) (dtype_of xsubs)
    | Some w ->
        if rank w <> 1 then invalid_arg "window must be 1-D";
        if (shape w).(0) <> nperseg_int then
          invalid_arg
            (Printf.sprintf "window must have length of %d" nperseg_int);
        w
  in
  let xsubs = Numpy.Ufuncs.multiply xsubs (R.sum win) in
  let nx = rank xsubs in
  let win_shape =
    Array.init nx (fun i -> if i = nx - 2 then (shape win).(0) else 1)
  in
  let win = L.broadcast_in_dim win win_shape [| nx - 2 |] in
  let x =
    overlap_and_add
      (NL.swapaxes (-2) (-1) (Numpy.Ufuncs.multiply xsubs win))
      nstep
  in
  let win_squared =
    NL.repeat ~axis:(-1) (Numpy.Ufuncs.multiply win win) (shape xsubs).(nx - 1)
  in
  let norm = overlap_and_add (NL.swapaxes (-2) (-1) win_squared) nstep in
  let x, norm =
    if boundary then
      let k = nperseg_int / 2 in
      let tx = trim_edges x k in
      (tx, trim_edges norm k)
    else (x, norm)
  in
  let py_float = Numpy.Jnp_util.py_float in
  let safe =
    NL.where_ (Numpy.Ufuncs.greater norm (py_float 1e-10)) norm (py_float 1.0)
  in
  let x = Numpy.Ufuncs.divide x safe in
  let x =
    if rank x > 1 && time_axis <> rankz - 1 then
      let target = if freq_axis < time_axis then time_axis - 1 else time_axis in
      NL.moveaxis [| -1 |] [| target |] x
    else x
  in
  let time =
    Numpy.Ufuncs.divide
      (NL.arange
         ~dtype:(real_part_dtype (dtype_of x))
         (float_of_int (shape x).(0)))
      (py_float fs)
  in
  (time, x)
