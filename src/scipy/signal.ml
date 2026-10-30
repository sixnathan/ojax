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
  let limit = Array.map2 ( + ) start out_shape in
  L.slice conv start limit

let fftconvolve ?(mode = "full") in1 in2 =
  if rank in1 <> rank in2 then
    invalid_arg "in1 and in2 should have the same dimensionality";
  let in1, in2 = promote in1 in2 in
  fftconvolve_unbatched in1 in2 mode

let convolve ?(mode = "full") ?(method_ = "auto") in1 in2 =
  match method_ with
  | "fft" -> fftconvolve ~mode in1 in2
  | "direct" | "auto" -> convolve_nd in1 in2 mode
  | other ->
      invalid_arg ("Got method=" ^ other ^ "; expected auto, fft, or direct.")

let convolve2d ?(mode = "full") in1 in2 =
  if rank in1 <> 2 || rank in2 <> 2 then
    invalid_arg "convolve2d() only supports 2-dimensional inputs.";
  convolve_nd in1 in2 mode

let correlate ?(mode = "full") ?method_ in1 in2 =
  convolve ~mode ?method_ in1 (flip_all (conj_if_complex in2))

let correlate2d ?(mode = "full") in1 in2 =
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

let detrend ?(axis = -1) ?(type_ = "linear") data =
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
      let design =
        Concrete
          (Ndarray.of_floats dt [| n; 2 |]
             (Array.init (2 * n) (fun i ->
                  if i mod 2 = 0 then 1.0
                  else float_of_int ((i / 2) + 1) /. float_of_int n)))
      in
      let coef, _, _, _ = Numpy.Linalg.lstsq design rows in
      let fit = L.dot design coef in
      let out = NL.reshape (b1 Sub [ rows; fit ]) moved_shape in
      NL.moveaxis [| 0 |] [| axis |] out
  | _ -> invalid_arg "Trend type must be 'linear' or 'constant'."
