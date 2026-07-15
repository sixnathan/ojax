module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Lax_numpy
module UF = Ufuncs
module IDX = Indexing
module LF = Lax.Fft

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let ndim v = Array.length (shape v)
let dtype v = (get_aval v).T.dtype
let is_complex = function D.Complex64 | D.Complex128 -> true | _ -> false

let complex_basetype = function
  | D.Complex64 -> D.F32
  | D.Complex128 -> D.F64
  | d -> d

let resolve_axis nd ax = if ax < 0 then ax + nd else ax
let idx_vec n = NL.arange ~start:0.0 ~step:1.0 ~dtype:D.I32 (float_of_int n)

let crop_prefix arr axis s =
  if (shape arr).(axis) = s then arr else IDX.take ~axis arr (idx_vec s)

let scalar_real dt f = T.Concrete (Nd.of_floats dt [||] [| f |])

let fft_norm s is_inverse norm transformed =
  match norm with
  | None | Some "backward" -> transformed
  | Some "ortho" ->
      let p = Array.fold_left (fun a x -> a *. float_of_int x) 1.0 s in
      let m = if is_inverse then sqrt p else 1.0 /. sqrt p in
      UF.multiply transformed
        (scalar_real (complex_basetype (dtype transformed)) m)
  | Some "forward" ->
      let p = Array.fold_left (fun a x -> a *. float_of_int x) 1.0 s in
      let m = if is_inverse then p else 1.0 /. p in
      UF.multiply transformed
        (scalar_real (complex_basetype (dtype transformed)) m)
  | Some other -> invalid_arg ("jax.numpy.fft: invalid norm value " ^ other)

let is_inverse_name func_name =
  String.length func_name > 0 && func_name.[0] = 'i'

let fft_core func_name fft_type a s axes norm =
  let arr = ref a in
  let nd0 = ndim a in
  (match s with
  | Some sv ->
      if Array.exists (fun x -> x <= 0) sv then
        invalid_arg "jax.numpy.fft: Shape should be positive."
  | None -> ());
  (match (s, axes) with
  | Some sv, Some av when Array.length sv <> Array.length av ->
      invalid_arg "jax.numpy.fft: Shape and axes have different lengths."
  | _ -> ());
  let orig_axes = axes in
  let axes =
    match axes with
    | Some av -> Array.map (resolve_axis nd0) av
    | None -> (
        match s with
        | None -> Array.init nd0 (fun i -> i)
        | Some sv ->
            Array.init (Array.length sv) (fun i -> nd0 - Array.length sv + i))
  in
  let seen = Hashtbl.create 8 in
  Array.iter
    (fun ax ->
      if Hashtbl.mem seen ax then
        invalid_arg
          (Printf.sprintf "jax.numpy.fft.%s does not support repeated axes."
             func_name);
      Hashtbl.add seen ax ())
    axes;
  let axes =
    match orig_axes with
    | None -> axes
    | Some _ ->
        let k = Array.length axes in
        let dst = Array.init k (fun i -> nd0 - k + i) in
        arr := NL.moveaxis axes dst !arr;
        dst
  in
  let s =
    match s with
    | Some sv ->
        let in_s = Array.copy (shape !arr) in
        Array.iteri (fun i ax -> in_s.(ax) <- sv.(i)) axes;
        (match fft_type with
        | T.Fft_irfft -> in_s.(nd0 - 1) <- (in_s.(nd0 - 1) / 2) + 1
        | _ -> ());
        Array.iteri
          (fun ax target ->
            if target < (shape !arr).(ax) then arr := crop_prefix !arr ax target)
          in_s;
        let cur = shape !arr in
        let pads = Array.init nd0 (fun ax -> (0, in_s.(ax) - cur.(ax))) in
        if Array.exists (fun (_, b) -> b > 0) pads then
          arr := NL.pad !arr pads 0.0;
        sv
    | None -> (
        match fft_type with
        | T.Fft_irfft ->
            let k = Array.length axes in
            if k = 0 then [||]
            else
              Array.init k (fun i ->
                  if i < k - 1 then (shape !arr).(axes.(i))
                  else max 0 (2 * ((shape !arr).(axes.(k - 1)) - 1)))
        | _ -> Array.map (fun ax -> (shape !arr).(ax)) axes)
  in
  let transformed = ref (LF.fft !arr fft_type s) in
  transformed := fft_norm s (is_inverse_name func_name) norm !transformed;
  (match orig_axes with
  | None -> ()
  | Some oav ->
      let k = Array.length axes in
      let dst = Array.init k (fun i -> nd0 - k + i) in
      transformed :=
        NL.moveaxis dst (Array.map (resolve_axis nd0) oav) !transformed);
  !transformed

let fft_core_1d func_name fft_type a n axis norm =
  let s = match n with None -> None | Some v -> Some [| v |] in
  let axes = match axis with None -> None | Some v -> Some [| v |] in
  fft_core func_name fft_type a s axes norm

let fft ?n ?(axis = -1) ?norm a =
  fft_core_1d "fft" T.Fft_fft a n (Some axis) norm

let ifft ?n ?(axis = -1) ?norm a =
  fft_core_1d "ifft" T.Fft_ifft a n (Some axis) norm

let rfft ?n ?(axis = -1) ?norm a =
  fft_core_1d "rfft" T.Fft_rfft a n (Some axis) norm

let irfft ?n ?(axis = -1) ?norm a =
  fft_core_1d "irfft" T.Fft_irfft a n (Some axis) norm

let fftn ?s ?axes ?norm a = fft_core "fftn" T.Fft_fft a s axes norm
let ifftn ?s ?axes ?norm a = fft_core "ifftn" T.Fft_ifft a s axes norm
let rfftn ?s ?axes ?norm a = fft_core "rfftn" T.Fft_rfft a s axes norm
let irfftn ?s ?axes ?norm a = fft_core "irfftn" T.Fft_irfft a s axes norm

let fft2 ?s ?(axes = [| -2; -1 |]) ?norm a =
  fft_core "fft2" T.Fft_fft a s (Some axes) norm

let fftfreq ?(d = 1.0) ?dtype n =
  let dt =
    match dtype with Some dt -> dt | None -> Dtypes.default_float_dtype ()
  in
  let denom = d *. float_of_int n in
  T.Concrete
    (Nd.of_floats dt [| n |]
       (Array.init n (fun j ->
            float_of_int (((j + (n / 2)) mod n) - (n / 2)) /. denom)))

let rfftfreq ?(d = 1.0) ?dtype n =
  let dt =
    match dtype with Some dt -> dt | None -> Dtypes.default_float_dtype ()
  in
  let denom = d *. float_of_int n in
  let m = (n / 2) + 1 in
  T.Concrete
    (Nd.of_floats dt [| m |] (Array.init m (fun j -> float_of_int j /. denom)))

let shift_of x axes sign =
  let sh = shape x in
  match axes with
  | None ->
      let ax = Array.init (Array.length sh) (fun i -> i) in
      (ax, Array.map (fun d -> sign * (d / 2)) sh)
  | Some av ->
      let ax = Array.map (resolve_axis (ndim x)) av in
      (ax, Array.map (fun a -> sign * (sh.(a) / 2)) ax)

let fftshift ?axes x =
  let ax, sh = shift_of x axes 1 in
  NL.roll ~axis:ax x sh

let ifftshift ?axes x =
  let ax, sh = shift_of x axes (-1) in
  NL.roll ~axis:ax x sh
