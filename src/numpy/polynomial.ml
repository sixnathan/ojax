module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module UF = Ufuncs
module NL = Lax_numpy
module R = Reductions

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let weak v = (get_aval v).T.weak_type
let bind1 = C.bind1
let prod sh = Array.fold_left ( * ) 1 sh
let const_full dt sh x = Core.splat dt sh x
let scalar dt x = const_full dt [||] x

let convert v dt =
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let is_inexact = function
  | D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn | D.F8_e3m4
  | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz | D.F8_e5m2
  | D.F8_e5m2fnuz | D.F8_e8m0fnu | D.F32 | D.F64 | D.Complex64 | D.Complex128 ->
      true
  | _ -> false

let to_inexact_dtype = Dtypes.to_inexact_dtype

let result_dtype vs =
  fst (Dtypes.result_type (List.map (fun v -> (dtype v, weak v)) vs))

let promote_dtypes vs =
  let dt = result_dtype vs in
  List.map (fun v -> convert v dt) vs

let promote_dtypes_inexact vs =
  let dt = to_inexact_dtype (result_dtype vs) in
  List.map (fun v -> convert v dt) vs

let two = function [ a; b ] -> (a, b) | _ -> assert false

let slice_axis v axis lo hi =
  let sh = shape v in
  let n = Array.length sh in
  let start = Array.make n 0 and limit = Array.copy sh in
  start.(axis) <- lo;
  limit.(axis) <- hi;
  bind1
    (T.Slice { start_indices = start; limit_indices = limit; strides = None })
    [ v ]

let neg v = bind1 T.Neg [ v ]

let polyval ?(unroll = 16) p x =
  ignore unroll;
  let p, x = two (promote_dtypes [ p; x ]) in
  let sh_p = shape p in
  let tail = Array.sub sh_p 1 (Array.length sh_p - 1) in
  let out_sh = NL.broadcast_shapes_n [ tail; shape x ] in
  let y0 = const_full (dtype x) out_sh 0.0 in
  let body carry =
    match carry with
    | [ y; pk ] -> [ UF.add (UF.multiply y x) pk ]
    | _ -> assert false
  in
  match Lax.scan body [ y0 ] [ p ] with [ y ] -> y | _ -> assert false

let pad_leading v k =
  if k = 0 then v
  else begin
    let d = Array.length (shape v) in
    let p = Array.make d (0, 0) in
    p.(0) <- (k, 0);
    NL.pad v p 0.0
  end

let polyadd a1 a2 =
  let a1, a2 = two (promote_dtypes [ a1; a2 ]) in
  let n1 = (shape a1).(0) and n2 = (shape a2).(0) in
  let n = max n1 n2 in
  bind1 T.Add [ pad_leading a1 (n - n1); pad_leading a2 (n - n2) ]

let polysub a1 a2 =
  let a1, a2 = two (promote_dtypes [ a1; a2 ]) in
  polyadd a1 (neg a2)

let polyint ?(m = 1) ?k p =
  if m < 0 then invalid_arg "polyint: order of integral must be non-negative";
  let k = match k with Some k -> k | None -> scalar D.I32 0.0 in
  let p =
    if dtype p = D.Bool then NL.astype p (Dtypes.default_float_dtype ()) else p
  in
  let p, k = two (promote_dtypes_inexact [ p; k ]) in
  if m = 0 then p
  else begin
    let dt = dtype p in
    let np_ = (shape p).(0) in
    let k = NL.atleast_1d k in
    let k = if (shape k).(0) = 1 then NL.broadcast_to k [| m |] else k in
    if (shape k).(0) <> m then
      invalid_arg "polyint: k must be a scalar or a rank-1 array of length m";
    let grid_a =
      NL.expand_dims (NL.arange ~dtype:dt (float_of_int (np_ + m))) [| 0 |]
    in
    let grid_b =
      NL.expand_dims (NL.arange ~dtype:dt (float_of_int m)) [| 1 |]
    in
    let grid = UF.subtract grid_a grid_b in
    let coeff =
      NL.flip ~axis:[| 0 |]
        (R.prod ~axis:[| 0 |] (UF.maximum (scalar dt 1.0) grid))
    in
    UF.true_divide (NL.concatenate [ p; k ]) coeff
  end

let polyder ?(m = 1) p =
  if m < 0 then invalid_arg "polyder: order of derivative must be non-negative";
  let p = List.hd (promote_dtypes_inexact [ p ]) in
  if m = 0 then p
  else begin
    let dt = dtype p in
    let np_ = (shape p).(0) in
    let grid_a =
      NL.expand_dims
        (NL.arange ~start:(float_of_int m) ~dtype:dt (float_of_int np_))
        [| 0 |]
    in
    let grid_b =
      NL.expand_dims (NL.arange ~dtype:dt (float_of_int m)) [| 1 |]
    in
    let coeff =
      NL.flip ~axis:[| 0 |] (R.prod ~axis:[| 0 |] (UF.subtract grid_a grid_b))
    in
    bind1 T.Mul [ slice_axis p 0 0 (np_ - m); coeff ]
  end

let poly seq =
  let s = NL.atleast_1d (List.hd (promote_dtypes_inexact [ seq ])) in
  let sh = shape s in
  let s =
    if Array.length sh = 2 && sh.(0) = sh.(1) && sh.(0) <> 0 then
      Linalg.eigvals s
    else s
  in
  if Array.length (shape s) <> 1 then
    invalid_arg "input must be 1d or non-empty square 2d array.";
  let dt = dtype s in
  let n = (shape s).(0) in
  if n = 0 then const_full dt [||] 1.0
  else begin
    let a = ref (const_full dt [| 1 |] 1.0) in
    for k = 0 to n - 1 do
      let sk = slice_axis s 0 k (k + 1) in
      let kernel = NL.concatenate [ const_full dt [| 1 |] 1.0; neg sk ] in
      a := NL.convolve ~mode:"full" !a kernel
    done;
    !a
  end

let trim_front v =
  match v with
  | T.Tracer _ ->
      invalid_arg "polymul: trim_leading_zeros needs a concrete array"
  | _ ->
      let n = (shape v).(0) in
      let nd =
        match v with
        | T.Concrete nd -> nd
        | T.Device b -> Pjrt.Buffer.to_host b
        | T.Tracer _ -> assert false
      in
      let rec first i =
        if i < n && Nd.get_c nd [| i |] = Complex.zero then first (i + 1) else i
      in
      let start = first 0 in
      if start = 0 then v
      else
        bind1
          (T.Slice
             {
               start_indices = [| start |];
               limit_indices = [| n |];
               strides = None;
             })
          [ v ]

let polymul ?(trim_leading_zeros = false) a1 a2 =
  let a1, a2 = two (promote_dtypes_inexact [ a1; a2 ]) in
  let a1, a2 =
    if trim_leading_zeros && ((shape a1).(0) > 1 || (shape a2).(0) > 1) then
      (trim_front a1, trim_front a2)
    else (a1, a2)
  in
  let a1 =
    if (shape a1).(0) = 0 then const_full (dtype a2) [| 1 |] 0.0 else a1
  in
  let a2 =
    if (shape a2).(0) = 0 then const_full (dtype a1) [| 1 |] 0.0 else a2
  in
  NL.convolve ~mode:"full" a1 a2

let poly p0 =
  Jnp_util.jit1 "poly"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0 ] -> poly p0 | _ -> invalid_arg "poly: arity")
    [ p0 ]

let polyval ?unroll p0 p1 =
  Jnp_util.jit1 "polyval"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string unroll [])))
    (function
      | [ p0; p1 ] -> polyval ?unroll p0 p1 | _ -> invalid_arg "polyval: arity")
    [ p0; p1 ]

let polyadd p0 p1 =
  Jnp_util.jit1 "polyadd"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> polyadd p0 p1 | _ -> invalid_arg "polyadd: arity")
    [ p0; p1 ]

let polyint ?m ?k p0 =
  let k_given = k <> None in
  if k_given then polyint ?m ?k p0
  else
    Jnp_util.jit1 "polyint"
      ~key:(Digest.to_hex (Digest.string (Marshal.to_string m [])))
      (function
        | [ p0 ] -> polyint ?m ?k p0 | _ -> invalid_arg "polyint: arity")
      [ p0 ]

let polyder ?m p0 =
  Jnp_util.jit1 "polyder"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string m [])))
    (function [ p0 ] -> polyder ?m p0 | _ -> invalid_arg "polyder: arity")
    [ p0 ]

let polysub p0 p1 =
  Jnp_util.jit1 "polysub"
    ~key:(Digest.to_hex (Digest.string (Marshal.to_string () [])))
    (function [ p0; p1 ] -> polysub p0 p1 | _ -> invalid_arg "polysub: arity")
    [ p0; p1 ]

let poly a0 =
  Eager_jit.one "numpy.polynomial.poly" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> poly a0 | _ -> assert false)

let polyval ?unroll a0 a1 =
  Eager_jit.one "numpy.polynomial.polyval" unroll [] [ a0; a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> polyval ?unroll a0 a1
      | _ -> assert false)

let polyadd a0 a1 =
  Eager_jit.one "numpy.polynomial.polyadd" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> polyadd a0 a1
      | _ -> assert false)

let polyint ?m ?k a0 =
  Eager_jit.one "numpy.polynomial.polyint" m [ k ] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ k ], [ a0 ] -> polyint ?m ?k a0
      | _ -> assert false)

let polyder ?m a0 =
  Eager_jit.one "numpy.polynomial.polyder" m [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> polyder ?m a0 | _ -> assert false)

let polysub a0 a1 =
  Eager_jit.one "numpy.polynomial.polysub" () [] [ a0; a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0; a1 ] -> polysub a0 a1
      | _ -> assert false)
