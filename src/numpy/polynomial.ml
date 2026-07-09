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

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (prod sh) x))

let scalar dt x = const_full dt [||] x

let convert v dt =
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let is_inexact = function D.F32 | D.F64 -> true | _ -> false

let to_inexact_dtype dt =
  if is_inexact dt then dt else Dtypes.default_float_dtype ()

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
  if Array.length (shape s) <> 1 then
    invalid_arg
      "poly: input must be 1d (square-matrix eigvals path deferred to M5)";
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

let polymul a1 a2 =
  let a1, a2 = two (promote_dtypes_inexact [ a1; a2 ]) in
  let a1 =
    if (shape a1).(0) = 0 then const_full (dtype a2) [| 1 |] 0.0 else a1
  in
  let a2 =
    if (shape a2).(0) = 0 then const_full (dtype a1) [| 1 |] 0.0 else a2
  in
  NL.convolve ~mode:"full" a1 a2
