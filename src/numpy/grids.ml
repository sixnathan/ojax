module T = Types
module D = Dtype
module Nd = Ndarray
module NL = Lax_numpy

type range = Step of float * float * float | Count of float * float * int

let default_float () = if Config.x64_enabled () then D.F64 else D.F32
let default_int () = if Config.x64_enabled () then D.I64 else D.I32

let all_integral ranges =
  List.for_all
    (function
      | Count _ -> false
      | Step (a, b, c) ->
          Float.is_integer a && Float.is_integer b && Float.is_integer c)
    ranges

let axis_values dt = function
  | Step (start, stop, step) ->
      let n =
        Stdlib.max 0 (int_of_float (Float.ceil ((stop -. start) /. step)))
      in
      T.Concrete
        (Nd.of_floats dt [| n |]
           (Array.init n (fun i -> start +. (float_of_int i *. step))))
  | Count (start, stop, n) ->
      let gap =
        if n > 1 then (stop -. start) /. float_of_int (n - 1) else 0.0
      in
      T.Concrete
        (Nd.of_floats dt [| n |]
           (Array.init n (fun i -> start +. (float_of_int i *. gap))))

let range_dtype ranges =
  if all_integral ranges then default_int () else default_float ()

let mgrid ranges =
  let dt = range_dtype ranges in
  match ranges with
  | [] -> T.Concrete (Nd.of_floats dt [| 0 |] [||])
  | [ only ] -> axis_values dt only
  | _ ->
      let axes = List.map (axis_values dt) ranges in
      NL.stack (NL.meshgrid ~indexing:"ij" axes)

let ogrid ranges =
  let dt = range_dtype ranges in
  let rank = List.length ranges in
  List.mapi
    (fun i r ->
      let v = axis_values dt r in
      let n = (Core.get_aval v).T.shape.(0) in
      NL.reshape v (Array.init rank (fun d -> if d = i then n else 1)))
    ranges

let promote_all arrays =
  let dt =
    List.fold_left
      (fun acc v -> Dtypes.promote_types acc (Core.get_aval v).T.dtype)
      (Core.get_aval (List.hd arrays)).T.dtype (List.tl arrays)
  in
  List.map (fun v -> NL.astype v dt) arrays

let r_ arrays =
  let lift v =
    if Array.length (Core.get_aval v).T.shape = 0 then NL.reshape v [| 1 |]
    else v
  in
  NL.concatenate ~axis:0 (promote_all (List.map lift arrays))

let c_ arrays =
  let lift v =
    let sh = (Core.get_aval v).T.shape in
    match Array.length sh with
    | 0 -> NL.reshape v [| 1; 1 |]
    | 1 -> NL.reshape v [| sh.(0); 1 |]
    | _ -> v
  in
  NL.concatenate ~axis:(-1) (promote_all (List.map lift arrays))
