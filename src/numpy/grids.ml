module T = Types
module D = Dtype
module Nd = Ndarray
module NL = Lax_numpy

type range = Step of float * float * float | Count of float * float * int

let default_float () = if Config.x64_enabled () then D.F64 else D.F32
let default_int () = if Config.x64_enabled () then D.I64 else D.I32

let is_integral_range = function
  | Count _ -> false
  | Step (a, b, c) ->
      Float.is_integer a && Float.is_integer b && Float.is_integer c

let promote_grids arrays =
  match arrays with
  | [] -> []
  | first :: rest ->
      let dt =
        List.fold_left
          (fun acc v -> Dtypes.promote_types acc (Core.get_aval v).T.dtype)
          (Core.get_aval first).T.dtype rest
      in
      List.map (fun v -> NL.astype v dt) arrays

let literal x =
  if Float.is_integer x then Jnp_util.py_int (int_of_float x)
  else Jnp_util.py_float x

let axis_values = function
  | Step (start, stop, step) as r ->
      let dtype =
        if is_integral_range r then default_int () else default_float ()
      in
      NL.arange ~start ~step ~dtype stop
  | Count (start, stop, n) ->
      Array_creation.linspace_v ~num:n (literal start) (literal stop)

let mgrid ranges =
  match ranges with
  | [] -> T.Concrete (Nd.of_floats (default_int ()) [| 0 |] [||])
  | [ only ] -> axis_values only
  | _ ->
      let axes = promote_grids (List.map axis_values ranges) in
      NL.stack (NL.meshgrid ~indexing:"ij" axes)

let ogrid ranges =
  let rank = List.length ranges in
  List.mapi
    (fun i v ->
      let n = (Core.get_aval v).T.shape.(0) in
      Core.bind1
        (T.Broadcast_in_dim
           {
             shape = Array.init rank (fun d -> if d = i then n else 1);
             dims = [| i |];
           })
        [ v ])
    (promote_grids (List.map axis_values ranges))

let promote_all arrays =
  let dt =
    List.fold_left
      (fun acc v -> Dtypes.promote_types acc (Core.get_aval v).T.dtype)
      (Core.get_aval (List.hd arrays)).T.dtype (List.tl arrays)
  in
  List.map (fun v -> NL.astype v dt) arrays

let r_ arrays =
  let lift v =
    if Array.length (Core.get_aval v).T.shape = 0 then NL.expand_dims v [| 0 |]
    else v
  in
  NL.concatenate ~axis:0 (promote_all (List.map lift arrays))

let c_ arrays =
  let lift v =
    match Array.length (Core.get_aval v).T.shape with
    | 0 -> Core.bind1 (T.Transpose [| 1; 0 |]) [ NL.expand_dims v [| 0; 1 |] ]
    | 1 -> Core.bind1 (T.Transpose [| 1; 0 |]) [ NL.expand_dims v [| 0 |] ]
    | _ -> v
  in
  NL.concatenate ~axis:(-1) (promote_all (List.map lift arrays))
