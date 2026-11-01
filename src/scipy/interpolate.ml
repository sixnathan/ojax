open Types
module L = Lax_api
module NL = Numpy.Lax_numpy
module U = Numpy.Ufuncs

type t = {
  points : value list;
  values : value;
  method_ : string;
  bounds_error : bool;
  fill_value : float option;
}

let shape v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype

let regular_grid_interpolator ?(method_ = "linear") ?(bounds_error = false)
    ?(fill_value = Some Float.nan) points values =
  if bounds_error then
    invalid_arg "`bounds_error` takes no effect under GPU and TPU";
  if method_ <> "linear" && method_ <> "nearest" then
    invalid_arg ("method " ^ method_ ^ " is not defined");
  if List.length points > Array.length (shape values) then
    invalid_arg "There are more dimensions in points than in values";
  { points; values; method_; bounds_error; fill_value }

let find_indices point x =
  let n = (shape point).(0) in
  let i =
    NL.clip ~min:1.0
      ~max:(float_of_int (n - 1))
      (NL.searchsorted ~side:"right" point x)
  in
  let lower = U.subtract i (L.full_like i 1.0) in
  let lo = Numpy.Indexing.take_index ~axis:0 point lower in
  let hi = Numpy.Indexing.take_index ~axis:0 point i in
  let weight = U.divide (U.subtract x lo) (U.subtract hi lo) in
  (lower, weight)

let unravel_grid t indices =
  let out_shape = shape (List.hd indices) in
  let n_points = Array.fold_left ( * ) 1 out_shape in
  let vshape = shape t.values in
  let d = List.length t.points in
  let tail = Array.sub vshape d (Array.length vshape - d) in
  let tail_size = Array.fold_left ( * ) 1 tail in
  let grid = Array.sub vshape 0 d in
  let strides = Array.make d 1 in
  for k = d - 2 downto 0 do
    strides.(k) <- strides.(k + 1) * grid.(k + 1)
  done;
  let flat_index =
    List.fold_left
      (fun acc (k, idx) ->
        U.add acc
          (U.multiply
             (NL.reshape idx [| n_points |])
             (L.full [| n_points |] (float_of_int strides.(k)) (dtype_of idx))))
      (L.full [| n_points |] 0.0 (dtype_of (List.hd indices)))
      (List.mapi (fun k idx -> (k, idx)) indices)
  in
  let cells = Array.fold_left ( * ) 1 grid in
  let flat_values = NL.reshape t.values [| cells; tail_size |] in
  let picked = Numpy.Indexing.take_index ~axis:0 flat_values flat_index in
  (picked, out_shape, tail)

let expand_weight w tail_rank =
  NL.reshape w (Array.append (shape w) (Array.make tail_rank 1))

let evaluate t xi =
  let d = List.length t.points in
  let xshape = shape xi in
  let n = Array.length xshape in
  if xshape.(n - 1) <> d then
    invalid_arg "the requested sample points xi have the wrong dimension";
  let coords =
    List.init d (fun k ->
        let c = L.index_in_dim ~axis:(n - 1) ~keepdims:false xi k in
        NL.astype c (dtype_of (List.hd t.points)))
  in
  let found = List.map2 find_indices t.points coords in
  let result =
    match t.method_ with
    | "nearest" ->
        let chosen =
          List.map
            (fun (lower, w) ->
              NL.where_
                (U.less_equal w (L.full_like w 0.5))
                lower
                (U.add lower (L.full_like lower 1.0)))
            found
        in
        let picked, out_shape, tail = unravel_grid t chosen in
        NL.reshape picked (Array.append out_shape tail)
    | _ ->
        let rec corners = function
          | [] -> [ ([], None) ]
          | (lower, w) :: rest ->
              let later = corners rest in
              let one = L.full_like w 1.0 in
              List.concat_map
                (fun (idx, weight) ->
                  let combine extra =
                    match weight with
                    | None -> Some extra
                    | Some acc -> Some (U.multiply extra acc)
                  in
                  [
                    (lower :: idx, combine (U.subtract one w));
                    (U.add lower (L.full_like lower 1.0) :: idx, combine w);
                  ])
                later
        in
        let terms =
          List.map
            (fun (idx, weight) ->
              let picked, out_shape, tail = unravel_grid t idx in
              let shaped = NL.reshape picked (Array.append out_shape tail) in
              let w =
                match weight with
                | Some w -> w
                | None -> failwith "interpolate: no dimensions"
              in
              let ew = expand_weight w (Array.length tail) in
              U.multiply (NL.astype ew (dtype_of shaped)) shaped)
            (corners found)
        in
        List.fold_left U.add (List.hd terms) (List.tl terms)
  in
  match t.fill_value with
  | None -> result
  | Some fill ->
      let tail_rank =
        Array.length (shape result) - Array.length (shape (List.hd coords))
      in
      let outside =
        List.fold_left2
          (fun acc c point ->
            let m = (shape point).(0) in
            let low = L.index_in_dim ~keepdims:false point 0 in
            let high = L.index_in_dim ~keepdims:false point (m - 1) in
            let broad v = L.broadcast_in_dim v (shape c) [||] in
            U.logical_or acc
              (U.logical_or (U.less c (broad low)) (U.greater c (broad high))))
          (L.full (shape (List.hd coords)) 0.0 Dtype.Bool)
          coords t.points
      in
      let mask =
        NL.reshape outside
          (Array.append (shape outside) (Array.make tail_rank 1))
      in
      NL.where_
        (NL.broadcast_to mask (shape result))
        (L.full (shape result) fill (dtype_of result))
        result
