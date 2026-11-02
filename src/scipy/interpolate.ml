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
    ?(fill_value = Some Float_format.nan) points values =
  if bounds_error then
    raise
      (Py_exn.Not_implemented_error "`bounds_error` takes no effect under JIT");
  if method_ <> "linear" && method_ <> "nearest" then
    Py_exn.value_error ("method '" ^ method_ ^ "' is not defined");
  if List.length points > Array.length (shape values) then
    Py_exn.value_error
      (Printf.sprintf "There are %d point arrays, but values has %d dimensions"
         (List.length points)
         (Array.length (shape values)));
  { points; values; method_; bounds_error; fill_value }

let py_int = Numpy.Jnp_util.py_int
let py_float = Numpy.Jnp_util.py_float
let where = NL.where_
let take1 g i = Numpy.Extras.fancy_take g [ i ]

let find_indices out_of_bounds x g =
  let i = U.subtract (NL.searchsorted g x) (py_int 1) in
  let i = where (U.less i (py_int 0)) (py_int 0) i in
  let size = (shape g).(0) in
  let i = where (U.greater i (py_int (size - 2))) (py_int (size - 2)) i in
  let lower = take1 g i in
  let delta = U.subtract x lower in
  let next = take1 g (U.add i (py_int 1)) in
  let base = take1 g i in
  let norm = U.true_divide delta (U.subtract next base) in
  let first = NL.index_at g 0 in
  let bounds = U.add out_of_bounds (U.less x first) in
  let last = NL.index_at g (size - 1) in
  let bounds = U.add bounds (U.greater x last) in
  (i, norm, bounds)

let rec edge_product = function
  | [] -> [ [] ]
  | (i, up) :: rest ->
      let later = edge_product rest in
      List.map (fun tail -> i :: tail) later
      @ List.map (fun tail -> up :: tail) later

let weight_slice t d weight =
  let extra = Array.length (shape t.values) - d in
  if extra = 0 then weight
  else L.expand_dims weight (Array.init extra (fun k -> k + 1))

let evaluate_linear t indices norms =
  let d = List.length indices in
  let uppers = List.map (fun i -> U.add i (py_int 1)) indices in
  let edges = edge_product (List.combine indices uppers) in
  List.fold_left
    (fun acc edge ->
      let weight =
        List.fold_left2
          (fun w (ei, i) yi ->
            let same = U.equal ei i in
            let complement = U.subtract (py_float 1.0) yi in
            U.multiply w (where same complement yi))
          (py_float 1.0)
          (List.combine edge indices)
          norms
      in
      let picked = Numpy.Extras.fancy_take t.values edge in
      U.add acc (U.multiply picked (weight_slice t d weight)))
    (py_float 0.0) edges

let evaluate_nearest t indices norms =
  let chosen =
    List.map2
      (fun i yi ->
        let near = U.less_equal yi (py_float 0.5) in
        where near i (U.add i (py_int 1)))
      indices norms
  in
  Numpy.Extras.fancy_take t.values chosen

let evaluate t xi =
  let d = List.length t.points in
  let xshape = shape xi in
  let n = Array.length xshape in
  if Validate.tuple_index xshape (-1) <> d then
    Py_exn.value_error
      (Printf.sprintf
         "the requested sample points xi have dimension %d, but this \
          RegularGridInterpolator has dimension %d"
         (Validate.tuple_index xshape 1)
         d);
  let flat_shape =
    [| Array.fold_left ( * ) 1 xshape / xshape.(n - 1); xshape.(n - 1) |]
  in
  let xi_flat = if xshape = flat_shape then xi else NL.reshape xi flat_shape in
  let rows = flat_shape.(0) in
  let transposed = NL.transpose xi_flat in
  let out_of_bounds = ref (L.full [| rows |] 0.0 Dtype.Bool) in
  let found =
    List.mapi
      (fun k g ->
        let sliced =
          L.slice ~strides:[| 1; 1 |] transposed [| k; 0 |] [| k + 1; rows |]
        in
        let x = L.squeeze sliced [| 0 |] in
        let i, norm, bounds = find_indices !out_of_bounds x g in
        out_of_bounds := bounds;
        (i, norm))
      t.points
  in
  let indices = List.map fst found and norms = List.map snd found in
  let result =
    match t.method_ with
    | "nearest" -> evaluate_nearest t indices norms
    | _ -> evaluate_linear t indices norms
  in
  let result =
    match t.fill_value with
    | None -> result
    | Some fill ->
        let rs = shape result in
        let bc =
          Array.init (Array.length rs) (fun k -> if k = 0 then rs.(0) else 1)
        in
        let mask =
          if bc = shape !out_of_bounds then !out_of_bounds
          else NL.reshape !out_of_bounds bc
        in
        where mask (py_float fill) result
  in
  let vs = shape t.values in
  let final =
    Array.append
      (Array.sub xshape 0 (n - 1))
      (Array.sub vs d (Array.length vs - d))
  in
  if shape result = final then result else NL.reshape result final
