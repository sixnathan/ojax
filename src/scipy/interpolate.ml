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

let py_int = Numpy.Jnp_util.py_int
let py_float = Numpy.Jnp_util.py_float

let normalized i dim =
  let dt = dtype_of i in
  let lit v = Concrete (Ndarray.of_floats dt [||] [| float_of_int v |]) in
  let negative = Core.bind1 Lt [ i; lit 0 ] in
  let wrapped = Core.bind1 Add [ i; lit dim ] in
  Core.bind1 Select_n [ negative; i; wrapped ]

let take_at values indices =
  let vshape = shape values in
  let d = List.length indices in
  let starts = List.mapi (fun k i -> normalized i vshape.(k)) indices in
  let batch = (shape (List.hd starts)).(0) in
  let columns =
    List.map (fun i -> L.broadcast_in_dim i [| batch; 1 |] [| 0 |]) starts
  in
  let index =
    match columns with
    | [ one ] -> one
    | many -> Core.bind1 (Concatenate 1) many
  in
  let tail = Array.length vshape - d in
  let dims = Array.init d Fun.id in
  Core.bind1
    (Gather
       {
         dimension_numbers =
           {
             offset_dims = Array.init tail (fun k -> k + 1);
             collapsed_slice_dims = dims;
             start_index_map = dims;
             g_operand_batching_dims = [||];
             g_start_indices_batching_dims = [||];
           };
         slice_sizes =
           Array.mapi (fun k size -> if k < d then 1 else size) vshape;
       })
    [ values; index ]

let scalar_at v k =
  let n = (shape v).(0) in
  let k = if k < 0 then k + n else k in
  Core.bind1 (Squeeze [| 0 |]) [ L.slice v [| k |] [| k + 1 |] ]

let find_indices xi_t points =
  let n_points = (shape xi_t).(1) in
  let out_of_bounds =
    Numpy.Array_creation.zeros ~dtype:Dtype.Bool [| n_points |]
  in
  let _, indices, distances, out_of_bounds =
    List.fold_left
      (fun (k, indices, distances, oob) g ->
        let row =
          L.slice ~strides:[| 1; 1 |] xi_t [| k; 0 |] [| k + 1; n_points |]
        in
        let x = Core.bind1 (Squeeze [| 0 |]) [ row ] in
        let size = (shape g).(0) in
        let i = U.subtract (NL.searchsorted g x) (py_int 1) in
        let i = NL.where_ (U.less i (py_int 0)) (py_int 0) i in
        let i =
          NL.where_ (U.greater i (py_int (size - 2))) (py_int (size - 2)) i
        in
        let numerator = U.subtract x (take_at g [ i ]) in
        let upper = take_at g [ U.add i (py_int 1) ] in
        let lower = take_at g [ i ] in
        let distance = U.divide numerator (U.subtract upper lower) in
        let oob = U.logical_or oob (U.less x (scalar_at g 0)) in
        let oob = U.logical_or oob (U.greater x (scalar_at g (-1))) in
        (k + 1, i :: indices, distance :: distances, oob))
      (0, [], [], out_of_bounds) points
  in
  (List.rev indices, List.rev distances, out_of_bounds)

let evaluate_linear t indices distances =
  let tail_rank = Array.length (shape t.values) - List.length indices in
  let options = List.map (fun i -> [ i; U.add i (py_int 1) ]) indices in
  let rec product = function
    | [] -> [ [] ]
    | first :: rest ->
        let later = product rest in
        List.concat_map (fun v -> List.map (fun tl -> v :: tl) later) first
  in
  List.fold_left
    (fun values edge ->
      let weight =
        List.fold_left2
          (fun weight (ei, i) yi ->
            let chosen =
              NL.where_ (U.equal ei i) (U.subtract (py_float 1.0) yi) yi
            in
            U.multiply weight chosen)
          (py_float 1.0)
          (List.combine edge indices)
          distances
      in
      let weight =
        if tail_rank = 0 then weight
        else NL.expand_dims weight (Array.init tail_rank (fun k -> k + 1))
      in
      U.add values (U.multiply (take_at t.values edge) weight))
    (py_float 0.0) (product options)

let evaluate t xi =
  let d = List.length t.points in
  let xshape = shape xi in
  let n = Array.length xshape in
  let xi = if n = 1 then NL.reshape xi [| -1; d |] else xi in
  let xshape = shape xi in
  let n = Array.length xshape in
  if xshape.(n - 1) <> d then
    invalid_arg "the requested sample points xi have the wrong dimension";
  let flat = NL.reshape xi [| -1; xshape.(n - 1) |] in
  let indices, distances, out_of_bounds =
    find_indices (NL.transpose flat) t.points
  in
  let result =
    match t.method_ with
    | "nearest" ->
        let chosen =
          List.map2
            (fun i yi ->
              NL.where_ (U.less_equal yi (py_float 0.5)) i (U.add i (py_int 1)))
            indices distances
        in
        take_at t.values chosen
    | _ -> evaluate_linear t indices distances
  in
  let result =
    match t.fill_value with
    | None -> result
    | Some fill ->
        let r = Array.length (shape result) in
        let mask_shape =
          Array.init r (fun k -> if k = 0 then (shape result).(0) else 1)
        in
        NL.where_ (NL.reshape out_of_bounds mask_shape) (py_float fill) result
  in
  NL.reshape result
    (Array.append
       (Array.sub xshape 0 (n - 1))
       (Array.sub (shape t.values) d (Array.length (shape t.values) - d)))
