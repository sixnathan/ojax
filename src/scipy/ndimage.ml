module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Numpy.Lax_numpy
module UF = Numpy.Ufuncs

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let ndim v = Array.length (shape v)
let bind1 = C.bind1
let numel sh = Array.fold_left ( * ) 1 sh

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (numel sh) x))

let scalar dt x = const_full dt [||] x

let is_integer = function
  | D.I32 | D.I64 | D.Uint32 -> true
  | D.F32 | D.F64 | D.Bool | D.Complex64 | D.Complex128 -> false

let round_half_away a = if is_integer (dtype a) then a else bind1 T.Round [ a ]
let to_i32 v = NL.astype v D.I32

let nearest_nodes coordinate =
  let index = to_i32 (round_half_away coordinate) in
  let weight = scalar (dtype coordinate) 1.0 in
  [ (index, weight) ]

let linear_nodes coordinate =
  let cdt = dtype coordinate in
  let lower = UF.floor coordinate in
  let upper_weight = UF.subtract coordinate lower in
  let lower_weight = UF.subtract (scalar cdt 1.0) upper_weight in
  let index = to_i32 lower in
  [ (index, lower_weight); (UF.add index (scalar D.I32 1.0), upper_weight) ]

let mirror_index_fixer index size =
  let idt = dtype index in
  let ci k = scalar idt (float_of_int k) in
  let s = size - 1 in
  UF.abs (UF.subtract (UF.remainder (UF.add index (ci s)) (ci (2 * s))) (ci s))

let reflect_index_fixer index size =
  let idt = dtype index in
  let ci k = scalar idt (float_of_int k) in
  let m =
    mirror_index_fixer
      (UF.add (UF.multiply (ci 2) index) (ci 1))
      ((2 * size) + 1)
  in
  UF.floor_divide (UF.subtract m (ci 1)) (ci 2)

let apply_index_fixer mode index size =
  let idt = dtype index in
  match mode with
  | "constant" -> index
  | "nearest" ->
      UF.minimum
        (UF.maximum index (scalar idt 0.0))
        (scalar idt (float_of_int (size - 1)))
  | "wrap" -> UF.remainder index (scalar idt (float_of_int size))
  | "mirror" -> mirror_index_fixer index size
  | "reflect" -> reflect_index_fixer index size
  | _ ->
      failwith
        (Printf.sprintf
           "jax.scipy.ndimage.map_coordinates does not yet support mode %s" mode)

let is_valid_constant index size =
  let idt = dtype index in
  UF.logical_and
    (UF.greater_equal index (scalar idt 0.0))
    (UF.less index (scalar idt (float_of_int size)))

let gather_nd input idx_arrays =
  let idxs = NL.broadcast_arrays idx_arrays in
  let s = shape (List.hd idxs) in
  let stacked = NL.stack ~axis:(Array.length s) idxs in
  let ash = shape input in
  let r = Array.length ash in
  let dnums =
    {
      T.offset_dims = [||];
      collapsed_slice_dims = Array.init r Fun.id;
      start_index_map = Array.init r Fun.id;
      g_operand_batching_dims = [||];
      g_start_indices_batching_dims = [||];
    }
  in
  bind1
    (T.Gather { dimension_numbers = dnums; slice_sizes = Array.make r 1 })
    [ input; stacked ]

let rec cartesian = function
  | [] -> [ [] ]
  | xs :: rest ->
      let tails = cartesian rest in
      List.concat_map (fun x -> List.map (fun t -> x :: t) tails) xs

let map_coordinates ?(mode = "constant") ?(cval = 0.0) input coordinates ~order
    =
  let n = ndim input in
  if List.length coordinates <> n then
    invalid_arg
      (Printf.sprintf
         "coordinates must be a sequence of length input.ndim, but %d != %d"
         (List.length coordinates) n);
  let idt = dtype input in
  let cval_v = scalar idt cval in
  let interp =
    match order with
    | 0 -> nearest_nodes
    | 1 -> linear_nodes
    | _ ->
        failwith "jax.scipy.ndimage.map_coordinates currently requires order<=1"
  in
  let sizes = Array.to_list (shape input) in
  let per_dim =
    List.map2
      (fun coordinate size ->
        List.map
          (fun (index, weight) ->
            let fixed = apply_index_fixer mode index size in
            let valid =
              if mode = "constant" then Some (is_valid_constant index size)
              else None
            in
            (fixed, valid, weight))
          (interp coordinate))
      coordinates sizes
  in
  let outputs =
    List.map
      (fun items ->
        let indices = List.map (fun (i, _, _) -> i) items in
        let validities = List.filter_map (fun (_, v, _) -> v) items in
        let weights = List.map (fun (_, _, w) -> w) items in
        let gathered = gather_nd input indices in
        let contribution =
          match validities with
          | [] -> gathered
          | v0 :: vs ->
              let all_valid = List.fold_left UF.logical_and v0 vs in
              NL.where_ all_valid gathered cval_v
        in
        let wprod =
          match weights with
          | w0 :: ws -> List.fold_left UF.multiply w0 ws
          | [] -> assert false
        in
        UF.multiply wprod contribution)
      (cartesian per_dim)
  in
  let result =
    match outputs with
    | o0 :: os -> List.fold_left UF.add o0 os
    | [] -> assert false
  in
  let result = if is_integer idt then round_half_away result else result in
  NL.astype result idt
