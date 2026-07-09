module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Numpy.Lax_numpy

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let bind1 = C.bind1

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (Array.fold_left ( * ) 1 sh) x))

let concrete_max_index v =
  match v with
  | T.Concrete nd ->
      let m = Nd.fold (fun acc x -> Float.max acc x) Float.neg_infinity nd in
      int_of_float m
  | T.Tracer _ ->
      invalid_arg "segment: num_segments must be static for traced segment_ids"

let identity_sum = 0.0
let identity_prod = 1.0

let identity_max dt =
  match dt with
  | D.F32 | D.F64 -> Float.neg_infinity
  | D.I32 -> -2147483648.0
  | D.I64 -> -9223372036854775808.0
  | D.Uint32 -> 0.0
  | D.Bool -> 0.0

let identity_min dt =
  match dt with
  | D.F32 | D.F64 -> Float.infinity
  | D.I32 -> 2147483647.0
  | D.I64 -> 9223372036854775807.0
  | D.Uint32 -> 4294967295.0
  | D.Bool -> 1.0

let segment_dnums rank =
  {
    T.update_window_dims = Array.init (rank - 1) (fun i -> i + 1);
    inserted_window_dims = [| 0 |];
    scatter_dims_to_operand_dims = [| 0 |];
    s_operand_batching_dims = [||];
    s_scatter_indices_batching_dims = [||];
  }

let segment_update prim identity ?num_segments data segment_ids =
  let dt = dtype data in
  let dsh = shape data in
  let n = if Array.length dsh = 0 then 0 else dsh.(0) in
  let nseg =
    match num_segments with
    | Some k -> k
    | None -> concrete_max_index segment_ids + 1
  in
  if nseg < 0 then invalid_arg "num_segments must be non-negative.";
  let out_shape = Array.copy dsh in
  if Array.length out_shape > 0 then out_shape.(0) <- nseg;
  let out = const_full dt out_shape identity in
  let scatter_indices = NL.reshape segment_ids [| n; 1 |] in
  let dnums = segment_dnums (Array.length dsh) in
  bind1 (prim dnums) [ out; scatter_indices; data ]

let segment_sum ?num_segments data segment_ids =
  segment_update
    (fun dnums -> T.Scatter_add { dimension_numbers = dnums })
    identity_sum ?num_segments data segment_ids

let segment_prod ?num_segments data segment_ids =
  segment_update
    (fun dnums ->
      T.Scatter_mul { dimension_numbers = dnums; unique_indices = false })
    identity_prod ?num_segments data segment_ids

let segment_max ?num_segments data segment_ids =
  segment_update
    (fun dnums -> T.Scatter_max { dimension_numbers = dnums })
    (identity_max (dtype data))
    ?num_segments data segment_ids

let segment_min ?num_segments data segment_ids =
  segment_update
    (fun dnums -> T.Scatter_min { dimension_numbers = dnums })
    (identity_min (dtype data))
    ?num_segments data segment_ids
