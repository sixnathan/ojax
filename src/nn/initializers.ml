open Types
module Nd = Ndarray
module C = Core
module D = Dtype
module RC = Random.Core

type t = value -> shape:int array -> value
type initializer_ = t
type mode = Fan_in | Fan_out | Fan_avg | Fan_geo_avg
type distribution = Truncated_normal | Normal | Uniform

let prod a = Array.fold_left ( * ) 1 a
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let concrete = function
  | Concrete nd -> nd
  | Tracer _ ->
      failwith "nn/initializers: initializer on a tracer not supported"
  | Device _ ->
      failwith "nn/initializers: initializer on a tracer not supported"

let const shape x = C.splat D.F32 shape x

let compute_fans shape ~in_axis ~out_axis ~batch_axis =
  let ndim = Array.length shape in
  if in_axis = -2 && ndim <= 1 then
    Py_exn.value_error
      (Printf.sprintf
         "Can't compute input and output sizes of a %d-dimensional weights \
          tensor with default in_axis. Must be at least 2D or specify in_axis \
          explicitly."
         ndim);
  let norm a = if a < 0 then a + ndim else a in
  let in_size = float_of_int shape.(norm in_axis) in
  let out_size = float_of_int shape.(norm out_axis) in
  let batch_size =
    Array.fold_left
      (fun acc a -> acc *. float_of_int shape.(norm a))
      1.0 batch_axis
  in
  let total = Array.fold_left (fun acc d -> acc *. float_of_int d) 1.0 shape in
  let receptive = total /. in_size /. out_size /. batch_size in
  (in_size *. receptive, out_size *. receptive)

let zeros : t = fun _key ~shape -> const shape 0.0
let ones : t = fun _key ~shape -> const shape 1.0

let zeros_dtype ?dtype _key ~shape =
  match dtype with
  | None -> const shape 0.0
  | Some d -> Lax_api.full shape 0.0 (Dtypes.canonicalize_dtype d)

let ones_dtype ?dtype _key ~shape =
  match dtype with
  | None -> const shape 1.0
  | Some d -> Lax_api.full shape 1.0 (Dtypes.canonicalize_dtype d)

let constant value : t = fun _key ~shape -> const shape (f32 value)
let dtype_of v = (C.get_aval v).dtype
let like v x = Concrete (Nd.of_floats (dtype_of v) [||] [| x |])
let times = Numpy.Ufuncs.multiply

let uniform ?(scale = 1e-2) () : t =
 fun key ~shape ->
  let samp = RC.uniform key ~shape ~minval:0.0 ~maxval:1.0 in
  times samp (like samp scale)

let normal ?(stddev = 1e-2) () : t =
 fun key ~shape ->
  let samp = RC.normal key ~shape in
  times samp (like samp stddev)

let truncated_normal ?(stddev = 1e-2) ?(lower = -2.0) ?(upper = 2.0) () : t =
 fun key ~shape ->
  let samp = RC.truncated_normal key ~lower ~upper ~shape in
  times samp (like samp stddev)

let variance_scaling scale mode distribution ?(in_axis = -2) ?(out_axis = -1)
    ?(batch_axis = [||]) () : t =
 fun key ~shape ->
  let fan_in, fan_out = compute_fans shape ~in_axis ~out_axis ~batch_axis in
  let denominator =
    match mode with
    | Fan_in -> fan_in
    | Fan_out -> fan_out
    | Fan_avg -> (fan_in +. fan_out) /. 2.0
    | Fan_geo_avg -> sqrt (fan_in *. fan_out)
  in
  let module UF = Numpy.Ufuncs in
  let dt = RC.float_dtype () in
  let variance = Concrete (Nd.of_floats dt [||] [| scale /. denominator |]) in
  let literal v = Concrete (Nd.of_floats dt [||] [| v |]) in
  match distribution with
  | Truncated_normal ->
      let root = UF.sqrt variance in
      let stddev = UF.true_divide root (literal 0.87962566103423978) in
      let samp = RC.truncated_normal_int key ~lower:(-2) ~upper:2 ~shape in
      times samp stddev
  | Normal ->
      let samp = RC.normal key ~shape in
      times samp (UF.sqrt variance)
  | Uniform ->
      let samp = RC.uniform_int_low key ~shape ~minval:(-1) in
      let tripled = UF.multiply (Numpy.Jnp_util.py_int 3) variance in
      times samp (UF.sqrt tripled)

let glorot_uniform ?in_axis ?out_axis ?batch_axis () : t =
  variance_scaling 1.0 Fan_avg Uniform ?in_axis ?out_axis ?batch_axis ()

let glorot_normal ?in_axis ?out_axis ?batch_axis () : t =
  variance_scaling 1.0 Fan_avg Truncated_normal ?in_axis ?out_axis ?batch_axis
    ()

let lecun_uniform ?in_axis ?out_axis ?batch_axis () : t =
  variance_scaling 1.0 Fan_in Uniform ?in_axis ?out_axis ?batch_axis ()

let lecun_normal ?in_axis ?out_axis ?batch_axis () : t =
  variance_scaling 1.0 Fan_in Truncated_normal ?in_axis ?out_axis ?batch_axis ()

let he_uniform ?in_axis ?out_axis ?batch_axis () : t =
  variance_scaling 2.0 Fan_in Uniform ?in_axis ?out_axis ?batch_axis ()

let he_normal ?in_axis ?out_axis ?batch_axis () : t =
  variance_scaling 2.0 Fan_in Truncated_normal ?in_axis ?out_axis ?batch_axis ()

let xavier_uniform = glorot_uniform
let xavier_normal = glorot_normal
let kaiming_uniform = he_uniform
let kaiming_normal = he_normal

let orthogonal ?(scale = 1.0) ?(column_axis = -1) () : t =
 fun key ~shape ->
  let rank = Array.length shape in
  if rank < 2 then
    Py_exn.value_error "orthogonal initializer requires at least a 2D shape";
  let axis = if column_axis < 0 then column_axis + rank else column_axis in
  let n_cols = shape.(axis) in
  let n_rows = prod shape / n_cols in
  let q = Random_extras.orthogonal ~m:n_cols key ~n:n_rows ~shape:[||] in
  let others =
    Array.of_list (List.filteri (fun i _ -> i <> axis) (Array.to_list shape))
  in
  let q = Numpy.Lax_numpy.reshape q (Array.append others [| n_cols |]) in
  let q = Numpy.Lax_numpy.moveaxis [| -1 |] [| column_axis |] q in
  let dt = (C.get_aval q).dtype in
  Numpy.Ufuncs.multiply (Concrete (Nd.of_floats dt [||] [| scale |])) q

let delta_orthogonal ?(scale = 1.0) ?(column_axis = -1) () : t =
 fun key ~shape ->
  let rank = Array.length shape in
  if rank < 3 || rank > 5 then
    Py_exn.value_error
      "Delta orthogonal initializer requires a 3D, 4D or 5D shape.";
  if shape.(rank - 1) < shape.(rank - 2) then
    Py_exn.value_error "`fan_in` must be less or equal than `fan_out`. ";
  let ortho =
    orthogonal ~scale ~column_axis () key
      ~shape:[| shape.(rank - 2); shape.(rank - 1) |]
  in
  let a = C.get_aval ortho in
  if rank = 3 then
    let w = C.splat a.dtype shape 0.0 in
    let idx =
      C.bind1
        (Broadcast_in_dim { shape = [| 1 |]; dims = [||] })
        [
          Concrete
            (Nd.of_floats Dtype.I32 [||]
               [| float_of_int ((shape.(0) - 1) / 2) |]);
        ]
    in
    C.bind1
      (Scatter
         {
           dimension_numbers =
             {
               update_window_dims = [| 0; 1 |];
               inserted_window_dims = [| 0 |];
               scatter_dims_to_operand_dims = [| 0 |];
               s_operand_batching_dims = [||];
               s_scatter_indices_batching_dims = [||];
               s_sorted_unique = true;
               s_clip = false;
             };
           unique_indices = true;
         })
      [ w; idx; ortho ]
  else
    let center = Array.sub shape 0 (rank - 2) in
    let mid = Array.map (fun d -> (d - 1) / 2) center in
    let cfg =
      Array.init rank (fun i ->
          if i < rank - 2 then (mid.(i), center.(i) - mid.(i) - 1, 0)
          else (0, 0, 0))
    in
    let expanded =
      C.bind1
        (Reshape (Array.append (Array.make (rank - 2) 1) a.shape))
        [ ortho ]
    in
    C.bind1 (Pad cfg)
      [ expanded; Concrete (Nd.of_floats a.dtype [||] [| 0.0 |]) ]

let with_dtype dtype (init : t) : t =
 fun key ~shape -> RC.with_dtypes ~float_dtype:dtype (fun () -> init key ~shape)
