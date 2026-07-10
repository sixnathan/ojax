open Types
module Nd = Ndarray
module C = Core
module D = Dtype
module RC = Random.Core

type t = value -> shape:int array -> value
type mode = Fan_in | Fan_out | Fan_avg | Fan_geo_avg
type distribution = Truncated_normal | Normal | Uniform

let prod a = Array.fold_left ( * ) 1 a
let f32 x = Int32.float_of_bits (Int32.bits_of_float x)

let concrete = function
  | Concrete nd -> nd
  | Tracer _ ->
      failwith "nn/initializers: initializer on a tracer not supported"

let const shape x =
  Concrete
    (Nd.canonicalize D.F32
       (Nd.of_floats D.F32 shape (Array.make (prod shape) x)))

let scalar_f v = Nd.get_f (concrete v) [||]
let sqrt_f32 x = scalar_f (C.bind1 Sqrt [ const [||] x ])
let scale_by shape v factor = C.bind1 Mul [ v; const shape factor ]

let compute_fans shape ~in_axis ~out_axis ~batch_axis =
  let ndim = Array.length shape in
  if in_axis = -2 && ndim <= 1 then
    invalid_arg
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
let constant value : t = fun _key ~shape -> const shape (f32 value)

let uniform ?(scale = 1e-2) () : t =
 fun key ~shape ->
  let samp = RC.uniform key ~shape ~minval:0.0 ~maxval:1.0 in
  scale_by shape samp (f32 scale)

let normal ?(stddev = 1e-2) () : t =
 fun key ~shape ->
  let samp = RC.normal key ~shape in
  scale_by shape samp (f32 stddev)

let truncated_normal ?(stddev = 1e-2) ?(lower = -2.0) ?(upper = 2.0) () : t =
 fun key ~shape ->
  let samp = RC.truncated_normal key ~lower ~upper ~shape in
  scale_by shape samp (f32 stddev)

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
  let variance = f32 (scale /. denominator) in
  match distribution with
  | Truncated_normal ->
      let stddev = f32 (sqrt_f32 variance /. f32 0.87962566103423978) in
      let samp = RC.truncated_normal key ~lower:(-2.0) ~upper:2.0 ~shape in
      scale_by shape samp stddev
  | Normal ->
      let samp = RC.normal key ~shape in
      scale_by shape samp (sqrt_f32 variance)
  | Uniform ->
      let samp = RC.uniform key ~shape ~minval:(-1.0) ~maxval:1.0 in
      scale_by shape samp (sqrt_f32 (f32 (3.0 *. variance)))

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
  let _ = scale and _ = column_axis in
  fun _key ~shape:_ ->
    failwith "nn.initializers.orthogonal: requires jnp.linalg.qr (M5)"

let delta_orthogonal ?(scale = 1.0) ?(column_axis = -1) () : t =
  let _ = scale and _ = column_axis in
  fun _key ~shape:_ ->
    failwith "nn.initializers.delta_orthogonal: requires jnp.linalg.qr (M5)"
