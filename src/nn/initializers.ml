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
  | Device _ ->
      failwith "nn/initializers: initializer on a tracer not supported"

let const shape x = C.splat D.F32 shape x
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
 fun key ~shape ->
  let rank = Array.length shape in
  if rank < 2 then
    invalid_arg "orthogonal initializer requires at least a 2D shape";
  let axis = if column_axis < 0 then column_axis + rank else column_axis in
  let n_cols = shape.(axis) in
  let n_rows = prod shape / n_cols in
  let matrix_shape =
    if n_rows < n_cols then [| n_cols; n_rows |] else [| n_rows; n_cols |]
  in
  let a = Random.Core.normal key ~shape:matrix_shape in
  let q, r =
    match Numpy.Linalg.qr a with
    | [ q; r ] -> (q, r)
    | _ -> failwith "orthogonal: qr arity"
  in
  let signs = C.bind1 Sign [ Numpy.Lax_numpy.diagonal r ] in
  let qs = C.get_aval q in
  let scaled =
    C.bind1 Mul
      [
        q;
        C.bind1
          (Broadcast_in_dim { shape = qs.shape; dims = [| 1 |] })
          [ signs ];
      ]
  in
  let oriented =
    if n_rows < n_cols then C.bind1 (Transpose [| 1; 0 |]) [ scaled ]
    else scaled
  in
  let others =
    Array.of_list (List.filteri (fun i _ -> i <> axis) (Array.to_list shape))
  in
  let reshaped =
    C.bind1 (Reshape (Array.append others [| n_cols |])) [ oriented ]
  in
  let moved = Numpy.Lax_numpy.moveaxis [| rank - 1 |] [| axis |] reshaped in
  let f = C.get_aval moved in
  C.bind1 Mul [ moved; C.splat f.dtype f.shape scale ]

let delta_orthogonal ?(scale = 1.0) ?(column_axis = -1) () : t =
 fun key ~shape ->
  let rank = Array.length shape in
  if rank < 3 || rank > 5 then
    invalid_arg "Delta orthogonal initializer requires a 3D, 4D or 5D shape.";
  if shape.(rank - 1) < shape.(rank - 2) then
    invalid_arg "`fan_in` must be less or equal than `fan_out`. ";
  let ortho =
    orthogonal ~scale ~column_axis () key
      ~shape:[| shape.(rank - 2); shape.(rank - 1) |]
  in
  let a = C.get_aval ortho in
  let center = Array.sub shape 0 (rank - 2) in
  let mid = Array.map (fun d -> (d - 1) / 2) center in
  let cfg =
    Array.init rank (fun i ->
        if i < rank - 2 then (mid.(i), center.(i) - mid.(i) - 1, 0)
        else (0, 0, 0))
  in
  let expanded =
    C.bind1 (Reshape (Array.append (Array.make (rank - 2) 1) a.shape)) [ ortho ]
  in
  C.bind1 (Pad cfg) [ expanded; Concrete (Nd.of_floats a.dtype [||] [| 0.0 |]) ]
