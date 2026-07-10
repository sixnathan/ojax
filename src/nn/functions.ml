module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module UF = Numpy.Ufuncs
module RED = Numpy.Reductions
module NL = Numpy.Lax_numpy

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let shape v = (get_aval v).T.shape
let bind1 = C.bind1
let numel sh = Array.fold_left ( * ) 1 sh

let const_full dt sh x =
  T.Concrete (Nd.of_floats dt sh (Array.make (numel sh) x))

let scalar_like v x = const_full (dtype v) (shape v) x
let where3 c a b = NL.where_ c a b

let to_inexact = function
  | D.F32 -> D.F32
  | D.F64 -> D.F64
  | D.I32 | D.Bool | D.Uint32 -> D.F32
  | D.I64 -> D.F64
  | (D.Complex64 | D.Complex128) as c -> c

let inexact v =
  let dt = to_inexact (dtype v) in
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let canon_axis ndim a = if a < 0 then a + ndim else a
let identity x = x
let relu x = UF.maximum x (scalar_like x 0.0)
let relu6 x = UF.minimum (relu x) (scalar_like x 6.0)
let softplus x = UF.logaddexp x (scalar_like x 0.0)
let soft_sign x = UF.divide x (UF.add (UF.abs x) (scalar_like x 1.0))
let sigmoid x = bind1 T.Logistic [ x ]
let silu x = UF.multiply x (sigmoid x)
let mish x = UF.multiply x (UF.tanh (softplus x))
let log_sigmoid x = UF.negative (softplus (UF.negative x))

let sparse_plus x =
  let one = scalar_like x 1.0 in
  let quad = UF.divide (UF.square (UF.add x one)) (scalar_like x 4.0) in
  where3
    (UF.less_equal x (scalar_like x (-1.0)))
    (scalar_like x 0.0)
    (where3 (UF.greater_equal x one) x quad)

let sparse_sigmoid x =
  UF.multiply (scalar_like x 0.5)
    (NL.clip ~min:0.0 ~max:2.0 (UF.add x (scalar_like x 1.0)))

let hard_tanh x =
  where3
    (UF.greater x (scalar_like x 1.0))
    (scalar_like x 1.0)
    (where3 (UF.less x (scalar_like x (-1.0))) (scalar_like x (-1.0)) x)

let hard_sigmoid x =
  UF.divide (relu6 (UF.add x (scalar_like x 3.0))) (scalar_like x 6.0)

let hard_silu x = UF.multiply x (hard_sigmoid x)

let elu ?(alpha = 1.0) x =
  let pos = UF.greater x (scalar_like x 0.0) in
  let safe = where3 pos (scalar_like x 0.0) x in
  where3 pos x (UF.multiply (scalar_like x alpha) (UF.expm1 safe))

let celu ?(alpha = 1.0) x =
  UF.add
    (UF.maximum x (scalar_like x 0.0))
    (UF.multiply (scalar_like x alpha)
       (UF.expm1
          (UF.divide (UF.minimum x (scalar_like x 0.0)) (scalar_like x alpha))))

let selu x =
  let alpha = 1.6732632423543772848170429916717 in
  let scale = 1.0507009873554804934193349852946 in
  UF.multiply (scalar_like x scale) (elu ~alpha x)

let leaky_relu ?(negative_slope = 0.01) x =
  where3
    (UF.greater_equal x (scalar_like x 0.0))
    x
    (UF.multiply (scalar_like x negative_slope) x)

let squareplus ?(b = 4.0) x =
  let y = UF.add x (UF.sqrt (UF.add (UF.square x) (scalar_like x b))) in
  UF.divide y (scalar_like x 2.0)

let log1mexp x =
  let c = scalar_like x (Float.log 2.0) in
  let nx = UF.negative x in
  where3 (UF.less x c)
    (UF.log (UF.negative (UF.expm1 nx)))
    (UF.log1p (UF.negative (UF.exp nx)))

let gelu ?(approximate = true) x =
  let x = inexact x in
  if approximate then begin
    let s2pi = scalar_like x (sqrt (2.0 /. Float.pi)) in
    let cube = bind1 (T.Integer_pow 3) [ x ] in
    let inner = UF.add x (UF.multiply (scalar_like x 0.044715) cube) in
    let cdf =
      UF.multiply (scalar_like x 0.5)
        (UF.add (scalar_like x 1.0) (UF.tanh (UF.multiply s2pi inner)))
    in
    UF.multiply x cdf
  end
  else begin
    let sh = scalar_like x (sqrt 0.5) in
    let e = bind1 T.Erfc [ UF.multiply (UF.negative x) sh ] in
    UF.multiply (UF.multiply (scalar_like x 0.5) x) e
  end

let glu ?(axis = -1) x =
  match NL.split ~axis x (NL.Count 2) with
  | [ x1; x2 ] -> UF.multiply x1 (sigmoid x2)
  | _ -> invalid_arg "glu: split did not produce two halves"

let softmax ?(axis = -1) x =
  let ax = [| canon_axis (Array.length (shape x)) axis |] in
  let x_max = RED.max ~axis:ax ~keepdims:true x in
  let unnormalized = UF.exp (UF.subtract x x_max) in
  UF.divide unnormalized (RED.sum ~axis:ax ~keepdims:true unnormalized)

let log_softmax ?(axis = -1) x =
  let ax = [| canon_axis (Array.length (shape x)) axis |] in
  let x_max = RED.max ~axis:ax ~keepdims:true x in
  let shifted = UF.subtract x x_max in
  let lse = UF.log (RED.sum ~axis:ax ~keepdims:true (UF.exp shifted)) in
  UF.subtract shifted lse

let standardize ?(axis = -1) ?(epsilon = 1e-5) x =
  let ax = [| canon_axis (Array.length (shape x)) axis |] in
  let mean = RED.mean ~axis:ax ~keepdims:true x in
  let variance =
    UF.subtract
      (RED.mean ~axis:ax ~keepdims:true (UF.square x))
      (UF.square mean)
  in
  let variance = NL.clip ~min:0.0 variance in
  UF.multiply (UF.subtract x mean)
    (bind1 T.Rsqrt [ UF.add variance (scalar_like variance epsilon) ])

let logsumexp ?axis ?(keepdims = false) x =
  let ndim = Array.length (shape x) in
  let dims =
    match axis with
    | None -> Array.init ndim (fun i -> i)
    | Some a -> Array.map (canon_axis ndim) a
  in
  let amax = RED.max ~axis:dims ~keepdims:true x in
  let amax = where3 (UF.isfinite amax) amax (scalar_like amax 0.0) in
  let shifted = UF.subtract x amax in
  let sumexp = RED.sum ~axis:dims ~keepdims:true (UF.exp shifted) in
  let out = UF.add (UF.log sumexp) amax in
  if keepdims then out else NL.squeeze ~axis:dims out

let logmeanexp ?axis ?(keepdims = false) x =
  let ndim = Array.length (shape x) in
  let dims =
    match axis with
    | None -> Array.init ndim (fun i -> i)
    | Some a -> Array.map (canon_axis ndim) a
  in
  let lse = logsumexp ?axis ~keepdims x in
  let n = Array.fold_left (fun acc d -> acc * (shape x).(d)) 1 dims in
  UF.subtract lse
    (const_full (dtype lse) (shape lse) (Float.log (float_of_int n)))

let one_hot ?(axis = -1) ~num_classes x =
  let ndim = Array.length (shape x) in
  let pos = if axis < 0 then axis + ndim + 1 else axis in
  let rhs_shape =
    Array.init (ndim + 1) (fun i -> if i = pos then num_classes else 1)
  in
  let lhs = NL.expand_dims x [| pos |] in
  let rhs =
    bind1 (T.Iota { dtype = dtype x; shape = rhs_shape; dimension = pos }) []
  in
  NL.astype (UF.equal lhs rhs) (Dtypes.default_float_dtype ())

let scaled_dot_general ~lhs_contract ~rhs_contract ~lhs_batch ~rhs_batch a b =
  bind1
    (T.Dot_general { lhs_contract; rhs_contract; lhs_batch; rhs_batch })
    [ a; b ]
