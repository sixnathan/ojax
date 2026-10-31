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
  | (D.F16 | D.Bf16) as narrow -> narrow
  | D.I32 | D.Bool | D.Uint32 | D.I8 | D.I16 | D.Uint8 | D.Uint16 -> D.F32
  | D.I64 | D.Uint64 -> D.F64
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

type block_scale_dtype = Float4_e2m1fn | Float8_e4m3fn | Float8_e8m0fnu

type block_scale_config = {
  mode : string;
  block_size : int;
  data_type : block_scale_dtype;
  scale_type : block_scale_dtype;
  global_scale : Types.value option;
  infer_only : bool;
}

let get_scaled_dot_general_config ?global_scale mode =
  match mode with
  | "nvfp4" ->
      let scale =
        match global_scale with
        | Some s -> s
        | None -> const_full D.F32 [| 1 |] 1.0
      in
      {
        mode;
        block_size = 16;
        data_type = Float4_e2m1fn;
        scale_type = Float8_e4m3fn;
        global_scale = Some scale;
        infer_only = false;
      }
  | "mxfp8" ->
      {
        mode;
        block_size = 32;
        data_type = Float8_e4m3fn;
        scale_type = Float8_e8m0fnu;
        global_scale = None;
        infer_only = false;
      }
  | other -> invalid_arg (Printf.sprintf "Unsupported mode: %s" other)

let scaled_dot_general ?(preferred_element_type = D.F32) ?configs ~lhs_contract
    ~rhs_contract ~lhs_batch ~rhs_batch a b =
  (match configs with
  | Some (_ : block_scale_config list) ->
      (match preferred_element_type with
      | D.F32 | D.Bf16 | D.F16 -> ()
      | _ ->
          invalid_arg "Only support preferred_element_type in (f32, bf16, f16)");
      failwith
        "MLIR translation rule for primitive 'scaled_matmul' not found for \
         platform cpu"
  | None -> ());
  Lax_api.dot_general ~preferred_element_type a b ~lhs_contract ~rhs_contract
    ~lhs_batch ~rhs_batch

let scaled_matmul ?(preferred_element_type = D.F32) lhs rhs lhs_scales
    rhs_scales =
  let dims v = shape v in
  if
    List.exists
      (fun v -> Array.length (dims v) <> 3)
      [ lhs; rhs; lhs_scales; rhs_scales ]
  then
    invalid_arg "scaled_matmul requires all inputs to be 3-dimensional arrays";
  let sa = dims lhs and sb = dims rhs in
  if sa.(2) <> sb.(2) || sa.(0) <> sb.(0) then
    invalid_arg
      "scaled_matmul requires inputs a and b to have matching batch (B) and \
       contract (K) dimensions";
  let ssa = dims lhs_scales and ssb = dims rhs_scales in
  if ssa.(2) <> ssb.(2) || ssa.(0) <> ssb.(0) then
    invalid_arg
      "scaled_matmul requires scales to have matching batch (B) and contract \
       (K) dimensions";
  if ssa.(1) <> sa.(1) || ssb.(1) <> sb.(1) then
    invalid_arg
      "scaled_matmul requires scales to match non-contract dimensions of inputs";
  let k = sa.(2) in
  if ssa.(2) = 0 || k mod ssa.(2) <> 0 then
    invalid_arg "scaled_matmul requires K to be divisible by the scale blocks";
  let block = k / ssa.(2) in
  let scaled operand scales =
    let sh = dims scales in
    let wide =
      Lax_api.broadcast_in_dim scales
        [| sh.(0); sh.(1); sh.(2); block |]
        [| 0; 1; 2 |]
    in
    let repeated = Lax_api.reshape wide [| sh.(0); sh.(1); k |] in
    UF.multiply (NL.astype operand D.F32) (NL.astype repeated D.F32)
  in
  let product =
    Numpy.Einsum.einsum "BMK,BNK->BMN"
      [ scaled lhs lhs_scales; scaled rhs rhs_scales ]
  in
  NL.astype product preferred_element_type

let swish = silu
let hard_swish = hard_silu
let tanh x = Core.bind1 T.Tanh [ x ]

let dot_product_attention ?bias ?mask ?scale ~query ~key ~value () =
  let module E = Numpy.Einsum in
  let module L = Lax_api in
  let q_shape = (Core.get_aval query).T.shape in
  let d = q_shape.(Array.length q_shape - 1) in
  let rank_of v = Array.length (Core.get_aval v).T.shape in
  let lift v =
    let r = rank_of v in
    if r >= 4 then v else L.expand_dims v (Array.init (4 - r) Fun.id)
  in
  let squeeze_back v = L.reshape v q_shape in
  let scale_value =
    match scale with Some s -> s | None -> 1.0 /. sqrt (float_of_int d)
  in
  let logits = E.einsum "BTNH,BSNH->BNTS" [ lift query; lift key ] in
  let logits = L.mul logits (L.full_like logits scale_value) in
  let logits = match bias with Some b -> L.add logits b | None -> logits in
  let logits =
    match mask with
    | Some m -> L.select m logits (L.full_like logits Float.neg_infinity)
    | None -> logits
  in
  let single = NL.astype logits D.F32 in
  let probs =
    NL.astype (softmax ~axis:(-1) single) (Core.get_aval key).T.dtype
  in
  squeeze_back (E.einsum "BNTS,BSNH->BTNH" [ probs; lift value ])

let as_float x =
  match dtype x with
  | D.Bool | D.I8 | D.I16 | D.I32 | D.I64 | D.Uint8 | D.Uint16 | D.Uint32
  | D.Uint64 ->
      NL.astype x (Dtypes.default_float_dtype ())
  | _ -> x

let as_int_if_bool x =
  if dtype x = D.Bool then
    NL.astype x (if Config.x64_enabled () then D.I64 else D.I32)
  else x

let relu x = relu (as_int_if_bool x)
let hard_tanh x = hard_tanh (as_int_if_bool x)
let relu6 x = relu6 (as_float x)
let leaky_relu ?negative_slope x = leaky_relu ?negative_slope (as_float x)
let elu ?alpha x = elu ?alpha (as_float x)
let celu ?alpha x = celu ?alpha (as_float x)
let selu x = selu (as_float x)
let hard_sigmoid x = hard_sigmoid (as_float x)
let hard_silu x = hard_silu (as_float x)
let sparse_plus x = sparse_plus (as_float x)
let sparse_sigmoid x = sparse_sigmoid (as_float x)
let as_inexact_dtype x = NL.astype x (Dtypes.to_inexact_dtype (dtype x))
let squareplus ?b x = squareplus ?b (as_inexact_dtype x)
let soft_sign x = soft_sign (if dtype x = D.Bool then as_float x else x)
let softplus x = softplus (if dtype x = D.Bool then as_float x else x)
let log1mexp x = log1mexp (if dtype x = D.Bool then as_float x else x)

let logmeanexp ?axis ?keepdims x =
  logmeanexp ?axis ?keepdims (as_inexact_dtype x)
