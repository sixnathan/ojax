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
let const_full dt sh x = Core.splat dt sh x
let scalar_like v x = Core.splat (dtype v) [||] x
let where3 c a b = NL.where_ c a b

let to_inexact = function
  | D.F32 -> D.F32
  | D.F64 -> D.F64
  | ( D.F16 | D.Bf16 | D.F4_e2m1fn | D.F6_e2m3fn | D.F6_e3m2fn | D.F8_e3m4
    | D.F8_e4m3 | D.F8_e4m3b11fnuz | D.F8_e4m3fn | D.F8_e4m3fnuz | D.F8_e5m2
    | D.F8_e5m2fnuz | D.F8_e8m0fnu ) as narrow ->
      narrow
  | D.I32 | D.Bool | D.Uint32 | D.I8 | D.I16 | D.I4 | D.Uint8 | D.Uint16
  | D.Uint4 ->
      D.F32
  | (D.I2 | D.Uint2) as narrow -> narrow
  | D.Float0 -> invalid_arg "float0 has no inexact dtype"
  | D.I64 | D.Uint64 -> D.F64
  | (D.Complex64 | D.Complex128) as c -> c

let inexact v =
  let dt = to_inexact (dtype v) in
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let canon_axis ndim a = if a < 0 then a + ndim else a
let identity x = x

let jit_unary name f x =
  Numpy.Jnp_util.jit1 name
    (function [ x ] -> f x | _ -> invalid_arg (name ^ ": arity"))
    [ x ]

let select_or_zero pred g =
  let zeros = Core.splat (dtype g) (shape g) 0.0 in
  bind1 T.Select_n [ pred; zeros; g ]

let custom_unary name primal rule =
  let call =
    Custom_derivatives.defjvps ~name
      (function [ x ] -> [ primal x ] | _ -> invalid_arg "custom: arity")
      [
        Some
          (fun g out primals ->
            match (out, primals) with
            | [ ans ], [ x ] -> [ rule g ans x ]
            | _ -> invalid_arg "custom: arity");
      ]
  in
  fun x -> List.hd (call [ x ])

let relu =
  custom_unary "relu"
    (jit_unary "relu" (fun x -> UF.maximum x (scalar_like x 0.0)))
    (fun g _ x ->
      let pred = UF.greater x (scalar_like x 0.0) in
      select_or_zero pred g)

let relu6 =
  custom_unary "relu6"
    (jit_unary "relu6" (fun x ->
         let high = UF.maximum x (scalar_like x 0.0) in
         UF.minimum high (scalar_like x 6.0)))
    (fun g _ x ->
      let above = UF.greater x (scalar_like x 0.0) in
      let below = UF.less x (scalar_like x 6.0) in
      let pred = UF.logical_and above below in
      select_or_zero pred g)

let softplus =
  jit_unary "softplus" (fun x -> UF.logaddexp x (scalar_like x 0.0))

let soft_sign =
  jit_unary "soft_sign" (fun x ->
      let magnitude = UF.abs x in
      let denom = UF.add magnitude (scalar_like x 1.0) in
      UF.divide x denom)

let sigmoid = jit_unary "sigmoid" (fun x -> bind1 T.Logistic [ x ])

let silu =
  jit_unary "silu" (fun x ->
      let s = sigmoid x in
      UF.multiply x s)

let mish =
  jit_unary "mish" (fun x ->
      let s = softplus x in
      let t = UF.tanh s in
      UF.multiply x t)

let log_sigmoid =
  jit_unary "log_sigmoid" (fun x ->
      let n = UF.negative x in
      let s = softplus n in
      UF.negative s)

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

let hard_sigmoid =
  jit_unary "hard_sigmoid" (fun x ->
      let shifted = UF.add x (scalar_like x 3.0) in
      let r = relu6 shifted in
      UF.divide r (scalar_like x 6.0))

let hard_silu =
  jit_unary "hard_silu" (fun x ->
      let h = hard_sigmoid x in
      UF.multiply x h)

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

let log1mexp_primal x =
  let c = scalar_like x (Float.log 2.0) in
  let pred = UF.less x c in
  let nx = UF.negative x in
  let a = UF.log (UF.negative (UF.expm1 nx)) in
  let nx2 = UF.negative x in
  let b = UF.log1p (UF.negative (UF.exp nx2)) in
  where3 pred a b

let log1mexp =
  custom_unary "log1mexp" log1mexp_primal (fun g _ x ->
      UF.divide g (UF.expm1 x))

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

let masked_max ax mask x =
  match mask with
  | None -> RED.max ~axis:ax ~keepdims:true x
  | Some m ->
      RED.max ~axis:ax ~keepdims:true ~where:m
        ~initial:(const_full (dtype x) [||] Float.neg_infinity)
        x

let softmax_body ~stop ax mask x =
  let x_max = masked_max ax mask x in
  let x_max = if stop then Custom_derivatives.stop_gradient x_max else x_max in
  let x_safe =
    match mask with
    | None -> x
    | Some m -> where3 m x (scalar_like x Float.neg_infinity)
  in
  let unnormalized = UF.exp (UF.subtract x_safe x_max) in
  let result =
    UF.divide unnormalized
      (RED.sum ~axis:ax ~keepdims:true ?where:mask unnormalized)
  in
  match mask with
  | None -> result
  | Some m -> where3 m result (scalar_like result 0.0)

let softmax ?(axis = -1) ?where:mask x =
  let ax = [| canon_axis (Array.length (shape x)) axis |] in
  if Config.value Config.softmax_custom_jvp then
    let primal = function
      | [ a ] -> [ softmax_body ~stop:false ax mask a ]
      | _ -> invalid_arg "softmax: arity"
    in
    let jvp primals tangents =
      match (primals, tangents) with
      | [ a ], [ a_dot ] ->
          let y = softmax_body ~stop:false ax mask a in
          let inner =
            RED.sum ~axis:ax ~keepdims:true ?where:mask (UF.multiply y a_dot)
          in
          ([ y ], [ UF.multiply y (UF.subtract a_dot inner) ])
      | _ -> invalid_arg "softmax: arity"
    in
    List.hd (Custom_derivatives.custom_jvp primal ~jvp [ x ])
  else softmax_body ~stop:true ax mask x

let log_softmax ?(axis = -1) ?where:mask x =
  let ax = [| canon_axis (Array.length (shape x)) axis |] in
  let x_max = masked_max ax mask x in
  let x_safe =
    match mask with
    | None -> x
    | Some m -> where3 m x (scalar_like x Float.neg_infinity)
  in
  let shifted = UF.subtract x_safe x_max in
  let lse =
    UF.log (RED.sum ~axis:ax ~keepdims:true ?where:mask (UF.exp shifted))
  in
  let result = UF.subtract shifted lse in
  match mask with
  | None -> result
  | Some m -> where3 m result (scalar_like result Float.neg_infinity)

let standardize ?(axis = -1) ?mean ?variance ?(epsilon = 1e-5) ?where:mask
    ?(algorithm = "fast") x =
  let ax = [| canon_axis (Array.length (shape x)) axis |] in
  let mean =
    match mean with
    | Some m -> m
    | None -> RED.mean ~axis:ax ~keepdims:true ?where:mask x
  in
  let variance =
    match variance with
    | Some v -> v
    | None -> (
        match algorithm with
        | "stable" ->
            RED.mean ~axis:ax ~keepdims:true ?where:mask
              (UF.square (UF.subtract x mean))
        | "fast" ->
            NL.clip ~min:0.0
              (UF.subtract
                 (RED.mean ~axis:ax ~keepdims:true ?where:mask (UF.square x))
                 (UF.square mean))
        | a ->
            invalid_arg
              ("Unknown algorithm '" ^ a ^ "'. Expected 'fast' or 'stable'."))
  in
  UF.multiply (UF.subtract x mean)
    (bind1 T.Rsqrt [ UF.add variance (scalar_like variance epsilon) ])

let logsumexp ?axis ?(keepdims = false) ?where:mask x =
  let ndim = Array.length (shape x) in
  let dims =
    match axis with
    | None -> Array.init ndim (fun i -> i)
    | Some a -> Array.map (canon_axis ndim) a
  in
  let amax = masked_max dims mask x in
  let amax = where3 (UF.isfinite amax) amax (scalar_like amax 0.0) in
  let shifted = UF.subtract x amax in
  let sumexp = RED.sum ~axis:dims ~keepdims:true ?where:mask (UF.exp shifted) in
  let out = UF.add (UF.log sumexp) amax in
  if keepdims then out else NL.squeeze ~axis:dims out

let logmeanexp ?axis ?(keepdims = false) ?where:mask x =
  let ndim = Array.length (shape x) in
  let dims =
    match axis with
    | None -> Array.init ndim (fun i -> i)
    | Some a -> Array.map (canon_axis ndim) a
  in
  let lse = logsumexp ?axis ~keepdims ?where:mask x in
  match mask with
  | Some m ->
      UF.subtract lse
        (UF.log
           (RED.sum ~axis:dims ~keepdims ~dtype:(dtype lse)
              (NL.broadcast_to m (shape x))))
  | None ->
      let n = Array.fold_left (fun acc d -> acc * (shape x).(d)) 1 dims in
      UF.subtract lse
        (const_full (dtype lse) (shape lse) (Float.log (float_of_int n)))

let one_hot ?(axis = -1) ?dtype:dt ~num_classes x =
  let ndim = Array.length (shape x) in
  let pos = if axis < 0 then axis + ndim + 1 else axis in
  let rhs_shape =
    Array.init (ndim + 1) (fun i -> if i = pos then num_classes else 1)
  in
  let lhs = NL.expand_dims x [| pos |] in
  let rhs =
    bind1 (T.Iota { dtype = dtype x; shape = rhs_shape; dimension = pos }) []
  in
  NL.astype (UF.equal lhs rhs)
    (match dt with Some d -> d | None -> Dtypes.default_float_dtype ())

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
let tanh x = Core.bind1 T.Tanh [ inexact x ]

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
  | D.Bool | D.I8 | D.I16 | D.I2 | D.I4 | D.I32 | D.I64 | D.Uint8 | D.Uint16
  | D.Uint2 | D.Uint4 | D.Uint32 | D.Uint64 ->
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
let hard_swish = hard_silu
let sparse_plus x = sparse_plus (as_float x)
let sparse_sigmoid x = sparse_sigmoid (as_float x)
let as_inexact_dtype x = NL.astype x (Dtypes.to_inexact_dtype (dtype x))
let squareplus ?b x = squareplus ?b (as_inexact_dtype x)
let soft_sign x = soft_sign (if dtype x = D.Bool then as_float x else x)
let softplus x = softplus (if dtype x = D.Bool then as_float x else x)
let mish x = mish (if dtype x = D.Bool then as_float x else x)

let logmeanexp ?axis ?keepdims ?where:mask x =
  logmeanexp ?axis ?keepdims ?where:mask (as_inexact_dtype x)

let identity a0 =
  Eager_jit.one "nn.functions.identity" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> identity a0 | _ -> assert false)

let relu a0 =
  Eager_jit.one "nn.functions.relu" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> relu a0 | _ -> assert false)

let squareplus ?b a0 =
  Eager_jit.one "nn.functions.squareplus" b [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> squareplus ?b a0
      | _ -> assert false)

let softplus a0 =
  Eager_jit.one "nn.functions.softplus" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> softplus a0 | _ -> assert false)

let sparse_plus a0 =
  Eager_jit.one "nn.functions.sparse_plus" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> sparse_plus a0 | _ -> assert false)

let soft_sign a0 =
  Eager_jit.one "nn.functions.soft_sign" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> soft_sign a0 | _ -> assert false)

let sigmoid a0 =
  Eager_jit.one "nn.functions.sigmoid" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> sigmoid a0 | _ -> assert false)

let sparse_sigmoid a0 =
  Eager_jit.one "nn.functions.sparse_sigmoid" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> sparse_sigmoid a0
      | _ -> assert false)

let silu a0 =
  Eager_jit.one "nn.functions.silu" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> silu a0 | _ -> assert false)

let mish a0 =
  Eager_jit.one "nn.functions.mish" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> mish a0 | _ -> assert false)

let log_sigmoid a0 =
  Eager_jit.one "nn.functions.log_sigmoid" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> log_sigmoid a0 | _ -> assert false)

let elu ?alpha a0 =
  Eager_jit.one "nn.functions.elu" alpha [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> elu ?alpha a0 | _ -> assert false)

let leaky_relu ?negative_slope a0 =
  Eager_jit.one "nn.functions.leaky_relu" negative_slope [] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> leaky_relu ?negative_slope a0
      | _ -> assert false)

let hard_tanh a0 =
  Eager_jit.one "nn.functions.hard_tanh" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> hard_tanh a0 | _ -> assert false)

let celu ?alpha a0 =
  Eager_jit.one "nn.functions.celu" alpha [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> celu ?alpha a0 | _ -> assert false)

let selu a0 =
  Eager_jit.one "nn.functions.selu" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> selu a0 | _ -> assert false)

let glu ?axis a0 =
  Eager_jit.one "nn.functions.glu" axis [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> glu ?axis a0 | _ -> assert false)

let logmeanexp ?axis ?keepdims ?where a0 =
  Eager_jit.one "nn.functions.logmeanexp" (axis, keepdims) [ where ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ where ], [ a0 ] -> logmeanexp ?axis ?keepdims ?where a0
      | _ -> assert false)

let log_softmax ?axis ?where a0 =
  Eager_jit.one "nn.functions.log_softmax" axis [ where ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ where ], [ a0 ] -> log_softmax ?axis ?where a0
      | _ -> assert false)

let standardize ?axis ?mean ?variance ?epsilon ?where ?algorithm a0 =
  Eager_jit.one "nn.functions.standardize" (axis, epsilon, algorithm)
    [ mean; variance; where ] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ mean; variance; where ], [ a0 ] ->
          standardize ?axis ?mean ?variance ?epsilon ?where ?algorithm a0
      | _ -> assert false)

let relu6 a0 =
  Eager_jit.one "nn.functions.relu6" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> relu6 a0 | _ -> assert false)

let hard_sigmoid a0 =
  Eager_jit.one "nn.functions.hard_sigmoid" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a0 ] -> hard_sigmoid a0
      | _ -> assert false)

let hard_silu a0 =
  Eager_jit.one "nn.functions.hard_silu" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> hard_silu a0 | _ -> assert false)

let log1mexp a0 =
  Eager_jit.one "nn.functions.log1mexp" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> log1mexp a0 | _ -> assert false)

let one_hot ?axis ?dtype ~num_classes a1 =
  Eager_jit.one "nn.functions.one_hot" (axis, dtype, num_classes) [] [ a1 ]
    (fun opts vals ->
      match (opts, vals) with
      | [], [ a1 ] -> one_hot ?axis ?dtype ~num_classes a1
      | _ -> assert false)

let softmax ?axis ?where a0 =
  Eager_jit.one "nn.functions.softmax" axis [ where ] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ where ], [ a0 ] -> softmax ?axis ?where a0
      | _ -> assert false)
