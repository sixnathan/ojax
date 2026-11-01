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
    (jit_unary "relu" (fun x -> UF.maximum x (Numpy.Jnp_util.py_int 0)))
    (fun g _ x ->
      let pred = UF.greater x (Numpy.Jnp_util.py_int 0) in
      select_or_zero pred g)

let relu6 =
  custom_unary "relu6"
    (jit_unary "relu6" (fun x ->
         let high = UF.maximum x (Numpy.Jnp_util.py_int 0) in
         UF.minimum high (Numpy.Jnp_util.py_float 6.0)))
    (fun g _ x ->
      let above = UF.greater x (Numpy.Jnp_util.py_int 0) in
      let below = UF.less x (Numpy.Jnp_util.py_int 6) in
      let pred = UF.logical_and above below in
      select_or_zero pred g)

let softplus =
  jit_unary "softplus" (fun x -> UF.logaddexp x (Numpy.Jnp_util.py_int 0))

let soft_sign =
  jit_unary "soft_sign" (fun x ->
      let magnitude = UF.abs x in
      let denom = UF.add magnitude (Numpy.Jnp_util.py_int 1) in
      UF.divide x denom)

let sigmoid x = bind1 T.Logistic [ x ]

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

let py_float = Numpy.Jnp_util.py_float
let py_int = Numpy.Jnp_util.py_int

let sparse_plus =
  jit_unary "sparse_plus" (fun x ->
      where3
        (UF.less_equal x (py_float (-1.0)))
        (py_float 0.0)
        (where3
           (UF.greater_equal x (py_float 1.0))
           x
           (UF.divide
              (UF.power (UF.add x (py_float 1.0)) (py_int 2))
              (py_int 4))))

let sparse_sigmoid =
  jit_unary "sparse_sigmoid" (fun x ->
      UF.multiply (py_float 0.5)
        (NL.clip ~min:0.0 ~max:2.0 (UF.add x (py_float 1.0))))

let hard_tanh =
  jit_unary "hard_tanh" (fun x ->
      where3
        (UF.greater x (py_int 1))
        (py_int 1)
        (where3 (UF.less x (py_int (-1))) (py_int (-1)) x))

let hard_sigmoid =
  jit_unary "hard_sigmoid" (fun x ->
      let shifted = UF.add x (py_float 3.0) in
      let r = relu6 shifted in
      UF.divide r (py_float 6.0))

let hard_silu =
  jit_unary "hard_silu" (fun x ->
      let h = hard_sigmoid x in
      UF.multiply x h)

let jit_scalar name ~default scalar f x =
  match scalar with
  | None ->
      Numpy.Jnp_util.jit1 name
        (function [ x ] -> f x default | _ -> invalid_arg (name ^ ": arity"))
        [ x ]
  | Some value ->
      Numpy.Jnp_util.jit1 name
        (function [ x; s ] -> f x s | _ -> invalid_arg (name ^ ": arity"))
        [ x; py_float value ]

let elu ?alpha x =
  jit_scalar "elu" ~default:(py_float 1.0) alpha
    (fun x alpha ->
      where3
        (UF.greater x (py_int 0))
        x
        (UF.multiply alpha
           (UF.expm1 (where3 (UF.greater x (py_int 0)) (py_float 0.0) x))))
    x

let celu ?alpha x =
  jit_scalar "celu" ~default:(py_float 1.0) alpha
    (fun x alpha ->
      UF.add
        (UF.maximum x (py_float 0.0))
        (UF.multiply alpha
           (UF.expm1 (UF.divide (UF.minimum x (py_float 0.0)) alpha))))
    x

let selu =
  jit_unary "selu" (fun x ->
      let alpha = 1.6732632423543772848170429916717 in
      let scale = 1.0507009873554804934193349852946 in
      UF.multiply (py_float scale) (elu ~alpha x))

let leaky_relu ?negative_slope x =
  jit_scalar "leaky_relu" ~default:(py_float 0.01) negative_slope
    (fun x slope ->
      where3 (UF.greater_equal x (py_int 0)) x (UF.multiply slope x))
    x

let squareplus ?b x =
  jit_scalar "squareplus" ~default:(py_int 4) b
    (fun x b ->
      let y = UF.add x (UF.sqrt (UF.add (UF.square x) b)) in
      UF.divide y (py_int 2))
    x

let log1mexp_primal x =
  let c = UF.log (py_float 2.0) in
  let pred = UF.less x c in
  let nx = UF.negative x in
  let a = UF.log (UF.negative (UF.expm1 nx)) in
  let nx2 = UF.negative x in
  let b = UF.log1p (UF.negative (UF.exp nx2)) in
  where3 pred a b

let log1mexp =
  custom_unary "log1mexp" (jit_unary "log1mexp" log1mexp_primal) (fun g _ x ->
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
    let half_x = UF.multiply (scalar_like x 0.5) x in
    let negated = UF.negative x in
    let e = bind1 T.Erfc [ UF.multiply negated sh ] in
    NL.copy (UF.multiply half_x e)
  end

let glu ?(axis = -1) x =
  match NL.split ~axis x (NL.Count 2) with
  | [ x1; x2 ] -> UF.multiply x1 (sigmoid x2)
  | _ -> invalid_arg "glu: split did not produce two halves"

let numpy_negative_infinity x =
  let dt = dtype x in
  let small_int lowest = if Core.staging () then 0.0 else lowest in
  let value =
    match dt with
    | D.I32 -> -2147483648.0
    | D.I64 -> -9223372036854775808.0
    | D.I8 -> small_int (-128.0)
    | D.I16 -> small_int (-32768.0)
    | D.Bool -> 1.0
    | D.Uint8 | D.Uint16 | D.Uint32 | D.Uint64 -> 0.0
    | _ -> Float.neg_infinity
  in
  const_full dt [||] value

let masked_max ax mask x =
  RED.max ~axis:ax ~keepdims:true ?where:mask
    ~initial:(numpy_negative_infinity x)
    x

let softmax_body ~stop ax mask x =
  let x_max = masked_max ax mask x in
  let x_max = if stop then Custom_derivatives.stop_gradient x_max else x_max in
  let x_safe =
    match mask with
    | None -> x
    | Some m -> where3 m x (py_float Float.neg_infinity)
  in
  let unnormalized = UF.exp (UF.subtract x_safe x_max) in
  let result =
    UF.divide unnormalized
      (RED.sum ~axis:ax ~keepdims:true ?where:mask unnormalized)
  in
  match mask with None -> result | Some m -> where3 m result (py_int 0)

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
  let x_max =
    RED.max ~axis:ax ~keepdims:true ?where:mask
      ~initial:(numpy_negative_infinity x)
      x
  in
  let x_safe =
    match mask with
    | None -> x
    | Some m -> where3 m x (py_float Float.neg_infinity)
  in
  let shifted = UF.subtract x_safe (Custom_derivatives.stop_gradient x_max) in
  let lse =
    UF.log (RED.sum ~axis:ax ~keepdims:true ?where:mask (UF.exp shifted))
  in
  let result = UF.subtract shifted lse in
  match mask with
  | None -> result
  | Some m -> where3 m result (py_float Float.neg_infinity)

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
            NL.clip_values ~min:(py_int 0)
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
  let x = NL.astype x (Dtypes.to_inexact_dtype (dtype x)) in
  let ndim = Array.length (shape x) in
  let dims =
    match axis with
    | None -> Array.init ndim (fun i -> i)
    | Some a -> Array.map (canon_axis ndim) a
  in
  let x =
    match mask with None -> x | Some m -> where3 m x (Numpy.Jnp_util.py_int 0)
  in
  let amax =
    RED.max ~axis:dims ~keepdims ?where:mask
      ~initial:(Numpy.Jnp_util.py_float Float.neg_infinity)
      (UF.real x)
  in
  let amax =
    Custom_derivatives.stop_gradient
      (Lax_api.select (UF.isfinite amax) amax (Lax_api.full_like amax 0.0))
  in
  let with_dims = if keepdims then amax else Lax_api.expand_dims amax dims in
  let exp_a = bind1 T.Exp [ bind1 T.Sub [ x; with_dims ] ] in
  let sumexp = RED.sum ~axis:dims ~keepdims ?where:mask exp_a in
  let _sign = bind1 T.Sign [ sumexp ] in
  let sumexp =
    match dtype sumexp with
    | D.Complex64 | D.Complex128 -> sumexp
    | _ -> UF.abs sumexp
  in
  bind1 T.Add [ bind1 T.Log [ sumexp ]; amax ]

let logmeanexp ?axis ?(keepdims = false) ?where:mask x =
  let ndim = Array.length (shape x) in
  let dims =
    match axis with
    | None -> Array.init ndim (fun i -> i)
    | Some a -> Array.map (canon_axis ndim) a
  in
  let lse = logsumexp ?axis ~keepdims ?where:mask x in
  let count =
    match mask with
    | Some m ->
        RED.sum ~axis:dims ~keepdims ~dtype:(dtype lse)
          (NL.broadcast_to m (shape x))
    | None ->
        let n = Array.fold_left (fun acc d -> acc * (shape x).(d)) 1 dims in
        const_full (dtype lse) [||] (float_of_int n)
  in
  UF.subtract lse (UF.log count)

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

let large_negative dt =
  let dtype_max =
    match dt with
    | D.F16 -> 65504.0
    | D.Bf16 -> 3.3895313892515355e38
    | _ -> 3.4028234663852886e38
  in
  const_full dt [||] (-0.7 *. dtype_max)

let attention_core ~scale ~query ~key ~value ~bias ~mask =
  let module E = Numpy.Einsum in
  let logits = E.einsum "BTNH,BSNH->BNTS" [ query; key ] in
  let logits = UF.multiply logits (const_full (dtype logits) [||] scale) in
  let logits =
    match bias with
    | Some b -> NL.astype (UF.add logits b) (dtype logits)
    | None -> logits
  in
  let padded =
    match mask with
    | None -> logits
    | Some m ->
        let combined =
          UF.logical_and (const_full D.Bool (shape logits) 1.0) m
        in
        where3 combined logits (large_negative (dtype logits))
  in
  let padded = NL.astype padded D.F32 in
  let probs = NL.astype (softmax ~axis:(-1) padded) (dtype key) in
  E.einsum "BNTS,BSNH->BTNH" [ probs; value ]

let dot_product_attention ?bias ?mask ?scale ~query ~key ~value () =
  let output_shape = shape query in
  let ensure_4d v =
    let r = Array.length (shape v) in
    if r < 4 then NL.expand_dims v (Array.init (4 - r) Fun.id) else v
  in
  let query = ensure_4d query in
  let key = ensure_4d key in
  let value = ensure_4d value in
  let bias = Option.map ensure_4d bias in
  let mask = Option.map ensure_4d mask in
  let q_shape = shape query and k_shape = shape key in
  let b, t, n, h = (q_shape.(0), q_shape.(1), q_shape.(2), q_shape.(3)) in
  let k = k_shape.(2) in
  if n mod k <> 0 then
    invalid_arg
      (Printf.sprintf
         "The number of query heads must be a multiple of key/value heads, but \
          got %d vs %d"
         n k);
  let g = n / k in
  let scale =
    match scale with Some s -> s | None -> 1.0 /. sqrt (float_of_int h)
  in
  let query = NL.reshape query [| b; t; k; g; h |] in
  let to_grouped v =
    let s = shape v in
    let tb, tn, tt, ts = (s.(0), s.(1), s.(2), s.(3)) in
    if tn = 1 then
      NL.broadcast_to (NL.expand_dims v [| 2 |]) [| tb; tn; g; tt; ts |]
    else NL.reshape v [| tb; k; g; tt; ts |]
  in
  let bias = Option.map to_grouped bias in
  let mask = Option.map to_grouped mask in
  let extras = Option.to_list bias @ Option.to_list mask in
  let output =
    Interpreters.Batching.vmap_flat ~out_axis:3
      (fun args ->
        match args with
        | query :: key :: value :: rest ->
            let bias, rest =
              match (bias, rest) with
              | Some _, b :: tl -> (Some b, tl)
              | _ -> (None, rest)
            in
            let mask = match rest with m :: _ -> Some m | [] -> None in
            [ attention_core ~scale ~query ~key ~value ~bias ~mask ]
        | _ -> invalid_arg "dot_product_attention: arity")
      ([ Some 3; None; None ] @ List.map (fun _ -> Some 2) extras)
      ([ query; key; value ] @ extras)
  in
  let encoded = NL.reshape (List.hd output) [| b; t; n; h |] in
  NL.reshape encoded output_shape

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

let hard_tanh a0 =
  Eager_jit.one "nn.functions.hard_tanh" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> hard_tanh a0 | _ -> assert false)

let selu a0 =
  Eager_jit.one "nn.functions.selu" () [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> selu a0 | _ -> assert false)

let glu ?axis a0 =
  Eager_jit.one ~jit:"glu" "nn.functions.glu" axis [] [ a0 ] (fun opts vals ->
      match (opts, vals) with [], [ a0 ] -> glu ?axis a0 | _ -> assert false)

let logmeanexp ?axis ?keepdims ?where a0 =
  Eager_jit.one ~jit:"logmeanexp" ~vals_first:true "nn.functions.logmeanexp"
    (axis, keepdims) [ where ] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ where ], [ a0 ] -> logmeanexp ?axis ?keepdims ?where a0
      | _ -> assert false)

let log_softmax ?axis ?where a0 =
  Eager_jit.one ~jit:"log_softmax" ~vals_first:true "nn.functions.log_softmax"
    axis [ where ] [ a0 ] (fun opts vals ->
      match (opts, vals) with
      | [ where ], [ a0 ] -> log_softmax ?axis ?where a0
      | _ -> assert false)

let standardize ?axis ?mean ?variance ?epsilon ?where ?algorithm a0 =
  Eager_jit.one ~jit:"standardize" ~vals_first:true "nn.functions.standardize"
    (axis, epsilon, algorithm) [ mean; variance; where ] [ a0 ]
    (fun opts vals ->
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
  Eager_jit.one ~jit:"_one_hot" "nn.functions.one_hot"
    (axis, dtype, num_classes) [] [ a1 ] (fun opts vals ->
      match (opts, vals) with
      | [], [ a1 ] -> one_hot ?axis ?dtype ~num_classes a1
      | _ -> assert false)
