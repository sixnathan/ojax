module T = Types
module D = Dtype
module L = Lax_api
module NL = Numpy.Lax_numpy
module RED = Numpy.Reductions

type rng_algorithm = Rng_default | Three_fry | Philox

let aval = Core.get_aval
let shape v = (aval v).T.shape
let dtype v = (aval v).T.dtype

let approx_max_k ?(reduction_dimension = -1) operand k =
  L.top_k ~axis:reduction_dimension operand k

let approx_min_k ?(reduction_dimension = -1) operand k =
  let values, indices = L.top_k ~axis:reduction_dimension (L.neg operand) k in
  (L.neg values, indices)

let broadcast_like arr like = NL.broadcast_to arr (shape like)
let create_token () = Core.bind1 T.Create_token []
let after_all tokens = Core.bind1 T.After_all tokens

let ragged_dot lhs rhs group_sizes =
  let m = (shape lhs).(0) in
  let g = (shape rhs).(0) in
  let idt = dtype group_sizes in
  let cumulative = L.cumsum ~axis:0 group_sizes in
  let rows =
    T.Concrete (Ndarray.of_floats idt [| m; 1 |] (Array.init m float_of_int))
  in
  let past =
    L.ge
      (NL.broadcast_to rows [| m; g |])
      (NL.broadcast_to (L.reshape cumulative [| 1; g |]) [| m; g |])
  in
  let group = RED.sum ~axis:[| 1 |] (NL.astype past D.I32) in
  let selected = Numpy.Indexing.take ~axis:0 rhs group in
  L.dot_general lhs selected ~lhs_contract:[| 1 |] ~rhs_contract:[| 1 |]
    ~lhs_batch:[| 0 |] ~rhs_batch:[| 0 |]

let conv_general_dilated_local ?window_strides ?padding ?lhs_dilation
    ?rhs_dilation ?dimension_numbers lhs rhs ~filter_shape =
  let rank = Array.length (shape lhs) in
  let identity = Array.init rank Fun.id in
  let dn =
    match dimension_numbers with
    | Some d -> d
    | None ->
        { T.lhs_spec = identity; rhs_spec = identity; out_spec = identity }
  in
  let patches =
    L.conv_general_dilated_patches ?window_strides ?padding ?lhs_dilation
      ?rhs_dilation ~dimension_numbers:dn lhs ~filter_shape
  in
  let spatial = Array.length filter_shape in
  let lhs_b = Array.sub dn.out_spec 2 spatial in
  let rhs_b = Array.sub dn.rhs_spec 2 spatial in
  let order = Array.init spatial Fun.id in
  Array.sort (fun a b -> compare lhs_b.(a) lhs_b.(b)) order;
  let out =
    L.dot_general patches rhs
      ~lhs_contract:[| dn.out_spec.(1) |]
      ~rhs_contract:[| dn.rhs_spec.(1) |]
      ~lhs_batch:(Array.map (fun i -> lhs_b.(i)) order)
      ~rhs_batch:(Array.map (fun i -> rhs_b.(i)) order)
  in
  NL.moveaxis [| -2; -1 |] [| dn.out_spec.(0); dn.out_spec.(1) |] out

let rng_bit_generator ?(dtype = D.Uint32) ?(algorithm = Rng_default) key
    out_shape =
  (match dtype with
  | D.Uint8 | D.Uint16 | D.Uint32 | D.Uint64 -> ()
  | _ -> invalid_arg "rng_bit_generator: unsupported dtype");
  let key_aval = aval key in
  let wide = Stablehlo.Ir.tensor_type D.Uint64 [| 2 |] in
  let vals_ty = Stablehlo.Ir.tensor_type dtype out_shape in
  let key_ty = Stablehlo.Ir.tensor_type key_aval.T.dtype key_aval.T.shape in
  let attr =
    match algorithm with
    | Rng_default -> "DEFAULT"
    | Three_fry -> "THREE_FRY"
    | Philox -> "PHILOX"
  in
  let narrow =
    match (key_aval.T.dtype, key_aval.T.shape) with
    | D.Uint32, [| 4 |] -> true
    | D.Uint64, [| 2 |] -> false
    | _ -> invalid_arg "rng_bit_generator: key must be u32[4] or u64[2]"
  in
  let pair = Stablehlo.Ir.tensor_type D.Uint32 [| 2; 2 |] in
  let pre =
    if narrow then
      Printf.sprintf
        "    %%r = stablehlo.reshape %%arg0 : (%s) -> %s\n\
        \    %%k = stablehlo.bitcast_convert %%r : (%s) -> %s\n"
        key_ty pair pair wide
    else ""
  in
  let input = if narrow then "%k" else "%arg0" in
  let post =
    if narrow then
      Printf.sprintf
        "    %%b = stablehlo.bitcast_convert %%o : (%s) -> %s\n\
        \    %%ok = stablehlo.reshape %%b : (%s) -> %s\n"
        wide pair pair key_ty
    else ""
  in
  let result = if narrow then "%ok" else "%o" in
  let text =
    Printf.sprintf
      "module {\n\
      \  func.func public @main(%%arg0: %s) -> (%s, %s) {\n\
       %s    %%o, %%v = \"stablehlo.rng_bit_generator\"(%s) <{rng_algorithm = \
       #stablehlo<rng_algorithm %s>}> : (%s) -> (%s, %s)\n\
       %s    return %s, %%v : %s, %s\n\
      \  }\n\
       }\n"
      key_ty key_ty vals_ty pre input attr wide wide vals_ty post result key_ty
      vals_ty
  in
  match
    Custom_derivatives.pure_callback
      (fun args -> Backend.run_text text args)
      ~result_avals:[ key_aval; { key_aval with T.shape = out_shape; dtype } ]
      [ key ]
  with
  | [ next_key; values ] -> (next_key, values)
  | _ -> failwith "rng_bit_generator: arity"
