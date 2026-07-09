open Types

let b1 = Core.bind1
let add a b = b1 Add [ a; b ]
let sub a b = b1 Sub [ a; b ]
let mul a b = b1 Mul [ a; b ]
let div a b = b1 Div [ a; b ]
let neg a = b1 Neg [ a ]
let zeros_like_value = Ad_util.zeros_like_value

let ones_like_aval a =
  let n = Array.fold_left ( * ) 1 a.shape in
  Concrete (Ndarray.of_floats a.dtype a.shape (Array.make n 1.0))

let ones_like_value v = ones_like_aval (Core.get_aval v)
let arity () = failwith "ad: rule arity mismatch"

let const_like v c =
  let a = Core.get_aval v in
  let n = Array.fold_left ( * ) 1 a.shape in
  Concrete (Ndarray.of_floats a.dtype a.shape (Array.make n c))

let square x = mul x x
let rsqrt z = b1 Pow [ z; const_like z (-0.5) ]
let recip z = div (ones_like_value z) z
let sinh_of x = div (sub (b1 Exp [ x ]) (b1 Exp [ neg x ])) (const_like x 2.0)
let pi_const = 4.0 *. Float.atan 1.0
let two_over_sqrt_pi = 2.0 /. Float.sqrt pi_const
let sqrt_pi_over_2 = Float.sqrt pi_const /. 2.0
let mem arr x = Array.exists (fun y -> y = x) arr

let gather_to_scatter (gd : gather_dims) : scatter_dims =
  {
    update_window_dims = gd.offset_dims;
    inserted_window_dims = gd.collapsed_slice_dims;
    scatter_dims_to_operand_dims = gd.start_index_map;
    s_operand_batching_dims = gd.g_operand_batching_dims;
    s_scatter_indices_batching_dims = gd.g_start_indices_batching_dims;
  }

let scatter_to_gather (sd : scatter_dims) : gather_dims =
  {
    offset_dims = sd.update_window_dims;
    collapsed_slice_dims = sd.inserted_window_dims;
    start_index_map = sd.scatter_dims_to_operand_dims;
    g_operand_batching_dims = sd.s_operand_batching_dims;
    g_start_indices_batching_dims = sd.s_scatter_indices_batching_dims;
  }

let scatter_gather_slice_sizes (sd : scatter_dims) operand_shape updates_shape =
  let op_rank = Array.length operand_shape in
  let wod =
    let out = ref [] in
    for o = op_rank - 1 downto 0 do
      if
        (not (mem sd.inserted_window_dims o))
        && not (mem sd.s_operand_batching_dims o)
      then out := o :: !out
    done;
    Array.of_list !out
  in
  Array.init op_rank (fun o ->
      if mem sd.inserted_window_dims o || mem sd.s_operand_batching_dims o then
        1
      else
        let rec find j =
          if wod.(j) = o then updates_shape.(sd.update_window_dims.(j))
          else find (j + 1)
        in
        find 0)

let reduce_chooser_jvp prim axes x tx =
  let a = Core.get_aval x in
  let shape = a.shape in
  let ndim = Array.length shape in
  let is_red = Array.make ndim false in
  Array.iter (fun i -> is_red.(i) <- true) axes;
  let shape_with_1 =
    Array.mapi (fun i d -> if is_red.(i) then 1 else d) shape
  in
  let ident = Array.init ndim (fun i -> i) in
  let ans = b1 prim [ x ] in
  let reshaped = b1 (Reshape shape_with_1) [ ans ] in
  let bcast = b1 (Broadcast_in_dim { shape; dims = ident }) [ reshaped ] in
  let loc = b1 (Convert_element_type a.dtype) [ b1 Eq [ x; bcast ] ] in
  let counts = b1 (Reduce_sum axes) [ loc ] in
  let numer = b1 (Reduce_sum axes) [ mul tx loc ] in
  (ans, div numer counts)

let jvp_rule prim (primals : value list) (tangents : value list) : value * value
    =
  match prim with
  | Add -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] -> (add x y, add tx ty)
      | _ -> arity ())
  | Sub -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] -> (sub x y, sub tx ty)
      | _ -> arity ())
  | Mul -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] -> (mul x y, add (mul tx y) (mul x ty))
      | _ -> arity ())
  | Div -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          (div x y, add (div tx y) (div (neg (mul x ty)) (mul y y)))
      | _ -> arity ())
  | Neg -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (neg x, neg tx)
      | _ -> arity ())
  | Sin -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Sin [ x ], mul (b1 Cos [ x ]) tx)
      | _ -> arity ())
  | Cos -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Cos [ x ], mul (neg (b1 Sin [ x ])) tx)
      | _ -> arity ())
  | Exp -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 Exp [ x ] in
          (po, mul po tx)
      | _ -> arity ())
  | Log -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Log [ x ], div tx x)
      | _ -> arity ())
  | Tanh -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 Tanh [ x ] in
          (po, mul (sub (ones_like_value po) (mul po po)) tx)
      | _ -> arity ())
  | Max -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          (b1 Max [ x; y ], b1 Select_n [ b1 Gt [ x; y ]; ty; tx ])
      | _ -> arity ())
  | Min -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          (b1 Min [ x; y ], b1 Select_n [ b1 Lt [ x; y ]; ty; tx ])
      | _ -> arity ())
  | Pow -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          let po = b1 Pow [ x; y ] in
          let one = ones_like_value y in
          let dx = mul (mul y (b1 Pow [ x; sub y one ])) tx in
          let dy = mul (mul (b1 Log [ x ]) po) ty in
          (po, add dx dy)
      | _ -> arity ())
  | Abs -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Abs [ x ], mul (b1 Sign [ x ]) tx)
      | _ -> arity ())
  | Sign -> (
      match (primals, tangents) with
      | [ x ], [ _ ] ->
          let po = b1 Sign [ x ] in
          (po, zeros_like_value po)
      | _ -> arity ())
  | Eq | Lt | Gt | Ge | Le | Eq_to | Le_to | Lt_to | And | Mulhi | Ne | Or | Xor
  | Shift_left | Shift_right_arithmetic | Shift_right_logical -> (
      match (primals, tangents) with
      | [ x; y ], [ _; _ ] ->
          let po = b1 prim [ x; y ] in
          (po, zeros_like_value po)
      | _ -> arity ())
  | Rem -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          let q = div x y in
          let corr = mul (b1 Sign [ q ]) (b1 Floor [ b1 Abs [ q ] ]) in
          (b1 Rem [ x; y ], add tx (mul (neg ty) corr))
      | _ -> arity ())
  | Atan2 -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          let denom = add (square x) (square y) in
          ( b1 Atan2 [ x; y ],
            add (mul tx (div y denom)) (mul ty (div (neg x) denom)) )
      | _ -> arity ())
  | Complex -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] -> (b1 Complex [ x; y ], b1 Complex [ tx; ty ])
      | _ -> arity ())
  | Select_n -> (
      match (primals, tangents) with
      | which :: cases, _ :: tcases ->
          (b1 Select_n (which :: cases), b1 Select_n (which :: tcases))
      | _ -> arity ())
  | Convert_element_type dt -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 (Convert_element_type dt) [ x ] in
          let to_ =
            match dt with
            | Dtype.F32 | Dtype.F64 -> b1 (Convert_element_type dt) [ tx ]
            | _ -> zeros_like_value po
          in
          (po, to_)
      | _ -> arity ())
  | Broadcast_in_dim p -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          (b1 (Broadcast_in_dim p) [ x ], b1 (Broadcast_in_dim p) [ tx ])
      | _ -> arity ())
  | Reshape ns -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 (Reshape ns) [ x ], b1 (Reshape ns) [ tx ])
      | _ -> arity ())
  | Reduce_sum ax -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 (Reduce_sum ax) [ x ], b1 (Reduce_sum ax) [ tx ])
      | _ -> arity ())
  | Dot_general dd -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          ( b1 (Dot_general dd) [ x; y ],
            add (b1 (Dot_general dd) [ tx; y ]) (b1 (Dot_general dd) [ x; ty ])
          )
      | _ -> arity ())
  | Conv_general_dilated _ -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] ->
          (b1 prim [ x; y ], add (b1 prim [ tx; y ]) (b1 prim [ x; ty ]))
      | _ -> arity ())
  | Asin -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          (b1 Asin [ x ], mul (rsqrt (sub (ones_like_value x) (square x))) tx)
      | _ -> arity ())
  | Acos -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          ( b1 Acos [ x ],
            mul (neg (rsqrt (sub (ones_like_value x) (square x)))) tx )
      | _ -> arity ())
  | Atan -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          (b1 Atan [ x ], div tx (add (ones_like_value x) (square x)))
      | _ -> arity ())
  | Asinh -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          (b1 Asinh [ x ], mul (rsqrt (add (square x) (ones_like_value x))) tx)
      | _ -> arity ())
  | Acosh -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          (b1 Acosh [ x ], mul (rsqrt (sub (square x) (ones_like_value x))) tx)
      | _ -> arity ())
  | Atanh -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          ( b1 Atanh [ x ],
            mul
              (recip (add (ones_like_value x) x))
              (div tx (sub (ones_like_value x) x)) )
      | _ -> arity ())
  | Cbrt -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 Cbrt [ x ] in
          ( po,
            mul
              (mul
                 (const_like x (1.0 /. 3.0))
                 (b1 Pow [ po; const_like po (-2.0) ]))
              tx )
      | _ -> arity ())
  | Cosh -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Cosh [ x ], mul (sinh_of x) tx)
      | _ -> arity ())
  | Exp2 -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 Exp2 [ x ] in
          (po, mul (const_like x (Float.log 2.0)) (mul tx po))
      | _ -> arity ())
  | Expm1 -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 Expm1 [ x ] in
          (po, mul tx (add po (ones_like_value po)))
      | _ -> arity ())
  | Ceil -> (
      match (primals, tangents) with
      | [ x ], [ _ ] ->
          let po = b1 Ceil [ x ] in
          (po, zeros_like_value po)
      | _ -> arity ())
  | Floor -> (
      match (primals, tangents) with
      | [ x ], [ _ ] ->
          let po = b1 Floor [ x ] in
          (po, zeros_like_value po)
      | _ -> arity ())
  | Conj -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Conj [ x ], b1 Conj [ tx ])
      | _ -> arity ())
  | Copy -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Copy [ x ], b1 Copy [ tx ])
      | _ -> arity ())
  | Integer_pow y -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 (Integer_pow y) [ x ] in
          let dydx =
            if y = 0 then zeros_like_value x
            else if y = 1 then tx
            else if y = 2 then mul tx (mul (const_like x 2.0) x)
            else
              mul tx
                (mul
                   (const_like x (float_of_int y))
                   (b1 (Integer_pow (y - 1)) [ x ]))
          in
          (po, dydx)
      | _ -> arity ())
  | Log1p -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Log1p [ x ], div tx (add x (ones_like_value x)))
      | _ -> arity ())
  | Logistic -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 Logistic [ x ] in
          (po, mul tx (mul po (sub (ones_like_value po) po)))
      | _ -> arity ())
  | Sinh -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Sinh [ x ], mul (b1 Cosh [ x ]) tx)
      | _ -> arity ())
  | Sqrt -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 Sqrt [ x ] in
          (po, mul tx (div (const_like x 0.5) po))
      | _ -> arity ())
  | Rsqrt -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 Rsqrt [ x ] in
          (po, mul tx (mul (const_like x (-0.5)) (div po x)))
      | _ -> arity ())
  | Square -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Square [ x ], mul tx (mul (const_like x 2.0) x))
      | _ -> arity ())
  | Tan -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let po = b1 Tan [ x ] in
          (po, mul tx (add (ones_like_value po) (mul po po)))
      | _ -> arity ())
  | Real -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Real [ x ], b1 Real [ tx ])
      | _ -> arity ())
  | Imag -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Imag [ x ], b1 Imag [ tx ])
      | _ -> arity ())
  | Is_finite -> (
      match (primals, tangents) with
      | [ x ], [ _ ] ->
          let po = b1 Is_finite [ x ] in
          (po, zeros_like_value po)
      | _ -> arity ())
  | Round -> (
      match (primals, tangents) with
      | [ x ], [ _ ] ->
          let po = b1 Round [ x ] in
          (po, zeros_like_value po)
      | _ -> arity ())
  | Not -> (
      match (primals, tangents) with
      | [ x ], [ _ ] ->
          let po = b1 Not [ x ] in
          (po, zeros_like_value po)
      | _ -> arity ())
  | Reduce_max axes -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> reduce_chooser_jvp (Reduce_max axes) axes x tx
      | _ -> arity ())
  | Reduce_min axes -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> reduce_chooser_jvp (Reduce_min axes) axes x tx
      | _ -> arity ())
  | Reduce_and _ | Reduce_or _ | Reduce_xor _ | Argmax _ | Argmin _ -> (
      match (primals, tangents) with
      | [ x ], [ _ ] ->
          let po = b1 prim [ x ] in
          (po, zeros_like_value po)
      | _ -> arity ())
  | Reduce_prod _ ->
      failwith "ad: reduce_prod jvp needs the variadic reduce tree (M2 gap)"
  | Reduce _ ->
      failwith "ad: general reduce jvp needs the variadic reduce tree (M2 gap)"
  | Population_count -> failwith "ad: population_count has no jvp rule"
  | Clz -> failwith "ad: clz has no jvp rule"
  | Nextafter -> failwith "ad: nextafter has no jvp rule"
  | Concatenate _ | Pad _ | Rev _ | Squeeze _ | Stack _ | Tile _ | Transpose _
  | Slice _ ->
      (b1 prim primals, b1 prim tangents)
  | Dynamic_slice p -> (
      match (primals, tangents) with
      | operand :: idx, t_op :: _ ->
          ( Core.bind1 (Dynamic_slice p) (operand :: idx),
            Core.bind1 (Dynamic_slice p) (t_op :: idx) )
      | _ -> arity ())
  | Dynamic_update_slice -> (
      match (primals, tangents) with
      | operand :: update :: idx, t_op :: t_up :: _ ->
          ( Core.bind1 Dynamic_update_slice (operand :: update :: idx),
            Core.bind1 Dynamic_update_slice (t_op :: t_up :: idx) )
      | _ -> arity ())
  | Gather { dimension_numbers; slice_sizes } -> (
      match (primals, tangents) with
      | [ operand; indices ], [ t_op; _ ] ->
          ( Core.bind1
              (Gather { dimension_numbers; slice_sizes })
              [ operand; indices ],
            Core.bind1
              (Gather { dimension_numbers; slice_sizes })
              [ t_op; indices ] )
      | _ -> arity ())
  | Scatter_add { dimension_numbers } -> (
      match (primals, tangents) with
      | [ op; idx; up ], [ t_op; _; t_up ] ->
          ( Core.bind1 (Scatter_add { dimension_numbers }) [ op; idx; up ],
            Core.bind1 (Scatter_add { dimension_numbers }) [ t_op; idx; t_up ]
          )
      | _ -> arity ())
  | Scatter_sub { dimension_numbers } -> (
      match (primals, tangents) with
      | [ op; idx; up ], [ t_op; _; t_up ] ->
          ( Core.bind1 (Scatter_sub { dimension_numbers }) [ op; idx; up ],
            Core.bind1 (Scatter_sub { dimension_numbers }) [ t_op; idx; t_up ]
          )
      | _ -> arity ())
  | Scatter { dimension_numbers; unique_indices } -> (
      if not unique_indices then
        failwith "ad: scatter jvp only implemented for unique_indices (M2)"
      else
        match (primals, tangents) with
        | [ op; idx; up ], [ t_op; _; t_up ] ->
            ( Core.bind1
                (Scatter { dimension_numbers; unique_indices })
                [ op; idx; up ],
              Core.bind1
                (Scatter { dimension_numbers; unique_indices })
                [ t_op; idx; t_up ] )
        | _ -> arity ())
  | Scatter_mul { dimension_numbers; unique_indices } -> (
      if not unique_indices then
        failwith "ad: scatter_mul jvp only implemented for unique_indices (M2)"
      else
        match (primals, tangents) with
        | [ op; idx; up ], [ t_op; _; t_up ] ->
            let po =
              Core.bind1
                (Scatter_mul { dimension_numbers; unique_indices })
                [ op; idx; up ]
            in
            let term_op =
              Core.bind1
                (Scatter_mul { dimension_numbers; unique_indices })
                [ t_op; idx; up ]
            in
            let zeros = zeros_like_value op in
            let term_up =
              mul op
                (Core.bind1
                   (Scatter_add { dimension_numbers })
                   [ zeros; idx; t_up ])
            in
            (po, add term_op term_up)
        | _ -> arity ())
  | Scatter_min _ | Scatter_max _ ->
      failwith
        "ad: scatter_min/scatter_max jvp needs the extremal averaging rule (M2 \
         gap)"
  | Split _ | Unstack _ | Optimization_barrier | Sort _ | Top_k _ | Scan _
  | While _ | Custom_linear_solve _ ->
      failwith "ad: multi-output jvp handled by jvp_process_primitive"
  | Reduce_precision p -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          (b1 (Reduce_precision p) [ x ], b1 (Reduce_precision p) [ tx ])
      | _ -> arity ())
  | Tie -> (
      match (primals, tangents) with
      | [ x; y ], [ _; ty ] -> (b1 Tie [ x; y ], ty)
      | _ -> arity ())
  | Clamp -> (
      match (primals, tangents) with
      | [ mn; x; mx ], [ tmn; tx; tmx ] ->
          let po = b1 Clamp [ mn; x; mx ] in
          let zeros = zeros_like_value x in
          let sel pred g = b1 Select_n [ pred; zeros; g ] in
          let t_mn = sel (b1 And [ b1 Gt [ mn; x ]; b1 Lt [ mn; mx ] ]) tmn in
          let t_x = sel (b1 And [ b1 Gt [ x; mn ]; b1 Lt [ x; mx ] ]) tx in
          let t_mx = sel (b1 Lt [ mx; x ]) tmx in
          (po, add (add t_mn t_x) t_mx)
      | _ -> arity ())
  | Bitcast_convert_type _ -> (
      match (primals, tangents) with
      | [ x ], [ _ ] ->
          let po = b1 prim [ x ] in
          (po, zeros_like_value po)
      | _ -> arity ())
  | Reduce_window_sum window -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          ( b1 (Reduce_window_sum window) [ x ],
            b1 (Reduce_window_sum window) [ tx ] )
      | _ -> arity ())
  | Reduce_window_max window -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          ( b1 (Reduce_window_max window) [ x ],
            b1 (Select_and_gather_add { select = Wge; window }) [ tx; x ] )
      | _ -> arity ())
  | Reduce_window_min window -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          ( b1 (Reduce_window_min window) [ x ],
            b1 (Select_and_gather_add { select = Wle; window }) [ tx; x ] )
      | _ -> arity ())
  | Select_and_gather_add { select; window } -> (
      match (primals, tangents) with
      | [ t; op ], [ g_t; _ ] ->
          ( b1 (Select_and_gather_add { select; window }) [ t; op ],
            b1 (Select_and_gather_add { select; window }) [ g_t; op ] )
      | _ -> arity ())
  | Select_and_scatter_add { select; window } -> (
      match (primals, tangents) with
      | [ src; op ], [ g_src; _ ] ->
          ( b1 (Select_and_scatter_add { select; window }) [ src; op ],
            b1 (Select_and_scatter_add { select; window }) [ g_src; op ] )
      | _ -> arity ())
  | Reduce_window _ ->
      failwith
        "ad: reduce_window jvp needs the variadic jvp reducer jaxpr (M2 gap)"
  | Select_and_scatter _ ->
      failwith "ad: select_and_scatter has no jvp rule in M1"
  | Lgamma -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 Lgamma [ x ], mul (b1 Digamma [ x ]) tx)
      | _ -> arity ())
  | Digamma -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          (b1 Digamma [ x ], mul (b1 Polygamma [ const_like x 1.0; x ]) tx)
      | _ -> arity ())
  | Erf -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          ( b1 Erf [ x ],
            mul
              (const_like x two_over_sqrt_pi)
              (mul tx (b1 Exp [ neg (square x) ])) )
      | _ -> arity ())
  | Erfc -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          ( b1 Erfc [ x ],
            mul
              (const_like x (-.two_over_sqrt_pi))
              (mul tx (b1 Exp [ neg (square x) ])) )
      | _ -> arity ())
  | Erf_inv -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let ans = b1 Erf_inv [ x ] in
          ( ans,
            mul (const_like x sqrt_pi_over_2) (mul tx (b1 Exp [ square ans ]))
          )
      | _ -> arity ())
  | Bessel_i0e -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let y = b1 Bessel_i0e [ x ] in
          (y, mul tx (sub (b1 Bessel_i1e [ x ]) (mul (b1 Sign [ x ]) y)))
      | _ -> arity ())
  | Bessel_i1e -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          let y = b1 Bessel_i1e [ x ] in
          let eps = const_like x 2.220446049250313e-16 in
          let not_tiny = b1 Gt [ b1 Abs [ x ]; eps ] in
          let safe_x = b1 Select_n [ not_tiny; eps; x ] in
          let dy_dx =
            sub (b1 Bessel_i0e [ safe_x ])
              (mul y (add (b1 Sign [ safe_x ]) (recip safe_x)))
          in
          let dy_dx = b1 Select_n [ not_tiny; const_like x 0.5; dy_dx ] in
          (y, mul tx dy_dx)
      | _ -> arity ())
  | Igamma -> (
      match (primals, tangents) with
      | [ a; x ], [ ta; tx ] ->
          let po = b1 Igamma [ a; x ] in
          let one = ones_like_value a in
          let gradx =
            b1 Exp
              [
                add
                  (add (neg x) (mul (sub a one) (b1 Log [ x ])))
                  (neg (b1 Lgamma [ a ]));
              ]
          in
          let da = mul ta (b1 Igamma_grad_a [ a; x ]) in
          let dx = mul tx gradx in
          (po, add da dx)
      | _ -> arity ())
  | Igammac -> (
      match (primals, tangents) with
      | [ a; x ], [ ta; tx ] ->
          let po = b1 Igammac [ a; x ] in
          let one = ones_like_value a in
          let gradx =
            b1 Exp
              [
                add
                  (add (neg x) (mul (sub a one) (b1 Log [ x ])))
                  (neg (b1 Lgamma [ a ]));
              ]
          in
          let da = mul ta (neg (b1 Igamma_grad_a [ a; x ])) in
          let dx = mul tx (neg gradx) in
          (po, add da dx)
      | _ -> arity ())
  | Polygamma -> (
      match (primals, tangents) with
      | [ m; x ], [ _; tx ] ->
          let po = b1 Polygamma [ m; x ] in
          (po, mul tx (b1 Polygamma [ add m (ones_like_value m); x ]))
      | _ -> arity ())
  | Regularized_incomplete_beta -> (
      match (primals, tangents) with
      | [ a; b; x ], [ _; _; tx ] ->
          let po = b1 Regularized_incomplete_beta [ a; b; x ] in
          let one = ones_like_value a in
          let lbeta =
            sub
              (add (b1 Lgamma [ a ]) (b1 Lgamma [ b ]))
              (b1 Lgamma [ add a b ])
          in
          let partial_x =
            b1 Exp
              [
                add
                  (add
                     (mul (sub b one) (b1 Log1p [ neg x ]))
                     (mul (sub a one) (b1 Log [ x ])))
                  (neg lbeta);
              ]
          in
          (po, mul partial_x tx)
      | _ -> arity ())
  | Cumsum { axis; reverse } -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          ( b1 (Cumsum { axis; reverse }) [ x ],
            b1 (Cumsum { axis; reverse }) [ tx ] )
      | _ -> arity ())
  | Cumprod _ | Cummax _ | Cummin _ | Cumlogsumexp _ ->
      failwith
        "ad: cumprod/cummax/cummin/cumlogsumexp jvp needs associative_scan (M2 \
         gap)"
  | Igamma_grad_a | Zeta ->
      failwith "ad: primitive has no jvp rule (matches jax)"
  | Iota _ | Empty _ | Empty2 _ | Create_token | After_all | Composite _
  | Dce_sink | From_edtype _ | Ragged_dot_general | Rng_bit_generator
  | Rng_uniform | To_edtype _ | Platform_index _ ->
      failwith "ad: primitive has no jvp rule in M1"
  | Cond _ -> failwith "ad: cond jvp handled by jvp_process_primitive"
  | Xla_call _ -> failwith "ad: jvp of xla_call not supported in M1"

let new_jvp_tracer trace primal tangent : tracer =
  {
    id = Core.fresh_id ();
    trace;
    aval = Core.get_aval primal;
    payload = JVP { primal; tangent };
  }

let jvp_pure trace v = Tracer (new_jvp_tracer trace v (zeros_like_value v))

let as_jvp trace v =
  match v with
  | Tracer t when t.trace.level = trace.level -> (
      match t.payload with
      | JVP { primal; tangent } -> (primal, tangent)
      | _ -> failwith "ad: expected a JVP tracer")
  | _ -> (v, zeros_like_value v)

let jvp (f : value list -> value list) (primals : value list)
    (tangents : value list) : value list * value list =
  Core.with_new_main KJVP GNone (fun main ->
      let ins =
        List.map2 (fun x t -> Tracer (new_jvp_tracer main x t)) primals tangents
      in
      let outs = f ins in
      let pairs =
        List.map
          (fun v ->
            match Core.full_raise main v with
            | Tracer { payload = JVP { primal; tangent }; _ } ->
                (primal, tangent)
            | _ -> failwith "ad.jvp: expected a JVP tracer")
          outs
      in
      List.split pairs)

let cond_tangent_branch jvp_f (branch : closed_jaxpr) (op_avals : aval list) :
    closed_jaxpr =
  let n = List.length op_avals in
  Jaxpr.make_jaxpr (op_avals @ op_avals) (fun args ->
      match Util.split_list args [ n ] with
      | [ prim_ops; tan_ops ] ->
          let _, tans =
            jvp_f (fun a -> Jaxpr.eval_closed_jaxpr branch a) prim_ops tan_ops
          in
          tans
      | _ -> failwith "ad: cond tangent branch split")

let jvp_process_primitive trace prim args =
  let pairs = List.map (as_jvp trace) args in
  let primals = List.map fst pairs and tangents = List.map snd pairs in
  match prim with
  | Split _ | Unstack _ | Optimization_barrier ->
      let pos = Core.bind prim primals in
      let tos = Core.bind prim tangents in
      List.map2 (fun po to_ -> Tracer (new_jvp_tracer trace po to_)) pos tos
  | Sort _ | Top_k _ -> failwith "ad: jvp of sort/top_k needs gather (M2 gap)"
  | Scan { length; reverse; num_carry; jaxpr } ->
      let split2 l n =
        match Util.split_list l [ n ] with [ a; b ] -> (a, b) | _ -> arity ()
      in
      let mapped_leading (a : aval) =
        let n = Array.length a.shape in
        { a with shape = Array.sub a.shape 1 (n - 1) }
      in
      let nc = num_carry in
      let carry, xs = split2 primals nc in
      let carry_t, xs_t = split2 tangents nc in
      let nx = List.length xs in
      let num_ys = List.length jaxpr.jaxpr.outs - nc in
      let carry_avals = List.map Core.get_aval carry in
      let x_slice_avals =
        List.map (fun x -> mapped_leading (Core.get_aval x)) xs
      in
      let new_body =
        Jaxpr.make_jaxpr
          (carry_avals @ carry_avals @ x_slice_avals @ x_slice_avals)
          (fun args ->
            let pc, r1 = split2 args nc in
            let tc, r2 = split2 r1 nc in
            let px, tx = split2 r2 nx in
            let po, to_ =
              jvp (fun a -> Jaxpr.eval_closed_jaxpr jaxpr a) (pc @ px) (tc @ tx)
            in
            let pc', py = split2 po nc in
            let tc', ty = split2 to_ nc in
            pc' @ tc' @ py @ ty)
      in
      let new_operands = carry @ carry_t @ xs @ xs_t in
      let out =
        Core.bind
          (Scan { length; reverse; num_carry = 2 * nc; jaxpr = new_body })
          new_operands
      in
      let pcarry, r1 = split2 out nc in
      let tcarry, r2 = split2 r1 nc in
      let pys, tys = split2 r2 num_ys in
      let primal_outs = pcarry @ pys in
      let tangent_outs = tcarry @ tys in
      List.map2
        (fun po to_ -> Tracer (new_jvp_tracer trace po to_))
        primal_outs tangent_outs
  | While { cond; body } ->
      let split2 l n =
        match Util.split_list l [ n ] with [ a; b ] -> (a, b) | _ -> arity ()
      in
      let nc = List.length primals in
      let carry_avals = List.map Core.get_aval primals in
      let new_body =
        Jaxpr.make_jaxpr (carry_avals @ carry_avals) (fun args ->
            let pc, tc = split2 args nc in
            let po, to_ = jvp (fun a -> Jaxpr.eval_closed_jaxpr body a) pc tc in
            po @ to_)
      in
      let new_cond =
        Jaxpr.make_jaxpr (carry_avals @ carry_avals) (fun args ->
            let pc, _ = split2 args nc in
            Jaxpr.eval_closed_jaxpr cond pc)
      in
      let out =
        Core.bind
          (While { cond = new_cond; body = new_body })
          (primals @ tangents)
      in
      let pcarry, tcarry = split2 out nc in
      List.map2
        (fun po to_ -> Tracer (new_jvp_tracer trace po to_))
        pcarry tcarry
  | Cond { t; f } -> (
      match (primals, tangents) with
      | pred :: prim_ops, _ :: tan_ops ->
          let op_avals = List.map Core.get_aval prim_ops in
          let primal_outs = Core.bind (Cond { t; f }) primals in
          let t_tan = cond_tangent_branch jvp t op_avals in
          let f_tan = cond_tangent_branch jvp f op_avals in
          let tangent_outs =
            Core.bind
              (Cond { t = t_tan; f = f_tan })
              ((pred :: prim_ops) @ tan_ops)
          in
          List.map2
            (fun po to_ -> Tracer (new_jvp_tracer trace po to_))
            primal_outs tangent_outs
      | _ -> arity ())
  | Custom_linear_solve _ ->
      let x = Core.bind prim primals in
      let x_dot = Core.bind prim tangents in
      List.map2 (fun po to_ -> Tracer (new_jvp_tracer trace po to_)) x x_dot
  | _ ->
      let po, to_ = jvp_rule prim primals tangents in
      [ Tracer (new_jvp_tracer trace po to_) ]

let interpreter : Core.interpreter =
  {
    i_pure = jvp_pure;
    i_lift = jvp_pure;
    i_full_lower = (fun v -> v);
    i_process_primitive = jvp_process_primitive;
    i_process_custom_jvp =
      (fun _ ~primal:_ ~jvp:_ _ ->
        failwith "ad: custom_jvp not supported in M1");
    i_process_custom_vjp =
      (fun _ ~primal:_ ~fwd:_ ~bwd:_ _ ->
        failwith "ad: custom_vjp not supported in M1");
  }

let install () = Core.register_interpreter KJVP interpreter
let () = install ()

let linearize (f : value list -> value list) (primals : value list) :
    value list * (value list -> value list) =
  let pvals_in =
    List.map Partial_eval.partial_val_known primals
    @ List.map
        (fun x -> Partial_eval.partial_val_unknown (Core.get_aval x))
        primals
  in
  let f_jvp inputs =
    let px, tx = Util.split_half inputs in
    let po, to_ = jvp f px tx in
    po @ to_
  in
  let jaxpr, consts, pvals_out =
    Partial_eval.partial_eval_flat f_jvp pvals_in
  in
  let primal_pvals, _ = Util.split_half pvals_out in
  let primals_out =
    List.map
      (fun pv ->
        match pv.pv_const with
        | Some c -> c
        | None -> failwith "ad.linearize: primal output not known")
      primal_pvals
  in
  let f_lin tangents = Jaxpr.eval_jaxpr jaxpr (consts @ tangents) in
  (primals_out, f_lin)

type tval = Prim of value | Undef of aval

let is_undef = function Undef _ -> true | Prim _ -> false

let prim_val = function
  | Prim v -> v
  | Undef _ -> failwith "ad: undefined primal"

let in_aval = function Undef a -> a | Prim v -> Core.get_aval v

let broadcast_transpose (shape : int array) (dims : int array)
    (in_shape : int array) ct =
  let out_ndim = Array.length shape in
  let bdim_of = Array.make out_ndim (-1) in
  Array.iteri (fun i d -> bdim_of.(d) <- i) dims;
  let sum_axes = ref [] in
  for d = out_ndim - 1 downto 0 do
    let i = bdim_of.(d) in
    if i < 0 then sum_axes := d :: !sum_axes
    else if in_shape.(i) = 1 && shape.(d) <> 1 then sum_axes := d :: !sum_axes
  done;
  let reduced = b1 (Reduce_sum (Array.of_list !sum_axes)) [ ct ] in
  b1 (Reshape in_shape) [ reduced ]

let reduce_sum_transpose (axes : int array) (in_shape : int array) ct =
  let in_ndim = Array.length in_shape in
  let is_red = Array.make in_ndim false in
  Array.iter (fun a -> is_red.(a) <- true) axes;
  let kept = ref [] in
  for d = in_ndim - 1 downto 0 do
    if not is_red.(d) then kept := d :: !kept
  done;
  b1 (Broadcast_in_dim { shape = in_shape; dims = Array.of_list !kept }) [ ct ]

let argsort perm =
  let n = Array.length perm in
  let out = Array.make n 0 in
  Array.iteri (fun i p -> out.(p) <- i) perm;
  out

let tile_transpose (reps : int array) (in_shape : int array) ct =
  let n = Array.length in_shape in
  let inter =
    Array.init (2 * n) (fun i ->
        if i mod 2 = 0 then reps.(i / 2) else in_shape.(i / 2))
  in
  let reshaped = b1 (Reshape inter) [ ct ] in
  let axes = Array.init n (fun i -> 2 * i) in
  b1 (Reduce_sum axes) [ reshaped ]

let rec transpose_rule prim (cts : value list) (primals : tval list) :
    value option list =
  let ct1 () =
    match cts with
    | [ c ] -> c
    | _ -> failwith "ad: transpose expects a single cotangent"
  in
  match prim with
  | Add ->
      let ct = ct1 () in
      [ Some ct; Some ct ]
  | Sub ->
      let ct = ct1 () in
      [ Some ct; Some (neg ct) ]
  | Neg -> [ Some (neg (ct1 ())) ]
  | Mul -> (
      let ct = ct1 () in
      match primals with
      | [ x; y ] ->
          if is_undef x then [ Some (mul ct (prim_val y)); None ]
          else [ None; Some (mul (prim_val x) ct) ]
      | _ -> arity ())
  | Div -> (
      let ct = ct1 () in
      match primals with
      | [ _; y ] -> [ Some (div ct (prim_val y)); None ]
      | _ -> arity ())
  | Select_n -> (
      let ct = ct1 () in
      match primals with
      | which :: cases ->
          let wv = prim_val which in
          None
          :: List.mapi
               (fun i p ->
                 if is_undef p then
                   let zs =
                     List.mapi
                       (fun j _ -> if j = i then ct else zeros_like_value ct)
                       cases
                   in
                   Some (b1 Select_n (wv :: zs))
                 else None)
               cases
      | _ -> arity ())
  | Convert_element_type _ -> (
      let ct = ct1 () in
      match primals with
      | [ x ] -> [ Some (b1 (Convert_element_type (in_aval x).dtype) [ ct ]) ]
      | _ -> arity ())
  | Broadcast_in_dim { shape; dims } -> (
      let ct = ct1 () in
      match primals with
      | [ x ] -> [ Some (broadcast_transpose shape dims (in_aval x).shape ct) ]
      | _ -> arity ())
  | Reshape _ -> (
      let ct = ct1 () in
      match primals with
      | [ x ] -> [ Some (b1 (Reshape (in_aval x).shape) [ ct ]) ]
      | _ -> arity ())
  | Reduce_sum axes -> (
      let ct = ct1 () in
      match primals with
      | [ x ] -> [ Some (reduce_sum_transpose axes (in_aval x).shape ct) ]
      | _ -> arity ())
  | Cumsum { axis; reverse } ->
      [ Some (b1 (Cumsum { axis; reverse = not reverse }) [ ct1 () ]) ]
  | Copy -> [ Some (b1 Copy [ ct1 () ]) ]
  | Conj -> [ Some (ct1 ()) ]
  | Rev dims -> [ Some (b1 (Rev dims) [ ct1 () ]) ]
  | Transpose perm -> [ Some (b1 (Transpose (argsort perm)) [ ct1 () ]) ]
  | Squeeze _ -> (
      match primals with
      | [ x ] -> [ Some (b1 (Reshape (in_aval x).shape) [ ct1 () ]) ]
      | _ -> arity ())
  | Tile reps -> (
      match primals with
      | [ x ] -> [ Some (tile_transpose reps (in_aval x).shape (ct1 ())) ]
      | _ -> arity ())
  | Concatenate dim ->
      let ct = ct1 () in
      let sizes =
        Array.of_list (List.map (fun p -> (in_aval p).shape.(dim)) primals)
      in
      let pieces = Core.bind (Split { sizes; axis = dim }) [ ct ] in
      List.map2
        (fun p piece -> if is_undef p then Some piece else None)
        primals pieces
  | Stack axis ->
      let ct = ct1 () in
      let pieces = Core.bind (Unstack axis) [ ct ] in
      List.map2
        (fun p piece -> if is_undef p then Some piece else None)
        primals pieces
  | Split { axis; _ } -> (
      match primals with
      | [ x ] ->
          [ (if is_undef x then Some (b1 (Concatenate axis) cts) else None) ]
      | _ -> arity ())
  | Unstack axis -> (
      match primals with
      | [ x ] -> [ (if is_undef x then Some (b1 (Stack axis) cts) else None) ]
      | _ -> arity ())
  | Optimization_barrier ->
      let bcts = Core.bind Optimization_barrier cts in
      List.map2 (fun p b -> if is_undef p then Some b else None) primals bcts
  | Tie -> [ None; Some (ct1 ()) ]
  | Reduce_precision p -> [ Some (b1 (Reduce_precision p) [ ct1 () ]) ]
  | Slice { start_indices; limit_indices; strides } -> (
      match primals with
      | [ x ] ->
          let os = (in_aval x).shape in
          let ct = ct1 () in
          let n = Array.length os in
          let strides =
            match strides with Some s -> s | None -> Array.make n 1
          in
          let ct_shape = (Core.get_aval ct).shape in
          let cfg =
            Array.init n (fun i ->
                let s = strides.(i) in
                let out_d = ct_shape.(i) in
                let real_limit =
                  start_indices.(i)
                  + if out_d = 0 then 0 else 1 + ((out_d - 1) * s)
                in
                (start_indices.(i), os.(i) - real_limit, s - 1))
          in
          let dt = (Core.get_aval ct).dtype in
          let zero = Concrete (Ndarray.of_floats dt [||] [| 0.0 |]) in
          [ Some (b1 (Pad cfg) [ ct; zero ]) ]
      | _ -> arity ())
  | Dynamic_slice _ -> (
      match primals with
      | operand :: idx ->
          let ct = ct1 () in
          let idx_vals = List.map prim_val idx in
          let op_t =
            if is_undef operand then
              let zeros = Ad_util.zeros_like_aval (in_aval operand) in
              Some (Core.bind1 Dynamic_update_slice (zeros :: ct :: idx_vals))
            else None
          in
          op_t :: List.map (fun _ -> None) idx
      | _ -> arity ())
  | Dynamic_update_slice -> (
      match primals with
      | operand :: update :: idx ->
          let ct = ct1 () in
          let ua = in_aval update in
          let idx_vals = List.map prim_val idx in
          let op_t =
            if is_undef operand then
              let zeros_u = Ad_util.zeros_like_aval ua in
              Some (Core.bind1 Dynamic_update_slice (ct :: zeros_u :: idx_vals))
            else None
          in
          let up_t =
            if is_undef update then
              Some
                (Core.bind1
                   (Dynamic_slice { slice_sizes = ua.shape })
                   (ct :: idx_vals))
            else None
          in
          op_t :: up_t :: List.map (fun _ -> None) idx
      | _ -> arity ())
  | Gather { dimension_numbers; _ } -> (
      match primals with
      | [ operand; indices ] ->
          let ct = ct1 () in
          let op_t =
            if is_undef operand then
              let zeros = Ad_util.zeros_like_aval (in_aval operand) in
              let sd = gather_to_scatter dimension_numbers in
              Some
                (Core.bind1
                   (Scatter_add { dimension_numbers = sd })
                   [ zeros; prim_val indices; ct ])
            else None
          in
          [ op_t; None ]
      | _ -> arity ())
  | Scatter_add { dimension_numbers } -> (
      match primals with
      | [ operand; indices; updates ] ->
          let ct = ct1 () in
          let op_t = if is_undef operand then Some ct else None in
          let up_t =
            if is_undef updates then
              let gd = scatter_to_gather dimension_numbers in
              let ss =
                scatter_gather_slice_sizes dimension_numbers
                  (in_aval operand).shape (in_aval updates).shape
              in
              Some
                (Core.bind1
                   (Gather { dimension_numbers = gd; slice_sizes = ss })
                   [ ct; prim_val indices ])
            else None
          in
          [ op_t; None; up_t ]
      | _ -> arity ())
  | Scatter_sub { dimension_numbers } -> (
      match primals with
      | [ operand; indices; updates ] ->
          let ct = ct1 () in
          let op_t = if is_undef operand then Some ct else None in
          let up_t =
            if is_undef updates then
              let gd = scatter_to_gather dimension_numbers in
              let ss =
                scatter_gather_slice_sizes dimension_numbers
                  (in_aval operand).shape (in_aval updates).shape
              in
              Some
                (neg
                   (Core.bind1
                      (Gather { dimension_numbers = gd; slice_sizes = ss })
                      [ ct; prim_val indices ]))
            else None
          in
          [ op_t; None; up_t ]
      | _ -> arity ())
  | Scatter_mul { dimension_numbers; unique_indices } -> (
      match primals with
      | [ operand; indices; updates ] ->
          let ct = ct1 () in
          let op_t =
            if is_undef operand then
              Some
                (Core.bind1
                   (Scatter_mul { dimension_numbers; unique_indices })
                   [ ct; prim_val indices; prim_val updates ])
            else None
          in
          let up_t =
            if is_undef updates then begin
              if not unique_indices then
                failwith "ad: scatter_mul transpose needs unique_indices";
              let gd = scatter_to_gather dimension_numbers in
              let ss =
                scatter_gather_slice_sizes dimension_numbers
                  (in_aval operand).shape (in_aval updates).shape
              in
              Some
                (Core.bind1
                   (Gather { dimension_numbers = gd; slice_sizes = ss })
                   [ mul ct (prim_val operand); prim_val indices ])
            end
            else None
          in
          [ op_t; None; up_t ]
      | _ -> arity ())
  | Scatter { dimension_numbers; unique_indices } -> (
      if not unique_indices then
        failwith "ad: scatter transpose needs unique_indices"
      else
        match primals with
        | [ operand; indices; updates ] ->
            let ct = ct1 () in
            let op_t =
              if is_undef operand then
                let zeros = Ad_util.zeros_like_aval (in_aval updates) in
                Some
                  (Core.bind1
                     (Scatter { dimension_numbers; unique_indices })
                     [ ct; prim_val indices; zeros ])
              else None
            in
            let up_t =
              if is_undef updates then
                let gd = scatter_to_gather dimension_numbers in
                let ss =
                  scatter_gather_slice_sizes dimension_numbers
                    (in_aval operand).shape (in_aval updates).shape
                in
                Some
                  (Core.bind1
                     (Gather { dimension_numbers = gd; slice_sizes = ss })
                     [ ct; prim_val indices ])
              else None
            in
            [ op_t; None; up_t ]
        | _ -> arity ())
  | Cond { t; f } -> (
      match primals with
      | pred :: rest ->
          if is_undef pred then
            failwith "ad: cond transpose needs a concrete predicate";
          let pred_nd =
            match prim_val pred with
            | Concrete nd -> nd
            | Tracer _ ->
                failwith "ad: cond transpose needs a concrete predicate"
          in
          let branch = if Ndarray.get_f pred_nd [||] <> 0.0 then t else f in
          let const_tvals =
            List.map (fun c -> Prim (Concrete c)) branch.consts
          in
          let in_cts =
            eval_jaxpr_transposed branch.jaxpr (const_tvals @ rest) cts
          in
          let remaining = ref in_cts in
          let rest_cts =
            List.map
              (fun tv ->
                if is_undef tv then
                  match !remaining with
                  | c :: tl ->
                      remaining := tl;
                      Some c
                  | [] -> None
                else None)
              rest
          in
          None :: rest_cts
      | _ -> arity ())
  | Custom_linear_solve { solve; transpose_solve } -> (
      match transpose_solve with
      | None ->
          failwith
            "ad: transpose_solve required for backwards mode automatic \
             differentiation of custom_linear_solve"
      | Some ts ->
          let ct_b =
            Core.bind
              (Custom_linear_solve { solve = ts; transpose_solve = Some solve })
              cts
          in
          List.map (fun c -> Some c) ct_b)
  | Sin | Cos | Exp | Log | Tanh | Max | Min | Pow | Abs | Sign | Eq | Lt | Gt
  | Acos | Acosh | Asin | Asinh | Atan | Atanh | Cbrt | Ceil | Clz | Cosh | Exp2
  | Expm1 | Floor | Imag | Integer_pow _ | Is_finite | Log1p | Logistic | Not
  | Population_count | Real | Round | Rsqrt | Sinh | Sqrt | Square | Tan | And
  | Atan2 | Complex | Eq_to | Ge | Le | Le_to | Lt_to | Mulhi | Ne | Nextafter
  | Or | Rem | Shift_left | Shift_right_arithmetic | Shift_right_logical | Xor
  | Pad _ | Dot_general _ | Conv_general_dilated _ | Argmax _ | Argmin _
  | Reduce _ | Reduce_and _ | Reduce_max _ | Reduce_min _ | Reduce_or _
  | Reduce_prod _ | Reduce_xor _ | After_all | Bitcast_convert_type _ | Clamp
  | Composite _ | Create_token | Dce_sink | Empty _ | Empty2 _ | From_edtype _
  | Iota _ | Ragged_dot_general | Rng_bit_generator | Rng_uniform | Sort _
  | To_edtype _ | Top_k _ | Scatter_min _ | Scatter_max _ | Reduce_window _
  | Reduce_window_max _ | Reduce_window_min _ | Reduce_window_sum _
  | Select_and_gather_add _ | Select_and_scatter _ | Select_and_scatter_add _
  | Bessel_i0e | Bessel_i1e | Digamma | Erf | Erf_inv | Erfc | Igamma
  | Igamma_grad_a | Igammac | Lgamma | Polygamma | Regularized_incomplete_beta
  | Zeta | Platform_index _ | Cumprod _ | Cummax _ | Cummin _ | Cumlogsumexp _
  | Xla_call _ ->
      failwith "ad: primitive has no transpose rule in M1"
  | Scan _ -> failwith "ad: scan transpose deferred to a later row (M2)"
  | While _ ->
      failwith
        "ad: reverse-mode differentiation does not work for lax.while_loop"

and eval_jaxpr_transposed (jx : jaxpr) (args : tval list) (cts : value list) :
    value list =
  let primal_env : (int, tval) Hashtbl.t = Hashtbl.create 32 in
  let read_primal = function
    | A_var v -> (
        match Hashtbl.find_opt primal_env v.vid with
        | Some tv -> tv
        | None -> Undef v.vaval)
    | A_lit nd -> Prim (Concrete nd)
    | DropVar a -> Undef a
  in
  let write_primal (v : var) tv =
    match tv with
    | Undef _ -> ()
    | Prim _ -> Hashtbl.replace primal_env v.vid tv
  in
  List.iter2 write_primal jx.in_binders args;
  List.iter
    (fun (e : eqn) ->
      let primals_in = List.map read_primal e.inputs in
      if not (List.exists is_undef primals_in) then begin
        let outs = Core.bind e.prim (List.map prim_val primals_in) in
        List.iter2 (fun (v : var) o -> write_primal v (Prim o)) e.outs outs
      end)
    jx.eqns;
  let ct_env : (int, value) Hashtbl.t = Hashtbl.create 32 in
  let read_ct (v : var) =
    match Hashtbl.find_opt ct_env v.vid with
    | Some c ->
        Hashtbl.remove ct_env v.vid;
        c
    | None -> Ad_util.zeros_like_aval v.vaval
  in
  let write_ct atom ct =
    match atom with
    | A_var v -> (
        match Hashtbl.find_opt ct_env v.vid with
        | Some c -> Hashtbl.replace ct_env v.vid (Ad_util.add_jaxvals c ct)
        | None -> Hashtbl.replace ct_env v.vid ct)
    | _ -> ()
  in
  List.iter2 write_ct jx.outs cts;
  List.iter
    (fun (e : eqn) ->
      let primals_in = List.map read_primal e.inputs in
      if List.exists is_undef primals_in then begin
        let cts_in = List.map read_ct e.outs in
        let cts_out = transpose_rule e.prim cts_in primals_in in
        List.iter2
          (fun atom cto ->
            match cto with Some ct -> write_ct atom ct | None -> ())
          e.inputs cts_out
      end)
    (List.rev jx.eqns);
  List.fold_right2
    (fun (v : var) tv acc ->
      match tv with Undef _ -> read_ct v :: acc | Prim _ -> acc)
    jx.in_binders args []

let vjp (f : value list -> value list) (primals : value list) :
    value list * (value list -> value list) =
  let pvals_in =
    List.map Partial_eval.partial_val_known primals
    @ List.map
        (fun x -> Partial_eval.partial_val_unknown (Core.get_aval x))
        primals
  in
  let f_jvp inputs =
    let px, tx = Util.split_half inputs in
    let po, to_ = jvp f px tx in
    po @ to_
  in
  let jaxpr, consts, pvals_out =
    Partial_eval.partial_eval_flat f_jvp pvals_in
  in
  let primal_pvals, _ = Util.split_half pvals_out in
  let primals_out =
    List.map
      (fun pv ->
        match pv.pv_const with
        | Some c -> c
        | None -> failwith "ad.vjp: primal output not known")
      primal_pvals
  in
  let transpose_inputs =
    List.map (fun c -> Prim c) consts
    @ List.map (fun x -> Undef (Core.get_aval x)) primals
  in
  let f_vjp cts = eval_jaxpr_transposed jaxpr transpose_inputs cts in
  (primals_out, f_vjp)

let grad (f : value list -> value) (xs : value list) : value list =
  let primals_out, f_vjp = vjp (fun args -> [ f args ]) xs in
  let y =
    match primals_out with
    | [ y ] -> y
    | _ -> failwith "ad.grad: expected a single output"
  in
  f_vjp [ ones_like_value y ]
