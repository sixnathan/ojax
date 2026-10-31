open Types

let b1 = Core.bind1
let is_zero = Ad_util.is_symbolic_zero
let zeros_like_value = Ad_util.symbolic_zero_like_value
let same_shape a b = (Core.get_aval a).shape = (Core.get_aval b).shape

let add a b =
  if is_zero a && same_shape a b then b
  else if is_zero b && same_shape a b then a
  else b1 Add [ a; b ]

let neg a = if is_zero a then a else b1 Neg [ a ]

let sub a b =
  if is_zero b && same_shape a b then a
  else if is_zero a && same_shape a b then neg b
  else b1 Sub [ a; b ]

let mul a b =
  if (is_zero a || is_zero b) && same_shape a b then if is_zero a then a else b
  else b1 Mul [ a; b ]

let div a b = if is_zero a && same_shape a b then a else b1 Div [ a; b ]

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

let swap_last2 x =
  let ndim = Array.length (Core.get_aval x).shape in
  let perm = Array.init ndim Fun.id in
  perm.(ndim - 1) <- ndim - 2;
  perm.(ndim - 2) <- ndim - 1;
  b1 (Transpose perm) [ x ]

let is_complex_value v =
  match (Core.get_aval v).dtype with
  | Dtype.Complex64 | Dtype.Complex128 -> true
  | _ -> false

let hconj x =
  let swapped = swap_last2 x in
  if is_complex_value x then b1 Conj [ swapped ] else swapped

let like_dtype reference v =
  let dt = (Core.get_aval reference).dtype in
  if (Core.get_aval v).dtype = dt then v else b1 (Convert_element_type dt) [ v ]

let matmul x y =
  let ndim = Array.length (Core.get_aval x).shape in
  let batch = Array.init (ndim - 2) Fun.id in
  b1
    (Dot_general
       {
         lhs_contract = [| ndim - 1 |];
         rhs_contract = [| ndim - 2 |];
         lhs_batch = batch;
         rhs_batch = batch;
       })
    [ x; y ]

let matrix_mask x keep =
  let a = Core.get_aval x in
  let ndim = Array.length a.shape in
  let rows = a.shape.(ndim - 2) and cols = a.shape.(ndim - 1) in
  let cell = rows * cols in
  let total = Array.fold_left ( * ) 1 a.shape in
  let data =
    Array.init total (fun i ->
        let local = i mod cell in
        if keep (local / cols) (local mod cols) then 1.0 else 0.0)
  in
  Concrete (Ndarray.of_floats a.dtype a.shape data)

let tril_of k x = mul x (matrix_mask x (fun i j -> j <= i + k))
let triu_of k x = mul x (matrix_mask x (fun i j -> j >= i + k))
let eye_like x = matrix_mask x (fun i j -> i = j)

let triangular_solve_with ~left_side ~lower ~transpose_a ~conjugate_a
    ~unit_diagonal a b =
  b1
    (Triangular_solve
       { left_side; lower; transpose_a; conjugate_a; unit_diagonal })
    [ a; b ]

let triangular_tangent_a ~left_side ~lower ~transpose_a ~conjugate_a
    ~unit_diagonal a ans ta =
  let k = if unit_diagonal then 1 else 0 in
  let g = if lower then tril_of (-k) ta else triu_of k ta in
  let g = neg g in
  let g = if transpose_a then swap_last2 g else g in
  let g = if conjugate_a && is_complex_value g then b1 Conj [ g ] else g in
  let inverse =
    triangular_solve_with ~left_side ~lower ~transpose_a ~conjugate_a
      ~unit_diagonal a
  in
  if left_side then matmul (inverse g) ans else matmul ans (inverse g)

let cholesky_jvp x t =
  let l = tril_of 0 (b1 Cholesky [ x ]) in
  let phi m =
    let half = matrix_mask m (fun i j -> if i = j then true else j < i) in
    let scale = sub half (mul (eye_like m) (const_like m 0.5)) in
    mul (tril_of 0 m) scale
  in
  let tmp =
    triangular_solve_with ~left_side:false ~lower:true ~transpose_a:true
      ~conjugate_a:true ~unit_diagonal:false l t
  in
  let inner =
    triangular_solve_with ~left_side:true ~lower:true ~transpose_a:false
      ~conjugate_a:false ~unit_diagonal:false l tmp
  in
  (l, matmul l (phi inner))

let diagonal_of x =
  let a = Core.get_aval x in
  let ndim = Array.length a.shape in
  let rows = a.shape.(ndim - 2) and cols = a.shape.(ndim - 1) in
  let k = min rows cols in
  let picks =
    List.init k (fun i ->
        let start = Array.make ndim 0 and limit = Array.copy a.shape in
        start.(ndim - 2) <- i;
        limit.(ndim - 2) <- i + 1;
        start.(ndim - 1) <- i;
        limit.(ndim - 1) <- i + 1;
        let piece =
          b1
            (Slice
               { start_indices = start; limit_indices = limit; strides = None })
            [ x ]
        in
        b1 (Squeeze [| ndim - 2; ndim - 1 |]) [ piece ])
  in
  b1 (Stack (ndim - 2)) picks

let expand_row w =
  let a = Core.get_aval w in
  let ndim = Array.length a.shape in
  let shape = Array.append a.shape [| 1 |] in
  ignore ndim;
  b1 (Reshape shape) [ w ]

let expand_col w =
  let a = Core.get_aval w in
  let ndim = Array.length a.shape in
  let shape =
    Array.concat
      [ Array.sub a.shape 0 (ndim - 1); [| 1 |]; [| a.shape.(ndim - 1) |] ]
  in
  b1 (Reshape shape) [ w ]

let broadcast_to_matrix x shape =
  let a = Core.get_aval x in
  b1
    (Broadcast_in_dim { shape; dims = Array.init (Array.length a.shape) Fun.id })
    [ x ]

let matrix_shape_of_vector w =
  let a = Core.get_aval w in
  let n = a.shape.(Array.length a.shape - 1) in
  Array.append a.shape [| n |]

let eigh_jvp ~lower a t =
  let outs = Core.bind (Eigh { lower }) [ a ] in
  match outs with
  | [ v; w ] ->
      let shape = matrix_shape_of_vector w in
      let ww_row = broadcast_to_matrix (expand_row w) shape in
      let ww_col = broadcast_to_matrix (expand_col w) shape in
      let eye = eye_like ww_row in
      let fmat = sub (recip (add eye (sub ww_col ww_row))) eye in
      let vt_a_v = matmul (matmul (hconj v) t) v in
      let dv = matmul v (mul (like_dtype vt_a_v fmat) vt_a_v) in
      let diag = diagonal_of vt_a_v in
      let dw = if is_complex_value diag then b1 Real [ diag ] else diag in
      (outs, [ dv; dw ])
  | _ -> failwith "ad: eigh arity"

let qr_jvp ~full_matrices a t =
  let outs = Core.bind (Qr full_matrices) [ a ] in
  match outs with
  | [ q; r ] ->
      let shape = (Core.get_aval a).shape in
      let ndim = Array.length shape in
      if full_matrices && shape.(ndim - 2) > shape.(ndim - 1) then
        failwith "Unimplemented case of QR decomposition derivative";
      let dx_rinv =
        triangular_solve_with ~left_side:false ~lower:false ~transpose_a:false
          ~conjugate_a:false ~unit_diagonal:false r t
      in
      let qt_dx_rinv = matmul (hconj q) dx_rinv in
      let lower_part = tril_of (-1) qt_dx_rinv in
      let skew = sub lower_part (hconj lower_part) in
      let d_omega =
        if is_complex_value qt_dx_rinv then
          add skew
            (mul (eye_like qt_dx_rinv)
               (sub qt_dx_rinv
                  (b1 (Convert_element_type (Core.get_aval qt_dx_rinv).dtype)
                     [ b1 Real [ qt_dx_rinv ] ])))
        else skew
      in
      let dq = add (matmul q (sub d_omega qt_dx_rinv)) dx_rinv in
      let dr = matmul (sub qt_dx_rinv d_omega) r in
      (outs, [ dq; dr ])
  | _ -> failwith "ad: qr arity"

let flat_gather v indices =
  let n = Array.fold_left ( * ) 1 (Core.get_aval v).shape in
  let m = Array.fold_left ( * ) 1 (Core.get_aval indices).shape in
  let flat = b1 (Reshape [| n |]) [ v ] in
  let idx = b1 (Reshape [| m; 1 |]) [ indices ] in
  b1
    (Gather
       {
         dimension_numbers =
           {
             offset_dims = [||];
             collapsed_slice_dims = [| 0 |];
             start_index_map = [| 0 |];
             g_operand_batching_dims = [||];
             g_start_indices_batching_dims = [||];
           };
         slice_sizes = [| 1 |];
       })
    [ flat; idx ]

let take_along_axis_value v perm ~axis =
  let shape = (Core.get_aval v).shape in
  let ndim = Array.length shape in
  let perm_shape = (Core.get_aval perm).shape in
  let idt = (Core.get_aval perm).dtype in
  let strides = Array.make ndim 1 in
  for d = ndim - 2 downto 0 do
    strides.(d) <- strides.(d + 1) * shape.(d + 1)
  done;
  let total = Array.fold_left ( * ) 1 perm_shape in
  let rest =
    Array.init total (fun i ->
        let offset = ref 0 and remaining = ref i in
        for d = ndim - 1 downto 0 do
          let coord = !remaining mod perm_shape.(d) in
          remaining := !remaining / perm_shape.(d);
          if d <> axis then offset := !offset + (coord * strides.(d))
        done;
        float_of_int !offset)
  in
  let offsets = Concrete (Ndarray.of_floats idt perm_shape rest) in
  let along = mul perm (const_like perm (float_of_int strides.(axis))) in
  let picked = flat_gather v (add along offsets) in
  b1 (Reshape perm_shape) [ picked ]

let lu_jvp a t =
  let outs = Core.bind Lu [ a ] in
  match outs with
  | [ lu; pivots; permutation ] ->
      let aval_a = Core.get_aval a in
      let shape = aval_a.shape in
      let ndim = Array.length shape in
      let batch = Array.sub shape 0 (ndim - 2) in
      let m = shape.(ndim - 2) and n = shape.(ndim - 1) in
      let k = min m n in
      let dt = aval_a.dtype in
      let zero = Concrete (Ndarray.of_floats dt [||] [| 0.0 |]) in
      let square size keep =
        matrix_mask
          (Ad_util.zeros_like_aval
             { aval_a with shape = Array.append batch [| size; size |] })
          keep
      in
      let perm_rank = Array.length (Core.get_aval permutation).shape in
      let expanded =
        b1
          (Reshape (Array.append (Core.get_aval permutation).shape [| 1 |]))
          [ permutation ]
      in
      let broadcast_perm =
        b1
          (Broadcast_in_dim { shape; dims = Array.init (perm_rank + 1) Fun.id })
          [ expanded ]
      in
      let x = take_along_axis_value t broadcast_perm ~axis:(ndim - 2) in
      let slice_last rows cols =
        let limit = Array.copy shape in
        limit.(ndim - 2) <- rows;
        limit.(ndim - 1) <- cols;
        b1
          (Slice
             {
               start_indices = Array.make ndim 0;
               limit_indices = limit;
               strides = None;
             })
          [ lu ]
      in
      let pad_last row_extra col_extra v =
        let cfg = Array.make ndim (0, 0, 0) in
        cfg.(ndim - 2) <- (0, row_extra, 0);
        cfg.(ndim - 1) <- (0, col_extra, 0);
        b1 (Pad cfg) [ v; zero ]
      in
      let l =
        add
          (pad_last 0 (m - k) (tril_of (-1) (slice_last m k)))
          (square m (fun i j -> i = j))
      in
      let u =
        add
          (pad_last (n - k) 0 (triu_of 0 (slice_last k n)))
          (square n (fun i j -> i = j && i >= k))
      in
      let la =
        triangular_solve_with ~left_side:true ~lower:true ~transpose_a:false
          ~conjugate_a:false ~unit_diagonal:true l x
      in
      let lau =
        triangular_solve_with ~left_side:false ~lower:false ~transpose_a:false
          ~conjugate_a:false ~unit_diagonal:false u la
      in
      let l_dot = matmul l (tril_of (-1) lau) in
      let u_dot = matmul (triu_of 0 lau) u in
      ( outs,
        [
          add l_dot u_dot; zeros_like_value pivots; zeros_like_value permutation;
        ] )
  | _ -> failwith "ad: lu arity"

let solve_general a b =
  match Core.bind Lu [ a ] with
  | [ lu; _; permutation ] ->
      let shape = (Core.get_aval b).shape in
      let ndim = Array.length shape in
      let perm_shape = (Core.get_aval permutation).shape in
      let perm_rank = Array.length perm_shape in
      let expanded =
        b1 (Reshape (Array.append perm_shape [| 1 |])) [ permutation ]
      in
      let broadcast_perm =
        b1
          (Broadcast_in_dim { shape; dims = Array.init (perm_rank + 1) Fun.id })
          [ expanded ]
      in
      let permuted = take_along_axis_value b broadcast_perm ~axis:(ndim - 2) in
      let y =
        triangular_solve_with ~left_side:true ~lower:true ~transpose_a:false
          ~conjugate_a:false ~unit_diagonal:true lu permuted
      in
      triangular_solve_with ~left_side:true ~lower:false ~transpose_a:false
        ~conjugate_a:false ~unit_diagonal:false lu y
  | _ -> failwith "ad: lu arity"

let eig_vec_jvp w v da =
  let shape = (Core.get_aval v).shape in
  let ndim = Array.length shape in
  let ww_row = broadcast_to_matrix (expand_row w) shape in
  let ww_col = broadcast_to_matrix (expand_col w) shape in
  let eye = eye_like ww_row in
  let fmat = sub (recip (add eye (sub ww_col ww_row))) eye in
  let p = matmul (solve_general v da) v in
  let dw = diagonal_of p in
  let u = matmul v (mul fmat p) in
  let k =
    b1 (Argmax { axis = ndim - 2; index_dtype = Dtype.I32 }) [ b1 Abs [ v ] ]
  in
  let iota = b1 (Iota { dtype = Dtype.I32; shape; dimension = ndim - 2 }) [] in
  let picked = b1 Eq [ iota; broadcast_to_matrix (expand_col k) shape ] in
  let mask = like_dtype v picked in
  let sum_rows x = b1 (Reduce_sum [| ndim - 2 |]) [ x ] in
  let c =
    b1 Complex
      [
        neg (b1 Real [ sum_rows (mul (b1 Conj [ v ]) u) ]);
        div
          (neg (b1 Imag [ sum_rows (mul mask u) ]))
          (b1 Real [ sum_rows (mul mask v) ]);
      ]
  in
  (dw, add u (mul v (broadcast_to_matrix (expand_col c) shape)))

let eig_jvp ~compute_left ~compute_right a t =
  let outs = Core.bind (Eig { compute_left; compute_right = true }) [ a ] in
  let w = List.hd outs and vr = List.nth outs (List.length outs - 1) in
  let da = like_dtype vr t in
  if not (compute_left || compute_right) then
    ([ w ], [ diagonal_of (matmul (solve_general vr da) vr) ])
  else
    let dw, dvr = eig_vec_jvp w vr da in
    let left =
      if compute_left then
        let vl = List.nth outs 1 in
        let _, dvl = eig_vec_jvp (b1 Conj [ w ]) vl (hconj da) in
        [ (vl, dvl) ]
      else []
    in
    let right = if compute_right then [ (vr, dvr) ] else [] in
    let vectors = left @ right in
    (w :: List.map fst vectors, dw :: List.map snd vectors)

let svd_jvp ~full_matrices ~compute_uv a t =
  if full_matrices then
    failwith
      "Singular value decomposition JVP not implemented for full matrices";
  let thin =
    Core.bind (Svd { full_matrices = false; compute_uv = true }) [ a ]
  in
  match thin with
  | [ u; s; vt ] ->
      let ut = hconj u and v = hconj vt in
      let d_s_mat = matmul (matmul ut t) v in
      let raw_diag = diagonal_of d_s_mat in
      let ds =
        if is_complex_value raw_diag then b1 Real [ raw_diag ] else raw_diag
      in
      if not compute_uv then ([ s ], [ ds ])
      else
        let shape = matrix_shape_of_vector s in
        let s_col = broadcast_to_matrix (expand_col s) shape in
        let s_row = broadcast_to_matrix (expand_row s) shape in
        let s_diffs = mul (add s_col s_row) (sub s_col s_row) in
        let eye = eye_like s_col in
        let f = like_dtype d_s_mat (sub (recip (add s_diffs eye)) eye) in
        let d_ss = mul (like_dtype d_s_mat s_col) d_s_mat in
        let sd_s = mul (like_dtype d_s_mat s_row) d_s_mat in
        let is_zero =
          b1 (Convert_element_type (Core.get_aval s).dtype)
            [ b1 Eq [ s; zeros_like_value s ] ]
        in
        let s_inv = sub (recip (add s is_zero)) is_zero in
        let s_inv_mat =
          mul eye (broadcast_to_matrix (expand_col s_inv) shape)
        in
        let half = const_like d_s_mat 0.5 in
        let d_ud_v = mul (mul half (sub d_s_mat (hconj d_s_mat))) s_inv_mat in
        let du = matmul u (add (mul f (add d_ss (hconj d_ss))) d_ud_v) in
        let dv = matmul v (mul f (add sd_s (hconj sd_s))) in
        let a_shape = (Core.get_aval a).shape in
        let ndim = Array.length a_shape in
        let m = a_shape.(ndim - 2) and n = a_shape.(ndim - 1) in
        let du =
          if m > n then
            let da_v = matmul t v in
            let residual = sub da_v (matmul u (matmul ut da_v)) in
            let denom =
              broadcast_to_matrix (expand_col s) (Core.get_aval residual).shape
            in
            add du (div residual denom)
          else du
        in
        let dv =
          if n > m then
            let da_u = matmul (hconj t) u in
            let residual = sub da_u (matmul v (matmul vt da_u)) in
            let denom =
              broadcast_to_matrix (expand_col s) (Core.get_aval residual).shape
            in
            add dv (div residual denom)
          else dv
        in
        (thin, [ du; ds; hconj dv ])
  | _ -> failwith "ad: svd arity"

let tridiagonal_matvec dl d du x =
  let m = (Core.get_aval d).shape.(0) in
  let rows = (Core.get_aval x).shape in
  let row_slice lo hi =
    let start = Array.make (Array.length rows) 0 in
    let limit = Array.copy rows in
    start.(0) <- lo;
    limit.(0) <- hi;
    b1
      (Slice { start_indices = start; limit_indices = limit; strides = None })
      [ x ]
  in
  let scale coeff part =
    let c =
      broadcast_to_matrix
        (b1
           (Reshape
              (Array.append (Core.get_aval coeff).shape
                 (Array.make (Array.length rows - 1) 1)))
           [ coeff ])
        (Core.get_aval part).shape
    in
    mul c part
  in
  let dt = (Core.get_aval d).dtype in
  let zero = Concrete (Ndarray.of_floats dt [||] [| 0.0 |]) in
  let shifted lo hi part =
    let cfg = Array.make (Array.length rows) (0, 0, 0) in
    cfg.(0) <- (lo, hi, 0);
    b1 (Pad cfg) [ part; zero ]
  in
  let vec_slice v lo hi =
    b1
      (Slice
         { start_indices = [| lo |]; limit_indices = [| hi |]; strides = None })
      [ v ]
  in
  let main = scale d x in
  let lower = shifted 1 0 (scale (vec_slice dl 1 m) (row_slice 0 (m - 1))) in
  let upper = shifted 0 1 (scale (vec_slice du 0 (m - 1)) (row_slice 1 m)) in
  add main (add lower upper)

let lane_fold_jvp x t ~axis step =
  let xs = Core.bind (Unstack axis) [ x ] in
  let ts = Core.bind (Unstack axis) [ t ] in
  step xs ts

let cumulative_jvp ~axis ~reverse x t combine =
  let ordered l = if reverse then List.rev l else l in
  lane_fold_jvp x t ~axis (fun xs ts ->
      let xs = ordered xs and ts = ordered ts in
      match (xs, ts) with
      | x0 :: xrest, t0 :: trest ->
          let _, _, outs, tans =
            List.fold_left2
              (fun (acc, acc_t, outs, tans) xi ti ->
                let v, vt = combine acc acc_t xi ti in
                (v, vt, v :: outs, vt :: tans))
              (x0, t0, [ x0 ], [ t0 ]) xrest trest
          in
          let final l = if reverse then l else List.rev l in
          (b1 (Stack axis) (final outs), b1 (Stack axis) (final tans))
      | _ -> failwith "ad: cumulative of an empty axis")

let product_step acc acc_t x t = (mul acc x, add (mul acc_t x) (mul acc t))

let choose_step test acc acc_t x t =
  let pick = b1 test [ x; acc ] in
  (b1 Select_n [ pick; acc; x ], b1 Select_n [ pick; acc_t; t ])

let logaddexp_step acc acc_t x t =
  let big = b1 Max [ acc; x ] in
  let result = add big (b1 Log1p [ b1 Exp [ sub (b1 Min [ acc; x ]) big ] ]) in
  let wa = b1 Exp [ sub acc result ] and wx = b1 Exp [ sub x result ] in
  (result, add (mul wa acc_t) (mul wx t))

let flatten_axes_last x axes =
  let shape = (Core.get_aval x).shape in
  let ndim = Array.length shape in
  let kept =
    List.filter (fun d -> not (Array.mem d axes)) (List.init ndim Fun.id)
  in
  let perm = Array.of_list (kept @ Array.to_list axes) in
  let moved = b1 (Transpose perm) [ x ] in
  let kept_shape = Array.of_list (List.map (fun d -> shape.(d)) kept) in
  let reduced = Array.fold_left (fun acc d -> acc * shape.(d)) 1 axes in
  b1 (Reshape (Array.append kept_shape [| reduced |])) [ moved ]

let reduce_fold_jvp x t axes step =
  let fx = flatten_axes_last x axes and ft = flatten_axes_last t axes in
  let last = Array.length (Core.get_aval fx).shape - 1 in
  lane_fold_jvp fx ft ~axis:last (fun xs ts ->
      match (xs, ts) with
      | x0 :: xrest, t0 :: trest ->
          List.fold_left2
            (fun (acc, acc_t) xi ti -> step acc acc_t xi ti)
            (x0, t0) xrest trest
      | _ -> failwith "ad: reduction over an empty axis")

let sort_jvp ~dimension ~is_stable ~num_keys primals tangents =
  let first = List.hd primals in
  let shape = (Core.get_aval first).shape in
  let idt = if Config.x64_enabled () then Dtype.I64 else Dtype.I32 in
  let iota = b1 (Iota { dtype = idt; shape; dimension }) [] in
  let outs =
    Core.bind (Sort { dimension; is_stable; num_keys }) (primals @ [ iota ])
  in
  let n = List.length primals in
  let sorted = List.filteri (fun i _ -> i < n) outs in
  let perm = List.nth outs n in
  let sorted_t =
    List.map (fun t -> take_along_axis_value t perm ~axis:dimension) tangents
  in
  (sorted, sorted_t)

let top_k_jvp ~k ~axis x t =
  let outs = Core.bind (Top_k { k; axis }) [ x ] in
  match outs with
  | [ values; indices ] ->
      let indices_i =
        let dt = if Config.x64_enabled () then Dtype.I64 else Dtype.I32 in
        b1 (Convert_element_type dt) [ indices ]
      in
      ( outs,
        [ take_along_axis_value t indices_i ~axis; zeros_like_value indices ] )
  | _ -> failwith "ad: top_k arity"

let reducer_jvp_ref :
    (closed_jaxpr -> value list -> value list -> value * value) ref =
  ref (fun _ _ _ -> failwith "ad: reducer jvp not installed")

let reducer_jvp j p t = !reducer_jvp_ref j p t

let concrete_all_zero = function
  | Concrete nd -> Ndarray.fold (fun acc x -> acc && x = 0.0) true nd
  | Device _ | Tracer _ -> false

let bilinear_jvp prim x y tx ty =
  let po = b1 prim [ x; y ] in
  let term t args = if is_zero t then zeros_like_value po else b1 prim args in
  (po, add (term tx [ tx; y ]) (term ty [ x; ty ]))

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
      | [ x ], [ tx ] -> (b1 Cos [ x ], neg (mul (b1 Sin [ x ]) tx))
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
          let dx () = mul (mul y (b1 Pow [ x; sub y one ])) tx in
          let dy () = mul (mul (b1 Log [ x ]) po) ty in
          if concrete_all_zero ty then (po, dx ())
          else if concrete_all_zero tx then (po, dy ())
          else (po, add (dx ()) (dy ()))
      | _ -> arity ())
  | Abs -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          if is_complex_value x then
            let po = b1 Abs [ x ] in
            let real_part v = b1 Real [ v ] and imag_part v = b1 Imag [ v ] in
            let numer =
              add
                (mul (real_part x) (real_part tx))
                (mul (imag_part x) (imag_part tx))
            in
            let safe =
              b1 Select_n
                [ b1 Eq [ po; zeros_like_value po ]; po; ones_like_value po ]
            in
            (po, div numer safe)
          else (b1 Abs [ x ], mul (b1 Sign [ x ]) tx)
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
            | Dtype.F32 | Dtype.F64 | Dtype.Complex64 | Dtype.Complex128 ->
                b1 (Convert_element_type dt) [ tx ]
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
      | [ x; y ], [ tx; ty ] -> bilinear_jvp prim x y tx ty
      | _ -> arity ())
  | Conv_general_dilated _ -> (
      match (primals, tangents) with
      | [ x; y ], [ tx; ty ] -> bilinear_jvp prim x y tx ty
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
  | Reduce_prod axes -> (
      match (primals, tangents) with
      | [ x ], [ t ] -> reduce_fold_jvp x t axes product_step
      | _ -> arity ())
  | Reduce { jaxpr; dimensions } -> (
      match (primals, tangents) with
      | [ x; init ], [ t; init_t ] ->
          let fx = flatten_axes_last x dimensions in
          let ft = flatten_axes_last t dimensions in
          let flat_shape = (Core.get_aval fx).shape in
          let rank = Array.length flat_shape in
          let n = flat_shape.(rank - 1) in
          let kept = Array.sub flat_shape 0 (rank - 1) in
          let lanes = Array.fold_left ( * ) 1 kept in
          let as_lanes v =
            Core.bind (Unstack 0) [ b1 (Reshape [| lanes; n |]) [ v ] ]
          in
          let results =
            List.map2
              (fun lane_x lane_t ->
                let xs = Core.bind (Unstack 0) [ lane_x ] in
                let ts = Core.bind (Unstack 0) [ lane_t ] in
                List.fold_left2
                  (fun (acc, acc_t) xi ti ->
                    reducer_jvp jaxpr [ acc; xi ] [ acc_t; ti ])
                  (init, init_t) xs ts)
              (as_lanes fx) (as_lanes ft)
          in
          let collect pick =
            b1 (Reshape kept) [ b1 (Stack 0) (List.map pick results) ]
          in
          (collect fst, collect snd)
      | _ -> arity ())
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
  | Zeta -> (
      match (primals, tangents) with
      | [ x; q ], [ _; tq ] ->
          let raised = add x (const_like x 1.0) in
          (b1 Zeta [ x; q ], mul tq (mul (neg x) (b1 Zeta [ raised; q ])))
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
  | Cumprod { axis; reverse } -> (
      match (primals, tangents) with
      | [ x ], [ t ] -> cumulative_jvp ~axis ~reverse x t product_step
      | _ -> arity ())
  | Cummax { axis; reverse } -> (
      match (primals, tangents) with
      | [ x ], [ t ] -> cumulative_jvp ~axis ~reverse x t (choose_step Gt)
      | _ -> arity ())
  | Cummin { axis; reverse } -> (
      match (primals, tangents) with
      | [ x ], [ t ] -> cumulative_jvp ~axis ~reverse x t (choose_step Lt)
      | _ -> arity ())
  | Cumlogsumexp { axis; reverse } -> (
      match (primals, tangents) with
      | [ x ], [ t ] -> cumulative_jvp ~axis ~reverse x t logaddexp_step
      | _ -> arity ())
  | Igamma_grad_a -> failwith "ad: primitive has no jvp rule (matches jax)"
  | Iota _ | Empty _ | Empty2 _ | Create_token | After_all | Composite _
  | Dce_sink | From_edtype _ | Ragged_dot_general | Rng_bit_generator
  | Rng_uniform | To_edtype _ | Platform_index _ | Threefry2x32
  | Iota_2x32_shape _ | Random_seed | Random_split _ | Random_fold_in
  | Random_bits _ | Random_wrap | Random_unwrap ->
      failwith "ad: primitive has no jvp rule in M1"
  | Cholesky -> (
      match (primals, tangents) with
      | [ x ], [ t ] -> cholesky_jvp x t
      | _ -> arity ())
  | Triangular_solve o -> (
      match (primals, tangents) with
      | [ a; b ], [ ta; tb ] ->
          let ans = b1 prim [ a; b ] in
          let from_b = b1 prim [ a; tb ] in
          let from_a =
            triangular_tangent_a ~left_side:o.left_side ~lower:o.lower
              ~transpose_a:o.transpose_a ~conjugate_a:o.conjugate_a
              ~unit_diagonal:o.unit_diagonal a ans ta
          in
          (ans, add from_a from_b)
      | _ -> arity ())
  | Tridiagonal_solve -> (
      match (primals, tangents) with
      | [ dl; d; du; b ], [ tdl; td; tdu; tb ] ->
          let ans = b1 prim [ dl; d; du; b ] in
          let rhs = sub tb (tridiagonal_matvec tdl td tdu ans) in
          (ans, b1 prim [ dl; d; du; rhs ])
      | _ -> arity ())
  | Householder_product | Lu | Lu_pivots_to_permutation _ | Qr _ | Eig _
  | Eigh _ | Hessenberg | Schur _ | Svd _ | Tridiagonal _ ->
      failwith "ad: this linalg primitive has no forward-mode rule"
  | Collective { kind; _ } -> (
      match (kind, primals, tangents) with
      | (Psum | Pbroadcast _ | Ppermute _ | All_gather _), [ x ], [ t ] ->
          (b1 prim [ x ], b1 prim [ t ])
      | Axis_index, [], [] ->
          let v = b1 prim [] in
          (v, zeros_like_value v)
      | (Pmax | Pmin), [ x ], [ _ ] ->
          let v = b1 prim [ x ] in
          (v, zeros_like_value v)
      | _ -> arity ())
  | Fft _ -> (
      match (primals, tangents) with
      | [ x ], [ t ] -> (b1 prim [ x ], b1 prim [ t ])
      | _ -> arity ())
  | Cond _ -> failwith "ad: cond jvp handled by jvp_process_primitive"
  | Custom_jvp_call _ | Custom_vjp_call _ | Custom_lin _ ->
      failwith "ad: custom rules handled by jvp_process_primitive"
  | Checkpoint_name _ -> (
      match (primals, tangents) with
      | [ x ], [ tx ] -> (b1 prim [ x ], tx)
      | _ -> arity ())
  | Xla_call _ -> failwith "ad: jvp of xla_call not supported in M1"

let is_concrete_zero = function
  | Concrete nd -> Ndarray.fold (fun acc x -> acc && x = 0.0) true nd
  | Device _ | Tracer _ -> false

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

let restore_symbolic_zeros zero_flags (tangents : value list) =
  List.map2
    (fun zero t -> if zero then Ad_util.symbolic_zero_like_value t else t)
    zero_flags tangents

let cond_tangent_branch jvp_f (branch : closed_jaxpr) (op_avals : aval list)
    (op_zero : bool list) : closed_jaxpr =
  let n = List.length op_avals in
  Jaxpr.make_jaxpr (op_avals @ op_avals) (fun args ->
      match Util.split_list args [ n ] with
      | [ prim_ops; tan_ops ] ->
          let _, tans =
            jvp_f
              (fun a -> Jaxpr.eval_closed_jaxpr branch a)
              prim_ops
              (restore_symbolic_zeros op_zero tan_ops)
          in
          tans
      | _ -> failwith "ad: cond tangent branch split")

let carry_zero_fixpoint trace_body flags =
  let rec go flags =
    let observed = ref flags in
    let traced = trace_body flags observed in
    if !observed = flags then (flags, traced) else go !observed
  in
  go flags

let jvp_process_with_tangents trace prim primals tangents =
  match prim with
  | Split _ | Unstack _ | Optimization_barrier ->
      let pos = Core.bind prim primals in
      let tos = Core.bind prim tangents in
      List.map2 (fun po to_ -> Tracer (new_jvp_tracer trace po to_)) pos tos
  | Sort { dimension; is_stable; num_keys } ->
      let po, to_ = sort_jvp ~dimension ~is_stable ~num_keys primals tangents in
      List.map2 (fun p d -> Tracer (new_jvp_tracer trace p d)) po to_
  | Top_k { k; axis } -> (
      match (primals, tangents) with
      | [ x ], [ t ] ->
          let po, to_ = top_k_jvp ~k ~axis x t in
          List.map2 (fun p d -> Tracer (new_jvp_tracer trace p d)) po to_
      | _ -> arity ())
  | Eig { compute_left; compute_right } -> (
      match (primals, tangents) with
      | [ a ], [ t ] ->
          let po, to_ = eig_jvp ~compute_left ~compute_right a t in
          List.map2 (fun p d -> Tracer (new_jvp_tracer trace p d)) po to_
      | _ -> arity ())
  | Eigh { lower } -> (
      match (primals, tangents) with
      | [ a ], [ t ] ->
          let po, to_ = eigh_jvp ~lower a t in
          List.map2 (fun p d -> Tracer (new_jvp_tracer trace p d)) po to_
      | _ -> arity ())
  | Qr full_matrices -> (
      match (primals, tangents) with
      | [ a ], [ t ] ->
          let po, to_ = qr_jvp ~full_matrices a t in
          List.map2 (fun p d -> Tracer (new_jvp_tracer trace p d)) po to_
      | _ -> arity ())
  | Lu -> (
      match (primals, tangents) with
      | [ a ], [ t ] ->
          let po, to_ = lu_jvp a t in
          List.map2 (fun p d -> Tracer (new_jvp_tracer trace p d)) po to_
      | _ -> arity ())
  | Svd { full_matrices; compute_uv } -> (
      match (primals, tangents) with
      | [ a ], [ t ] ->
          let po, to_ = svd_jvp ~full_matrices ~compute_uv a t in
          List.map2 (fun p d -> Tracer (new_jvp_tracer trace p d)) po to_
      | _ -> arity ())
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
      let x_zero = List.map Ad_util.is_symbolic_zero xs_t in
      let ys_zero = ref [] in
      let trace_body carry_zero observed =
        Jaxpr.make_jaxpr
          (carry_avals @ carry_avals @ x_slice_avals @ x_slice_avals)
          (fun args ->
            let pc, r1 = split2 args nc in
            let tc, r2 = split2 r1 nc in
            let px, tx = split2 r2 nx in
            let po, to_ =
              jvp
                (fun a -> Jaxpr.eval_closed_jaxpr jaxpr a)
                (pc @ px)
                (restore_symbolic_zeros carry_zero tc
                @ restore_symbolic_zeros x_zero tx)
            in
            let pc', py = split2 po nc in
            let tc', ty = split2 to_ nc in
            observed :=
              List.map2
                (fun z t -> z && Ad_util.is_symbolic_zero t)
                carry_zero tc';
            ys_zero := List.map Ad_util.is_symbolic_zero ty;
            pc' @ tc' @ py @ ty)
      in
      let carry_zero, new_body =
        carry_zero_fixpoint trace_body
          (List.map Ad_util.is_symbolic_zero carry_t)
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
      let tangent_outs =
        restore_symbolic_zeros carry_zero tcarry
        @ restore_symbolic_zeros !ys_zero tys
      in
      List.map2
        (fun po to_ -> Tracer (new_jvp_tracer trace po to_))
        primal_outs tangent_outs
  | While { cond; body } ->
      let split2 l n =
        match Util.split_list l [ n ] with [ a; b ] -> (a, b) | _ -> arity ()
      in
      let nc = List.length primals in
      let carry_avals = List.map Core.get_aval primals in
      let trace_body carry_zero observed =
        Jaxpr.make_jaxpr (carry_avals @ carry_avals) (fun args ->
            let pc, tc = split2 args nc in
            let po, to_ =
              jvp
                (fun a -> Jaxpr.eval_closed_jaxpr body a)
                pc
                (restore_symbolic_zeros carry_zero tc)
            in
            observed :=
              List.map2
                (fun z t -> z && Ad_util.is_symbolic_zero t)
                carry_zero to_;
            po @ to_)
      in
      let carry_zero, new_body =
        carry_zero_fixpoint trace_body
          (List.map Ad_util.is_symbolic_zero tangents)
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
        pcarry
        (restore_symbolic_zeros carry_zero tcarry)
  | Cond { t; f } -> (
      match (primals, tangents) with
      | pred :: prim_ops, _ :: tan_ops ->
          let op_avals = List.map Core.get_aval prim_ops in
          let primal_outs = Core.bind (Cond { t; f }) primals in
          let op_zero = List.map Ad_util.is_symbolic_zero tan_ops in
          let t_tan = cond_tangent_branch jvp t op_avals op_zero in
          let f_tan = cond_tangent_branch jvp f op_avals op_zero in
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
  | Custom_jvp_call id -> (
      match Core.custom_rule id with
      | Core.Jvp_rule { jvp; _ } ->
          let po, to_ = jvp (List.map Core.full_lower primals) tangents in
          List.map2 (fun p t -> Tracer (new_jvp_tracer trace p t)) po to_
      | Core.Vjp_rule _ | Core.Linear_rule _ | Core.Callback_rule _ ->
          failwith "ad: custom_jvp_call with a non-jvp rule")
  | Custom_vjp_call id -> (
      match Core.custom_rule id with
      | Core.Vjp_rule { fwd; fwd_perturbed; _ } ->
          let primals = List.map Core.full_lower primals in
          let po, res =
            match fwd_perturbed with
            | None -> fwd primals
            | Some rule ->
                let perturbed =
                  List.map (fun t -> not (Ad_util.is_symbolic_zero t)) tangents
                in
                rule perturbed primals
          in
          let out_avals = List.map Core.get_aval po in
          let to_ =
            if List.for_all is_concrete_zero tangents then
              List.map Ad_util.zeros_like_aval out_avals
            else
              Core.bind
                (Custom_lin { rule = id; n_res = List.length res; out_avals })
                (res @ tangents)
          in
          List.map2 (fun p t -> Tracer (new_jvp_tracer trace p t)) po to_
      | Core.Callback_rule { result_avals = []; _ } ->
          ignore (Core.bind prim primals);
          []
      | Core.Callback_rule _ ->
          invalid_arg
            "Pure callbacks do not support JVP. Please use `jax.custom_jvp` to \
             use callbacks while taking gradients."
      | Core.Jvp_rule _ | Core.Linear_rule _ ->
          failwith "ad: custom_vjp_call with a non-vjp rule")
  | Custom_lin { n_res; rule; _ } -> (
      let res, linear =
        match Util.split_list primals [ n_res ] with
        | [ a; b ] -> (a, b)
        | _ -> arity ()
      in
      let res_t, linear_t =
        match Util.split_list tangents [ n_res ] with
        | [ a; b ] -> (a, b)
        | _ -> arity ()
      in
      match Core.custom_rule rule with
      | Core.Linear_rule { forward; res_differentiable = true; _ } ->
          let wrap p t = Tracer (new_jvp_tracer trace p t) in
          forward (List.map2 wrap res res_t) (List.map2 wrap linear linear_t)
      | _ ->
          let po = Core.bind prim (res @ linear) in
          let to_ = Core.bind prim (res @ linear_t) in
          List.map2 (fun p t -> Tracer (new_jvp_tracer trace p t)) po to_)
  | _ ->
      let po, to_ = jvp_rule prim primals tangents in
      [ Tracer (new_jvp_tracer trace po to_) ]

let jvp_process_primitive trace prim args =
  let pairs = List.map (as_jvp trace) args in
  let primals = List.map fst pairs and tangents = List.map snd pairs in
  if List.for_all Ad_util.is_symbolic_zero tangents then Core.bind prim primals
  else jvp_process_with_tangents trace prim primals tangents

let jvp_full_lower = function
  | Tracer { payload = JVP { primal; tangent }; _ }
    when Ad_util.is_symbolic_zero tangent ->
      Core.full_lower primal
  | v -> v

let interpreter : Core.interpreter =
  {
    i_pure = jvp_pure;
    i_lift = jvp_pure;
    i_full_lower = jvp_full_lower;
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

let () =
  reducer_jvp_ref :=
    fun jaxpr primals tangents ->
      match jvp (fun a -> Jaxpr.eval_closed_jaxpr jaxpr a) primals tangents with
      | [ p ], [ t ] -> (p, t)
      | _ -> failwith "ad: reducer must return one value"

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
  let primal_pvals, tangent_pvals = Util.split_half pvals_out in
  let primals_out =
    List.map
      (fun pv ->
        match pv.pv_const with
        | Some c -> c
        | None -> failwith "ad.linearize: primal output not known")
      primal_pvals
  in
  let f_lin tangents =
    let computed = ref (Jaxpr.eval_jaxpr jaxpr (consts @ tangents)) in
    List.map
      (fun pv ->
        match pv.pv_const with
        | Some c -> c
        | None -> (
            match !computed with
            | v :: rest ->
                computed := rest;
                v
            | [] -> failwith "ad.linearize: tangent output arity"))
      tangent_pvals
  in
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

let undef_aval = function Undef a -> a | Prim v -> Core.get_aval v

let remaining_dims ndim (used : int array list) =
  List.filter
    (fun d -> not (List.exists (fun a -> Array.mem d a) used))
    (List.init ndim Fun.id)

let int_range start len = Array.init len (fun i -> start + i)

let invert_permutation perm =
  let inv = Array.make (Array.length perm) 0 in
  Array.iteri (fun i p -> inv.(p) <- i) perm;
  inv

let argsort_ints a =
  let idx = Array.init (Array.length a) Fun.id in
  Array.stable_sort (fun i j -> compare a.(i) a.(j)) idx;
  idx

let dot_transpose_lhs g (x : aval) y (dims : dot_dims) swap_ans =
  let x_ndim = Array.length x.shape in
  let y_ndim = Array.length (Core.get_aval y).shape in
  let x_kept =
    Array.of_list (remaining_dims x_ndim [ dims.lhs_contract; dims.lhs_batch ])
  in
  let y_kept =
    Array.of_list (remaining_dims y_ndim [ dims.rhs_contract; dims.rhs_batch ])
  in
  let n_batch = Array.length dims.lhs_batch in
  let ans_batch = int_range 0 n_batch in
  let ans_y =
    if swap_ans then int_range n_batch (Array.length y_kept)
    else int_range (n_batch + Array.length x_kept) (Array.length y_kept)
  in
  let g_dims =
    {
      lhs_contract = ans_y;
      rhs_contract = y_kept;
      lhs_batch = ans_batch;
      rhs_batch = dims.rhs_batch;
    }
  in
  let x_contract_sorted =
    Array.map (fun i -> dims.lhs_contract.(i)) (argsort_ints dims.rhs_contract)
  in
  let order = Array.concat [ dims.lhs_batch; x_kept; x_contract_sorted ] in
  b1 (Transpose (invert_permutation order)) [ b1 (Dot_general g_dims) [ g; y ] ]

let dilate_shape shape dilation =
  Array.mapi
    (fun i s -> if s = 0 then 0 else 1 + (dilation.(i) * (s - 1)))
    shape

let conv_vjp_lhs_padding in_shape window_dimensions window_strides out_shape
    padding lhs_dilation rhs_dilation =
  let lhs_d = dilate_shape in_shape lhs_dilation in
  let rhs_d = dilate_shape window_dimensions rhs_dilation in
  let out_d = dilate_shape out_shape window_strides in
  Array.mapi
    (fun i (lo, _) ->
      let before = rhs_d.(i) - lo - 1 in
      (before, lhs_d.(i) + rhs_d.(i) - 1 - out_d.(i) - before))
    padding

let conv_vjp_rhs_padding in_shape window_dimensions window_strides out_shape
    padding lhs_dilation rhs_dilation =
  let lhs_d = dilate_shape in_shape lhs_dilation in
  let rhs_d = dilate_shape window_dimensions rhs_dilation in
  let out_d = dilate_shape out_shape window_strides in
  Array.mapi
    (fun i (lo, _) -> (lo, out_d.(i) - lhs_d.(i) + (rhs_d.(i) - lo - 1)))
    padding

let conv_sdims spec = Array.sub spec 2 (Array.length spec - 2)

let conv_spec_transpose spec =
  let t = Array.copy spec in
  t.(0) <- spec.(1);
  t.(1) <- spec.(0);
  t

let take_dims shape dims = Array.map (fun d -> shape.(d)) dims

let reshape_axis_out_of src size1 x =
  let shape = (Core.get_aval x).shape in
  let before = Array.sub shape 0 src in
  let after = Array.sub shape (src + 1) (Array.length shape - src - 1) in
  let size2 = shape.(src) / size1 in
  b1 (Reshape (Array.concat [ before; [| size1; size2 |]; after ])) [ x ]

let reshape_axis_into src dst x =
  let shape = (Core.get_aval x).shape in
  let others =
    List.filter (fun i -> i <> src) (List.init (Array.length shape) Fun.id)
  in
  let rec insert i l =
    match (i, l) with
    | 0, _ -> src :: l
    | _, h :: t -> h :: insert (i - 1) t
    | _, [] -> [ src ]
  in
  let perm = Array.of_list (insert dst others) in
  let new_shape = Array.of_list (List.map (fun i -> shape.(i)) others) in
  new_shape.(dst) <- new_shape.(dst) * shape.(src);
  b1 (Reshape new_shape) [ b1 (Transpose perm) [ x ] ]

let conv_transpose_lhs g (lhs : aval) rhs ~window_strides ~padding ~lhs_dilation
    ~rhs_dilation ~(dn : conv_dims) ~feature_group_count ~batch_group_count =
  let rhs_shape = (Core.get_aval rhs).shape in
  let rhs', groups =
    if feature_group_count > 1 then
      ( reshape_axis_into dn.rhs_spec.(0) dn.rhs_spec.(1)
          (reshape_axis_out_of dn.rhs_spec.(0) feature_group_count rhs),
        feature_group_count )
    else if batch_group_count > 1 then
      ( reshape_axis_into dn.rhs_spec.(0) dn.rhs_spec.(1)
          (reshape_axis_out_of dn.rhs_spec.(0) batch_group_count rhs),
        batch_group_count )
    else (rhs, 1)
  in
  let vjp_padding =
    conv_vjp_lhs_padding
      (take_dims lhs.shape (conv_sdims dn.lhs_spec))
      (take_dims rhs_shape (conv_sdims dn.rhs_spec))
      window_strides
      (take_dims (Core.get_aval g).shape (conv_sdims dn.out_spec))
      padding lhs_dilation rhs_dilation
  in
  let revd = b1 (Rev (conv_sdims dn.rhs_spec)) [ rhs' ] in
  let out =
    b1
      (Conv_general_dilated
         {
           window_strides = lhs_dilation;
           padding = vjp_padding;
           lhs_dilation = window_strides;
           rhs_dilation;
           dimension_numbers =
             {
               lhs_spec = dn.out_spec;
               rhs_spec = conv_spec_transpose dn.rhs_spec;
               out_spec = dn.lhs_spec;
             };
           feature_group_count = groups;
           batch_group_count = 1;
         })
      [ g; revd ]
  in
  if batch_group_count > 1 then
    reshape_axis_into dn.lhs_spec.(1) dn.lhs_spec.(0)
      (reshape_axis_out_of dn.lhs_spec.(1) batch_group_count out)
  else out

let conv_transpose_rhs g lhs (rhs : aval) ~window_strides ~padding ~lhs_dilation
    ~rhs_dilation ~(dn : conv_dims) ~feature_group_count ~batch_group_count =
  let lhs_shape = (Core.get_aval lhs).shape in
  let new_feature, new_batch =
    if batch_group_count > 1 then (batch_group_count, 1)
    else if feature_group_count > 1 then (1, feature_group_count)
    else (1, 1)
  in
  let vjp_padding =
    conv_vjp_rhs_padding
      (take_dims lhs_shape (conv_sdims dn.lhs_spec))
      (take_dims rhs.shape (conv_sdims dn.rhs_spec))
      window_strides
      (take_dims (Core.get_aval g).shape (conv_sdims dn.out_spec))
      padding lhs_dilation rhs_dilation
  in
  b1
    (Conv_general_dilated
       {
         window_strides = rhs_dilation;
         padding = vjp_padding;
         lhs_dilation;
         rhs_dilation = window_strides;
         dimension_numbers =
           {
             lhs_spec = conv_spec_transpose dn.lhs_spec;
             rhs_spec = conv_spec_transpose dn.out_spec;
             out_spec = conv_spec_transpose dn.rhs_spec;
           };
         feature_group_count = new_feature;
         batch_group_count = new_batch;
       })
    [ lhs; g ]

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
  | Conj ->
      let t = ct1 () in
      [ Some (if is_complex_value t then b1 Conj [ t ] else t) ]
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
            failwith "ad: cond transpose needs a known predicate";
          let residuals =
            List.filter_map
              (fun tv -> if is_undef tv then None else Some (prim_val tv))
              rest
          in
          let n_cts = List.length cts in
          let avals =
            List.map Core.get_aval cts @ List.map Core.get_aval residuals
          in
          let transposed (branch : closed_jaxpr) =
            Jaxpr.make_jaxpr avals (fun args ->
                let body_cts, res =
                  match Util.split_list args [ n_cts ] with
                  | [ a; b ] -> (a, b)
                  | _ -> arity ()
                in
                let remaining = ref res in
                let n_consts = List.length branch.consts in
                let binders =
                  List.filteri
                    (fun i _ -> i >= n_consts)
                    branch.jaxpr.in_binders
                in
                let inner =
                  List.map2
                    (fun tv (v : var) ->
                      if is_undef tv then Undef v.vaval
                      else
                        match !remaining with
                        | r :: tl ->
                            remaining := tl;
                            Prim r
                        | [] -> arity ())
                    rest binders
                in
                eval_jaxpr_transposed branch.jaxpr
                  (List.map (fun c -> Prim (Concrete c)) branch.consts @ inner)
                  body_cts)
          in
          let in_cts =
            Core.bind
              (Cond { t = transposed t; f = transposed f })
              ((prim_val pred :: cts) @ residuals)
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
  | Collective { kind; axis_name } -> (
      match kind with
      | Psum -> [ Some (b1 prim [ ct1 () ]) ]
      | Ppermute pairs ->
          [
            Some
              (b1
                 (Collective
                    {
                      kind =
                        Ppermute
                          (Array.map (fun (src, dst) -> (dst, src)) pairs);
                      axis_name;
                    })
                 [ ct1 () ]);
          ]
      | Pmax | Pmin | Pbroadcast _ | All_gather _ | Axis_index ->
          failwith "ad: this collective has no transpose rule")
  | Real ->
      let t = ct1 () in
      [ Some (b1 Complex [ t; zeros_like_value t ]) ]
  | Imag ->
      let t = ct1 () in
      [ Some (b1 Complex [ zeros_like_value t; neg t ]) ]
  | Complex -> (
      let t = ct1 () in
      match primals with
      | [ x; y ] ->
          let x_ct = if is_undef x then Some (b1 Real [ t ]) else None in
          let y_ct = if is_undef y then Some (b1 Imag [ neg t ]) else None in
          [ x_ct; y_ct ]
      | _ -> arity ())
  | Fft { fft_type; fft_lengths } -> (
      let t = ct1 () in
      let ndim = Array.length (Core.get_aval t).shape in
      let m = (Core.get_aval t).shape.(ndim - 1) in
      let n = fft_lengths.(Array.length fft_lengths - 1) in
      let dt = (Core.get_aval t).dtype in
      let hermitian_mask kept =
        Array.init kept (fun i ->
            if i = 0 || (n mod 2 = 0 && i = kept - 1) then 1.0 else 2.0)
      in
      let expanded values dtype =
        let shape =
          Array.append (Array.make (ndim - 1) 1) [| Array.length values |]
        in
        match dtype with
        | Dtype.Complex64 | Dtype.Complex128 ->
            Concrete
              (Ndarray.of_complex dtype shape
                 (Array.map (fun re -> { Complex.re; im = 0.0 }) values))
        | _ -> Concrete (Ndarray.of_floats dtype shape values)
      in
      let length_product = Array.fold_left ( * ) 1 fft_lengths in
      match fft_type with
      | Fft_forward | Fft_inverse -> [ Some (b1 prim [ t ]) ]
      | Fft_real ->
          let mask = expanded (hermitian_mask m) dt in
          let inner =
            b1
              (Fft { fft_type = Fft_inverse_real; fft_lengths })
              [
                b1 Div
                  [
                    b1 Conj [ t ];
                    broadcast_to_matrix mask (Core.get_aval t).shape;
                  ];
              ]
          in
          [ Some (mul inner (const_like inner (float_of_int length_product))) ]
      | Fft_inverse_real ->
          let x = b1 (Fft { fft_type = Fft_real; fft_lengths }) [ t ] in
          let xa = Core.get_aval x in
          let kept = xa.shape.(Array.length xa.shape - 1) in
          let mask =
            expanded
              (Array.map
                 (fun v -> v /. float_of_int length_product)
                 (hermitian_mask kept))
              xa.dtype
          in
          [ Some (b1 Conj [ b1 Mul [ broadcast_to_matrix mask xa.shape; x ] ]) ]
      )
  | Dot_general dims -> (
      match primals with
      | [ x; y ] when is_undef x && not (is_undef y) ->
          [
            Some
              (dot_transpose_lhs (ct1 ()) (undef_aval x) (prim_val y) dims false);
            None;
          ]
      | [ x; y ] when is_undef y && not (is_undef x) ->
          let swapped =
            {
              lhs_contract = dims.rhs_contract;
              rhs_contract = dims.lhs_contract;
              lhs_batch = dims.rhs_batch;
              rhs_batch = dims.lhs_batch;
            }
          in
          [
            None;
            Some
              (dot_transpose_lhs (ct1 ()) (undef_aval y) (prim_val x) swapped
                 true);
          ]
      | _ -> failwith "ad: dot_general transpose must be linear in one operand")
  | Conv_general_dilated c -> (
      match primals with
      | [ lhs; rhs ] when is_undef lhs && not (is_undef rhs) ->
          [
            Some
              (conv_transpose_lhs (ct1 ()) (undef_aval lhs) (prim_val rhs)
                 ~window_strides:c.window_strides ~padding:c.padding
                 ~lhs_dilation:c.lhs_dilation ~rhs_dilation:c.rhs_dilation
                 ~dn:c.dimension_numbers
                 ~feature_group_count:c.feature_group_count
                 ~batch_group_count:c.batch_group_count);
            None;
          ]
      | [ lhs; rhs ] when is_undef rhs && not (is_undef lhs) ->
          [
            None;
            Some
              (conv_transpose_rhs (ct1 ()) (prim_val lhs) (undef_aval rhs)
                 ~window_strides:c.window_strides ~padding:c.padding
                 ~lhs_dilation:c.lhs_dilation ~rhs_dilation:c.rhs_dilation
                 ~dn:c.dimension_numbers
                 ~feature_group_count:c.feature_group_count
                 ~batch_group_count:c.batch_group_count);
          ]
      | _ -> failwith "ad: conv transpose must be linear in one operand")
  | Pad cfg -> (
      match primals with
      | [ operand; padding_value ] ->
          let t = ct1 () in
          let ndim = Array.length cfg in
          let t_operand () =
            let unpad = Array.map (fun (lo, hi, _) -> (-lo, -hi, 0)) cfg in
            let dt = (Core.get_aval t).dtype in
            let zero = Concrete (Ndarray.of_floats dt [||] [| 0.0 |]) in
            let unpadded = b1 (Pad unpad) [ t; zero ] in
            b1
              (Slice
                 {
                   start_indices = Array.make ndim 0;
                   limit_indices = (Core.get_aval unpadded).shape;
                   strides = Some (Array.map (fun (_, _, i) -> i + 1) cfg);
                 })
              [ unpadded ]
          in
          let total x = b1 (Reduce_sum (Array.init ndim Fun.id)) [ x ] in
          let operand_ct =
            if is_undef operand then Some (t_operand ()) else None
          in
          let padding_ct =
            if is_undef padding_value then
              Some (b1 Sub [ total t; total (t_operand ()) ])
            else None
          in
          [ operand_ct; padding_ct ]
      | _ -> arity ())
  | Reduce_window_sum w -> (
      match primals with
      | [ operand ] ->
          let t = ct1 () in
          let input_shape = (undef_aval operand).shape in
          let pads =
            conv_vjp_lhs_padding input_shape w.window_dimensions
              w.window_strides (Core.get_aval t).shape w.w_padding
              w.base_dilation w.window_dilation
          in
          let cfg =
            Array.mapi
              (fun i (lo, hi) -> (lo, hi, w.window_strides.(i) - 1))
              pads
          in
          let dt = (Core.get_aval t).dtype in
          let zero = Concrete (Ndarray.of_floats dt [||] [| 0.0 |]) in
          let padded = b1 (Pad cfg) [ t; zero ] in
          let ndim = Array.length input_shape in
          [
            Some
              (b1
                 (Reduce_window_sum
                    {
                      window_dimensions = w.window_dimensions;
                      window_strides = w.base_dilation;
                      w_padding = Array.make ndim (0, 0);
                      base_dilation = Array.make ndim 1;
                      window_dilation = w.window_dilation;
                    })
                 [ padded ]);
          ]
      | _ -> arity ())
  | Select_and_scatter_add { select; window } -> (
      match primals with
      | [ source; operand ] when is_undef source ->
          let ndim = Array.length window.window_dimensions in
          let ones = Array.make ndim 1 in
          let gathered =
            b1
              (Select_and_gather_add
                 {
                   select;
                   window =
                     {
                       window with
                       base_dilation = ones;
                       window_dilation = ones;
                     };
                 })
              [ ct1 (); prim_val operand ]
          in
          [ Some gathered; None ]
      | _ ->
          failwith
            "ad: select_and_scatter_add transpose expects a linear source")
  | Select_and_gather_add { select; window } -> (
      match primals with
      | [ tangents; operand ] when is_undef tangents ->
          if Array.exists (fun d -> d <> 1) window.window_dilation then
            failwith
              "VJP not implemented for select_and_gather (MaxPool) with window \
               dilation";
          if Array.exists (fun d -> d <> 1) window.base_dilation then
            failwith
              "ad: select_and_gather transpose with base dilation is not \
               implemented";
          let result =
            b1
              (Select_and_scatter_add { select; window })
              [ ct1 (); prim_val operand ]
          in
          [ Some result; None ]
      | _ -> failwith "ad: select_and_gather transpose expects linear tangents")
  | Custom_lin { rule; n_res; _ } ->
      let transpose =
        match Core.custom_rule rule with
        | Core.Vjp_rule { bwd; _ } -> bwd
        | Core.Linear_rule { transpose; _ } -> transpose
        | Core.Jvp_rule _ | Core.Callback_rule _ ->
            failwith "ad: custom_lin with a non-linear rule"
      in
      let res_tv, linear_tv =
        match Util.split_list primals [ n_res ] with
        | [ a; b ] -> (a, b)
        | _ -> arity ()
      in
      let cts_out = transpose (List.map prim_val res_tv) cts in
      let keep =
        List.map2
          (fun tv ct -> if is_undef tv then Some ct else None)
          linear_tv cts_out
      in
      List.map (fun _ -> None) res_tv @ keep
  | Custom_jvp_call _ | Custom_vjp_call _ ->
      failwith "ad: custom call cannot appear in a linear program"
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
  | Expm1 | Floor | Integer_pow _ | Is_finite | Log1p | Logistic | Not
  | Population_count | Round | Rsqrt | Sinh | Sqrt | Square | Tan | And | Atan2
  | Eq_to | Ge | Le | Le_to | Lt_to | Mulhi | Ne | Nextafter | Or | Rem
  | Shift_left | Shift_right_arithmetic | Shift_right_logical | Xor | Argmax _
  | Argmin _ | Reduce _ | Reduce_and _ | Reduce_max _ | Reduce_min _
  | Reduce_or _ | Reduce_prod _ | Reduce_xor _ | After_all
  | Bitcast_convert_type _ | Clamp | Composite _ | Create_token | Dce_sink
  | Empty _ | Empty2 _ | From_edtype _ | Iota _ | Ragged_dot_general
  | Rng_bit_generator | Rng_uniform | Sort _ | To_edtype _ | Top_k _
  | Scatter_min _ | Scatter_max _ | Reduce_window _ | Reduce_window_max _
  | Reduce_window_min _ | Select_and_scatter _ | Bessel_i0e | Bessel_i1e
  | Digamma | Erf | Erf_inv | Erfc | Igamma | Igamma_grad_a | Igammac | Lgamma
  | Polygamma | Regularized_incomplete_beta | Zeta | Platform_index _
  | Cumprod _ | Cummax _ | Cummin _ | Cumlogsumexp _ | Threefry2x32
  | Iota_2x32_shape _ | Random_seed | Random_split _ | Random_fold_in
  | Random_bits _ | Random_wrap | Random_unwrap | Xla_call _ ->
      failwith "ad: primitive has no transpose rule (deferred, matches jax gap)"
  | Scan { length; reverse; num_carry; jaxpr } ->
      let split_at l n =
        match Util.split_list l [ n ] with [ a; b ] -> (a, b) | _ -> arity ()
      in
      let slice (a : aval) =
        { a with shape = Array.sub a.shape 1 (Array.length a.shape - 1) }
      in
      let carry_tv, xs_tv = split_at primals num_carry in
      let ct_carry, ct_ys = split_at cts num_carry in
      let residuals =
        List.filter_map
          (fun tv -> if is_undef tv then None else Some (prim_val tv))
          xs_tv
      in
      let n_ct = List.length cts in
      let const_tvals = List.map (fun c -> Prim (Concrete c)) jaxpr.consts in
      let carry_undef =
        List.filteri
          (fun i _ ->
            i >= List.length const_tvals
            && i < List.length const_tvals + num_carry)
          jaxpr.jaxpr.in_binders
        |> List.map (fun (v : var) -> Undef v.vaval)
      in
      let x_binders =
        List.filteri
          (fun i _ -> i >= List.length const_tvals + num_carry)
          jaxpr.jaxpr.in_binders
      in
      let transposed_body =
        Jaxpr.make_jaxpr
          (List.map Core.get_aval ct_carry
          @ List.map (fun c -> slice (Core.get_aval c)) ct_ys
          @ List.map (fun r -> slice (Core.get_aval r)) residuals)
          (fun args ->
            let body_cts, res = split_at args n_ct in
            let rest = ref res in
            let x_tvals =
              List.map2
                (fun tv (v : var) ->
                  if is_undef tv then Undef v.vaval
                  else
                    match !rest with
                    | r :: tl ->
                        rest := tl;
                        Prim r
                    | [] -> arity ())
                xs_tv x_binders
            in
            eval_jaxpr_transposed jaxpr.jaxpr
              (const_tvals @ carry_undef @ x_tvals)
              body_cts)
      in
      let outs =
        Core.bind
          (Scan
             {
               length;
               reverse = not reverse;
               num_carry;
               jaxpr = transposed_body;
             })
          (ct_carry @ ct_ys @ residuals)
      in
      let carry_out, xs_out = split_at outs num_carry in
      let remaining = ref xs_out in
      let xs_cts =
        List.map
          (fun tv ->
            if is_undef tv then
              match !remaining with
              | c :: tl ->
                  remaining := tl;
                  Some c
              | [] -> None
            else None)
          xs_tv
      in
      List.map2
        (fun tv c -> if is_undef tv then Some c else None)
        carry_tv carry_out
      @ xs_cts
  | While _ ->
      failwith
        "ad: reverse-mode differentiation does not work for lax.while_loop"
  | Triangular_solve o -> (
      match primals with
      | [ a; b ] when is_undef b && not (is_undef a) ->
          [
            None;
            Some
              (triangular_solve_with ~left_side:o.left_side ~lower:o.lower
                 ~transpose_a:(not o.transpose_a) ~conjugate_a:o.conjugate_a
                 ~unit_diagonal:o.unit_diagonal (prim_val a) (ct1 ()));
          ]
      | _ -> failwith "ad: triangular_solve is linear only in its right side")
  | Tridiagonal_solve -> (
      match primals with
      | [ dl; d; du; b ]
        when is_undef b && not (is_undef dl || is_undef d || is_undef du) ->
          let dl = prim_val dl and d = prim_val d and du = prim_val du in
          let m = (Core.get_aval d).shape.(0) in
          let dt = (Core.get_aval d).dtype in
          let zero = Concrete (Ndarray.of_floats dt [||] [| 0.0 |]) in
          let vec_slice v lo hi =
            b1
              (Slice
                 {
                   start_indices = [| lo |];
                   limit_indices = [| hi |];
                   strides = None;
                 })
              [ v ]
          in
          let dl_t =
            b1 (Pad [| (1, 0, 0) |]) [ vec_slice du 0 (m - 1); zero ]
          in
          let du_t = b1 (Pad [| (0, 1, 0) |]) [ vec_slice dl 1 m; zero ] in
          [
            None;
            None;
            None;
            Some (b1 Tridiagonal_solve [ dl_t; d; du_t; ct1 () ]);
          ]
      | _ -> failwith "ad: tridiagonal_solve is linear only in its right side")
  | Checkpoint_name _ -> [ Some (ct1 ()) ]
  | Cholesky | Householder_product | Lu | Lu_pivots_to_permutation _ | Qr _
  | Eig _ | Eigh _ | Hessenberg | Schur _ | Svd _ | Tridiagonal _ ->
      failwith "ad: nonlinear linalg primitive has no transpose rule"

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
    | None -> Ad_util.symbolic_zero_of_aval v.vaval
  in
  let write_ct atom ct =
    match atom with
    | A_var v when not (Ad_util.is_symbolic_zero ct) -> (
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
        if not (List.for_all Ad_util.is_symbolic_zero cts_in) then
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

let vjp_staged (f : value list -> value list) (primals : value list) :
    value list * value list * (value list -> value list -> value list) =
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
  let primal_pvals, tangent_pvals = Util.split_half pvals_out in
  let primals_out =
    List.map
      (fun pv ->
        match pv.pv_const with
        | Some c -> c
        | None -> failwith "ad.vjp: primal output not known")
      primal_pvals
  in
  let undefined = List.map (fun x -> Undef (Core.get_aval x)) primals in
  let backward residuals cts =
    let transpose_inputs = List.map (fun c -> Prim c) residuals @ undefined in
    let unknown_cts =
      List.concat
        (List.map2
           (fun pv ct -> if Partial_eval.is_unknown pv then [ ct ] else [])
           tangent_pvals cts)
    in
    eval_jaxpr_transposed jaxpr transpose_inputs unknown_cts
  in
  (primals_out, consts, backward)

let vjp f primals =
  let primals_out, residuals, backward = vjp_staged f primals in
  (primals_out, backward residuals)

let aux_primal = function
  | Tracer { trace = { kind = KJVP; level; _ }; payload = JVP { primal; _ }; _ }
    when level = Core.current_level () ->
      Core.full_lower primal
  | v -> v

let grad (f : value list -> value) (xs : value list) : value list =
  let primals_out, f_vjp = vjp (fun args -> [ f args ]) xs in
  let y =
    match primals_out with
    | [ y ] -> y
    | _ -> failwith "ad.grad: expected a single output"
  in
  f_vjp [ ones_like_value y ]
