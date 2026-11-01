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
    s_sorted_unique = false;
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
  let shape_with_1 =
    Array.mapi (fun i d -> if Array.mem i axes then 1 else d) a.shape
  in
  let ans = b1 prim [ x ] in
  if is_zero tx then (ans, zeros_like_value ans)
  else
    let reshaped = b1 (Reshape shape_with_1) [ ans ] in
    let loc =
      b1 (Convert_element_type (Core.get_aval tx).dtype)
        [ b1 Eq [ x; reshaped ] ]
    in
    let counts = b1 (Reduce_sum axes) [ loc ] in
    let numer = b1 (Reduce_sum axes) [ b1 Mul [ tx; loc ] ] in
    (ans, b1 Div [ numer; counts ])

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

let matmul_at dot_precision x y =
  let ndim = Array.length (Core.get_aval x).shape in
  let batch = Array.init (ndim - 2) Fun.id in
  b1
    (Dot_general
       {
         lhs_contract = [| ndim - 1 |];
         rhs_contract = [| ndim - 2 |];
         lhs_batch = batch;
         rhs_batch = batch;
         dot_precision;
       })
    [ x; y ]

let matmul_plain = matmul_at None
let matmul = matmul_at (Some ("HIGHEST", "HIGHEST"))

let broadcast_to_matrix x shape =
  let a = Core.get_aval x in
  b1
    (Broadcast_in_dim { shape; dims = Array.init (Array.length a.shape) Fun.id })
    [ x ]

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

let slice_in_dim ~axis ?(stride = 1) start stop x =
  let shape = (Core.get_aval x).shape in
  let n = Array.length shape in
  let d = shape.(axis) in
  let stop = if stop < 0 then d + stop else min stop d in
  let start_indices = Array.make n 0 and limit_indices = Array.copy shape in
  start_indices.(axis) <- start;
  limit_indices.(axis) <- stop;
  let strides =
    if stride = 1 then None
    else begin
      let st = Array.make n 1 in
      st.(axis) <- stride;
      Some st
    end
  in
  b1 (Slice { start_indices; limit_indices; strides }) [ x ]

let interleave a b axis =
  let an = (Core.get_aval a).shape.(axis) in
  let bn = (Core.get_aval b).shape.(axis) in
  let dt = (Core.get_aval a).dtype in
  let n = Array.length (Core.get_aval a).shape in
  let zero = Core.splat dt [||] 0.0 in
  let cfg lo hi =
    let c = Array.make n (0, 0, 0) in
    c.(axis) <- (lo, hi, 1);
    c
  in
  let a_pad = if bn = an then 1 else 0 in
  let pa = b1 (Pad (cfg 0 a_pad)) [ a; zero ] in
  let pb = b1 (Pad (cfg 1 (1 - a_pad))) [ b; zero ] in
  ignore bn;
  if dt = Dtype.Bool then b1 Or [ pa; pb ] else b1 Add [ pa; pb ]

let associative_scan_flat combine ~axis ~reverse elems =
  let elems =
    if reverse then List.map (fun e -> b1 (Rev [| axis |]) [ e ]) elems
    else elems
  in
  let rec scan elems =
    let num = (Core.get_aval (List.hd elems)).shape.(axis) in
    if num < 2 then elems
    else begin
      let left = List.map (slice_in_dim ~axis ~stride:2 0 (-1)) elems in
      let right = List.map (slice_in_dim ~axis ~stride:2 1 max_int) elems in
      let reduced = combine left right in
      let odd = scan reduced in
      let even =
        if num mod 2 = 0 then begin
          let a = List.map (slice_in_dim ~axis 0 (-1)) odd in
          let b = List.map (slice_in_dim ~axis ~stride:2 2 max_int) elems in
          combine a b
        end
        else
          let b = List.map (slice_in_dim ~axis ~stride:2 2 max_int) elems in
          combine odd b
      in
      let even =
        List.map2
          (fun elem result ->
            let head = slice_in_dim ~axis 0 1 elem in
            b1 (Concatenate axis) [ head; result ])
          elems even
      in
      List.map2 (fun e o -> interleave e o axis) even odd
    end
  in
  let scans = scan elems in
  if reverse then List.map (fun e -> b1 (Rev [| axis |]) [ e ]) scans else scans

let p2tz v = zeros_like_value v

let add_tangents x y =
  if is_zero x then y else if is_zero y then x else Ad_util.add_any x y

let instantiate t =
  if is_zero t then
    let a = Core.get_aval t in
    Core.splat a.dtype a.shape 0.0
  else t

let scalar_const v c = Core.splat (Core.get_aval v).dtype [||] c

let full_like v c =
  let a = Core.get_aval v in
  Core.splat a.dtype a.shape c

let lmul a b = b1 Mul [ a; b ]
let ldiv a b = b1 Div [ a; b ]
let ladd a b = b1 Add [ a; b ]
let lsub a b = b1 Sub [ a; b ]
let lneg a = b1 Neg [ a ]
let lsquare a = b1 Square [ a ]
let select pred on_true on_false = b1 Select_n [ pred; on_false; on_true ]
let lreciprocal x = b1 (Integer_pow (-1)) [ x ]

let lreshape shape x =
  let current = (Core.get_aval x).shape in
  if Array.length current > 0 && current = shape then x
  else b1 (Reshape shape) [ x ]

let expand_dims x dims =
  if Array.length dims = 0 then x
  else
    let shape = (Core.get_aval x).shape in
    let ndim_out = Array.length shape + Array.length dims in
    let is_new = Array.make ndim_out false in
    Array.iter (fun d -> is_new.(d) <- true) dims;
    let next = ref 0 in
    let out =
      Array.init ndim_out (fun i ->
          if is_new.(i) then 1
          else begin
            let d = shape.(!next) in
            incr next;
            d
          end)
    in
    let kept =
      List.filter (fun i -> not is_new.(i)) (List.init ndim_out Fun.id)
    in
    let bdims = Array.of_list kept in
    if ndim_out = Array.length shape && Array.length bdims = 0 then x
    else b1 (Broadcast_in_dim { shape = out; dims = bdims }) [ x ]

let maybe_broadcast target x =
  let shape = (Core.get_aval x).shape in
  if shape = target then x
  else if Array.length shape = 0 then
    b1 (Broadcast_in_dim { shape = target; dims = [||] }) [ x ]
  else
    let dims =
      List.filter
        (fun i -> shape.(i) = target.(i))
        (List.init (Array.length shape) Fun.id)
    in
    let squeeze = Array.of_list (List.map (fun i -> shape.(i)) dims) in
    let reshaped = lreshape squeeze x in
    b1
      (Broadcast_in_dim { shape = target; dims = Array.of_list dims })
      [ reshaped ]

let unbroadcast (target : int array) x =
  let shape = (Core.get_aval x).shape in
  if shape = target then x
  else if Array.length target = 0 then
    b1 (Reduce_sum (Array.init (Array.length shape) Fun.id)) [ x ]
  else
    let dims =
      List.filter
        (fun i -> shape.(i) <> target.(i))
        (List.init (Array.length shape) Fun.id)
    in
    let reduced =
      match dims with
      | [] -> x
      | _ -> b1 (Reduce_sum (Array.of_list dims)) [ x ]
    in
    lreshape target reduced

let replace_zero x =
  let is_z = b1 Eq [ x; scalar_const x 0.0 ] in
  let ones = full_like x 1.0 in
  select is_z ones x

let eq_meet a b =
  let da = (Core.get_aval a).dtype and db = (Core.get_aval b).dtype in
  if da = db then b1 Eq [ a; b ]
  else if Dtypes.promote_types da db = da then
    b1 Eq [ b1 (Convert_element_type db) [ a ]; b ]
  else b1 Eq [ a; b1 (Convert_element_type da) [ b ] ]

let balanced_eq x z y =
  let e1 = eq_meet x z in
  let o1 = full_like z 1.0 in
  let z1 = full_like z 0.0 in
  let num = select e1 o1 z1 in
  let e2 = eq_meet y z in
  let t2 = full_like z 2.0 in
  let o2 = full_like z 1.0 in
  let den = select e2 t2 o2 in
  ldiv num den

let standard_jvp prim primals tangents rules =
  let out = b1 prim primals in
  let terms =
    List.concat
      (List.map2
         (fun rule t ->
           match rule with
           | Some r when not (is_zero t) -> [ r t out ]
           | _ -> [])
         rules tangents)
  in
  (out, List.fold_left add_tangents (p2tz out) terms)

let linear_jvp prim primals tangents =
  let out = b1 prim primals in
  if List.for_all is_zero tangents then (out, p2tz out)
  else
    let ts = List.map instantiate tangents in
    (out, b1 prim ts)

let zero_jvp prim primals =
  let out = b1 prim primals in
  (out, p2tz out)

let is_inexact_dtype dt = Dtypes.issubdtype dt `Inexact

let pow_jvp_lhs g x y =
  let y_dtype = (Core.get_aval y).dtype in
  let rd = Dtypes.promote_types (Core.get_aval x).dtype y_dtype in
  let conv v =
    if (Core.get_aval v).dtype = rd then v
    else b1 (Convert_element_type rd) [ v ]
  in
  let x = conv x in
  let y = conv y in
  if not (is_inexact_dtype y_dtype) then begin
    let sx = (Core.get_aval x).shape and sy = (Core.get_aval y).shape in
    let x, y =
      if sx <> sy then
        let shape = Core.elementwise_shape [ sx; sy ] in
        let x = maybe_broadcast shape x in
        let y = maybe_broadcast shape y in
        (x, y)
      else (x, y)
    in
    let is_z = b1 Eq [ y; scalar_const y 0.0 ] in
    let zs = full_like y 0.0 in
    let rz = replace_zero y in
    let ones = full_like y 1.0 in
    let dec = lsub y ones in
    let powed = b1 Pow [ x; dec ] in
    let m = lmul rz powed in
    let jac = select is_z zs m in
    lmul g jac
  end
  else
    let ones = full_like y 1.0 in
    let dec = lsub y ones in
    let powed = b1 Pow [ x; dec ] in
    let jac = lmul y powed in
    lmul g jac

let pow_jvp_rhs g ans x y =
  let y_dtype = (Core.get_aval y).dtype in
  let rz = replace_zero x in
  let lg = b1 Log [ rz ] in
  let m = lmul lg ans in
  let out = lmul g m in
  if (Core.get_aval out).dtype = y_dtype then out
  else b1 (Convert_element_type y_dtype) [ out ]

let abs_jvp g ans x =
  if is_complex_value x then begin
    let cx = b1 Conj [ x ] in
    let dt = (Core.get_aval x).dtype in
    let conv = b1 (Convert_element_type dt) [ ans ] in
    let rz = replace_zero conv in
    let q = ldiv cx rz in
    let m = lmul g q in
    b1 Real [ m ]
  end
  else
    let z = scalar_const x 0.0 in
    let pred = b1 Ge [ x; z ] in
    let ng = lneg g in
    select pred g ng

let integer_pow_jvp g x y =
  if y = 0 then full_like g 0.0
  else if y = 1 then g
  else if y = 2 then
    let m = lmul (scalar_const x 2.0) x in
    lmul g m
  else
    let p = b1 (Integer_pow (y - 1)) [ x ] in
    let m = lmul (scalar_const x (float_of_int y)) p in
    lmul g m

let arg1 = function [ x ] -> x | _ -> arity ()
let arg2 = function [ x; y ] -> (x, y) | _ -> arity ()

let unary_rule prim primals tangents rule =
  let x = arg1 primals in
  standard_jvp prim primals tangents [ Some (fun g ans -> rule g ans x) ]

let reduce_chooser_jvp_rule axes g ans operand =
  let shape =
    Array.mapi
      (fun i d -> if Array.mem i axes then 1 else d)
      (Core.get_aval operand).shape
  in
  let reshaped = lreshape shape ans in
  let m = eq_meet operand reshaped in
  let loc = b1 (Convert_element_type (Core.get_aval g).dtype) [ m ] in
  let counts = b1 (Reduce_sum axes) [ loc ] in
  let gl = lmul g loc in
  let num = b1 (Reduce_sum axes) [ gl ] in
  ldiv num counts

let lax_iota dtype shape dimension = b1 (Iota { dtype; shape; dimension }) []
let scalar_i32 n = Core.splat Dtype.I32 [||] (float_of_int n)

let lax_eye dtype n m =
  let i = lax_iota Dtype.I32 [| n; m |] 0 in
  let plus = ladd i (scalar_i32 0) in
  let j = lax_iota Dtype.I32 [| n; m |] 1 in
  let e = b1 Eq [ plus; j ] in
  b1 (Convert_element_type dtype) [ e ]

let ndim_of v = Array.length (Core.get_aval v).shape
let shape_of v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype
let batch_of v = Array.sub (shape_of v) 0 (ndim_of v - 2)

let batch_broadcast mask batch =
  if Array.length batch = 0 then mask
  else
    let b = Array.length batch in
    let ms = shape_of mask in
    b1
      (Broadcast_in_dim
         {
           shape = Array.append batch ms;
           dims = Array.init (Array.length ms) (fun i -> b + i);
         })
      [ mask ]

let tri_mask n m k =
  let i = lax_iota Dtype.I32 [| n; m |] 0 in
  let plus = ladd i (scalar_i32 k) in
  let j = lax_iota Dtype.I32 [| n; m |] 1 in
  b1 Ge [ plus; j ]

let ltril ?(k = 0) x =
  let s = shape_of x and r = ndim_of x in
  let mask = tri_mask s.(r - 2) s.(r - 1) k in
  let mask = batch_broadcast mask (batch_of x) in
  let zeros = full_like x 0.0 in
  select mask x zeros

let ltriu ?(k = 0) x =
  let s = shape_of x and r = ndim_of x in
  let mask = tri_mask s.(r - 2) s.(r - 1) (k - 1) in
  let mask = batch_broadcast mask (batch_of x) in
  let zeros = full_like x 0.0 in
  select mask zeros x

let lconj x = if is_complex_value x then b1 Conj [ x ] else x
let lreal x = if is_complex_value x then b1 Real [ x ] else x
let t_of x = swap_last2 x

let h_of x =
  let t = t_of x in
  lconj t

let astype x dt =
  if dtype_of x = dt then x else b1 (Convert_element_type dt) [ x ]

let expand_leading x rank =
  let k = rank - ndim_of x in
  if k = 0 then x else expand_dims x (Array.init k Fun.id)

let symmetrize x =
  let h = h_of x in
  let s = ladd x h in
  ldiv s (scalar_const x 2.0)

let triangular ?(left_side = false) ?(lower = false) ?(transpose_a = false)
    ?(conjugate_a = false) ?(unit_diagonal = false) a b =
  let conjugate_a = conjugate_a && is_complex_value a in
  b1
    (Triangular_solve
       { left_side; lower; transpose_a; conjugate_a; unit_diagonal })
    [ a; b ]

let normalize_index n x =
  let lt = b1 Lt [ x; scalar_i32 0 ] in
  let plus = ladd x (scalar_i32 n) in
  select lt plus x

let diagonal_indices n =
  let i = lax_iota Dtype.I32 [| n |] 0 in
  let a = normalize_index n i in
  let b = normalize_index n i in
  let c1 = b1 (Broadcast_in_dim { shape = [| n; 1 |]; dims = [| 0 |] }) [ a ] in
  let c2 = b1 (Broadcast_in_dim { shape = [| n; 1 |]; dims = [| 0 |] }) [ b ] in
  b1 (Concatenate 1) [ c1; c2 ]

let extract_diagonal s =
  let sh = shape_of s and r = ndim_of s in
  let n = min sh.(r - 2) sh.(r - 1) in
  let idx = diagonal_indices n in
  let b = r - 2 in
  b1
    (Gather
       {
         dimension_numbers =
           {
             offset_dims = Array.init b Fun.id;
             collapsed_slice_dims = [| b; b + 1 |];
             start_index_map = [| b; b + 1 |];
             g_operand_batching_dims = [||];
             g_start_indices_batching_dims = [||];
           };
         slice_sizes = Array.init r (fun i -> if i < b then sh.(i) else 1);
       })
    [ s; idx ]

let construct_diagonal s =
  let sh = shape_of s and r = ndim_of s in
  let n = sh.(r - 1) in
  let i = lax_iota Dtype.I32 [| n |] 0 in
  let zeros = Core.splat (dtype_of s) (Array.append sh [| n |]) 0.0 in
  let a = normalize_index n i in
  let b = normalize_index n i in
  let c1 = b1 (Broadcast_in_dim { shape = [| n; 1 |]; dims = [| 0 |] }) [ a ] in
  let c2 = b1 (Broadcast_in_dim { shape = [| n; 1 |]; dims = [| 0 |] }) [ b ] in
  let idx = b1 (Concatenate 1) [ c1; c2 ] in
  let bd = r - 1 in
  b1
    (Scatter
       {
         dimension_numbers =
           {
             update_window_dims = Array.init bd Fun.id;
             inserted_window_dims = [| bd; bd + 1 |];
             scatter_dims_to_operand_dims = [| bd; bd + 1 |];
             s_operand_batching_dims = [||];
             s_scatter_indices_batching_dims = [||];
             s_sorted_unique = false;
           };
         unique_indices = false;
       })
    [ zeros; idx; s ]

let promote_rank_add a b =
  let r = max (ndim_of a) (ndim_of b) in
  let a = expand_leading a r in
  let b = expand_leading b r in
  ladd a b

let promote_rank_sub a b =
  let r = max (ndim_of a) (ndim_of b) in
  let a = expand_leading a r in
  let b = expand_leading b r in
  lsub a b

let row_vector w =
  let sh = shape_of w and r = ndim_of w in
  let n = sh.(r - 1) in
  let b = r - 1 in
  b1
    (Broadcast_in_dim
       {
         shape = Array.concat [ Array.sub sh 0 b; [| 1; n |] ];
         dims = Array.init r (fun i -> if i < b then i else i + 1);
       })
    [ w ]

let column_vector w =
  let sh = shape_of w and r = ndim_of w in
  b1
    (Broadcast_in_dim
       { shape = Array.append sh [| 1 |]; dims = Array.init r Fun.id })
    [ w ]

let cholesky_jvp x t =
  let c = b1 Cholesky [ x ] in
  let l = ltril c in
  let phi m =
    let lo = ltril m in
    let n = (shape_of m).(ndim_of m - 1) in
    let eye = lax_eye (dtype_of m) n n in
    let s = ladd (scalar_const m 1.0) eye in
    let d = expand_leading s (ndim_of lo) in
    ldiv lo d
  in
  let tmp =
    triangular ~left_side:false ~transpose_a:true ~conjugate_a:true ~lower:true
      l t
  in
  let inner = triangular ~left_side:true ~lower:true l tmp in
  let p = phi inner in
  (l, matmul l p)

let eigh_jvp ~lower a t =
  let n = (shape_of a).(ndim_of a - 1) in
  let sym = symmetrize a in
  let outs = Core.bind (Eigh { lower }) [ sym ] in
  match outs with
  | [ v; w_real ] ->
      let w = astype w_real (dtype_of a) in
      let eye_n = lax_eye (dtype_of a) n n in
      let row = row_vector w in
      let s1 = promote_rank_add eye_n row in
      let col = column_vector w in
      let s2 = promote_rank_sub s1 col in
      let inv = b1 (Integer_pow (-1)) [ s2 ] in
      let fmat = promote_rank_sub inv eye_n in
      let hv = h_of v in
      let d1 = matmul hv t in
      let vav = matmul d1 v in
      let m = lmul fmat vav in
      let dv = matmul v m in
      let re = lreal vav in
      let dw = extract_diagonal re in
      (outs, [ dv; dw ])
  | _ -> failwith "ad: eigh arity"

let slice_block x row col =
  let sh = shape_of x and r = ndim_of x in
  let rlo, rhi = row and clo, chi = col in
  if rlo = 0 && rhi = sh.(r - 2) && clo = 0 && chi = sh.(r - 1) then x
  else
    let start_indices = Array.make r 0 and limit_indices = Array.copy sh in
    start_indices.(r - 2) <- rlo;
    limit_indices.(r - 2) <- rhi;
    start_indices.(r - 1) <- clo;
    limit_indices.(r - 1) <- chi;
    b1 (Slice { start_indices; limit_indices; strides = None }) [ x ]

let thin_qr_jvp q r dx =
  let dx_rinv = triangular r dx in
  let hq = h_of q in
  let qt = matmul_plain hq dx_rinv in
  let lower = ltril ~k:(-1) qt in
  let hl = h_of lower in
  let d = lsub lower hl in
  let n = (shape_of r).(ndim_of r - 1) in
  let i = lax_eye (dtype_of d) n n in
  let i = expand_leading i (ndim_of qt) in
  let re = astype (lreal qt) (dtype_of qt) in
  let diff = lsub qt re in
  let corr = lmul i diff in
  let d = ladd d corr in
  let dq_in = lsub d qt in
  let m = matmul_plain q dq_in in
  let dq = ladd m dx_rinv in
  let dr_in = lsub qt d in
  let dr = matmul_plain dr_in r in
  (dq, dr)

let qr_tangents ~full_matrices x dx outs =
  match outs with
  | q :: r :: _ ->
      let sh = shape_of x and rk = ndim_of x in
      let m = sh.(rk - 2) and n = sh.(rk - 1) in
      if m < n then begin
        let x2 = slice_block x (0, m) (m, n) in
        let dx1 = slice_block dx (0, m) (0, m) in
        let dx2 = slice_block dx (0, m) (m, n) in
        let r1 = slice_block r (0, m) (0, m) in
        let dq, dr1 = thin_qr_jvp q r1 dx1 in
        let hdq = h_of dq in
        let a1 = matmul_plain hdq x2 in
        let hq = h_of q in
        let a2 = matmul_plain hq dx2 in
        let dr2 = ladd a1 a2 in
        let dr = b1 (Concatenate (ndim_of dr1 - 1)) [ dr1; dr2 ] in
        [ dq; dr ]
      end
      else if full_matrices && m <> n then
        failwith "Unimplemented case of QR decomposition derivative"
      else
        let dq, dr = thin_qr_jvp q r dx in
        [ dq; dr ]
  | _ -> failwith "ad: qr arity"

let qr_jvp ~full_matrices x dx =
  let outs = Core.bind (Qr full_matrices) [ x ] in
  (outs, qr_tangents ~full_matrices x dx outs)

let qr_pivoted_jvp ~full_matrices x dx =
  let outs = Core.bind (Qr_pivoted full_matrices) [ x ] in
  let p = List.nth outs 2 in
  let sh = shape_of x in
  if Array.length sh <> 2 then
    failwith "Unimplemented case of pivoted QR decomposition derivative";
  let columns =
    b1
      (Gather
         {
           dimension_numbers =
             {
               offset_dims = [| 0 |];
               collapsed_slice_dims = [| 1 |];
               start_index_map = [| 1 |];
               g_operand_batching_dims = [||];
               g_start_indices_batching_dims = [||];
             };
           slice_sizes = [| sh.(0); 1 |];
         })
      [ dx; lreshape [| sh.(1); 1 |] p ]
  in
  (outs, qr_tangents ~full_matrices x columns outs @ [ p2tz p ])

let lu_jvp_inner lu a_dot permutation =
  let sh = shape_of a_dot in
  let m = sh.(0) and n = sh.(1) in
  let dtype = dtype_of a_dot in
  let k = min m n in
  let zero = scalar_const lu 0.0 in
  let lu_k = slice_block lu (0, m) (0, k) in
  let tl = ltril ~k:(-1) lu_k in
  let l0 = b1 (Pad [| (0, 0, 0); (0, m - k, 0) |]) [ tl; zero ] in
  let eye_m = lax_eye dtype m m in
  let l = ladd l0 eye_m in
  let e = lax_eye dtype (n - k) (n - k) in
  let u_eye = b1 (Pad [| (k, 0, 0); (k, 0, 0) |]) [ e; zero ] in
  let lu_u = slice_block lu (0, k) (0, n) in
  let tu = ltriu lu_u in
  let u0 = b1 (Pad [| (0, n - k, 0); (0, 0, 0) |]) [ tu; zero ] in
  let u = ladd u0 u_eye in
  let idx = normalize_index m permutation in
  let idx =
    b1 (Broadcast_in_dim { shape = [| m; 1 |]; dims = [| 0 |] }) [ idx ]
  in
  let permuted =
    b1
      (Gather
         {
           dimension_numbers =
             {
               offset_dims = [| 1 |];
               collapsed_slice_dims = [| 0 |];
               start_index_map = [| 0 |];
               g_operand_batching_dims = [||];
               g_start_indices_batching_dims = [||];
             };
           slice_sizes = [| 1; n |];
         })
      [ a_dot; idx ]
  in
  let la =
    triangular ~left_side:true ~transpose_a:false ~lower:true
      ~unit_diagonal:true l permuted
  in
  let lau = triangular ~left_side:false ~transpose_a:false ~lower:false u la in
  let t1 = ltril ~k:(-1) lau in
  let l_dot = matmul l t1 in
  let t2 = ltriu lau in
  let u_dot = matmul t2 u in
  ladd l_dot u_dot

let lu_jvp a t =
  let outs = Core.bind Lu [ a ] in
  match outs with
  | [ lu; pivots; permutation ] ->
      let rec lifted d args =
        if d = 0 then
          match args with
          | [ l; x; p ] -> [ lu_jvp_inner l x p ]
          | _ -> arity ()
        else Batching.vmap (lifted (d - 1)) [ Some 0; Some 0; Some 0 ] args
      in
      let lu_dot =
        match lifted (ndim_of a - 2) [ lu; t; permutation ] with
        | [ d ] -> d
        | _ -> arity ()
      in
      (outs, [ lu_dot; p2tz pivots; p2tz permutation ])
  | _ -> failwith "ad: lu arity"

let solve_ref : (value -> value -> value) ref =
  ref (fun _ _ -> failwith "ad: linear solve not installed")

let eig_vec_jvp w v da =
  let n = (shape_of w).(ndim_of w - 1) in
  let eye_n = lax_eye (dtype_of w) n n in
  let row = row_vector w in
  let s1 = promote_rank_add eye_n row in
  let col = column_vector w in
  let s2 = promote_rank_sub s1 col in
  let rc = lreciprocal s2 in
  let fmat = promote_rank_sub rc eye_n in
  let solved = !solve_ref v da in
  let p = matmul solved v in
  let dw = extract_diagonal p in
  let fp = lmul fmat p in
  let u = matmul v fp in
  let r = ndim_of v in
  let ab = b1 Abs [ v ] in
  let k = b1 (Argmax { axis = r - 2; index_dtype = Dtype.I32 }) [ ab ] in
  let iota = lax_iota Dtype.I32 (shape_of v) (r - 2) in
  let ek = expand_dims k [| r - 2 |] in
  let eqm = b1 Eq [ iota; ek ] in
  let mask = astype eqm (dtype_of v) in
  let sum_rows x = b1 (Reduce_sum [| r - 2 |]) [ x ] in
  let vc = lconj v in
  let m1 = lmul vc u in
  let s_1 = sum_rows m1 in
  let re1 = lreal s_1 in
  let c_re = lneg re1 in
  let m2 = lmul mask u in
  let s_2 = sum_rows m2 in
  let im2 = b1 Imag [ s_2 ] in
  let ni = lneg im2 in
  let m3 = lmul mask v in
  let s_3 = sum_rows m3 in
  let re3 = lreal s_3 in
  let c_im = ldiv ni re3 in
  let c = b1 Complex [ c_re; c_im ] in
  let ec = expand_dims c [| r - 2 |] in
  let vc2 = lmul v ec in
  (dw, ladd u vc2)

let eig_jvp ~compute_left ~compute_right ~enable_eigvec_derivs a t =
  if (compute_left || compute_right) && not enable_eigvec_derivs then
    failwith
      "Derivatives of non-symmetric eigenvectors are only valid under \
       assumptions on the input that JAX cannot check. Pass \
       enable_eigvec_derivs=true to lax.linalg.eig to opt in.";
  let outs =
    Core.bind
      (Eig { compute_left; compute_right = true; enable_eigvec_derivs })
      [ a ]
  in
  let w = List.hd outs and vr = List.nth outs (List.length outs - 1) in
  let da = astype t (dtype_of vr) in
  if not (compute_left || compute_right) then begin
    let s = !solve_ref vr da in
    let tr = t_of vr in
    let m = lmul s tr in
    ([ w ], [ b1 (Reduce_sum [| ndim_of m - 1 |]) [ m ] ])
  end
  else
    let dw, dvr = eig_vec_jvp w vr da in
    let left =
      if compute_left then begin
        let vl = List.nth outs 1 in
        let wc = lconj w in
        let hda = h_of da in
        let _, dvl = eig_vec_jvp wc vl hda in
        [ (vl, dvl) ]
      end
      else []
    in
    let right = if compute_right then [ (vr, dvr) ] else [] in
    let vectors = left @ right in
    (w :: List.map fst vectors, dw :: List.map snd vectors)

let svd_jvp ~full_matrices ~compute_uv a t =
  let thin =
    Core.bind
      (Svd { full_matrices = false; compute_uv = true; gesvd = false })
      [ a ]
  in
  match thin with
  | [ u; s; vt ] ->
      let sh = shape_of a and r = ndim_of a in
      let m = sh.(r - 2) and n = sh.(r - 1) in
      if compute_uv && full_matrices && m <> n then
        failwith
          "Singular value decomposition JVP not implemented for full matrices";
      let ut = h_of u in
      let v = h_of vt in
      let s_dim = row_vector s in
      let p1 = matmul ut t in
      let d_s = matmul p1 v in
      let re = lreal d_s in
      let ds = extract_diagonal re in
      if not compute_uv then ([ s ], [ ds ])
      else begin
        let adt = dtype_of a in
        let t1 = t_of s_dim in
        let plus = ladd s_dim t1 in
        let t2 = t_of s_dim in
        let minus = lsub s_dim t2 in
        let diffs = lmul plus minus in
        let k = (shape_of s).(ndim_of s - 1) in
        let z = lax_eye (dtype_of s) k k in
        let z = expand_leading z (ndim_of diffs) in
        let dz = ladd diffs z in
        let inv = ldiv (scalar_const dz 1.0) dz in
        let f = lsub inv z in
        let sa = astype s_dim adt in
        let d_ss = lmul sa d_s in
        let sa2 = astype s_dim adt in
        let tsa = t_of sa2 in
        let sd_s = lmul tsa d_s in
        let eq0 = b1 Eq [ s; scalar_const s 0.0 ] in
        let s_zeros = astype eq0 (dtype_of s) in
        let sz = ladd s s_zeros in
        let invs = ldiv (scalar_const sz 1.0) sz in
        let s_inv = lsub invs s_zeros in
        let s_inv_mat = construct_diagonal s_inv in
        let hds = h_of d_s in
        let dd = lsub d_s hds in
        let half = lmul (scalar_const dd 0.5) dd in
        let sim = astype s_inv_mat adt in
        let dudv = lmul half sim in
        let fa = astype f adt in
        let h1 = h_of d_ss in
        let s1 = ladd d_ss h1 in
        let m1 = lmul fa s1 in
        let s2 = ladd m1 dudv in
        let du = matmul u s2 in
        let fa2 = astype f adt in
        let h2 = h_of sd_s in
        let s3 = ladd sd_s h2 in
        let m2 = lmul fa2 s3 in
        let dv = matmul v m2 in
        let du =
          if m > n then begin
            let dav = matmul t v in
            let q1 = matmul ut dav in
            let q2 = matmul u q1 in
            let d1 = lsub dav q2 in
            let sa3 = astype s_dim adt in
            let q3 = ldiv d1 sa3 in
            ladd du q3
          end
          else du
        in
        let dv =
          if n > m then begin
            let hda = h_of t in
            let dahu = matmul hda u in
            let q1 = matmul vt dahu in
            let q2 = matmul v q1 in
            let d1 = lsub dahu q2 in
            let sa3 = astype s_dim adt in
            let q3 = ldiv d1 sa3 in
            ladd dv q3
          end
          else dv
        in
        let hdv = h_of dv in
        (thin, [ du; ds; hdv ])
      end
  | _ -> failwith "ad: svd arity"

let triangular_jvp_a ~left_side ~lower ~transpose_a ~conjugate_a ~unit_diagonal
    g_a ans a b =
  let r = ndim_of b in
  let m = (shape_of b).(r - 2) and n = (shape_of b).(r - 1) in
  let k = if unit_diagonal then 1 else 0 in
  let g = if lower then ltril ~k:(-k) g_a else ltriu ~k g_a in
  let g = lneg g in
  let g = if transpose_a then t_of g else g in
  let g = if conjugate_a then lconj g else g in
  let a_inverse rhs =
    triangular ~left_side ~lower ~transpose_a ~conjugate_a ~unit_diagonal a rhs
  in
  if left_side then
    if m > n then a_inverse (matmul g ans)
    else
      let i = a_inverse g in
      matmul i ans
  else if m < n then a_inverse (matmul ans g)
  else
    let i = a_inverse g in
    matmul ans i

let gather_along ~axis ~rank t idx =
  let batch =
    Array.of_list (List.filter (fun i -> i <> axis) (List.init rank Fun.id))
  in
  b1
    (Gather
       {
         dimension_numbers =
           {
             offset_dims = [||];
             collapsed_slice_dims = [| axis |];
             start_index_map = [| axis |];
             g_operand_batching_dims = batch;
             g_start_indices_batching_dims = batch;
           };
         slice_sizes = Array.make rank 1;
       })
    [ t; idx ]

let index_dtype_for_shape shape =
  if Array.exists (fun d -> d > 0xFFFFFFFF) shape then Dtype.Uint64
  else Dtype.Uint32

let sort_jvp ~dimension ~is_stable ~num_keys primals tangents =
  let first = List.hd primals in
  let shape = (Core.get_aval first).shape in
  let idt = index_dtype_for_shape shape in
  let iota = b1 (Iota { dtype = idt; shape; dimension }) [] in
  let outs =
    Core.bind (Sort { dimension; is_stable; num_keys }) (primals @ [ iota ])
  in
  let n = List.length primals in
  let sorted = List.filteri (fun i _ -> i < n) outs in
  let perm = List.nth outs n in
  let rank = Array.length shape in
  let idx = expand_dims perm [| rank |] in
  let sorted_t =
    List.map
      (fun t ->
        if is_zero t then t else gather_along ~axis:dimension ~rank t idx)
      tangents
  in
  (sorted, sorted_t)

let top_k_jvp ~k ~axis x t =
  let outs = Core.bind (Top_k { k; axis }) [ x ] in
  match outs with
  | [ values; indices ] ->
      let t_out =
        if is_zero t then p2tz values
        else
          let idx_shape = (Core.get_aval indices).shape in
          let rank = Array.length idx_shape in
          let gather_indices =
            lreshape (Array.append idx_shape [| 1 |]) indices
          in
          gather_along ~axis ~rank t gather_indices
      in
      (outs, [ t_out; p2tz indices ])
  | _ -> failwith "ad: top_k arity"

let cumulative_jvp_ref :
    ((value -> value -> value) ->
    axis:int ->
    reverse:bool ->
    value list ->
    value list ->
    value * value)
    ref =
  ref (fun _ ~axis:_ ~reverse:_ _ _ ->
      failwith "ad: cumulative jvp not installed")

let cumulative_jvp ~axis ~reverse primals tangents combine =
  !cumulative_jvp_ref combine ~axis ~reverse primals tangents

let reduce_jvp_ref :
    ((value list -> value list) ->
    value list ->
    value list ->
    value list ->
    int array ->
    value * value)
    ref =
  ref (fun _ _ _ _ _ -> failwith "ad: reduce jvp not installed")

let reduce_jvp reducer init primals tangents axes =
  !reduce_jvp_ref reducer init primals tangents axes

let jvp_rule_core prim (primals : value list) (tangents : value list) :
    (value * value) option =
  let std1 rule = Some (unary_rule prim primals tangents rule) in
  let std rules = Some (standard_jvp prim primals tangents rules) in
  let linear () = Some (linear_jvp prim primals tangents) in
  let zero () = Some (zero_jvp prim primals) in
  match prim with
  | Add | Sub -> (
      let x, y = arg2 primals and xdot, ydot = arg2 tangents in
      let out = b1 prim [ x; y ] in
      let shape = (Core.get_aval out).shape in
      match (is_zero xdot, is_zero ydot) with
      | true, true -> Some (out, p2tz out)
      | true, false ->
          let t = if prim = Sub then lneg ydot else ydot in
          Some (out, maybe_broadcast shape t)
      | false, true -> Some (out, maybe_broadcast shape xdot)
      | false, false -> Some (out, b1 prim [ xdot; ydot ]))
  | Add_any | Neg | Real | Imag | Complex | Conj | Squeeze _ | Reshape _
  | Reshape_perm _ | Rev _ | Transpose _ | Reduce_sum _ | Stack _
  | Concatenate _ | Pad _ | Copy | Reduce_precision _ | Tile _ ->
      linear ()
  | Stop_gradient ->
      let x = arg1 primals in
      let out = Custom_derivatives.stop_gradient x in
      Some (out, p2tz x)
  | Sign | Floor | Ceil | Round | Round_nearest_even | Is_finite | Not | And
  | Or | Xor | Eq | Ne | Ge | Gt | Le | Lt | Eq_to | Le_to | Lt_to | Mulhi
  | Shift_left | Shift_right_arithmetic | Shift_right_logical | Argmax _
  | Argmin _ | Reduce_and _ | Reduce_or _ | Reduce_xor _
  | Bitcast_convert_type _ ->
      zero ()
  | Mul ->
      let x, y = arg2 primals in
      std [ Some (fun g _ -> lmul g y); Some (fun g _ -> lmul x g) ]
  | Div ->
      let x, y = arg2 primals in
      std
        [
          Some (fun g _ -> ldiv g y);
          Some
            (fun g _ ->
              let ng = lneg g in
              let m = lmul ng x in
              let p = b1 (Integer_pow (-2)) [ y ] in
              lmul m p);
        ]
  | Rem ->
      let x, y = arg2 primals in
      std
        [
          Some (fun g ans -> maybe_broadcast (Core.get_aval ans).shape g);
          Some
            (fun g _ ->
              let ng = lneg g in
              let d1 = ldiv x y in
              let sg = b1 Sign [ d1 ] in
              let d2 = ldiv x y in
              let ab = b1 Abs [ d2 ] in
              let fl = b1 Floor [ ab ] in
              let m = lmul sg fl in
              lmul ng m);
        ]
  | Max | Min ->
      let x, y = arg2 primals in
      std
        [
          Some
            (fun g ans ->
              let b = balanced_eq x ans y in
              lmul g b);
          Some
            (fun g ans ->
              let b = balanced_eq y ans x in
              lmul g b);
        ]
  | Pow ->
      let x, y = arg2 primals in
      std
        [
          Some (fun g _ -> pow_jvp_lhs g x y);
          Some (fun g ans -> pow_jvp_rhs g ans x y);
        ]
  | Atan2 ->
      let x, y = arg2 primals in
      let denom () =
        let sx = lsquare x in
        let sy = lsquare y in
        ladd sx sy
      in
      std
        [
          Some
            (fun g _ ->
              let d = denom () in
              let q = ldiv y d in
              lmul g q);
          Some
            (fun g _ ->
              let nx = lneg x in
              let d = denom () in
              let q = ldiv nx d in
              lmul g q);
        ]
  | Sin -> std1 (fun g _ x -> lmul g (b1 Cos [ x ]))
  | Cos ->
      std1 (fun g _ x ->
          let s = b1 Sin [ x ] in
          lneg (lmul g s))
  | Tan ->
      std1 (fun g ans x ->
          let c = scalar_const x 1.0 in
          let sq = lsquare ans in
          let s = ladd c sq in
          lmul g s)
  | Exp -> std1 (fun g ans _ -> lmul g ans)
  | Exp2 ->
      std1 (fun g ans x ->
          let lg = b1 Log [ scalar_const x 2.0 ] in
          let m = lmul g ans in
          lmul lg m)
  | Log -> std1 (fun g _ x -> ldiv g x)
  | Expm1 ->
      std1 (fun g ans _ ->
          let s = ladd ans (scalar_const ans 1.0) in
          lmul g s)
  | Log1p ->
      std1 (fun g _ x ->
          let s = ladd x (scalar_const x 1.0) in
          ldiv g s)
  | Tanh ->
      std1 (fun g ans x ->
          let m = lmul g ans in
          let s = ladd g m in
          let d = lsub (scalar_const x 1.0) ans in
          lmul s d)
  | Logistic ->
      std1 (fun g ans _ ->
          let d = lsub (scalar_const ans 1.0) ans in
          let m = lmul ans d in
          lmul g m)
  | Asin ->
      std1 (fun g _ x ->
          let sq = lsquare x in
          let d = lsub (scalar_const x 1.0) sq in
          let r = b1 Rsqrt [ d ] in
          lmul g r)
  | Acos ->
      std1 (fun g _ x ->
          let sq = lsquare x in
          let d = lsub (scalar_const x 1.0) sq in
          let r = b1 Rsqrt [ d ] in
          let n = lneg r in
          lmul g n)
  | Atan ->
      std1 (fun g _ x ->
          let sq = lsquare x in
          let d = ladd (scalar_const x 1.0) sq in
          ldiv g d)
  | Sinh -> std1 (fun g _ x -> lmul g (b1 Cosh [ x ]))
  | Cosh -> std1 (fun g _ x -> lmul g (b1 Sinh [ x ]))
  | Asinh ->
      std1 (fun g _ x ->
          let sq = lsquare x in
          let s = ladd sq (scalar_const x 1.0) in
          let r = b1 Rsqrt [ s ] in
          lmul g r)
  | Acosh ->
      std1 (fun g _ x ->
          let sq = lsquare x in
          let s = lsub sq (scalar_const x 1.0) in
          let r = b1 Rsqrt [ s ] in
          lmul g r)
  | Atanh ->
      std1 (fun g _ x ->
          let s = ladd (scalar_const x 1.0) x in
          let r = lreciprocal s in
          let d = lsub (scalar_const x 1.0) x in
          let q = ldiv g d in
          lmul r q)
  | Abs -> std1 (fun g ans x -> abs_jvp g ans x)
  | Sqrt ->
      std1 (fun g ans x ->
          let q = ldiv (scalar_const x 0.5) ans in
          lmul g q)
  | Rsqrt ->
      std1 (fun g ans x ->
          let q = ldiv ans x in
          let m = lmul (scalar_const x (-0.5)) q in
          lmul g m)
  | Cbrt ->
      std1 (fun g ans x ->
          let p = b1 (Integer_pow (-2)) [ ans ] in
          let m = lmul (scalar_const x (1.0 /. 3.0)) p in
          lmul g m)
  | Square ->
      std1 (fun g _ x ->
          let m = lmul (scalar_const x 2.0) x in
          lmul g m)
  | Integer_pow y -> std1 (fun g _ x -> integer_pow_jvp g x y)
  | Select_n -> (
      match (primals, tangents) with
      | which :: cases, _ :: tcases ->
          let out = b1 Select_n (which :: cases) in
          if List.for_all is_zero tcases then Some (out, List.hd tcases)
          else
            let nz = List.find (fun t -> not (is_zero t)) tcases in
            let z = full_like nz 0.0 in
            let filled =
              List.map (fun t -> if is_zero t then z else t) tcases
            in
            Some (out, b1 Select_n (which :: filled))
      | _ -> arity ())
  | Clamp -> (
      match primals with
      | [ mn; x; mx ] ->
          let zeros () = full_like x 0.0 in
          std
            [
              Some
                (fun g _ ->
                  let a = b1 Gt [ mn; x ] in
                  let b = b1 Lt [ mn; mx ] in
                  let p = b1 And [ a; b ] in
                  let z = zeros () in
                  select p g z);
              Some
                (fun g _ ->
                  let a = b1 Gt [ x; mn ] in
                  let b = b1 Lt [ x; mx ] in
                  let p = b1 And [ a; b ] in
                  let z = zeros () in
                  select p g z);
              Some
                (fun g _ ->
                  let p = b1 Lt [ mx; x ] in
                  let z = zeros () in
                  select p g z);
            ]
      | _ -> arity ())
  | Convert_element_type dt ->
      std
        [
          Some
            (fun g ans ->
              if is_inexact_dtype dt then b1 (Convert_element_type dt) [ g ]
              else p2tz ans);
        ]
  | Broadcast_in_dim _ ->
      let x = arg1 primals and t = arg1 tangents in
      let y = b1 prim [ x ] in
      Some (y, if is_zero t then p2tz y else b1 prim [ t ])
  | Dot_general _ | Conv_general_dilated _ ->
      let x, y = arg2 primals in
      std
        [
          Some (fun g _ -> b1 prim [ g; y ]); Some (fun g _ -> b1 prim [ x; g ]);
        ]
  | Reduce_max axes | Reduce_min axes ->
      std1 (fun g ans x -> reduce_chooser_jvp_rule axes g ans x)
  | Slice _ ->
      let x = arg1 primals and t = arg1 tangents in
      let out = b1 prim [ x ] in
      Some (out, if is_zero t then p2tz out else b1 prim [ t ])
  | Dynamic_slice _ -> (
      match (primals, tangents) with
      | operand :: idx, t_op :: _ ->
          let t_out = if is_zero t_op then t_op else b1 prim (t_op :: idx) in
          let out = b1 prim (operand :: idx) in
          Some (out, t_out)
      | _ -> arity ())
  | Dynamic_update_slice -> (
      match (primals, tangents) with
      | operand :: update :: idx, g_op :: g_up :: _ ->
          let out = b1 prim (operand :: update :: idx) in
          if is_zero g_op && is_zero g_up then Some (out, p2tz out)
          else
            let g_op = instantiate g_op in
            let g_up = instantiate g_up in
            Some (out, b1 prim (g_op :: g_up :: idx))
      | _ -> arity ())
  | Gather _ -> (
      match primals with
      | [ _; indices ] -> std [ Some (fun g _ -> b1 prim [ g; indices ]); None ]
      | _ -> arity ())
  | Scatter_add _ | Scatter_sub _ -> (
      match (primals, tangents) with
      | [ op; idx; up ], [ g_op; _; g_up ] ->
          let out = b1 prim [ op; idx; up ] in
          if is_zero g_op && is_zero g_up then Some (out, p2tz out)
          else
            let g_op = instantiate g_op in
            let g_up = instantiate g_up in
            Some (out, b1 prim [ g_op; idx; g_up ])
      | _ -> arity ())
  | Scatter_mul { dimension_numbers; unique_indices } -> (
      match primals with
      | [ x; i; y ] ->
          std
            [
              Some (fun g _ -> b1 prim [ g; i; y ]);
              None;
              Some
                (fun g _ ->
                  if not unique_indices then
                    failwith
                      "scatter_mul gradients are only implemented if \
                       `unique_indices=True`";
                  let zeros = full_like x 0.0 in
                  let s =
                    b1 (Scatter_add { dimension_numbers }) [ zeros; i; g ]
                  in
                  lmul x s);
            ]
      | _ -> arity ())
  | Lgamma -> std1 (fun g _ x -> lmul g (b1 Digamma [ x ]))
  | Digamma ->
      std1 (fun g _ x -> lmul g (b1 Polygamma [ scalar_const x 1.0; x ]))
  | Polygamma ->
      let m, x = arg2 primals in
      std
        [
          Some
            (fun _ _ ->
              invalid_arg
                "polygamma gradient with respect to m is not supported");
          Some
            (fun g _ ->
              let s = ladd m (scalar_const m 1.0) in
              let p = b1 Polygamma [ s; x ] in
              lmul g p);
        ]
  | Igamma | Igammac ->
      let a, x = arg2 primals in
      let negate v = if prim = Igammac then lneg v else v in
      std
        [
          Some
            (fun g _ ->
              let d = b1 Igamma_grad_a [ a; x ] in
              negate (lmul g d));
          Some
            (fun g _ ->
              let nx = lneg x in
              let ones = full_like a 1.0 in
              let am = lsub a ones in
              let lx = b1 Log [ x ] in
              let m = lmul am lx in
              let s = ladd nx m in
              let lg = b1 Lgamma [ a ] in
              let d = lsub s lg in
              let e = b1 Exp [ d ] in
              negate (lmul g e));
        ]
  | Bessel_i0e ->
      std1 (fun g y x ->
          let i1 = b1 Bessel_i1e [ x ] in
          let sg = b1 Sign [ x ] in
          let m = lmul sg y in
          let d = lsub i1 m in
          lmul g d)
  | Bessel_i1e ->
      std1 (fun g y x ->
          let dt = (Core.get_aval x).dtype in
          let eps = (Dtypes.finfo dt).eps in
          let ab = b1 Abs [ x ] in
          let not_tiny = b1 Gt [ ab; scalar_const x eps ] in
          let fe = full_like x eps in
          let safe_x = select not_tiny x fe in
          let i0 = b1 Bessel_i0e [ safe_x ] in
          let sg = b1 Sign [ safe_x ] in
          let rc = lreciprocal safe_x in
          let s = ladd sg rc in
          let m = lmul y s in
          let dy = lsub i0 m in
          let half = full_like x 0.5 in
          let dy = select not_tiny dy half in
          lmul g dy)
  | Erf | Erfc ->
      let c = if prim = Erf then two_over_sqrt_pi else -.two_over_sqrt_pi in
      std1 (fun g _ x ->
          let sq = lsquare x in
          let n = lneg sq in
          let e = b1 Exp [ n ] in
          let m = lmul g e in
          lmul (scalar_const x c) m)
  | Erf_inv ->
      std1 (fun g ans x ->
          let sq = lsquare ans in
          let e = b1 Exp [ sq ] in
          let m = lmul g e in
          lmul (scalar_const x sqrt_pi_over_2) m)
  | Regularized_incomplete_beta -> (
      match primals with
      | [ a; b; x ] ->
          let not_supported _ _ =
            invalid_arg
              "Betainc gradient with respect to a and b not supported."
          in
          std
            [
              Some not_supported;
              Some not_supported;
              Some
                (fun g _ ->
                  let la = b1 Lgamma [ a ] in
                  let lb = b1 Lgamma [ b ] in
                  let s = ladd la lb in
                  let ab = ladd a b in
                  let lab = b1 Lgamma [ ab ] in
                  let lbeta = lsub s lab in
                  let bm = lsub b (scalar_const b 1.0) in
                  let nx = lneg x in
                  let l1 = b1 Log1p [ nx ] in
                  let t1 = lmul bm l1 in
                  let am = lsub a (scalar_const a 1.0) in
                  let lx = b1 Log [ x ] in
                  let t2 = lmul am lx in
                  let s2 = ladd t1 t2 in
                  let d = lsub s2 lbeta in
                  let px = b1 Exp [ d ] in
                  lmul px g);
            ]
      | _ -> arity ())
  | _ -> None

let jvp_rule_rest prim (primals : value list) (tangents : value list) :
    value * value =
  match prim with
  | Reduce_prod axes -> (
      match (primals, tangents) with
      | [ x ], [ _ ] ->
          let init = scalar_const x 1.0 in
          reduce_jvp
            (fun args ->
              match args with [ a; b ] -> [ lmul a b ] | _ -> arity ())
            [ init ] primals tangents axes
      | _ -> arity ())
  | Reduce { jaxpr; dimensions } -> (
      match (primals, tangents) with
      | [ x; init ], [ t; init_t ] ->
          if not (is_zero init_t) then
            failwith
              "Gradient of general lax.reduce with non-zero tangents for \
               initial values to reduction not implemented";
          reduce_jvp
            (fun args -> Jaxpr.eval_closed_jaxpr jaxpr args)
            [ init ] [ x ] [ t ] dimensions
      | _ -> arity ())
  | Population_count -> failwith "ad: population_count has no jvp rule"
  | Clz -> failwith "ad: clz has no jvp rule"
  | Nextafter -> failwith "ad: nextafter has no jvp rule"
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
  | Scatter_min _ | Scatter_max _ ->
      failwith
        "ad: scatter_min/scatter_max jvp needs the extremal averaging rule (M2 \
         gap)"
  | Split _ | Unstack _ | Optimization_barrier | Sort _ | Top_k _ | Scan _
  | While _ | Custom_linear_solve _ ->
      failwith "ad: multi-output jvp handled by jvp_process_primitive"
  | Tie -> (
      match (primals, tangents) with
      | [ x; y ], [ _; ty ] -> (b1 Tie [ x; y ], ty)
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
  | Cumsum { axis; reverse } -> (
      match (primals, tangents) with
      | [ x ], [ tx ] ->
          ( b1 (Cumsum { axis; reverse }) [ x ],
            b1 (Cumsum { axis; reverse }) [ tx ] )
      | _ -> arity ())
  | Cumprod { axis; reverse } ->
      cumulative_jvp ~axis ~reverse primals tangents (fun a b -> lmul a b)
  | Cummax { axis; reverse } ->
      cumulative_jvp ~axis ~reverse primals tangents (fun a b ->
          b1 Max [ a; b ])
  | Cummin { axis; reverse } ->
      cumulative_jvp ~axis ~reverse primals tangents (fun a b ->
          b1 Min [ a; b ])
  | Cumlogsumexp { axis; reverse } ->
      cumulative_jvp ~axis ~reverse primals tangents (fun a b ->
          Lax_other.logaddexp a b)
  | Igamma_grad_a -> failwith "ad: primitive has no jvp rule (matches jax)"
  | Iota _ | Empty _ | Empty2 _ | Create_token | After_all | Composite _
  | Dce_sink | From_edtype _ | Ragged_dot_general | Rng_bit_generator _
  | Rng_uniform | To_edtype _ | Platform_index _ | Threefry2x32
  | Iota_2x32_shape _ | Random_seed | Random_split _ | Random_fold_in
  | Random_bits _ | Random_wrap | Random_unwrap ->
      failwith "ad: primitive has no jvp rule in M1"
  | Cholesky -> (
      match (primals, tangents) with
      | [ x ], [ t ] -> cholesky_jvp x t
      | _ -> arity ())
  | Triangular_solve o -> (
      match primals with
      | [ a; b ] ->
          let rule_a g_a ans =
            triangular_jvp_a ~left_side:o.left_side ~lower:o.lower
              ~transpose_a:o.transpose_a ~conjugate_a:o.conjugate_a
              ~unit_diagonal:o.unit_diagonal g_a ans a b
          in
          standard_jvp prim primals tangents
            [ Some rule_a; Some (fun g_b _ -> b1 prim [ a; g_b ]) ]
      | _ -> arity ())
  | Tridiagonal_solve _ -> (
      match (primals, tangents) with
      | [ dl; d; du; b ], [ tdl; td; tdu; tb ] ->
          let ans = b1 prim [ dl; d; du; b ] in
          let rhs =
            if is_zero tdl && is_zero td && is_zero tdu then tb
            else
              let idl = instantiate tdl in
              let id = instantiate td in
              let idu = instantiate tdu in
              let mv = tridiagonal_matvec idl id idu ans in
              add_tangents tb (lneg mv)
          in
          (ans, b1 prim [ dl; d; du; instantiate rhs ])
      | _ -> arity ())
  | Householder_product | Lu | Lu_pivots_to_permutation _ | Qr _ | Eig _
  | Eigh _ | Hessenberg | Schur _ | Svd _ | Tridiagonal _ ->
      failwith "ad: this linalg primitive has no forward-mode rule"
  | Geqrf -> failwith "Differentiation rule for 'geqrf' not implemented"
  | Geqp3 -> failwith "Differentiation rule for 'geqp3' not implemented"
  | Ormqr _ -> failwith "Differentiation rule for 'ormqr' not implemented"
  | Collective { kind; _ } -> (
      match (kind, primals, tangents) with
      | ( ( Psum | Pbroadcast _ | Ppermute _ | All_gather _ | All_to_all _
          | Reduce_scatter _ ),
          [ x ],
          [ t ] ) ->
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
  | Xla_call _ -> failwith "ad: xla_call jvp handled by jvp_pairs"
  | Ffi_call _ -> failwith "ad: ffi_call jvp handled by jvp_process_primitive"
  | _ -> failwith "ad: rule handled by jvp_rule_core"

let jvp_rule prim primals tangents =
  match jvp_rule_core prim primals tangents with
  | Some r -> r
  | None -> jvp_rule_rest prim primals tangents

let ffi_not_differentiable (c : ffi_call) =
  invalid_arg
    (Printf.sprintf
       "The FFI call to `%s` cannot be differentiated. You can use \
        `jax.custom_jvp` or `jax.custom_jvp` to add support."
       c.ffi_target)

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

let jvp_instantiated f primals tangents =
  let pos, tos = jvp f primals tangents in
  ( pos,
    List.map
      (fun t ->
        if is_zero t && Dtypes.issubdtype (Core.get_aval t).dtype `Inexact then
          instantiate t
        else t)
      tos )

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

let jvp_jaxpr_cache :
    (int * bool list * bool, closed_jaxpr * bool list) Hashtbl.t =
  Hashtbl.create 64

let jvp_jaxpr (c : closed_jaxpr) primal_avals nzs =
  let key = (c.jid, nzs, Config.x64_enabled ()) in
  match Hashtbl.find_opt jvp_jaxpr_cache key with
  | Some r -> r
  | None ->
      let nz_avals =
        List.concat
          (List.map2 (fun a nz -> if nz then [ a ] else []) primal_avals nzs)
      in
      let n = List.length primal_avals in
      let out_nzs = ref [] in
      let closed =
        Jaxpr.make_jaxpr (primal_avals @ nz_avals) (fun args ->
            let primals, tan_args =
              match Util.split_list args [ n ] with
              | [ a; b ] -> (a, b)
              | _ -> arity ()
            in
            let rest = ref tan_args in
            let tangents =
              List.map2
                (fun p nz ->
                  if nz then
                    begin match !rest with
                    | t :: tl ->
                        rest := tl;
                        t
                    | [] -> arity ()
                    end
                  else p2tz p)
                primals nzs
            in
            let pos, tos =
              jvp (fun a -> Jaxpr.eval_closed_jaxpr c a) primals tangents
            in
            out_nzs := List.map (fun t -> not (is_zero t)) tos;
            pos @ List.filter (fun t -> not (is_zero t)) tos)
      in
      Pretty_printer.register_call_name closed.jid
        (Pretty_printer.call_name c.jid);
      Core.inherit_inline ~from:c.jid closed.jid;
      let r = (closed, !out_nzs) in
      Hashtbl.replace jvp_jaxpr_cache key r;
      r

let xla_call_jvp (c : closed_jaxpr) primals tangents =
  let nzs = List.map (fun t -> not (is_zero t)) tangents in
  let closed, out_nzs = jvp_jaxpr c (List.map Core.get_aval primals) nzs in
  let nz_in = List.filter (fun t -> not (is_zero t)) tangents in
  let outs = Core.bind (Xla_call closed) (primals @ nz_in) in
  let n_out = List.length out_nzs in
  let pos, tos =
    match Util.split_list outs [ n_out ] with
    | [ a; b ] -> (a, b)
    | _ -> arity ()
  in
  let rest = ref tos in
  ( pos,
    List.map2
      (fun po nz ->
        if nz then
          begin match !rest with
          | t :: tl ->
              rest := tl;
              t
          | [] -> arity ()
          end
        else p2tz po)
      pos out_nzs )

let rec jvp_pairs prim primals tangents =
  match prim with
  | Xla_call c -> xla_call_jvp c primals tangents
  | Ffi_call c ->
      if List.for_all is_concrete_zero tangents then
        let pos = Core.bind prim primals in
        (pos, List.map p2tz pos)
      else ffi_not_differentiable c
  | Split _ | Unstack _ | Optimization_barrier ->
      let pos = Core.bind prim primals in
      let ts = List.map instantiate tangents in
      let tos = Core.bind prim ts in
      (pos, tos)
  | Sort { dimension; is_stable; num_keys } ->
      sort_jvp ~dimension ~is_stable ~num_keys primals tangents
  | Top_k { k; axis } -> (
      match (primals, tangents) with
      | [ x ], [ t ] -> top_k_jvp ~k ~axis x t
      | _ -> arity ())
  | Eig { compute_left; compute_right; enable_eigvec_derivs } -> (
      match (primals, tangents) with
      | [ a ], [ t ] ->
          eig_jvp ~compute_left ~compute_right ~enable_eigvec_derivs a t
      | _ -> arity ())
  | Eigh { lower } -> (
      match (primals, tangents) with
      | [ a ], [ t ] -> eigh_jvp ~lower a t
      | _ -> arity ())
  | Qr full_matrices -> (
      match (primals, tangents) with
      | [ a ], [ t ] -> qr_jvp ~full_matrices a t
      | _ -> arity ())
  | Qr_pivoted full_matrices -> (
      match (primals, tangents) with
      | [ a ], [ t ] -> qr_pivoted_jvp ~full_matrices a t
      | _ -> arity ())
  | Lu -> (
      match (primals, tangents) with
      | [ a ], [ t ] -> lu_jvp a t
      | _ -> arity ())
  | Svd { full_matrices; compute_uv; _ } -> (
      match (primals, tangents) with
      | [ a ], [ t ] -> svd_jvp ~full_matrices ~compute_uv a t
      | _ -> arity ())
  | Scan ({ num_consts; num_carry; jaxpr; _ } as s) when num_consts > 0 ->
      let lifted =
        Scan
          {
            s with
            num_consts = 0;
            num_carry = num_consts + num_carry;
            jaxpr = Core.scan_lift_consts ~num_consts jaxpr;
          }
      in
      let drop = List.filteri (fun i _ -> i >= num_consts) in
      let p, t = jvp_pairs lifted primals tangents in
      (drop p, drop t)
  | Scan { length; reverse; num_carry; jaxpr; unroll; _ } ->
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
          (Scan
             {
               length;
               reverse;
               num_consts = 0;
               num_carry = 2 * nc;
               jaxpr = new_body;
               unroll;
             })
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
      (primal_outs, tangent_outs)
  | While { cond_nconsts; body_nconsts; cond; body }
    when cond_nconsts + body_nconsts > 0 ->
      let n = cond_nconsts + body_nconsts in
      let cond, body =
        Jaxpr.lift_while ~cond_nconsts ~body_nconsts cond body
          (List.map Core.get_aval primals)
      in
      let drop = List.filteri (fun i _ -> i >= n) in
      let p, t =
        jvp_pairs
          (While { cond_nconsts = 0; body_nconsts = 0; cond; body })
          primals tangents
      in
      (drop p, drop t)
  | While { cond; body; _ } ->
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
          (While
             {
               cond_nconsts = 0;
               body_nconsts = 0;
               cond = new_cond;
               body = new_body;
             })
          (primals @ tangents)
      in
      let pcarry, tcarry = split2 out nc in
      (pcarry, restore_symbolic_zeros carry_zero tcarry)
  | Cond { branches; platforms } -> (
      match (primals, tangents) with
      | pred :: prim_ops, _ :: tan_ops ->
          let op_avals = List.map Core.get_aval prim_ops in
          let primal_outs = Core.bind (Cond { branches; platforms }) primals in
          let op_zero = List.map Ad_util.is_symbolic_zero tan_ops in
          let tan_branches =
            List.map
              (fun b -> cond_tangent_branch jvp b op_avals op_zero)
              branches
          in
          let tangent_outs =
            Core.bind
              (Cond { branches = tan_branches; platforms })
              ((pred :: prim_ops) @ tan_ops)
          in
          (primal_outs, tangent_outs)
      | _ -> arity ())
  | Custom_linear_solve { num_consts; _ } ->
      let x = Core.bind prim primals in
      let keep i = i < num_consts in
      let x_dot =
        Core.bind prim
          (List.filteri (fun i _ -> keep i) primals
          @ List.filteri
              (fun i _ -> not (keep i))
              (List.map instantiate tangents))
      in
      (x, x_dot)
  | Custom_jvp_call id -> (
      match Core.custom_rule id with
      | Core.Jvp_rule { jvp; _ } ->
          jvp (List.map Core.full_lower primals) tangents
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
            Core.bind
              (Custom_lin { rule = id; n_res = List.length res; out_avals })
              (res @ List.map instantiate tangents)
          in
          (po, to_)
      | Core.Callback_rule { result_avals = []; _ } ->
          ignore (Core.bind prim primals);
          ([], [])
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
          let n = List.length res in
          jvp
            (fun args ->
              match Util.split_list args [ n ] with
              | [ r; l ] -> forward r l
              | _ -> arity ())
            (res @ linear) (res_t @ linear_t)
      | _ ->
          let po = Core.bind prim (res @ linear) in
          let to_ = Core.bind prim (res @ List.map instantiate linear_t) in
          (po, to_))
  | _ ->
      let po, to_ = jvp_rule prim primals tangents in
      ([ po ], [ to_ ])

let wrap_jvp trace po to_ =
  if is_zero to_ then po else Tracer (new_jvp_tracer trace po to_)

let tape_tracer (b : jaxpr_builder) trace aval =
  let t = { id = Core.fresh_id (); trace; aval; payload = Jaxpr () } in
  let v = { vid = Core.fresh_id (); vaval = aval } in
  b.jb_tracer_to_var <- (t.id, v) :: b.jb_tracer_to_var;
  t

let tape_trace (b : jaxpr_builder) =
  { level = max_int; kind = KJaxpr; global_data = GBuilder b }

let tape_var (b : jaxpr_builder) (t : tracer) =
  List.assoc t.id b.jb_tracer_to_var

let same_const a b =
  match (a, b) with Tracer x, Tracer y -> x.id = y.id | _ -> a == b

let tape_const (b : jaxpr_builder) v =
  match v with
  | Concrete nd when Array.length (Ndarray.shape nd) = 0 -> A_lit nd
  | _ -> (
      match List.find_opt (fun (_, c) -> same_const c v) b.jb_constvals with
      | Some (var, _) -> A_var var
      | None ->
          let var = { vid = Core.fresh_id (); vaval = Core.get_aval v } in
          b.jb_constvals <- (var, v) :: b.jb_constvals;
          A_var var)

let rec tape_eval_atoms (b : jaxpr_builder) (jx : jaxpr) args =
  let env : (int, atom) Hashtbl.t = Hashtbl.create 16 in
  List.iter2 (fun (v : var) a -> Hashtbl.replace env v.vid a) jx.in_binders args;
  let read = function A_var v -> Hashtbl.find env v.vid | a -> a in
  List.iter
    (fun (e : eqn) ->
      let inputs = List.map read e.inputs in
      match e.prim with
      | Xla_call c when Core.is_inline_call c.jid ->
          let const_atoms =
            List.map (fun nd -> tape_const b (Concrete nd)) c.consts
          in
          let outs = tape_eval_atoms b c.jaxpr (const_atoms @ inputs) in
          List.iter2
            (fun (v : var) a -> Hashtbl.replace env v.vid a)
            e.outs outs
      | _ ->
          let outs =
            List.map
              (fun (v : var) ->
                let fresh = { vid = Core.fresh_id (); vaval = v.vaval } in
                Hashtbl.replace env v.vid (A_var fresh);
                fresh)
              e.outs
          in
          b.jb_eqns <- { e with inputs; outs } :: b.jb_eqns)
    jx.eqns;
  List.map read jx.outs

let tape_eval (b : jaxpr_builder) (jx : jaxpr) consts tangents =
  let args =
    List.map (tape_const b) consts
    @ List.map
        (function
          | Tracer t -> A_var (tape_var b t)
          | _ -> failwith "ad: expected a tangent on the tape")
        tangents
  in
  tape_eval_atoms b jx args

let rec tape_value (b : jaxpr_builder) atom =
  match atom with
  | A_var v ->
      let t =
        {
          id = Core.fresh_id ();
          trace = tape_trace b;
          aval = v.vaval;
          payload = Jaxpr ();
        }
      in
      b.jb_tracer_to_var <- (t.id, v) :: b.jb_tracer_to_var;
      Tracer t
  | A_lit nd ->
      let aval =
        {
          shape = Ndarray.shape nd;
          dtype = Ndarray.dtype nd;
          weak_type = false;
        }
      in
      let var = { vid = Core.fresh_id (); vaval = aval } in
      b.jb_constvals <- (var, Concrete nd) :: b.jb_constvals;
      tape_value_of_var b var
  | DropVar _ -> failwith "ad: dropped tangent output"

and tape_value_of_var (b : jaxpr_builder) (v : var) =
  let t =
    {
      id = Core.fresh_id ();
      trace = tape_trace b;
      aval = v.vaval;
      payload = Jaxpr ();
    }
  in
  b.jb_tracer_to_var <- (t.id, v) :: b.jb_tracer_to_var;
  Tracer t

let known_of_pe main v =
  match Core.full_raise main v with
  | Tracer ({ payload = PE { pval; _ }; _ } as t) -> (
      match pval.pv_const with Some c -> `Known c | None -> `Unknown t)
  | _ -> failwith "ad: expected a partial eval tracer"

let lin_process (trace : trace) (b : jaxpr_builder) prim primals tangents =
  let pos, nzs, consts, jx =
    Core.with_new_main KPE GNone (fun main ->
        let args =
          List.map2
            (fun p t ->
              if is_zero t then (t, None)
              else
                let a = Partial_eval.new_unknown_arg main (Core.get_aval p) in
                (Tracer a, Some a))
            primals tangents
        in
        let tangent_args = List.map fst args in
        let in_tracers = List.filter_map snd args in
        let pos, tos = jvp_pairs prim primals tangent_args in
        let pos =
          List.map
            (fun p ->
              match p with
              | Tracer { trace = t; _ } when t == main -> (
                  match known_of_pe main p with
                  | `Known c -> c
                  | `Unknown _ ->
                      failwith
                        "Linearization failed to produce known values for all \
                         output primals. This is typically caused by \
                         attempting to differentiate a function uses an \
                         operation that does not support reverse-mode \
                         autodiff.")
              | _ -> p)
            pos
        in
        let classified =
          List.map
            (fun t ->
              if is_zero t then None
              else
                match known_of_pe main t with
                | `Known _ -> None
                | `Unknown tr -> Some tr)
            tos
        in
        let out_tracers = List.filter_map Fun.id classified in
        let jx, consts = Partial_eval.tracers_to_jaxpr in_tracers out_tracers in
        (pos, List.map Option.is_some classified, consts, jx))
  in
  let nz_in = List.filter (fun t -> not (is_zero t)) tangents in
  let atoms = tape_eval b jx consts nz_in in
  let rest = ref atoms in
  List.map2
    (fun po nz ->
      if nz then
        begin match !rest with
        | a :: tl ->
            rest := tl;
            Tracer (new_jvp_tracer trace po (tape_value b a))
        | [] -> failwith "ad: tangent output arity"
        end
      else po)
    pos nzs

type lin_call = {
  lc_primal : closed_jaxpr;
  lc_tangent : closed_jaxpr;
  lc_nzs_out : bool list;
  lc_in_fwd : int option list;
  lc_out_fwd : int option list;
  lc_res_in : [ `Input of int | `Output | `Same of int | `Value of value ] list;
}

let trace_linearized_ref :
    ((value list -> value list) ->
    value list ->
    bool list ->
    value list * bool list * jaxpr * value list)
    ref =
  ref (fun _ _ _ -> failwith "ad: linearize not installed")

let lin_call_cache : (int * bool list * bool, lin_call) Hashtbl.t =
  Hashtbl.create 64

let linearize_call (c : closed_jaxpr) primal_avals nzs =
  let key = (c.jid, nzs, Config.x64_enabled ()) in
  match Hashtbl.find_opt lin_call_cache key with
  | Some r -> r
  | None ->
      let holder = ref None in
      let primal =
        Jaxpr.make_jaxpr primal_avals (fun inputs ->
            let pos, nzs_out, jx, consts =
              !trace_linearized_ref
                (fun a -> Jaxpr.eval_closed_jaxpr c a)
                inputs nzs
            in
            let index_in l v =
              let rec go i = function
                | [] -> None
                | a :: tl -> if same_const a v then Some i else go (i + 1) tl
              in
              go 0 l
            in
            let in_fwd = List.map (index_in inputs) pos in
            let kept_pos =
              List.concat
                (List.map2
                   (fun p f -> if f = None then [ p ] else [])
                   pos in_fwd)
            in
            let res_in =
              List.map
                (fun v ->
                  match v with
                  | Concrete _ | Device _ -> `Value v
                  | Tracer _ -> (
                      match index_in inputs v with
                      | Some i -> `Input i
                      | None -> (
                          match index_in kept_pos v with
                          | Some i -> `Same i
                          | None -> `Output)))
                consts
            in
            let residual_outs =
              List.concat
                (List.map2
                   (fun v r -> if r = `Output then [ v ] else [])
                   consts res_in)
            in
            holder := Some (nzs_out, jx, in_fwd, res_in);
            kept_pos @ residual_outs)
      in
      let nzs_out, jx, in_fwd, res_in =
        match !holder with
        | Some h -> h
        | None -> failwith "ad: linearized call was not traced"
      in
      let name = Pretty_printer.call_name c.jid in
      let primal = Jaxpr.dce_keep_args primal in
      Pretty_printer.register_call_name primal.jid name;
      let tangent = { jid = Core.fresh_id (); jaxpr = jx; consts = [] } in
      Pretty_printer.register_call_name tangent.jid name;
      Core.inherit_inline ~from:c.jid primal.jid;
      Core.inherit_inline ~from:c.jid tangent.jid;
      let r =
        {
          lc_primal = primal;
          lc_tangent = tangent;
          lc_nzs_out = nzs_out;
          lc_in_fwd = in_fwd;
          lc_out_fwd = [];
          lc_res_in = res_in;
        }
      in
      Hashtbl.replace lin_call_cache key r;
      r

let lin_process_call (trace : trace) (b : jaxpr_builder) (c : closed_jaxpr)
    primals tangents =
  let nzs = List.map (fun t -> not (is_zero t)) tangents in
  let lc = linearize_call c (List.map Core.get_aval primals) nzs in
  let outs = Core.bind (Xla_call lc.lc_primal) primals in
  let n_kept = List.length (List.filter (fun f -> f = None) lc.lc_in_fwd) in
  let kept, res_outs =
    match Util.split_list outs [ n_kept ] with
    | [ a; b ] -> (a, b)
    | _ -> arity ()
  in
  let rest = ref kept in
  let pos =
    List.map
      (function
        | Some i -> List.nth primals i
        | None -> (
            match !rest with
            | v :: tl ->
                rest := tl;
                v
            | [] -> arity ()))
      lc.lc_in_fwd
  in
  let res_rest = ref res_outs in
  let residuals =
    List.map
      (function
        | `Input i -> List.nth primals i
        | `Same i -> List.nth kept i
        | `Value v -> v
        | `Output -> (
            match !res_rest with
            | v :: tl ->
                res_rest := tl;
                v
            | [] -> arity ()))
      lc.lc_res_in
  in
  let nz_in = List.filter (fun t -> not (is_zero t)) tangents in
  let res_atoms = List.map (tape_const b) residuals in
  let tan_atoms =
    List.map
      (function
        | Tracer t -> A_var (tape_var b t)
        | _ -> failwith "ad: expected a tangent on the tape")
      nz_in
  in
  let out_atoms =
    if Core.is_inline_call lc.lc_tangent.jid then
      tape_eval_atoms b lc.lc_tangent.jaxpr (res_atoms @ tan_atoms)
    else begin
      let out_binders =
        List.map
          (fun a ->
            { vid = Core.fresh_id (); vaval = Pretty_printer.atom_type a })
          lc.lc_tangent.jaxpr.outs
      in
      b.jb_eqns <-
        {
          prim = Xla_call lc.lc_tangent;
          inputs = res_atoms @ tan_atoms;
          outs = out_binders;
          multiple_results = List.length out_binders > 1;
        }
        :: b.jb_eqns;
      List.map (fun v -> A_var v) out_binders
    end
  in
  let tan_rest = ref out_atoms in
  List.map2
    (fun po nz ->
      if nz then
        begin match !tan_rest with
        | a :: tl ->
            tan_rest := tl;
            Tracer (new_jvp_tracer trace po (tape_value b a))
        | [] -> arity ()
        end
      else po)
    pos lc.lc_nzs_out

let joint_branch_jvp rebuild branches primals tangents =
  match (primals, tangents) with
  | index :: prim_ops, _ :: tan_ops ->
      let nzs = List.map (fun t -> not (is_zero t)) tan_ops in
      let keep flags vs =
        List.filter_map
          (fun (nz, v) -> if nz then Some v else None)
          (List.combine flags vs)
      in
      let op_avals = List.map Core.get_aval prim_ops in
      let n = List.length op_avals in
      let in_avals = op_avals @ keep nzs op_avals in
      let run (branch : closed_jaxpr) args =
        let ops, tans =
          match Util.split_list args [ n ] with
          | [ a; b ] -> (a, b)
          | _ -> arity ()
        in
        let rest = ref tans in
        let tangent_ops =
          List.map2
            (fun p nz ->
              if nz then
                begin match !rest with
                | t :: tl ->
                    rest := tl;
                    t
                | [] -> arity ()
                end
              else p2tz p)
            ops nzs
        in
        jvp (fun a -> Jaxpr.eval_closed_jaxpr branch a) ops tangent_ops
      in
      let flags_of branch =
        let flags = ref [] in
        ignore
          (Jaxpr.make_jaxpr in_avals (fun args ->
               let pos, tos = run branch args in
               flags := List.map (fun t -> not (is_zero t)) tos;
               pos));
        !flags
      in
      let out_nzs =
        match List.map flags_of branches with
        | [] -> []
        | first :: others -> List.fold_left (List.map2 ( || )) first others
      in
      let joint =
        List.map
          (fun branch ->
            Jaxpr.make_jaxpr in_avals (fun args ->
                let pos, tos = run branch args in
                pos @ keep out_nzs (List.map instantiate tos)))
          branches
      in
      let outs =
        Core.bind (rebuild joint) ((index :: prim_ops) @ keep nzs tan_ops)
      in
      let pos, tos =
        match Util.split_list outs [ List.length out_nzs ] with
        | [ a; b ] -> (a, b)
        | _ -> arity ()
      in
      let rest = ref tos in
      ( pos,
        List.map2
          (fun po nz ->
            if nz then
              begin match !rest with
              | t :: tl ->
                  rest := tl;
                  t
              | [] -> arity ()
              end
            else p2tz po)
          pos out_nzs )
  | _ -> arity ()

let jvp_process_primitive trace prim args =
  let pairs = List.map (as_jvp trace) args in
  let primals = List.map fst pairs and tangents = List.map snd pairs in
  if List.for_all is_zero tangents then Core.bind prim primals
  else
    match (trace.global_data, prim) with
    | GBuilder b, Xla_call c -> lin_process_call trace b c primals tangents
    | GBuilder b, _ -> lin_process trace b prim primals tangents
    | _ ->
        let pos, tos =
          match prim with
          | Cond { branches; platforms } ->
              joint_branch_jvp
                (fun branches -> Cond { branches; platforms })
                branches primals tangents
          | _ -> jvp_pairs prim primals tangents
        in
        List.map2 (wrap_jvp trace) pos tos

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
  reduce_jvp_ref :=
    fun reducer init_values primals tangents axes ->
      let first = List.hd primals in
      let shape = (Core.get_aval first).shape in
      let ndim = Array.length shape in
      let n = Array.fold_left (fun acc a -> acc * shape.(a)) 1 axes in
      let non_axes =
        List.filter (fun d -> not (Array.mem d axes)) (List.init ndim Fun.id)
      in
      let permutation = Array.append axes (Array.of_list non_axes) in
      let new_shape =
        Array.of_list (n :: List.map (fun d -> shape.(d)) non_axes)
      in
      let flat x =
        let moved = b1 (Transpose permutation) [ x ] in
        lreshape new_shape moved
      in
      let primals = List.map flat primals in
      let tangents = List.map flat (List.map instantiate tangents) in
      let rec vmapped d f =
        if d = 0 then f
        else
          vmapped (d - 1) (fun args ->
              Batching.vmap f (List.map (fun _ -> Some 0) args) args)
      in
      let reducer = vmapped (List.length non_axes + 1) reducer in
      let rec reduce_tree (xs : value list) =
        let len = (Core.get_aval (List.hd xs)).shape.(0) in
        if len > 1 then begin
          let n1 = (len + 1) / 2 in
          let n2 = len - n1 in
          let slice_dim lo hi x =
            let sh = (Core.get_aval x).shape in
            let start = Array.make (Array.length sh) 0 in
            let limit = Array.copy sh in
            start.(0) <- lo;
            limit.(0) <- hi;
            b1
              (Slice
                 {
                   start_indices = start;
                   limit_indices = limit;
                   strides = None;
                 })
              [ x ]
          in
          let xs1 = List.map (slice_dim 0 n1) xs in
          let xs2 = List.map (slice_dim n1 len) xs in
          let xs2 =
            if n2 <> n1 then
              List.map2
                (fun x2 i ->
                  let cfg =
                    Array.make (Array.length (Core.get_aval x2).shape) (0, 0, 0)
                  in
                  cfg.(0) <- (0, 1, 0);
                  b1 (Pad cfg) [ x2; i ])
                xs2 init_values
            else xs2
          in
          reduce_tree (reducer (xs1 @ xs2))
        end
        else if len = 0 then
          List.map
            (fun i ->
              let kept =
                Array.of_list (List.map (fun d -> shape.(d)) non_axes)
              in
              b1 (Broadcast_in_dim { shape = kept; dims = [||] }) [ i ])
            init_values
        else List.map (fun x -> b1 (Squeeze [| 0 |]) [ x ]) xs
      in
      match jvp reduce_tree primals tangents with
      | [ p ], [ t ] -> (p, t)
      | _ -> failwith "ad: reduce must return one value"

let () =
  cumulative_jvp_ref :=
    fun combine ~axis ~reverse primals tangents ->
      let body args =
        associative_scan_flat
          (fun a b -> List.map2 combine a b)
          ~axis ~reverse args
      in
      match jvp_instantiated body primals tangents with
      | [ p ], [ t ] -> (p, t)
      | _ -> failwith "ad: cumulative arity"

let is_differentiable_dtype dt = Dtypes.issubdtype dt `Inexact

let dce_tape (eqns : eqn list) (outs : atom list) =
  let live : (int, unit) Hashtbl.t = Hashtbl.create 64 in
  let mark = function
    | A_var (v : var) -> Hashtbl.replace live v.vid ()
    | _ -> ()
  in
  let is_live (v : var) = Hashtbl.mem live v.vid in
  List.iter mark outs;
  List.fold_left
    (fun kept (e : eqn) ->
      match e.prim with
      | Xla_call c when not (Jaxpr.eqn_has_effect e) ->
          let used = List.map is_live e.outs in
          if not (List.exists Fun.id used) then kept
          else begin
            let pruned, ins = Jaxpr.dce_closed c used in
            let inputs =
              List.concat
                (List.map2
                   (fun a keep -> if keep then [ a ] else [])
                   e.inputs ins)
            in
            List.iter mark inputs;
            let outs = List.filter is_live e.outs in
            {
              prim = Xla_call pruned;
              inputs;
              outs;
              multiple_results = List.length outs > 1;
            }
            :: kept
          end
      | _ ->
          if List.exists is_live e.outs || e.outs = [] then begin
            List.iter mark e.inputs;
            e :: kept
          end
          else kept)
    [] eqns

let trace_linearized_nz (f : value list -> value list) (primals : value list)
    nzs =
  let b =
    {
      jb_eqns = [];
      jb_tracer_to_var = [];
      jb_const_tracers = [];
      jb_constvals = [];
      jb_tracers = [];
      jb_invars = [];
      jb_debug =
        {
          func_src_info = "<unnamed function>";
          traced_for = "jit";
          arg_names = None;
        };
    }
  in
  let tangent_trace = tape_trace b in
  let in_vars = ref [] in
  let pos, tos =
    Core.with_new_main KJVP (GBuilder b) (fun main ->
        let ins =
          List.map2
            (fun p nz ->
              let a = Core.get_aval p in
              if nz then begin
                let t = tape_tracer b tangent_trace a in
                in_vars := tape_var b t :: !in_vars;
                Tracer (new_jvp_tracer main p (Tracer t))
              end
              else p)
            primals nzs
        in
        let outs = f ins in
        let pairs =
          List.map
            (fun v ->
              match Core.full_raise main v with
              | Tracer { payload = JVP { primal; tangent }; _ } ->
                  (Core.full_lower primal, tangent)
              | _ -> failwith "ad.linearize: expected a tangent tracer")
            outs
        in
        List.split pairs)
  in
  let nzs = List.map (fun t -> not (is_zero t)) tos in
  let out_atoms =
    List.filter_map
      (function
        | Tracer t when not (is_zero (Tracer t)) -> Some (A_var (tape_var b t))
        | _ -> None)
      tos
  in
  let eqns = dce_tape b.jb_eqns out_atoms in
  let live : (int, unit) Hashtbl.t = Hashtbl.create 16 in
  let mark = function
    | A_var (v : var) -> Hashtbl.replace live v.vid ()
    | _ -> ()
  in
  List.iter mark out_atoms;
  List.iter (fun (e : eqn) -> List.iter mark e.inputs) eqns;
  let constvals =
    List.filter
      (fun ((v : var), _) -> Hashtbl.mem live v.vid)
      (List.rev b.jb_constvals)
  in
  let in_binders = List.rev !in_vars in
  let jx =
    { in_binders = List.map fst constvals @ in_binders; eqns; outs = out_atoms }
  in
  (pos, nzs, jx, List.map snd constvals)

let trace_linearized f primals =
  trace_linearized_nz f primals
    (List.map
       (fun p -> is_differentiable_dtype (Core.get_aval p).dtype)
       primals)

let () = trace_linearized_ref := trace_linearized_nz

let linearize (f : value list -> value list) (primals : value list) :
    value list * (value list -> value list) =
  let pos, nzs, jx, consts = trace_linearized f primals in
  let f_lin tangents =
    let nz_in =
      List.filter_map
        (fun (p, t) ->
          if is_differentiable_dtype (Core.get_aval p).dtype then Some t
          else None)
        (List.combine primals tangents)
    in
    let computed = ref (Jaxpr.eval_jaxpr jx (consts @ nz_in)) in
    List.map2
      (fun po nz ->
        if nz then
          begin match !computed with
          | v :: rest ->
              computed := rest;
              v
          | [] -> failwith "ad.linearize: tangent output arity"
          end
        else instantiate (p2tz po))
      pos nzs
  in
  (pos, f_lin)

type tval = Prim of value | Undef of aval

let is_undef = function Undef _ -> true | Prim _ -> false

let prim_val = function
  | Prim v -> v
  | Undef _ -> failwith "ad: undefined primal"

let in_aval = function Undef a -> a | Prim v -> Core.get_aval v

let broadcast_transpose (shape : int array) (dims : int array)
    (in_shape : int array) ct =
  let is_unit i = in_shape.(i) = 1 in
  let unit_dims =
    List.filter is_unit (List.init (Array.length in_shape) Fun.id)
  in
  let bdims =
    List.filter
      (fun i -> not (is_unit i))
      (List.init (Array.length in_shape) Fun.id)
    |> List.map (fun i -> dims.(i))
  in
  let axes =
    List.filter
      (fun d -> not (List.mem d bdims))
      (List.init (Array.length shape) Fun.id)
  in
  let reduced = b1 (Reduce_sum (Array.of_list axes)) [ ct ] in
  if unit_dims = [] then reduced
  else expand_dims reduced (Array.of_list unit_dims)

let reduce_sum_transpose (axes : int array) (in_shape : int array) ct =
  let in_ndim = Array.length in_shape in
  let is_red = Array.make in_ndim false in
  Array.iter (fun a -> is_red.(a) <- true) axes;
  let kept = ref [] in
  for d = in_ndim - 1 downto 0 do
    if not is_red.(d) then kept := d :: !kept
  done;
  if Array.length in_shape = 0 && Array.length (Core.get_aval ct).shape = 0 then
    ct
  else
    b1
      (Broadcast_in_dim
         { shape = in_shape; dims = Np_ints.mark (Array.of_list !kept) })
      [ ct ]

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
      dot_precision = dims.dot_precision;
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

let transpose_cache :
    ( int * bool list * aval list * bool list * bool,
      closed_jaxpr * bool list )
    Hashtbl.t =
  Hashtbl.create 64

let rec transpose_rule prim (cts : value list) (primals : tval list) :
    value option list =
  let ct1 () =
    match cts with
    | [ c ] -> c
    | _ -> failwith "ad: transpose expects a single cotangent"
  in
  match prim with
  | Add | Add_any -> (
      let ct = ct1 () in
      match primals with
      | [ x; y ] ->
          let cx = unbroadcast (in_aval x).shape ct in
          let cy = unbroadcast (in_aval y).shape ct in
          [ Some cx; Some cy ]
      | _ -> arity ())
  | Sub -> (
      let ct = ct1 () in
      match primals with
      | [ x; y ] ->
          let cx = unbroadcast (in_aval x).shape ct in
          let n = lneg ct in
          let cy = unbroadcast (in_aval y).shape n in
          [ Some cx; Some cy ]
      | _ -> arity ())
  | Neg -> [ Some (lneg (ct1 ())) ]
  | Mul -> (
      let ct = ct1 () in
      match primals with
      | [ x; y ] ->
          if is_undef x then
            [
              Some (unbroadcast (in_aval x).shape (lmul ct (prim_val y))); None;
            ]
          else
            [
              None; Some (unbroadcast (in_aval y).shape (lmul (prim_val x) ct));
            ]
      | _ -> arity ())
  | Div -> (
      let ct = ct1 () in
      match primals with
      | [ x; y ] ->
          [ Some (unbroadcast (in_aval x).shape (ldiv ct (prim_val y))); None ]
      | _ -> arity ())
  | Select_n -> (
      let ct = ct1 () in
      match primals with
      | which :: cases ->
          let wv = prim_val which in
          let zeros = full_like ct 0.0 in
          if (Core.get_aval wv).dtype = Dtype.Bool then
            let ct_at i p =
              if not (is_undef p) then None
              else if i = 0 then Some (select wv zeros ct)
              else Some (select wv ct zeros)
            in
            let first = ct_at 0 (List.nth cases 0) in
            let second = ct_at 1 (List.nth cases 1) in
            [ None; first; second ]
          else
            None
            :: List.mapi
                 (fun i case ->
                   if is_undef case then
                     let p = b1 Eq [ wv; scalar_const wv (float_of_int i) ] in
                     Some (select p ct zeros)
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
      | [ x ] -> [ Some (lreshape (in_aval x).shape ct) ]
      | _ -> arity ())
  | Reshape_perm { dimensions; _ } -> (
      let ct = ct1 () in
      match primals with
      | [ x ] ->
          let shape = (in_aval x).shape in
          let n = Array.length dimensions in
          let inverse = Array.make n 0 in
          Array.iteri (fun i d -> inverse.(d) <- i) dimensions;
          [
            Some
              (b1 (Transpose inverse)
                 [ lreshape (Array.map (fun d -> shape.(d)) dimensions) ct ]);
          ]
      | _ -> arity ())
  | Reduce_sum axes -> (
      let ct = ct1 () in
      match primals with
      | [ x ] -> [ Some (reduce_sum_transpose axes (in_aval x).shape ct) ]
      | _ -> arity ())
  | Cumsum { axis; reverse } ->
      [ Some (b1 (Cumsum { axis; reverse = not reverse }) [ ct1 () ]) ]
  | Copy -> [ Some (b1 Copy [ ct1 () ]) ]
  | Conj -> (
      let t = ct1 () in
      match primals with
      | [ x ] ->
          if Dtypes.issubdtype (in_aval x).dtype `Complex then
            [ Some (b1 Conj [ t ]) ]
          else [ Some (b1 Real [ t ]) ]
      | _ -> arity ())
  | Rev dims -> [ Some (b1 (Rev dims) [ ct1 () ]) ]
  | Transpose perm -> [ Some (b1 (Transpose (argsort perm)) [ ct1 () ]) ]
  | Squeeze dims -> [ Some (expand_dims (ct1 ()) dims) ]
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
          let unit_strides = Array.for_all (fun s -> s = 1) strides in
          let cfg =
            Np_ints.mark_triples
              (fun _ j -> j = 1 || (j = 2 && not unit_strides))
              (Array.init n (fun i ->
                   let s = strides.(i) in
                   let out_d = ct_shape.(i) in
                   let real_limit =
                     start_indices.(i)
                     + if out_d = 0 then 0 else 1 + ((out_d - 1) * s)
                   in
                   (start_indices.(i), os.(i) - real_limit, s - 1)))
          in
          [ Some (b1 (Pad cfg) [ ct; scalar_const ct 0.0 ]) ]
      | _ -> arity ())
  | Dynamic_slice _ -> (
      match primals with
      | operand :: idx ->
          let ct = ct1 () in
          let idx_vals = List.map prim_val idx in
          let op_t =
            if is_undef operand then
              let a = in_aval operand in
              let zeros = Core.splat a.dtype a.shape 0.0 in
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
          let zeros_u = Core.splat (Core.get_aval ct).dtype ua.shape 0.0 in
          let op_t =
            if is_undef operand then
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
              let zeros =
                Core.splat (Core.get_aval ct).dtype (in_aval operand).shape 0.0
              in
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
                (lneg
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
                   [ lmul ct (prim_val operand); prim_val indices ])
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
                let zeros =
                  Core.splat (Core.get_aval ct).dtype (in_aval updates).shape
                    0.0
                in
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
  | Cond { branches; platforms } -> (
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
              (Cond { branches = List.map transposed branches; platforms })
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
      | All_to_all { split_axis; concat_axis; size } ->
          [
            Some
              (b1
                 (Collective
                    {
                      kind =
                        All_to_all
                          {
                            split_axis = concat_axis;
                            concat_axis = split_axis;
                            size;
                          };
                      axis_name;
                    })
                 [ ct1 () ]);
          ]
      | Pmax | Pmin | Pbroadcast _ | All_gather _ | Axis_index
      | Reduce_scatter _ ->
          failwith "ad: this collective has no transpose rule")
  | Real ->
      let t = ct1 () in
      [ Some (b1 Complex [ t; scalar_const t 0.0 ]) ]
  | Imag ->
      let t = ct1 () in
      let n = lneg t in
      [ Some (b1 Complex [ scalar_const t 0.0; n ]) ]
  | Complex -> (
      let t = ct1 () in
      match primals with
      | [ x; y ] ->
          let x_ct =
            if is_undef x then
              Some (unbroadcast (in_aval x).shape (b1 Real [ t ]))
            else None
          in
          let y_ct =
            if is_undef y then
              let n = lneg t in
              let im = b1 Imag [ n ] in
              Some (unbroadcast (in_aval y).shape im)
            else None
          in
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
              dot_precision = dims.dot_precision;
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
  | Custom_jvp_call id -> (
      let avals =
        List.map (function Prim v -> Core.get_aval v | Undef a -> a) primals
      in
      match Jaxpr.custom_call_jaxpr id avals with
      | Some c -> xla_call_transpose c cts primals
      | None -> failwith "ad: custom call cannot appear in a linear program")
  | Custom_vjp_call _ ->
      failwith "ad: custom call cannot appear in a linear program"
  | Ffi_call c -> ffi_not_differentiable c
  | Custom_linear_solve { solve; transpose_solve; num_consts } -> (
      match transpose_solve with
      | None ->
          failwith
            "ad: transpose_solve required for backwards mode automatic \
             differentiation of custom_linear_solve"
      | Some ts ->
          let consts =
            List.filteri (fun i _ -> i < num_consts) primals
            |> List.map prim_val
          in
          let ct_b =
            Core.bind
              (Custom_linear_solve
                 { solve = ts; transpose_solve = Some solve; num_consts })
              (consts @ cts)
          in
          List.map (fun _ -> None) consts @ List.map (fun c -> Some c) ct_b)
  | Sin | Cos | Exp | Log | Tanh | Max | Min | Pow | Abs | Sign | Eq | Lt | Gt
  | Acos | Acosh | Asin | Asinh | Atan | Atanh | Cbrt | Ceil | Clz | Cosh | Exp2
  | Expm1 | Floor | Integer_pow _ | Is_finite | Log1p | Logistic | Not
  | Population_count | Round | Round_nearest_even | Rsqrt | Sinh | Sqrt | Square
  | Tan | And | Atan2 | Eq_to | Ge | Le | Le_to | Lt_to | Mulhi | Ne | Nextafter
  | Or | Rem | Shift_left | Shift_right_arithmetic | Shift_right_logical | Xor
  | Argmax _ | Argmin _ | Reduce _ | Reduce_and _ | Reduce_max _ | Reduce_min _
  | Reduce_or _ | Reduce_prod _ | Reduce_xor _ | After_all
  | Bitcast_convert_type _ | Clamp | Composite _ | Create_token | Dce_sink
  | Empty _ | Empty2 _ | From_edtype _ | Iota _ | Ragged_dot_general
  | Rng_bit_generator _ | Rng_uniform | Sort _ | To_edtype _ | Top_k _
  | Scatter_min _ | Scatter_max _ | Reduce_window _ | Reduce_window_max _
  | Reduce_window_min _ | Select_and_scatter _ | Bessel_i0e | Bessel_i1e
  | Digamma | Erf | Erf_inv | Erfc | Igamma | Igamma_grad_a | Igammac | Lgamma
  | Polygamma | Regularized_incomplete_beta | Zeta | Platform_index _
  | Cumprod _ | Cummax _ | Cummin _ | Cumlogsumexp _ | Threefry2x32
  | Iota_2x32_shape _ | Random_seed | Random_split _ | Random_fold_in
  | Random_bits _ | Random_wrap | Random_unwrap ->
      failwith "ad: primitive has no transpose rule (deferred, matches jax gap)"
  | Scan { length; reverse; num_carry; jaxpr; unroll; _ } ->
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
               num_consts = 0;
               num_carry;
               jaxpr = transposed_body;
               unroll;
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
              (triangular ~left_side:o.left_side ~lower:o.lower
                 ~transpose_a:(not o.transpose_a) ~conjugate_a:o.conjugate_a
                 ~unit_diagonal:o.unit_diagonal (prim_val a) (ct1 ()));
          ]
      | _ -> failwith "ad: triangular_solve is linear only in its right side")
  | Tridiagonal_solve { perturb_singular } -> (
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
            Some
              (b1
                 (Tridiagonal_solve { perturb_singular })
                 [ dl_t; d; du_t; ct1 () ]);
          ]
      | _ -> failwith "ad: tridiagonal_solve is linear only in its right side")
  | Checkpoint_name _ -> [ Some (ct1 ()) ]
  | Xla_call c -> xla_call_transpose c cts primals
  | Stop_gradient -> failwith "ad: stop_gradient has no transpose rule"
  | Cholesky | Householder_product | Lu | Lu_pivots_to_permutation _ | Qr _
  | Qr_pivoted _ | Geqrf | Geqp3 | Ormqr _ | Eig _ | Eigh _ | Hessenberg
  | Schur _ | Svd _ | Tridiagonal _ ->
      failwith "ad: nonlinear linalg primitive has no transpose rule"
  | Decomposable _ -> failwith "ad: decomposed before differentiation"

and xla_call_transpose (c : closed_jaxpr) cts primals =
  let known =
    List.filter_map (function Prim v -> Some v | Undef _ -> None) primals
  in
  let specs =
    List.map (function Prim _ -> None | Undef a -> Some a) primals
  in
  let ct_nz = List.map (fun t -> not (is_zero t)) cts in
  let cts_in = List.filter (fun t -> not (is_zero t)) cts in
  let key =
    ( c.jid,
      List.map Option.is_some specs,
      List.map Core.get_aval known,
      ct_nz,
      Config.x64_enabled () )
  in
  let closed, out_nz =
    match Hashtbl.find_opt transpose_cache key with
    | Some r -> r
    | None ->
        let n_known = List.length known in
        let out_nz = ref [] in
        let closed =
          Jaxpr.make_jaxpr
            (List.map Core.get_aval known @ List.map Core.get_aval cts_in)
            (fun args ->
              let k_args, c_args =
                match Util.split_list args [ n_known ] with
                | [ a; b ] -> (a, b)
                | _ -> arity ()
              in
              let k_rest = ref k_args in
              let tvals =
                List.map
                  (function
                    | Some a -> Undef a
                    | None -> (
                        match !k_rest with
                        | v :: tl ->
                            k_rest := tl;
                            Prim v
                        | [] -> arity ()))
                  specs
              in
              let c_rest = ref c_args in
              let full_cts =
                List.map2
                  (fun nz t ->
                    if nz then
                      begin match !c_rest with
                      | v :: tl ->
                          c_rest := tl;
                          v
                      | [] -> arity ()
                      end
                    else t)
                  ct_nz cts
              in
              let const_in = List.map (fun nd -> Prim (Concrete nd)) c.consts in
              let bars =
                eval_jaxpr_transposed c.jaxpr (const_in @ tvals) full_cts
              in
              out_nz := List.map (fun v -> not (is_zero v)) bars;
              List.filter (fun v -> not (is_zero v)) bars)
        in
        Pretty_printer.register_call_name closed.jid
          (Pretty_printer.call_name c.jid);
        Core.inherit_inline ~from:c.jid closed.jid;
        let r = (closed, !out_nz) in
        Hashtbl.replace transpose_cache key r;
        r
  in
  let outs = Core.bind (Xla_call closed) (known @ cts_in) in
  let rest = ref outs and flags = ref out_nz in
  List.map
    (function
      | None -> None
      | Some a -> (
          match !flags with
          | true :: ftl -> (
              flags := ftl;
              match !rest with
              | v :: tl ->
                  rest := tl;
                  Some v
              | [] -> arity ())
          | false :: ftl ->
              flags := ftl;
              Some (Ad_util.symbolic_zero_of_aval a)
          | [] -> arity ()))
    specs

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
        | Some c -> Hashtbl.replace ct_env v.vid (Ad_util.add_any c ct)
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
  let pos, nzs, jx, consts = trace_linearized f primals in
  let backward residuals cts =
    let nz_cts =
      List.concat (List.map2 (fun nz ct -> if nz then [ ct ] else []) nzs cts)
    in
    let const_inputs = List.map (fun c -> Prim c) residuals in
    let n_const = List.length residuals in
    let in_tvals =
      List.filteri (fun i _ -> i >= n_const) jx.in_binders
      |> List.map (fun (v : var) -> Undef v.vaval)
    in
    let bars = eval_jaxpr_transposed jx (const_inputs @ in_tvals) nz_cts in
    let rest = ref bars in
    List.map
      (fun p ->
        let a = Core.get_aval p in
        if is_differentiable_dtype a.dtype then
          begin match !rest with
          | v :: tl ->
              rest := tl;
              instantiate v
          | [] -> failwith "ad.vjp: cotangent arity"
          end
        else p2tz p)
      primals
  in
  (pos, consts, backward)

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
  f_vjp [ Core.splat (Core.get_aval y).dtype [||] 1.0 ]
