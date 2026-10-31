module T = Types
module D = Dtype
module L = Lax_api
module LL = Lax.Linalg
module LN = Numpy.Linalg
module NL = Numpy.Lax_numpy
module UF = Numpy.Ufuncs
module RED = Numpy.Reductions

let aval = Core.get_aval
let shape v = (aval v).T.shape
let dtype v = (aval v).T.dtype

let is_complex v =
  match dtype v with D.Complex64 | D.Complex128 -> true | _ -> false

let real_dtype v =
  match dtype v with
  | D.Complex128 | D.F64 -> D.F64
  | D.Complex64 | D.F32 -> D.F32
  | d -> d

let const_like v x = L.full [||] x (dtype v)
let hconj x = UF.conj (NL.matrix_transpose x)

let host = function
  | T.Concrete nd -> nd
  | T.Device b -> Pjrt.Buffer.to_host b
  | _ -> failwith "linalg_extras: callback received a tracer"

let on_host f ~result_avals a =
  Custom_derivatives.pure_callback
    (fun args -> List.map (fun nd -> T.Concrete nd) (f (host (List.hd args))))
    ~result_avals [ a ]

let vmap_matrix f a =
  List.hd
    (Interpreters.Batching.vmap
       (fun args -> [ f (List.hd args) ])
       [ Some 0 ] [ a ])

let rec geqrf a =
  let sh = shape a in
  if Array.length sh > 2 then
    let outs =
      Interpreters.Batching.vmap
        (fun args ->
          let r, t = geqrf (List.hd args) in
          [ r; t ])
        [ Some 0 ] [ a ]
    in
    (List.nth outs 0, List.nth outs 1)
  else
    let k = min sh.(0) sh.(1) in
    match
      on_host LL.geqrf_impl
        ~result_avals:[ aval a; { (aval a) with T.shape = [| k |] } ]
        a
    with
    | [ r; t ] -> (r, t)
    | _ -> failwith "geqrf: arity"

let rec geqp3 a =
  let sh = shape a in
  if Array.length sh > 2 then
    let outs =
      Interpreters.Batching.vmap
        (fun args ->
          let r, p, t = geqp3 (List.hd args) in
          [ r; p; t ])
        [ Some 0 ] [ a ]
    in
    (List.nth outs 0, List.nth outs 1, List.nth outs 2)
  else
    let n = sh.(1) in
    let k = min sh.(0) n in
    match
      on_host LL.geqp3_impl
        ~result_avals:
          [
            aval a;
            { (aval a) with T.shape = [| n |]; dtype = D.I32 };
            { (aval a) with T.shape = [| k |] };
          ]
        a
    with
    | [ r; p; t ] -> (r, p, t)
    | _ -> failwith "geqp3: arity"

let rec ormqr ?(left = true) ?(transpose = false) a taus c =
  if Array.length (shape a) > 2 then
    List.hd
      (Interpreters.Batching.vmap
         (fun args ->
           match args with
           | [ a; t; c ] -> [ ormqr ~left ~transpose a t c ]
           | _ -> failwith "ormqr: arity")
         [ Some 0; Some 0; Some 0 ] [ a; taus; c ])
  else begin
    let m = (shape a).(0) in
    let k = (shape taus).(0) in
    if left && (shape c).(0) <> m then
      invalid_arg
        "ormqr with left=True expects c to have the same number of rows as the \
         Householder matrix a";
    if (not left) && (shape c).(1) <> m then
      invalid_arg
        "ormqr with left=False expects c to have the same number of columns as \
         the Householder matrix a has rows";
    let dt = dtype a in
    let v_all =
      UF.add
        (NL.tril ~k:(-1) (L.slice_in_dim ~axis:1 a 0 k))
        (NL.eye ~m:k ~dtype:dt m)
    in
    let reverse = left <> transpose in
    let conj_if v = if is_complex a then L.conj v else v in
    let step c i =
      let idx = if reverse then k - 1 - i else i in
      let tau = L.index_in_dim ~axis:0 ~keepdims:false taus idx in
      let tau = if transpose then conj_if tau else tau in
      let v = L.slice_in_dim ~axis:1 v_all idx (idx + 1) in
      let v_h = conj_if (NL.matrix_transpose v) in
      let update =
        if left then L.dot v (L.dot v_h c) else L.dot (L.dot c v) v_h
      in
      UF.subtract c (UF.multiply tau update)
    in
    List.fold_left step c (List.init k Fun.id)
  end

let rec cholesky_update r w =
  if Array.length (shape r) > 2 then
    List.hd
      (Interpreters.Batching.vmap
         (fun args ->
           match args with
           | [ r; w ] -> [ cholesky_update r w ]
           | _ -> failwith "cholesky_update: arity")
         [ Some 0; Some 0 ] [ r; w ])
  else begin
    let sh = shape r in
    if sh.(0) <> sh.(1) || (shape w).(0) <> sh.(1) then
      invalid_arg
        "Rank-1 update to Cholesky decomposition takes a square matrix and a \
         vector of the same size as input";
    let n = sh.(0) in
    let scalar v i = L.reshape (L.slice_in_dim ~axis:0 v i (i + 1)) [||] in
    let rotation x y =
      let nonzero args =
        match args with
        | [ x; y ] ->
            let abs_x = L.abs x and abs_y = L.abs y in
            let denominator = L.select (L.gt abs_x abs_y) abs_x abs_y in
            let x = L.div x denominator and y = L.div y denominator in
            let rh =
              L.div (const_like x 1.0) (L.sqrt (L.add (L.mul x x) (L.mul y y)))
            in
            [ L.mul x rh; L.neg (L.mul y rh) ]
        | _ -> failwith "cholesky_update: arity"
      in
      L.cond
        (L.eq y (const_like y 0.0))
        (fun _ -> [ const_like x 1.0; const_like x 0.0 ])
        nonzero [ x; y ]
    in
    let rows =
      Array.init n (fun i -> L.index_in_dim ~axis:0 ~keepdims:false r i)
    in
    let z = ref w in
    for k = 0 to n - 1 do
      match rotation (scalar rows.(k) k) (scalar !z k) with
      | [ c; s ] ->
          let row = rows.(k) in
          rows.(k) <- L.sub (L.mul c row) (L.mul s !z);
          z := L.add (L.mul c !z) (L.mul s row)
      | _ -> failwith "cholesky_update: arity"
    done;
    NL.stack (Array.to_list rows)
  end

let symmetric_product ?(alpha = 1.0) ?(beta = 0.0) ?(symmetrize_output = false)
    a c =
  let scale x v = UF.multiply (const_like v x) v in
  let result =
    UF.add
      (scale alpha (L.batch_matmul a (NL.matrix_transpose a)))
      (scale beta c)
  in
  if symmetrize_output then
    UF.add (NL.tril result) (NL.matrix_transpose (NL.tril ~k:(-1) result))
  else result

let use_qr u (a_minus_e_by_sqrt_c, sqrt_c, e) =
  let sh = shape u in
  let big_m = sh.(0) and big_n = sh.(1) in
  let dt = dtype u in
  let y =
    NL.concatenate ~axis:0
      [ UF.multiply (const_like u sqrt_c) u; NL.eye ~dtype:dt big_n ]
  in
  let q, _ = LL.qr ~full_matrices:false y in
  let q1 = L.slice q [| 0; 0 |] [| big_m; big_n |] in
  let q2 = hconj (L.slice q [| big_m; 0 |] [| big_m + big_n; big_n |]) in
  UF.add
    (UF.multiply (const_like u e) u)
    (UF.multiply (const_like u a_minus_e_by_sqrt_c) (L.dot q1 q2))

let use_cholesky u (a_minus_e, c, e) =
  let big_n = (shape u).(1) in
  let dt = dtype u in
  let x =
    UF.add
      (UF.multiply (const_like u c) (L.dot (hconj u) u))
      (NL.eye ~dtype:dt big_n)
  in
  let y = LL.cholesky ~symmetrize_input:false x in
  let z =
    UF.conj
      (LL.triangular_solve ~left_side:true ~lower:true ~conjugate_a:true y
         (NL.matrix_transpose u))
  in
  let z =
    hconj
      (LL.triangular_solve ~left_side:true ~lower:true ~transpose_a:true
         ~conjugate_a:true y z)
  in
  UF.add
    (UF.multiply (const_like u e) u)
    (UF.multiply (const_like u a_minus_e) z)

let frobenius v =
  let m = UF.abs v in
  L.sqrt (RED.sum (UF.multiply m m))

let qdwh ?(is_hermitian = false) ?(max_iterations = 10) ?eps x =
  ignore is_hermitian;
  let sh = shape x in
  if Array.length sh <> 2 then invalid_arg "qdwh: expected a matrix";
  if sh.(0) < sh.(1) then
    invalid_arg "The input matrix of shape M x N must have M >= N";
  let rdt = real_dtype x in
  let eps =
    match eps with
    | Some e -> e
    | None -> if rdt = D.F64 then epsilon_float else 1.1920928955078125e-07
  in
  let abs_x = UF.abs x in
  let one_norm = RED.amax (RED.sum ~axis:[| 0 |] abs_x) in
  let inf_norm = RED.amax (RED.sum ~axis:[| 1 |] abs_x) in
  let alpha_inverse = L.mul (L.rsqrt one_norm) (L.rsqrt inf_norm) in
  let alpha_inverse =
    NL.where_
      (L.eq one_norm (const_like one_norm 0.0))
      (const_like one_norm 1.0) alpha_inverse
  in
  let u0 = UF.multiply x (NL.astype alpha_inverse (dtype x)) in
  let tol_l = 10.0 *. eps /. 2.0 in
  let tol_norm = L.full [||] (Float.cbrt tol_l) rdt in
  let qr_coefs = ref [] and chol_coefs = ref [] in
  let l = ref eps and k = ref 0 in
  while !l +. tol_l < 1.0 && !k < max_iterations do
    incr k;
    let l2 = !l *. !l in
    let dd = Float.pow (4.0 *. ((1.0 /. l2) -. 1.0) /. l2) (1.0 /. 3.0) in
    let sqd = Float.sqrt (1.0 +. dd) in
    let a =
      sqd +. Float.sqrt (2.0 -. dd +. (2.0 *. (2.0 -. l2) /. (l2 *. sqd)))
    in
    let b = (a -. 1.0) *. (a -. 1.0) /. 4.0 in
    let c = a +. b -. 1.0 in
    l := !l *. (a +. (b *. l2)) /. (1.0 +. (c *. l2));
    let e = b /. c in
    if c > 100.0 then
      qr_coefs := ((a -. e) /. Float.sqrt c, Float.sqrt c, e) :: !qr_coefs
    else chol_coefs := (a -. e, c, e) :: !chol_coefs
  done;
  let qr_list = List.rev !qr_coefs and chol_list = List.rev !chol_coefs in
  let u = List.fold_left use_qr u0 qr_list in
  let not_converged u prev = L.gt (frobenius (UF.subtract u prev)) tol_norm in
  let u, flag =
    List.fold_left
      (fun (u, _) coef ->
        let next = use_cholesky u coef in
        (next, not_converged next u))
      (u, T.Concrete (Ndarray.of_floats D.Bool [||] [| 1.0 |]))
      chol_list
  in
  let start = List.length qr_list + List.length chol_list in
  let idx v = L.full [||] (float_of_int v) D.I32 in
  let fixed = (2.0, 3.0, 1.0 /. 3.0) in
  let final =
    L.while_loop
      (fun state ->
        match state with
        | [ k; _; flag ] -> L.bitwise_and flag (L.lt k (idx max_iterations))
        | _ -> failwith "qdwh: arity")
      (fun state ->
        match state with
        | [ k; u; _ ] ->
            let next = use_cholesky u fixed in
            [ L.add k (idx 1); next; not_converged next u ]
        | _ -> failwith "qdwh: arity")
      [ idx start; u; flag ]
  in
  match final with
  | [ num_iters; u; flag ] ->
      let cube = L.dot u (L.dot (hconj u) u) in
      let u =
        UF.subtract
          (UF.multiply (const_like u 1.5) u)
          (UF.multiply (const_like u 0.5) cube)
      in
      let h = L.dot (hconj u) x in
      let h = UF.divide (UF.add h (hconj h)) (const_like h 2.0) in
      (u, h, num_iters, L.bitwise_not flag)
  | _ -> failwith "qdwh: arity"
