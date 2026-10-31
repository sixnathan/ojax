module T = Types
module D = Dtype
module L = Lax_api
module LL = Lax.Linalg

let aval = Core.get_aval
let shape v = (aval v).T.shape
let dtype v = (aval v).T.dtype

let real_dtype v =
  match dtype v with
  | D.Complex128 | D.F64 -> D.F64
  | D.Complex64 | D.F32 -> D.F32
  | d -> d

let const_like v x = L.full [||] x (dtype v)

let conj v =
  match dtype v with D.Complex64 | D.Complex128 -> L.conj v | _ -> v

let matrix_transpose v = L.transpose v [| 1; 0 |]
let hconj x = conj (matrix_transpose x)
let scale s v = L.mul (L.broadcast_in_dim (const_like v s) (shape v) [||]) v

let eye dt n =
  let index dimension =
    List.hd
      (Core.bind (T.Iota { dtype = D.I32; shape = [| n; n |]; dimension }) [])
  in
  L.convert_element_type (L.eq (index 0) (index 1)) dt

let use_qr u (a_minus_e_by_sqrt_c, sqrt_c, e) =
  let sh = shape u in
  let big_m = sh.(0) and big_n = sh.(1) in
  let dt = dtype u in
  let y = L.concatenate [ scale sqrt_c u; eye dt big_n ] 0 in
  let q, _ = LL.qr ~full_matrices:false y in
  let q1 = L.slice q [| 0; 0 |] [| big_m; big_n |] in
  let q2 = hconj (L.slice q [| big_m; 0 |] [| big_m + big_n; big_n |]) in
  L.add (scale e u) (scale a_minus_e_by_sqrt_c (L.dot q1 q2))

let use_cholesky u (a_minus_e, c, e) =
  let big_n = (shape u).(1) in
  let dt = dtype u in
  let x = L.add (scale c (L.dot (hconj u) u)) (eye dt big_n) in
  let y = LL.cholesky x in
  let z =
    conj
      (LL.triangular_solve ~left_side:true ~lower:true ~conjugate_a:true y
         (matrix_transpose u))
  in
  let z =
    hconj
      (LL.triangular_solve ~left_side:true ~lower:true ~transpose_a:true
         ~conjugate_a:true y z)
  in
  L.add (scale e u) (scale a_minus_e z)

let frobenius v =
  let m = L.abs v in
  L.sqrt (L.reduce_sum (L.mul m m) (Array.init (Array.length (shape m)) Fun.id))

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
  let abs_x = L.abs x in
  let one_norm = L.reduce_max (L.reduce_sum abs_x [| 0 |]) [| 0 |] in
  let inf_norm = L.reduce_max (L.reduce_sum abs_x [| 1 |]) [| 0 |] in
  let alpha_inverse = L.mul (L.rsqrt one_norm) (L.rsqrt inf_norm) in
  let alpha_inverse =
    L.select
      (L.eq one_norm (const_like one_norm 0.0))
      (const_like one_norm 1.0) alpha_inverse
  in
  let u0 =
    L.mul x
      (L.broadcast_in_dim
         (L.convert_element_type alpha_inverse (dtype x))
         sh [||])
  in
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
  let not_converged u prev = L.gt (frobenius (L.sub u prev)) tol_norm in
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
  let fixed = (3.0 -. (1.0 /. 3.0), 3.0, 1.0 /. 3.0) in
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
      let u = L.sub (scale 1.5 u) (scale 0.5 cube) in
      let h = L.dot (hconj u) x in
      let sum = L.add h (hconj h) in
      let h =
        L.div sum (L.broadcast_in_dim (const_like sum 2.0) (shape sum) [||])
      in
      (u, h, num_iters, L.bitwise_not flag)
  | _ -> failwith "qdwh: arity"
