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
let i32 v = L.full [||] (float_of_int v) D.I32

let iota_at sh dimension =
  List.hd (Core.bind (T.Iota { dtype = D.I32; shape = sh; dimension }) [])

let eye dt n =
  L.convert_element_type (L.eq (iota_at [| n; n |] 0) (iota_at [| n; n |] 1)) dt

let spread s like = L.broadcast_in_dim s (shape like) [||]

let scalar_times s v =
  let s = L.convert_element_type s (dtype v) in
  L.mul (spread s v) v

let mask x (d0, d1) alternative =
  let sh = shape x in
  let below i d = L.lt (iota_at sh i) (L.broadcast_in_dim (i32 d) sh [||]) in
  L.select (L.bitwise_and (below 0 d0) (below 1 d1)) x alternative

let zeros_like v = spread (const_like v 0.0) v

let use_qr u (a_minus_e_by_sqrt_c, sqrt_c, e) =
  let sh = shape u in
  let big_m = sh.(0) and big_n = sh.(1) in
  let dt = dtype u in
  let scaled = scalar_times sqrt_c u in
  let id = eye dt big_n in
  let padded =
    L.pad scaled (const_like scaled 0.0) [| (0, big_n, 0); (0, 0, 0) |]
  in
  let y = L.dynamic_update_slice padded id [ i32 big_m; i32 0 ] in
  let q, _ = LL.qr ~full_matrices:false y in
  let q1_raw = L.slice q [| 0; 0 |] [| big_m; big_n |] in
  let q1 = mask q1_raw (big_m, big_n) (zeros_like q1_raw) in
  let q2_raw = L.dynamic_slice q [ i32 big_m; i32 0 ] [| big_n; big_n |] in
  let q2 = hconj (mask q2_raw (big_n, big_n) (zeros_like q2_raw)) in
  let first = scalar_times e u in
  L.add first (scalar_times a_minus_e_by_sqrt_c (L.dot q1 q2))

let use_cholesky u (a_minus_e, c, e) =
  let big_n = (shape u).(1) in
  let dt = dtype u in
  let gram = scalar_times c (L.dot (hconj u) u) in
  let x = L.add gram (eye dt big_n) in
  let x = mask x (big_n, big_n) (eye dt big_n) in
  let y = LL.cholesky ~symmetrize_input:false x in
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
  let first = scalar_times e u in
  L.add first (scalar_times a_minus_e z)

let frobenius v =
  let squared =
    match dtype v with
    | D.Complex64 | D.Complex128 -> L.real (L.mul v (L.conj v))
    | _ -> L.mul v v
  in
  L.sqrt
    (L.reduce_sum squared (Array.init (Array.length (shape squared)) Fun.id))

let coefficient_table dt rows =
  let flat = Array.concat (List.map Array.of_list rows) in
  T.Concrete (Ndarray.of_floats dt [| List.length rows; 3 |] flat)

let row_params table k =
  let row = L.dynamic_slice table [ k; i32 0 ] [| 1; 3 |] in
  let cell j = L.reshape (L.slice row [| 0; j |] [| 1; j + 1 |]) [||] in
  (cell 0, cell 1, cell 2)

let qdwh_body ~max_iterations ~eps x =
  let sh = shape x in
  let rdt = real_dtype x in
  let eps =
    match eps with
    | Some e -> e
    | None -> if rdt = D.F64 then epsilon_float else 1.1920928955078125e-07
  in
  let abs_x = L.abs x in
  let one_norm = L.reduce_max (L.reduce_sum abs_x [| 0 |]) [| 0 |] in
  let inf_norm = L.reduce_max (L.reduce_sum abs_x [| 1 |]) [| 0 |] in
  let alpha_inverse =
    let one = L.rsqrt one_norm in
    let inf = L.rsqrt inf_norm in
    L.mul one inf
  in
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
  let tol_norm = L.cbrt (L.full [||] tol_l rdt) in
  let qr_coefs = ref [] and chol_coefs = ref [] in
  let l = ref eps and k = ref 0 in
  let half = 1.0 /. 2.0 in
  while !l +. tol_l < 1.0 && !k < max_iterations do
    incr k;
    let l2 = !l *. !l in
    let dd = Float.pow (4.0 *. ((1.0 /. l2) -. 1.0) /. l2) (1.0 /. 3.0) in
    let sqd = Float.pow (1.0 +. dd) half in
    let a =
      sqd +. Float.pow (2.0 -. dd +. (2.0 *. (2.0 -. l2) /. (l2 *. sqd))) half
    in
    let b = Float.pow (a -. 1.0) 2.0 /. 4.0 in
    let c = a +. b -. 1.0 in
    l := !l *. (a +. (b *. l2)) /. (1.0 +. (c *. l2));
    let e = b /. c in
    if c > 100.0 then
      let sqrt_c = Float.pow c half in
      qr_coefs := [ (a -. e) /. sqrt_c; sqrt_c; e ] :: !qr_coefs
    else chol_coefs := [ a -. e; c; e ] :: !chol_coefs
  done;
  let qr_list = List.rev !qr_coefs and chol_list = List.rev !chol_coefs in
  let fori table count step init =
    let final =
      L.while_loop
        (fun state ->
          match state with
          | k :: _ -> L.lt k (i32 count)
          | [] -> failwith "qdwh: arity")
        (fun state ->
          match state with
          | k :: rest -> L.add k (i32 1) :: step (row_params table k) rest
          | [] -> failwith "qdwh: arity")
        (i32 0 :: init)
    in
    List.tl final
  in
  let u =
    if qr_list = [] then u0
    else
      let table = coefficient_table rdt qr_list in
      match
        fori table (List.length qr_list)
          (fun params state ->
            match state with
            | [ u ] -> [ use_qr u params ]
            | _ -> failwith "qdwh: arity")
          [ u0 ]
      with
      | [ u ] -> u
      | _ -> failwith "qdwh: arity"
  in
  let not_converged u prev = L.gt (frobenius (L.sub u prev)) tol_norm in
  let always = T.Concrete (Ndarray.of_floats D.Bool [||] [| 1.0 |]) in
  let u, flag =
    if chol_list = [] then (u, always)
    else
      let table = coefficient_table rdt chol_list in
      match
        fori table (List.length chol_list)
          (fun params state ->
            match state with
            | [ u; _ ] ->
                let next = use_cholesky u params in
                [ next; not_converged next u ]
            | _ -> failwith "qdwh: arity")
          [ u; always ]
      with
      | [ u; flag ] -> (u, flag)
      | _ -> failwith "qdwh: arity"
  in
  let start = List.length qr_list + List.length chol_list in
  let real v = L.full [||] v rdt in
  let a = 3.0 and b = 1.0 and c = 3.0 in
  let e = b /. c in
  let halley = (real (a -. e), real c, real e) in
  let final =
    L.while_loop
      (fun state ->
        match state with
        | [ k; _; flag ] -> L.bitwise_and flag (L.lt k (i32 max_iterations))
        | _ -> failwith "qdwh: arity")
      (fun state ->
        match state with
        | [ k; u; _ ] ->
            let next = use_cholesky u halley in
            [ L.add k (i32 1); next; not_converged next u ]
        | _ -> failwith "qdwh: arity")
      [ i32 start; u; flag ]
  in
  match final with
  | [ num_iters; u; flag ] ->
      let times v s = scalar_times (real s) v in
      let cube = L.dot (times u 0.5) (L.dot (hconj u) u) in
      let u = L.sub (times u 1.5) cube in
      let h = L.dot (hconj u) x in
      let sum = L.add h (hconj h) in
      let h =
        L.div sum (L.broadcast_in_dim (const_like sum 2.0) (shape sum) [||])
      in
      [ u; h; num_iters; L.bitwise_not flag ]
  | _ -> failwith "qdwh: arity"

let qdwh ?(is_hermitian = false) ?(max_iterations = 10) ?eps x =
  ignore is_hermitian;
  let sh = shape x in
  if Array.length sh <> 2 then invalid_arg "qdwh: expected a matrix";
  if sh.(0) < sh.(1) then
    invalid_arg "The input matrix of shape M x N must have M >= N";
  match
    Jaxpr.named_jit ~name:"qdwh"
      ~key:
        (Digest.to_hex
           (Digest.string (Marshal.to_string (max_iterations, eps) [])))
      (function
        | [ x ] -> qdwh_body ~max_iterations ~eps x
        | _ -> failwith "qdwh: arity")
      [ x ]
  with
  | [ u; h; num_iters; ok ] -> (u, h, num_iters, ok)
  | _ -> failwith "qdwh: arity"
