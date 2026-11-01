module T = Types
module D = Dtype
module L = Lax_api
module LL = Lax.Linalg

let b1 = Core.bind1
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

let scalar dt x = Core.splat dt [||] x

let jit1 name ?key f args =
  match Jaxpr.named_jit ~name ?key (fun a -> [ f a ]) args with
  | [ v ] -> v
  | _ -> invalid_arg "Qdwh: arity"

let where c x y =
  jit1 "_where"
    (function
      | [ c; x; y ] ->
          let dt =
            match (shape x, shape y) with
            | [||], _ when dtype y <> dtype x -> dtype y
            | _ -> dtype x
          in
          let widen v =
            let v =
              if dtype v = dt then v else b1 (T.Convert_element_type dt) [ v ]
            in
            let target = shape c in
            if shape v = target then v
            else b1 (T.Broadcast_in_dim { shape = target; dims = [||] }) [ v ]
          in
          let x = widen x in
          let y = widen y in
          b1 T.Select_n [ c; y; x ]
      | _ -> invalid_arg "where: arity")
    [ c; x; y ]

let index i = T.Concrete (Ndarray.of_floats D.I32 [||] [| float_of_int i |])
let conj v = if is_complex v then b1 T.Conj [ v ] else v
let transpose v = b1 (T.Transpose [| 1; 0 |]) [ v ]
let hconj x = conj (transpose x)
let iota sh dimension = b1 (T.Iota { dtype = D.I32; shape = sh; dimension }) []

let eye dt n =
  let sh = [| n; n |] in
  let rows = iota sh 0 in
  let cols = iota sh 1 in
  let shifted = b1 T.Add [ rows; index 0 ] in
  b1 (T.Convert_element_type dt) [ b1 T.Eq [ shifted; cols ] ]

let mask x dims alternative =
  let sh = shape x in
  let below i d = b1 T.Lt [ iota sh i; index d ] in
  let first = below 0 dims.(0) in
  let second = below 1 dims.(1) in
  let m = b1 T.And [ first; second ] in
  where m x alternative

let use_qr u (a_minus_e_by_sqrt_c, sqrt_c, e) =
  let sh = shape u in
  let m = sh.(0) and n = sh.(1) in
  let dt = dtype u in
  let scaled = b1 T.Mul [ sqrt_c; u ] in
  let identity = eye dt n in
  let padded =
    b1 (T.Pad [| (0, n, 0); (0, 0, 0) |]) [ scaled; scalar dt 0.0 ]
  in
  let y = b1 T.Dynamic_update_slice [ padded; identity; index m; index 0 ] in
  let q, _ = LL.qr ~full_matrices:false y in
  let q1 =
    b1
      (T.Slice
         {
           start_indices = [| 0; 0 |];
           limit_indices = [| m; n |];
           strides = None;
         })
      [ q ]
  in
  let q1 = mask q1 [| m; n |] (index 0) in
  let q2 =
    b1 (T.Dynamic_slice { slice_sizes = [| n; n |] }) [ q; index m; index 0 ]
  in
  let q2 = hconj (mask q2 [| n; n |] (index 0)) in
  let left = b1 T.Mul [ e; u ] in
  let product = L.dot q1 q2 in
  b1 T.Add [ left; b1 T.Mul [ a_minus_e_by_sqrt_c; product ] ]

let use_cholesky u (a_minus_e, c, e) =
  let n = (shape u).(1) in
  let dt = dtype u in
  let gram = L.dot (hconj u) u in
  let scaled = b1 T.Mul [ c; gram ] in
  let x = b1 T.Add [ scaled; eye dt n ] in
  let alternative = eye dt n in
  let x = mask x [| n; n |] alternative in
  let y = LL.cholesky ~symmetrize_input:false x in
  let z =
    conj
      (LL.triangular_solve ~left_side:true ~lower:true ~conjugate_a:true y
         (transpose u))
  in
  let z =
    hconj
      (LL.triangular_solve ~left_side:true ~lower:true ~transpose_a:true
         ~conjugate_a:true y z)
  in
  let left = b1 T.Mul [ e; u ] in
  b1 T.Add [ left; b1 T.Mul [ a_minus_e; z ] ]

let matrix_norm name axis v =
  jit1 "norm" ~key:name
    (function
      | [ v ] ->
          let magnitude = b1 T.Abs [ v ] in
          let summed = b1 (T.Reduce_sum [| axis |]) [ magnitude ] in
          let top = b1 (T.Reduce_max [| 0 |]) [ summed ] in
          b1 T.Max [ scalar (dtype top) 0.0; top ]
      | _ -> invalid_arg "norm: arity")
    [ v ]

let frobenius v =
  jit1 "norm" ~key:"fro"
    (function
      | [ v ] ->
          let squared =
            if is_complex v then b1 T.Real [ b1 T.Mul [ v; b1 T.Conj [ v ] ] ]
            else b1 T.Mul [ v; v ]
          in
          let axes = Array.init (Array.length (shape squared)) Fun.id in
          b1 T.Sqrt [ b1 (T.Reduce_sum axes) [ squared ] ]
      | _ -> invalid_arg "norm: arity")
    [ v ]

let unpack_row row =
  let piece i =
    let sliced =
      b1
        (T.Slice
           {
             start_indices = [| i |];
             limit_indices = [| i + 1 |];
             strides = Some [| 1 |];
           })
        [ row ]
    in
    b1 (T.Squeeze [| 0 |]) [ sliced ]
  in
  let first = piece 0 in
  let second = piece 1 in
  let third = piece 2 in
  (first, second, third)

let dynamic_index coefs k =
  jit1 "dynamic_index_in_dim"
    (function
      | [ coefs; k ] ->
          let sh = shape coefs in
          let negative = b1 T.Lt [ k; index 0 ] in
          let strong =
            if (aval k).T.weak_type then b1 (T.Convert_element_type D.I32) [ k ]
            else k
          in
          let wrapped = b1 T.Add [ strong; index sh.(0) ] in
          let start = b1 T.Select_n [ negative; k; wrapped ] in
          let picked =
            b1
              (T.Dynamic_slice { slice_sizes = [| 1; sh.(1) |] })
              [ coefs; start; index 0 ]
          in
          b1 (T.Squeeze [| 0 |]) [ picked ]
      | _ -> invalid_arg "dynamic_index_in_dim: arity")
    [ coefs; k ]

let not_converged tol_norm next prev =
  let norm = frobenius (b1 T.Sub [ next; prev ]) in
  b1 T.Gt [ norm; b1 (T.Convert_element_type (dtype norm)) [ tol_norm ] ]

let iterate u flag coefs update test_convergence =
  match coefs with
  | [] -> (u, flag)
  | _ -> (
      let dt = dtype u in
      let n_rows = List.length coefs in
      let table =
        let flat = List.concat_map (fun (a, b, c) -> [ a; b; c ]) coefs in
        let rdt = Dtypes.default_float_dtype () in
        let host =
          T.Concrete
            (Ndarray.of_floats rdt [| n_rows; 3 |] (Array.of_list flat))
        in
        if rdt = dt then host else b1 (T.Convert_element_type dt) [ host ]
      in
      let body = function
        | [ i; u; _ ] ->
            let next_i = b1 T.Add [ i; Core.py_scalar D.I32 1.0 ] in
            let row = dynamic_index table i in
            let params = unpack_row row in
            let next = update u params in
            [ next_i; next; test_convergence next u ]
        | _ -> invalid_arg "qdwh: arity"
      in
      let start =
        T.Concrete (Ndarray.as_weak (Ndarray.of_floats D.I32 [||] [| 0.0 |]))
      in
      match Lax.scan ~length:n_rows body [ start; u; flag ] [] with
      | [ _; u; flag ] -> (u, flag)
      | _ -> invalid_arg "qdwh: arity")

let qdwh_body max_iterations eps x =
  let rdt = real_dtype x in
  let eps =
    match eps with
    | Some e -> e
    | None -> if rdt = D.F64 then epsilon_float else 1.1920928955078125e-07
  in
  let dt = dtype x in
  let one_norm = matrix_norm "1" 0 x in
  let inf_norm = matrix_norm "inf" 1 x in
  let first = b1 T.Rsqrt [ one_norm ] in
  let second = b1 T.Rsqrt [ inf_norm ] in
  let alpha_inverse = b1 T.Mul [ first; second ] in
  let is_zero = b1 T.Eq [ one_norm; scalar (dtype one_norm) 0.0 ] in
  let alpha_inverse = where is_zero (index 1) alpha_inverse in
  let scale =
    if dtype alpha_inverse = dt then alpha_inverse
    else b1 (T.Convert_element_type dt) [ alpha_inverse ]
  in
  let u0 = b1 T.Mul [ x; scale ] in
  let tol_l = 10.0 *. eps /. 2.0 in
  let tol_norm = b1 T.Cbrt [ scalar (Dtypes.default_float_dtype ()) tol_l ] in
  let qr_coefs = ref [] and chol_coefs = ref [] in
  let l = ref eps and k = ref 0 in
  while !l +. tol_l < 1.0 && !k < max_iterations do
    incr k;
    let l2 = !l *. !l in
    let dd = Float.pow (4.0 *. ((1.0 /. l2) -. 1.0) /. l2) (1.0 /. 3.0) in
    let sqd = Float.pow (1.0 +. dd) 0.5 in
    let a =
      sqd +. Float.pow (2.0 -. dd +. (2.0 *. (2.0 -. l2) /. (l2 *. sqd))) 0.5
    in
    let b = Float.pow (a -. 1.0) 2.0 /. 4.0 in
    let c = a +. b -. 1.0 in
    l := !l *. (a +. (b *. l2)) /. (1.0 +. (c *. l2));
    let e = b /. c in
    if c > 100.0 then begin
      let sqrt_c = Float.pow c 0.5 in
      qr_coefs := ((a -. e) /. sqrt_c, sqrt_c, e) :: !qr_coefs
    end
    else chol_coefs := (a -. e, c, e) :: !chol_coefs
  done;
  let qr_list = List.rev !qr_coefs and chol_list = List.rev !chol_coefs in
  let always = T.Concrete (Ndarray.of_floats D.Bool [||] [| 1.0 |]) in
  let u, _ = iterate u0 always qr_list use_qr (fun _ _ -> always) in
  let u, flag =
    iterate u always chol_list use_cholesky (not_converged tol_norm)
  in
  let start = List.length qr_list + List.length chol_list in
  let final_params =
    let e = 1.0 /. 3.0 in
    (scalar dt (3.0 -. e), scalar dt 3.0, scalar dt e)
  in
  let final =
    Lax.while_loop
      (function
        | [ k; _; flag ] ->
            let lt = b1 T.Lt [ k; index max_iterations ] in
            b1 T.And [ flag; b1 (T.Convert_element_type D.Bool) [ lt ] ]
        | _ -> invalid_arg "qdwh: arity")
      (function
        | [ k; u; _ ] ->
            let next = use_cholesky u final_params in
            let flag = not_converged tol_norm next u in
            [ b1 T.Add [ k; index 1 ]; next; flag ]
        | _ -> invalid_arg "qdwh: arity")
      [ index start; u; flag ]
  in
  match final with
  | [ num_iters; u; flag ] ->
      let left = b1 T.Mul [ scalar dt 1.5; u ] in
      let half = b1 T.Mul [ scalar dt 0.5; u ] in
      let cube = L.dot half (L.dot (hconj u) u) in
      let u = b1 T.Sub [ left; cube ] in
      let h = L.dot (hconj u) x in
      let sum = b1 T.Add [ h; hconj h ] in
      let h = b1 T.Div [ sum; scalar dt 2.0 ] in
      [ u; h; num_iters; b1 T.Not [ flag ] ]
  | _ -> invalid_arg "qdwh: arity"

let qdwh ?(is_hermitian = false) ?(max_iterations = 10) ?eps x =
  ignore is_hermitian;
  let sh = shape x in
  if Array.length sh <> 2 then invalid_arg "qdwh: expected a matrix";
  if sh.(0) < sh.(1) then
    invalid_arg "The input matrix of shape M x N must have M >= N";
  let key =
    Printf.sprintf "%d %s" max_iterations
      (match eps with Some e -> string_of_float e | None -> "none")
  in
  let outs =
    Jaxpr.named_jit ~name:"qdwh" ~key
      (function
        | [ x ] ->
            Config.with_value Config.default_matmul_precision (Some "float32")
              (fun () -> qdwh_body max_iterations eps x)
        | _ -> invalid_arg "qdwh: arity")
      [ x ]
  in
  match outs with
  | [ u; h; num_iters; ok ] -> (u, h, num_iters, ok)
  | _ -> invalid_arg "qdwh: arity"
