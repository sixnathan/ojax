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

let vmap_matrix f a =
  List.hd
    (Interpreters.Batching.vmap
       (fun args -> [ f (List.hd args) ])
       [ Some 0 ] [ a ])

let geqrf a =
  match Core.bind T.Geqrf [ a ] with
  | [ r; t ] -> (r, t)
  | _ -> failwith "geqrf: arity"

let geqp3 a jpvt =
  match Core.bind T.Geqp3 [ a; jpvt ] with
  | [ r; p; t ] -> (r, p, t)
  | _ -> failwith "geqp3: arity"

let ormqr ?(left = true) ?(transpose = false) a taus c =
  Core.bind1
    (T.Ormqr { ormqr_left = left; ormqr_transpose = transpose })
    [ a; taus; c ]

let cholesky_update_jax_fn r w =
  let n = (shape r).(0) in
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
          [ L.mul x rh; L.mul (L.neg y) rh ]
      | _ -> failwith "cholesky_update: arity"
    in
    L.cond
      (L.eq y (const_like y 0.0))
      (fun _ -> [ const_like x 1.0; const_like x 0.0 ])
      nonzero [ x; y ]
  in
  let at_scalar v starts =
    L.reshape (L.slice v starts (Array.map (fun i -> i + 1) starts)) [||]
  in
  let current = ref r and z = ref w in
  for k = 0 to n - 1 do
    let x = at_scalar !current [| k; k |] and y = at_scalar !z [| k |] in
    match rotation x y with
    | [ c; s ] ->
        let row = L.index_in_dim ~axis:0 ~keepdims:false !current k in
        let spread v = L.broadcast_in_dim v [| n |] [||] in
        let new_row = L.sub (L.mul (spread c) row) (L.mul (spread s) !z) in
        z := L.add (L.mul (spread c) !z) (L.mul (spread s) row);
        let index =
          L.broadcast_in_dim
            (Core.splat D.I32 [||] (float_of_int k))
            [| 1 |] [||]
        in
        current :=
          Core.bind1
            (T.Scatter
               {
                 dimension_numbers =
                   {
                     T.update_window_dims = [| 0 |];
                     inserted_window_dims = [| 0 |];
                     scatter_dims_to_operand_dims = [| 0 |];
                     s_operand_batching_dims = [||];
                     s_scatter_indices_batching_dims = [||];
                     s_sorted_unique = true;
                     s_clip = false;
                   };
                 unique_indices = true;
               })
            [ !current; index; new_row ]
    | _ -> failwith "cholesky_update: arity"
  done;
  !current

let symmetric_product_jax_fn ~alpha ~beta a c =
  let scale x v = UF.multiply (const_like v x) v in
  UF.add
    (scale alpha
       (L.batch_matmul
          ~precision:
            (Lax_config.precision_of_level Lax_config.Precision_highest)
          a (NL.matrix_transpose a)))
    (scale beta c)

let () =
  Core.lowered_linalg_hook :=
    fun prim vals ->
      match (prim, vals) with
      | T.Cholesky_update, [ r; w ] -> [ cholesky_update_jax_fn r w ]
      | T.Symmetric_product { alpha; beta }, [ a; c ] ->
          [ symmetric_product_jax_fn ~alpha ~beta a c ]
      | _ -> failwith "lowered linalg: unexpected primitive"

let cholesky_update r w =
  let check name v i expected =
    if Array.length (shape v) <> expected then
      invalid_arg
        (Printf.sprintf
           "Input %d to %s must have a rank of exactly %d, but got shape=%s" i
           name expected
           ("("
           ^ String.concat ", "
               (List.map string_of_int (Array.to_list (shape v)))
           ^ ")"))
  in
  check "cholesky_update" r 0 2;
  check "cholesky_update" w 1 1;
  Eager_jit.one "lax.linalg.cholesky_update" () [] [ r; w ] (fun _ vals ->
      Core.bind1 T.Cholesky_update vals)

let symmetric_product ?(alpha = 1.0) ?(beta = 0.0) ?(symmetrize_output = false)
    a c =
  let result =
    Eager_jit.one "lax.linalg.symmetric_product" (alpha, beta) [] [ a; c ]
      (fun _ vals -> Core.bind1 (T.Symmetric_product { alpha; beta }) vals)
  in
  if symmetrize_output then
    let n = Array.length (Core.get_aval result).shape in
    let perm =
      Array.init n (fun i ->
          if i = n - 1 then n - 2 else if i = n - 2 then n - 1 else i)
    in
    let upper_half =
      Core.bind1 (Types.Transpose perm) [ LL.tril ~k:(-1) result ]
    in
    UF.add (LL.tril result) upper_half
  else result

let qdwh = Qdwh.qdwh
