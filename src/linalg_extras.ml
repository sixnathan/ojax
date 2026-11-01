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
    Eager_jit.one "lax.linalg.cholesky_update" () [] [ r; w ] (fun _ vals ->
        let r, w = match vals with [ r; w ] -> (r, w) | _ -> assert false in
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
              let new_row =
                L.sub (L.mul (spread c) row) (L.mul (spread s) !z)
              in
              z := L.add (L.mul (spread c) !z) (L.mul (spread s) row);
              current :=
                L.dynamic_update_index_in_dim ~axis:0 !current new_row
                  (Core.splat D.I32 [||] (float_of_int k))
          | _ -> failwith "cholesky_update: arity"
        done;
        !current)
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

let qdwh = Qdwh.qdwh
