open Types
module C = Random.Core
module NL = Numpy.Lax_numpy
module L = Lax_api

let shape v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype

let split2 key =
  let keys = C.split key 2 in
  (L.index_in_dim ~keepdims:false keys 0, L.index_in_dim ~keepdims:false keys 1)

let maxwell key ~shape:sh =
  let samples = C.normal key ~shape:(Array.append sh [| 3 |]) in
  let last = Array.length (shape samples) - 1 in
  L.sqrt (Numpy.Reductions.sum ~axis:[| last |] (L.mul samples samples))

let double_sided_maxwell key ~loc ~scale ~shape:sh =
  let k1, k2 = split2 key in
  let magnitude = maxwell k1 ~shape:sh in
  let signs = NL.astype (C.rademacher k2 ~shape:sh) (dtype_of magnitude) in
  let dt = dtype_of magnitude in
  L.add
    (L.mul (L.mul signs magnitude) (L.full [||] scale dt))
    (L.full [||] loc dt)

let ball ?(p = 2.0) key ~d ~shape:sh =
  let k1, k2 = split2 key in
  let g = C.generalized_normal k1 ~shape:(Array.append sh [| d |]) ~p in
  let e = C.exponential k2 ~shape:sh in
  let dt = dtype_of g in
  let power = L.pow (L.abs g) (L.full [||] p dt) in
  let total =
    Numpy.Reductions.sum ~axis:[| Array.length (shape g) - 1 |] power
  in
  let denom = L.pow (L.add total e) (L.full [||] (1.0 /. p) dt) in
  let expanded = L.expand_dims denom [| -1 |] in
  L.div g (NL.broadcast_to expanded (shape g))

let orthogonal ?m key ~n ~shape:sh =
  let m = Option.value m ~default:n in
  let big = Stdlib.max n m and small = Stdlib.min n m in
  let z = C.normal key ~shape:(Array.append sh [| big; small |]) in
  let q, r =
    match Numpy.Linalg.qr z with
    | [ q; r ] -> (q, r)
    | _ -> failwith "orthogonal: qr arity"
  in
  let d = NL.diagonal ~axis1:(-2) ~axis2:(-1) r in
  let signs = L.expand_dims (L.sign d) [| -2 |] in
  let x = L.mul q (NL.broadcast_to signs (shape q)) in
  if n < m then NL.swapaxes (-1) (-2) x else x

let multivariate_normal ?shape:shape_opt ?(method_ = "cholesky") key ~mean ~cov
    =
  let mean_shape = shape mean in
  let rank = Array.length mean_shape in
  let event = mean_shape.(rank - 1) in
  let batch = Array.sub mean_shape 0 (rank - 1) in
  let sh = Option.value shape_opt ~default:batch in
  let float_dtype = NL.result_type [ mean; cov ] in
  let z =
    C.with_dtypes ~float_dtype (fun () ->
        C.normal key ~shape:(Array.append sh [| event |]))
  in
  let factor =
    match method_ with
    | "cholesky" -> Numpy.Linalg.cholesky cov
    | "eigh" ->
        let w, v = Numpy.Linalg.eigh cov in
        let root = L.expand_dims (L.sqrt w) [| -2 |] in
        L.mul v (NL.broadcast_to root (shape v))
    | "svd" -> (
        match Numpy.Linalg.svd cov with
        | [ u; s; _ ] ->
            let root = L.expand_dims (L.sqrt s) [| -2 |] in
            L.mul u (NL.broadcast_to root (shape u))
        | _ -> failwith "multivariate_normal: svd arity")
    | other ->
        invalid_arg
          ("method must be one of {'svd', 'eigh', 'cholesky'}; got " ^ other)
  in
  let product = Numpy.Einsum.einsum "...ij,...j->...i" [ factor; z ] in
  Numpy.Ufuncs.add mean product
