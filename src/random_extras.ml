open Types
module C = Random.Core
module NL = Numpy.Lax_numpy
module L = Lax_api

module Core_like = struct
  let scalar dt x = Concrete (Ndarray.of_floats dt [||] [| x |])
end

let shape v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype

let split2 key =
  let keys = C.split key 2 in
  (L.index_in_dim ~keepdims:false keys 0, L.index_in_dim ~keepdims:false keys 1)

let maxwell key ~shape:sh =
  let dt = C.float_dtype () in
  C.jit_region "_maxwell" (sh, dt) [ key ] (function
    | [ key ] ->
        let samples = C.normal key ~shape:(Array.append sh [| 3 |]) in
        Numpy.Linalg.norm ~axis:(Numpy.Linalg.Aint (-1)) samples
    | _ -> assert false)

let double_sided_maxwell key ~loc ~scale ~shape:sh =
  let dt = C.float_dtype () in
  let pyf = Numpy.Jnp_util.py_float in
  C.jit_region "_double_sided_maxwell" (sh, dt)
    [ key; pyf loc; pyf scale ]
    (function
      | [ key; loc; scale ] ->
          let k1, k2 = split2 key in
          let magnitude = maxwell k1 ~shape:sh in
          let signs = C.rademacher_as k2 ~shape:sh ~dt in
          let open Numpy.Ufuncs in
          let conv v = NL.astype v dt in
          add (multiply (multiply signs magnitude) (conv scale)) (conv loc)
      | _ -> assert false)

let ball ?p key ~d ~shape:sh =
  let dt = C.float_dtype () in
  let pv =
    match p with
    | None -> Numpy.Jnp_util.py_int 2
    | Some p -> Numpy.Jnp_util.py_float p
  in
  C.jit_region "_ball" (sh, dt, d) [ key; pv ] (function
    | [ key; pv ] ->
        let k1, k2 = split2 key in
        let g =
          C.generalized_normal_value k1 ~shape:(Array.append sh [| d |]) ~p:pv
        in
        let e = C.exponential k2 ~shape:sh in
        let open Numpy.Ufuncs in
        let powered = power (abs g) pv in
        let total = Numpy.Reductions.sum ~axis:[| -1 |] powered in
        let total = add total e in
        let inv = true_divide (Numpy.Jnp_util.py_float 1.0) pv in
        let denom = power total inv in
        true_divide g (L.expand_dims denom [| -1 |])
    | _ -> assert false)

let orthogonal ?m key ~n ~shape:sh =
  let m = Option.value m ~default:n in
  let big = Stdlib.max n m and small = Stdlib.min n m in
  let dt = C.float_dtype () in
  C.jit_region "_orthogonal" (sh, dt, n, m) [ key ] (function
    | [ key ] ->
        let z = C.normal key ~shape:(Array.append sh [| big; small |]) in
        let q, r =
          match Numpy.Linalg.qr z with
          | [ q; r ] -> (q, r)
          | _ -> failwith "orthogonal: qr arity"
        in
        let d = Numpy.Linalg.diagonal r in
        let signs = L.expand_dims (Numpy.Ufuncs.sign d) [| -2 |] in
        let x = Numpy.Ufuncs.multiply q signs in
        if n < m then NL.matrix_transpose x else x
    | _ -> assert false)

let multivariate_normal ?shape:shape_opt ?(method_ = "cholesky") ?dtype key
    ~mean ~cov =
  if not (List.mem method_ [ "svd"; "eigh"; "cholesky" ]) then
    raise
      (Py_exn.Value_error "method must be one of {'svd', 'eigh', 'cholesky'}");
  C.check_float_dtype "multivariate_normal" dtype;
  let mean_shape = shape mean in
  let rank = Array.length mean_shape in
  let event = mean_shape.(rank - 1) in
  let batch = Array.sub mean_shape 0 (rank - 1) in
  let sh = Option.value shape_opt ~default:batch in
  let float_dtype =
    match dtype with
    | Some d -> Dtypes.canonicalize_dtype d
    | None -> Dtypes.to_inexact_dtype (NL.result_type [ mean; cov ])
  in
  C.jit_region "_multivariate_normal" (sh, float_dtype, method_)
    [ key; mean; cov ] (function
    | [ key; mean; cov ] ->
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
                ("method must be one of {'svd', 'eigh', 'cholesky'}; got "
               ^ other)
        in
        let z =
          C.with_dtypes ~float_dtype (fun () ->
              C.normal key ~shape:(Array.append sh [| event |]))
        in
        let product = Numpy.Einsum.einsum "...ij,...j->...i" [ factor; z ] in
        Numpy.Ufuncs.add mean product
    | _ -> assert false)

let with_dt dtype f =
  match dtype with None -> f () | Some d -> C.with_dtypes ~float_dtype:d f

let maxwell ?dtype key ~shape =
  C.check_float_dtype "maxwell" dtype;
  with_dt dtype (fun () -> maxwell key ~shape)

let double_sided_maxwell ?dtype key ~loc ~scale ~shape =
  C.check_float_dtype "double_sided_maxwell" dtype;
  with_dt dtype (fun () -> double_sided_maxwell key ~loc ~scale ~shape)

let ball ?dtype ?p key ~d ~shape =
  C.check_float_dtype "gamma" dtype;
  with_dt dtype (fun () -> ball ?p key ~d ~shape)

let orthogonal ?dtype ?m key ~n ~shape =
  C.check_inexact_dtype "normal" dtype;
  with_dt dtype (fun () -> orthogonal ?m key ~n ~shape)
