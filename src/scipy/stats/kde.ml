open Types
module L = Lax_api
module NL = Numpy.Lax_numpy
module U = Numpy.Ufuncs
module R = Numpy.Reductions
module LA = Numpy.Linalg

type t = {
  dataset : value;
  weights : value;
  neff : float;
  d : int;
  n : int;
  factor : float;
  covariance : value;
  inv_cov : value;
}

let shape v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype

let host_scalar v =
  match v with
  | Concrete nd -> Ndarray.get_f nd [||]
  | Device b -> Ndarray.get_f (Pjrt.Buffer.to_host b) [||]
  | Tracer _ -> invalid_arg "gaussian_kde: bandwidth requires concrete weights"

let gaussian_kde ?(bw_method = "scott") ?weights dataset =
  let sh = shape dataset in
  if Array.length sh <> 2 then
    invalid_arg "`dataset` input should have multiple elements.";
  let d = sh.(0) and n = sh.(1) in
  if n < 2 then invalid_arg "`dataset` input should have multiple elements.";
  let dt = dtype_of dataset in
  let raw_weights =
    match weights with
    | Some w -> NL.astype w dt
    | None -> L.full [| n |] 1.0 dt
  in
  let total = R.sum raw_weights in
  let w = U.divide raw_weights (L.broadcast_in_dim total [| n |] [||]) in
  let neff = 1.0 /. host_scalar (R.sum (U.multiply w w)) in
  let dim = float_of_int d in
  let factor =
    match bw_method with
    | "scott" -> neff ** (-1.0 /. (dim +. 4.0))
    | "silverman" -> (neff *. (dim +. 2.0) /. 4.0) ** (-1.0 /. (dim +. 4.0))
    | other ->
        invalid_arg ("`bw_method` should be 'scott' or 'silverman': " ^ other)
  in
  let mean = NL.reshape (L.dot dataset w) [| d; 1 |] in
  let centered = U.subtract dataset (NL.broadcast_to mean sh) in
  let weighted =
    U.multiply centered (NL.broadcast_to (NL.reshape w [| 1; n |]) sh)
  in
  let raw_cov = L.dot weighted (L.transpose centered [| 1; 0 |]) in
  let denom = 1.0 -. host_scalar (R.sum (U.multiply w w)) in
  let data_cov = L.div raw_cov (L.full_like raw_cov denom) in
  let covariance = L.mul data_cov (L.full_like data_cov (factor *. factor)) in
  let inv_cov = LA.inv covariance in
  { dataset; weights = w; neff; d; n; factor; covariance; inv_cov }

let log_norm t =
  let dim = float_of_int t.d in
  let sign, logdet = LA.slogdet t.covariance in
  ignore sign;
  let two_pi = 8.0 *. Float.atan 1.0 in
  L.mul
    (L.add logdet (L.full_like logdet (dim *. log two_pi)))
    (L.full_like logdet 0.5)

let squared_mahalanobis t points =
  let m = (shape points).(1) in
  let expand_points =
    NL.broadcast_to (NL.reshape points [| t.d; m; 1 |]) [| t.d; m; t.n |]
  in
  let expand_data =
    NL.broadcast_to (NL.reshape t.dataset [| t.d; 1; t.n |]) [| t.d; m; t.n |]
  in
  let diff = U.subtract expand_points expand_data in
  let product = Numpy.Einsum.einsum "ij,jmn->imn" [ t.inv_cov; diff ] in
  R.sum ~axis:[| 0 |] (U.multiply diff product)

let as_points t points =
  let points = NL.astype points (dtype_of t.dataset) in
  match Array.length (shape points) with
  | 1 -> NL.reshape points [| t.d; (shape points).(0) / t.d |]
  | 2 -> points
  | _ -> invalid_arg "points must be at most two dimensional"

let evaluate t points =
  let points = as_points t points in
  let m = (shape points).(1) in
  let maha = squared_mahalanobis t points in
  let exponent = L.mul maha (L.full_like maha (-0.5)) in
  let kernel = U.exp exponent in
  let weights =
    NL.broadcast_to (NL.reshape t.weights [| 1; t.n |]) [| m; t.n |]
  in
  let total = R.sum ~axis:[| 1 |] (U.multiply weights kernel) in
  let norm = U.exp (L.neg (log_norm t)) in
  U.multiply total (L.broadcast_in_dim norm (shape total) [||])

let pdf = evaluate

let logpdf t points =
  let points = as_points t points in
  let m = (shape points).(1) in
  let maha = squared_mahalanobis t points in
  let exponent = L.mul maha (L.full_like maha (-0.5)) in
  let log_w =
    NL.broadcast_to (NL.reshape (U.log t.weights) [| 1; t.n |]) [| m; t.n |]
  in
  let combined = U.add exponent log_w in
  let summed = Ops.Special.logsumexp ~axis:[| 1 |] combined in
  U.subtract summed (L.broadcast_in_dim (log_norm t) (shape summed) [||])

let integrate_box_1d t low high =
  if t.d <> 1 then invalid_arg "integrate_box_1d() only handles 1D pdfs";
  let variance =
    L.index_in_dim ~keepdims:false
      (L.index_in_dim ~keepdims:false t.covariance 0)
      0
  in
  let sigma = L.sqrt variance in
  let flat = NL.reshape t.dataset [| t.n |] in
  let edge x =
    U.divide
      (U.subtract (L.full [| t.n |] x (dtype_of flat)) flat)
      (L.broadcast_in_dim sigma [| t.n |] [||])
  in
  let cdf z =
    let half = L.full_like z 0.5 in
    let root2 = L.full_like z (sqrt 2.0) in
    U.multiply half
      (L.sub (L.full_like z 1.0) (Core.bind1 Erf [ L.neg (L.div z root2) ]))
  in
  R.sum (U.multiply t.weights (U.subtract (cdf (edge high)) (cdf (edge low))))
