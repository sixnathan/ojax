open Types
module L = Lax_api
module NL = Numpy.Lax_numpy
module U = Numpy.Ufuncs
module R = Numpy.Reductions
module LA = Numpy.Linalg

type t = {
  dataset : value;
  weights : value;
  neff : value;
  d : int;
  n : int;
  factor : value;
  covariance : value;
  inv_cov : value;
}

let shape v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype

let gaussian_kde ?(bw_method = "scott") ?weights dataset =
  let sh = shape dataset in
  if Array.length sh <> 2 then
    invalid_arg "`dataset` input should have multiple elements.";
  let d = sh.(0) and n = sh.(1) in
  if n < 2 then invalid_arg "`dataset` input should have multiple elements.";
  let dt = dtype_of dataset in
  let w =
    match weights with
    | Some raw ->
        let raw = NL.astype raw dt in
        U.divide raw (L.broadcast_in_dim (R.sum raw) [| n |] [||])
    | None -> L.full [| n |] (1.0 /. float_of_int n) dt
  in
  let sum_sq = R.sum (Core.bind1 (Integer_pow 2) [ w ]) in
  let neff = L.div (L.full_like sum_sq 1.0) sum_sq in
  let dim = float_of_int d in
  let exponent = L.full_like neff (-1.0 /. (dim +. 4.0)) in
  let factor =
    match bw_method with
    | "scott" -> L.pow neff exponent
    | "silverman" ->
        L.pow (L.mul neff (L.full_like neff ((dim +. 2.0) /. 4.0))) exponent
    | other ->
        invalid_arg ("`bw_method` should be 'scott' or 'silverman': " ^ other)
  in
  let data_cov = NL.atleast_2d (NL.cov ~aweights:w dataset) in
  let data_inv = LA.inv data_cov in
  let squared = Core.bind1 (Integer_pow 2) [ factor ] in
  let spread v = L.broadcast_in_dim squared (shape v) [||] in
  let covariance = L.mul data_cov (spread data_cov) in
  let inv_cov = L.div data_inv (spread data_inv) in
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
  let dt = dtype_of t.dataset in
  let bound v = L.full [||] v dt in
  match
    Jaxpr.named_jit ~name:"integrate_box_1d"
      (function
        | [ _neff; dataset; weights; covariance; _inv_cov; low; high ] ->
            let sigma = NL.squeeze (L.sqrt covariance) in
            let edge x = NL.squeeze (U.divide (U.subtract x dataset) sigma) in
            let lower = edge low in
            let upper = edge high in
            let ndtr_upper = Special.ndtr upper in
            let ndtr_lower = Special.ndtr lower in
            [ R.sum (U.multiply weights (U.subtract ndtr_upper ndtr_lower)) ]
        | _ -> invalid_arg "integrate_box_1d: arity")
      [
        t.neff;
        t.dataset;
        t.weights;
        t.covariance;
        t.inv_cov;
        bound low;
        bound high;
      ]
  with
  | [ out ] -> out
  | _ -> invalid_arg "integrate_box_1d: arity"
