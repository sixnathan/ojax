module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module RED = Numpy.Reductions
module AC = Numpy.Array_creation
module TC = Numpy.Tensor_contractions
module SL = Linalg
module NLA = Numpy.Linalg
module SP = Special
module OPS = Ops.Special
open DU

type bw_method = Scott | Silverman | Scalar of float

type t = {
  neff : Types.value;
  dataset : Types.value;
  weights : Types.value;
  covariance : Types.value;
  inv_cov : Types.value;
}

let two_pi = 2.0 *. Float.pi
let dataset t = t.dataset
let weights t = t.weights
let covariance t = t.covariance
let inv_cov t = t.inv_cov
let neff t = t.neff
let d t = (shape t.dataset).(0)
let n t = (shape t.dataset).(1)

let promote1 v =
  match promote_dtypes [ v ] with [ v ] -> v | _ -> assert false

let promote2 a b =
  match promote_dtypes [ a; b ] with [ a; b ] -> (a, b) | _ -> assert false

let weighted_cov x w =
  let w_sum = RED.sum w in
  let w_row = NL.expand_dims w [| 0 |] in
  let avg = UF.divide (RED.sum ~axis:[| 1 |] (UF.multiply x w_row)) w_sum in
  let xc = UF.subtract x (NL.expand_dims avg [| 1 |]) in
  let fact = UF.subtract w_sum (UF.divide (RED.sum (UF.square w)) w_sum) in
  let xcw = UF.multiply xc w_row in
  NL.atleast_2d (UF.divide (TC.matmul xc (NL.transpose xcw)) fact)

let make ?(bw_method = Scott) ?weights dataset =
  let dataset = NL.atleast_2d dataset in
  let sh = shape dataset in
  let dd = sh.(0) and nn = sh.(1) in
  let dataset, w =
    match weights with
    | Some w ->
        let dataset, w = promote2 dataset w in
        let w = NL.atleast_1d w in
        (dataset, UF.divide w (RED.sum w))
    | None ->
        let dataset = promote1 dataset in
        ( dataset,
          AC.full ~dtype:(dtype dataset) [| nn |] (1.0 /. float_of_int nn) )
  in
  let neff = UF.divide (sc dataset 1.0) (RED.sum (UF.square w)) in
  let df = float_of_int (dd + 4) in
  let factor =
    match bw_method with
    | Scott -> UF.power neff (sc neff (-1.0 /. df))
    | Silverman ->
        UF.power
          (UF.multiply neff (sc neff (float_of_int (dd + 2) /. 4.0)))
          (sc neff (-1.0 /. df))
    | Scalar s -> sc dataset s
  in
  let data_covariance = weighted_cov dataset w in
  let data_inv_cov = NLA.inv data_covariance in
  let f2 = UF.square factor in
  {
    neff;
    dataset;
    weights = w;
    covariance = UF.multiply data_covariance f2;
    inv_cov = UF.divide data_inv_cov f2;
  }

let reshape_points t points =
  let points = NL.atleast_2d points in
  let sh = shape points in
  let pd = sh.(0) and pm = sh.(1) in
  let selfd = d t in
  if pd <> selfd then
    if pd = 1 && pm = selfd then NL.reshape points [| selfd; 1 |]
    else
      failwith
        (Printf.sprintf "points have dimension %d, dataset has dimension %d" pd
           selfd)
  else points

let gaussian_kernel_eval ~in_log ~points ~w ~xi ~precision =
  let points = promote1 points and xi = promote1 xi in
  let precision = promote1 precision and w = promote1 w in
  let dd = (shape points).(1) in
  if (shape xi).(1) <> dd then
    failwith "points and xi must have same trailing dim";
  let ps = shape precision in
  if Array.length ps <> 2 || ps.(0) <> dd || ps.(1) <> dd then
    failwith "precision matrix must match data dims";
  let whitening = SL.cholesky ~lower:true precision in
  let points = TC.matmul points whitening in
  let xi = TC.matmul xi whitening in
  let log_norm =
    UF.subtract
      (RED.sum (UF.log (NL.diag whitening)))
      (sc points (0.5 *. float_of_int dd *. Float.log two_pi))
  in
  let mm = (shape xi).(0) and nn = (shape points).(0) in
  let xi3 = NL.reshape xi [| mm; 1; dd |] in
  let pts3 = NL.reshape points [| 1; nn; dd |] in
  let diff = UF.subtract xi3 pts3 in
  let sq = RED.sum ~axis:[| 2 |] (UF.square diff) in
  let arg = UF.subtract log_norm (UF.multiply (sc sq 0.5) sq) in
  let w_row = NL.expand_dims w [| 0 |] in
  if in_log then OPS.logsumexp ~axis:[| 1 |] (UF.add arg (UF.log w_row))
  else RED.sum ~axis:[| 1 |] (UF.multiply (UF.exp arg) w_row)

let evaluate t points =
  let points = reshape_points t points in
  gaussian_kernel_eval ~in_log:false ~points:(NL.transpose t.dataset)
    ~w:t.weights ~xi:(NL.transpose points) ~precision:t.inv_cov

let pdf = evaluate

let logpdf t x =
  let x = reshape_points t x in
  gaussian_kernel_eval ~in_log:true ~points:(NL.transpose t.dataset)
    ~w:t.weights ~xi:(NL.transpose x) ~precision:t.inv_cov

let gaussian_kernel_convolve ~chol ~norm ~target ~w ~mean =
  let diff = UF.subtract target (NL.expand_dims mean [| 1 |]) in
  let alpha = SL.cho_solve chol diff in
  let arg =
    UF.multiply (sc diff 0.5) (RED.sum ~axis:[| 0 |] (UF.multiply diff alpha))
  in
  UF.multiply norm (RED.sum (UF.multiply (UF.exp (UF.negative arg)) w))

let cho_norm chol dd =
  let coef = Float.pow (sqrt two_pi) (float_of_int dd) in
  let pd = RED.prod (NL.diag chol) in
  let norm0 = UF.multiply (sc pd coef) pd in
  UF.divide (sc norm0 1.0) norm0

let integrate_gaussian t mean cov =
  let mean = NL.atleast_1d (NL.squeeze mean) in
  let cov = NL.atleast_2d cov in
  let dd = d t in
  if shape mean <> [| dd |] then
    failwith (Printf.sprintf "mean does not have dimension %d" dd);
  if shape cov <> [| dd; dd |] then
    failwith (Printf.sprintf "covariance does not have dimension %d" dd);
  let chol, lower = SL.cho_factor (UF.add t.covariance cov) in
  let norm = cho_norm chol dd in
  gaussian_kernel_convolve ~chol:(chol, lower) ~norm ~target:t.dataset
    ~w:t.weights ~mean

let integrate_box_1d t low high =
  if d t <> 1 then failwith "integrate_box_1d() only handles 1D pdfs";
  let sigma = NL.squeeze (UF.sqrt t.covariance) in
  let f v =
    NL.squeeze (UF.divide (UF.subtract (sc t.dataset v) t.dataset) sigma)
  in
  RED.sum
    (UF.multiply t.weights (UF.subtract (SP.ndtr (f high)) (SP.ndtr (f low))))

let integrate_kde self other =
  if d other <> d self then failwith "KDEs are not the same dimensionality";
  let dd = d self in
  let chol, lower = SL.cho_factor (UF.add self.covariance other.covariance) in
  let norm = cho_norm chol dd in
  let sm, lg = if n self < n other then (self, other) else (other, self) in
  let cols = NL.split ~axis:1 sm.dataset (NL.Count (n sm)) in
  let results =
    List.map
      (fun col ->
        let mean = NL.squeeze ~axis:[| 1 |] col in
        gaussian_kernel_convolve ~chol:(chol, lower) ~norm ~target:lg.dataset
          ~w:lg.weights ~mean)
      cols
  in
  RED.sum (UF.multiply (NL.stack results) sm.weights)

let resample ?(shape = [||]) _t _key =
  ignore shape;
  failwith
    "gaussian_kde.resample requires random.multivariate_normal and weighted \
     random.choice (M5 row 106); not yet available"
