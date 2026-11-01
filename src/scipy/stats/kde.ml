open Types
module L = Lax_api
module NL = Numpy.Lax_numpy
module U = Numpy.Ufuncs
module R = Numpy.Reductions
module LA = Numpy.Linalg
module JU = Numpy.Jnp_util

type t = {
  dataset : value;
  weights : value;
  neff : value;
  d : int;
  n : int;
  covariance : value;
  inv_cov : value;
}

let shape v = (Core.get_aval v).shape
let dtype_of v = (Core.get_aval v).dtype
let lit v x = Concrete (Ndarray.of_floats (dtype_of v) [||] [| x |])
let py_int2 = lazy (JU.py_int 2)

let gaussian_kde ?(bw_method = "scott") ?weights dataset =
  let dataset = NL.atleast_2d dataset in
  let sh = shape dataset in
  if Array.length sh <> 2 || Array.fold_left ( * ) 1 sh <= 1 then
    invalid_arg "`dataset` input should have multiple elements.";
  let d = sh.(0) and n = sh.(1) in
  let dataset, w =
    match weights with
    | Some w -> (
        match Dist_util.promote_dtypes [ dataset; w ] with
        | [ dataset; w ] ->
            let w = NL.atleast_1d w in
            (dataset, U.true_divide w (R.sum w))
        | _ -> assert false)
    | None -> (
        match Dist_util.promote_dtypes [ dataset ] with
        | [ dataset ] ->
            (dataset, L.full [| n |] (1.0 /. float_of_int n) (dtype_of dataset))
        | _ -> assert false)
  in
  let neff =
    U.true_divide (lit w 1.0) (R.sum (U.power w (Lazy.force py_int2)))
  in
  let exponent = -1.0 /. float_of_int (d + 4) in
  let factor =
    match bw_method with
    | "scott" -> U.power neff (lit neff exponent)
    | "silverman" ->
        let scaled =
          U.true_divide
            (U.multiply neff (lit neff (float_of_int (d + 2))))
            (lit neff 4.0)
        in
        U.power scaled (lit neff exponent)
    | other ->
        invalid_arg
          ("`bw_method` should be 'scott', 'silverman', a scalar, or a \
            callable: " ^ other)
  in
  let cov = NL.cov ~rowvar:true ~bias:false ~aweights:w dataset in
  let data_covariance = NL.atleast_2d cov in
  let data_inv_cov = LA.inv data_covariance in
  let covariance =
    U.multiply data_covariance (U.power factor (Lazy.force py_int2))
  in
  let inv_cov =
    U.true_divide data_inv_cov (U.power factor (Lazy.force py_int2))
  in
  { dataset; weights = w; neff; d; n; covariance; inv_cov }

let reshape_points t points =
  let points = NL.atleast_2d points in
  let sh = shape points in
  if sh.(0) <> t.d then
    if sh.(0) = 1 && sh.(1) = t.d then NL.reshape points [| t.d; 1 |]
    else
      invalid_arg
        (Printf.sprintf "points have dimension %d, dataset has dimension %d"
           sh.(0) t.d)
  else points

let kernel_eval in_log points values xi precision =
  Eager_jit.one ~jit:"_gaussian_kernel_eval" "scipy.stats.kde._gaussian_kernel_eval" in_log []
    [ points; values; xi; precision ] (fun _ vals ->
      JU.jit1 "_gaussian_kernel_eval" ~key:(string_of_bool in_log)
        (fun args ->
          match Dist_util.promote_dtypes args with
          | [ points; values; xi; precision ] ->
              let d = (shape points).(1) in
              let whitening = LA.cholesky ~symmetrize_input:false precision in
              let points = Numpy.Tensor_contractions.dot points whitening in
              let xi = Numpy.Tensor_contractions.dot xi whitening in
              let diag = NL.diag whitening in
              let summed = R.sum (U.log diag) in
              let two_pi = Numpy.Jnp_util.py_float (2.0 *. Float.pi) in
              let lg = U.log two_pi in
              let half_d = lit summed (0.5 *. float_of_int d) in
              let log_norm =
                U.subtract summed
                  (U.multiply half_d (NL.astype lg (dtype_of summed)))
              in
              let kernel x_test x_train y_train =
                let delta = U.subtract x_train x_test in
                let total = R.sum (U.square delta) in
                let arg =
                  U.subtract log_norm (U.multiply (lit total 0.5) total)
                in
                if in_log then U.add (U.log y_train) arg
                else U.multiply y_train (U.exp arg)
              in
              let reduced x =
                let mapped =
                  List.hd
                    (Interpreters.Batching.vmap
                       (function
                         | [ tr; y ] -> [ kernel x tr y ] | _ -> assert false)
                       [ Some 0; Some 0 ] [ points; values ])
                in
                if in_log then Ops.Special.logsumexp ~axis:[| 0 |] mapped
                else R.sum ~axis:[| 0 |] mapped
              in
              List.hd
                (Interpreters.Batching.vmap
                   (function [ x ] -> [ reduced x ] | _ -> assert false)
                   [ Some 0 ] [ xi ])
          | _ -> assert false)
        vals)

let values_of t = L.expand_dims t.weights [| 1 |]

let evaluate t points =
  let points = reshape_points t points in
  let result =
    kernel_eval false (NL.transpose t.dataset) (values_of t)
      (NL.transpose points) t.inv_cov
  in
  L.index_in_dim ~axis:1 ~keepdims:false result 0

let pdf = evaluate

let logpdf t points =
  let points = reshape_points t points in
  let result =
    kernel_eval true (NL.transpose t.dataset) (values_of t)
      (NL.transpose points) t.inv_cov
  in
  L.index_in_dim ~axis:1 ~keepdims:false result 0

let integrate_box_1d t low high =
  if t.d <> 1 then invalid_arg "integrate_box_1d() only handles 1D pdfs";
  let pyf = JU.py_float in
  Eager_jit.one ~jit:"integrate_box_1d" "scipy.stats.kde.integrate_box_1d" () []
    [ t.dataset; t.weights; t.covariance; pyf low; pyf high ]
    (fun _ vals ->
      match vals with
      | [ dataset; weights; covariance; low; high ] ->
          let sigma = NL.squeeze (U.sqrt covariance) in
          let low = NL.astype low (dtype_of dataset) in
          let high = NL.astype high (dtype_of dataset) in
          let low_z =
            NL.squeeze (U.true_divide (U.subtract low dataset) sigma)
          in
          let high_z =
            NL.squeeze (U.true_divide (U.subtract high dataset) sigma)
          in
          let cdf_high = Special.ndtr high_z in
          let cdf_low = Special.ndtr low_z in
          R.sum (U.multiply weights (U.subtract cdf_high cdf_low))
      | _ -> assert false)
