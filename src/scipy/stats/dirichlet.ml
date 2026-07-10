module DU = Dist_util
module C = Core
module T = Types
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module RED = Numpy.Reductions
module SP = Special
open DU

let is_simplex x =
  let x_sum = RED.sum ~axis:[| 0 |] x in
  UF.logical_and
    (RED.all ~axis:[| 0 |] (UF.greater x (sc x 0.0)))
    (UF.less (UF.abs (UF.subtract x_sum (sc x_sum 1.0))) (sc x_sum 1e-6))

let logpdf x alpha =
  match promote_dtypes [ x; alpha ] with
  | [ x; alpha ] ->
      let ash = shape alpha in
      if Array.length ash <> 1 then
        invalid_arg "dirichlet.logpdf: alpha must be one-dimensional";
      let one = sc x 1.0 in
      let x =
        if (shape x).(0) <> ash.(0) then
          NL.concatenate ~axis:0
            [ x; UF.subtract one (RED.sum ~axis:[| 0 |] ~keepdims:true x) ]
        else x
      in
      let normalize_term =
        UF.subtract (RED.sum (SP.gammaln alpha)) (SP.gammaln (RED.sum alpha))
      in
      let alpha =
        if ndim x > 1 then
          let target = Array.append ash (Array.make (ndim x - 1) 1) in
          C.bind1
            (T.Broadcast_in_dim { shape = target; dims = [| 0 |] })
            [ alpha ]
        else alpha
      in
      let log_probs =
        UF.subtract
          (RED.sum ~axis:[| 0 |] (SP.xlogy (UF.subtract alpha one) x))
          normalize_term
      in
      NL.where_ (is_simplex x) log_probs (ninf x)
  | _ -> assert false

let pdf x alpha = UF.exp (logpdf x alpha)
