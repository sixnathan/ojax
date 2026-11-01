module UF = Numpy.Ufuncs
module L = Lax_api
open Dist_util

let promote2 a b =
  match promote [ a; b ] with [ a; b ] -> (a, b) | _ -> assert false

let logpdf x c =
  let x, c = promote2 x c in
  let cc = UF.multiply c c in
  let log_two_pi = UF.log (py_float (2.0 *. Float.pi)) in
  let first = UF.log (UF.subtract (sc c 1.0) cc) in
  let first_less =
    UF.subtract first (Numpy.Lax_numpy.astype log_two_pi (dtype first))
  in
  let denom =
    UF.subtract
      (UF.add (sc c 1.0) (UF.multiply c c))
      (UF.multiply (UF.multiply (sc c 2.0) c) (UF.cos x))
  in
  let inner = UF.subtract first_less (UF.log denom) in
  let in_support =
    UF.bitwise_and (L.ge x (sc x 0.0)) (L.le x (sc x (Float.pi *. 2.0)))
  in
  let valid_c = UF.bitwise_and (L.gt c (sc c 0.0)) (L.lt c (sc c 1.0)) in
  where_or_nan valid_c (where_or_ninf in_support inner)

let pdf x c = L.exp (logpdf x c)
