module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
open Dist_util

let promote2 a b =
  match promote [ a; b ] with [ a; b ] -> (a, b) | _ -> assert false

let logpdf x c =
  let x, c = promote2 x c in
  let zero_c = sc c 0.0 in
  let one_c = sc c 1.0 in
  let zero_x = sc x 0.0 in
  let two_pi = sc x (2.0 *. Float.pi) in
  let cc = UF.multiply c c in
  let inner =
    UF.subtract
      (UF.subtract
         (UF.log (UF.subtract one_c cc))
         (sc x (Float.log (2.0 *. Float.pi))))
      (UF.log
         (UF.add
            (UF.subtract one_c
               (UF.multiply (sc x 2.0) (UF.multiply c (UF.cos x))))
            cc))
  in
  let in_support =
    UF.logical_and (UF.greater_equal x zero_x) (UF.less_equal x two_pi)
  in
  let valid_c = UF.logical_and (UF.greater c zero_c) (UF.less c one_c) in
  NL.where_ valid_c (NL.where_ in_support inner (ninf x)) (nan c)

let pdf x c = UF.exp (logpdf x c)
