module NL = Numpy.Lax_numpy

let trapezoid ?x ?dx ?axis y = NL.trapezoid ?x ?dx ?axis y

let trapezoid ?x ?dx ?axis a0 =
  Eager_jit.one "scipy.integrate.trapezoid" (dx, axis) [ x ] [ a0 ]
    (fun opts vals ->
      match (opts, vals) with
      | [ x ], [ a0 ] -> trapezoid ?x ?dx ?axis a0
      | _ -> assert false)
