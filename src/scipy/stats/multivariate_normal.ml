module DU = Dist_util
module UF = Numpy.Ufuncs
module NL = Numpy.Lax_numpy
module RED = Numpy.Reductions
module LL = Lax.Linalg
open DU

let two_pi = 2.0 *. Float.pi

let promote3 x mean cov =
  match promote_dtypes [ x; mean; cov ] with
  | [ x; mean; cov ] -> (x, mean, cov)
  | _ -> assert false

let last_axis_sq y =
  let s = shape y in
  RED.sum ~axis:[| Array.length s - 1 |] (UF.square y)

let logpdf x mean cov =
  let x, mean, cov = promote3 x mean cov in
  let ms = shape mean in
  if Array.length ms = 0 then
    UF.subtract
      (UF.divide
         (UF.multiply (sc x (-0.5)) (UF.square (UF.subtract x mean)))
         cov)
      (UF.multiply (sc x 0.5) (UF.add (UF.log (sc x two_pi)) (UF.log cov)))
  else begin
    let n = ms.(Array.length ms - 1) in
    let cs = shape cov in
    if Array.length cs = 0 then begin
      let y = UF.subtract x mean in
      UF.subtract
        (UF.divide (UF.multiply (sc x (-0.5)) (last_axis_sq y)) cov)
        (UF.multiply
           (sc x (float_of_int n /. 2.0))
           (UF.add (UF.log (sc x two_pi)) (UF.log cov)))
    end
    else begin
      let cn = Array.length cs in
      if cn < 2 || cs.(cn - 1) <> n || cs.(cn - 2) <> n then
        failwith "multivariate_normal.logpdf got incompatible shapes";
      let l = LL.cholesky cov in
      let diff = UF.subtract x mean in
      let ds = shape diff in
      let ndf = Array.length ds in
      let batch = Array.fold_left ( * ) 1 (Array.sub ds 0 (ndf - 1)) in
      let solved =
        LL.triangular_solve ~lower:true ~transpose_a:true l
          (NL.reshape diff [| batch; n |])
      in
      let y = NL.reshape solved ds in
      UF.subtract
        (UF.subtract
           (UF.multiply (sc x (-0.5)) (last_axis_sq y))
           (UF.multiply (sc x (float_of_int n /. 2.0)) (UF.log (sc x two_pi))))
        (RED.sum ~axis:[| -1 |] (UF.log (NL.diag l)))
    end
  end

let pdf x mean cov = UF.exp (logpdf x mean cov)
