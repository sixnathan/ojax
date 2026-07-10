module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module LL = Lax.Linalg
module LN = Numpy.Linalg
module NL = Numpy.Lax_numpy
module UF = Numpy.Ufuncs
module TC = Numpy.Tensor_contractions
module IDX = Numpy.Indexing
module SORT = Numpy.Sorting

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let ndim v = Array.length (shape v)

let is_inexact = function
  | D.F32 | D.F64 | D.Complex64 | D.Complex128 -> true
  | _ -> false

let default_float () = Dtypes.default_float_dtype ()

let to_inexact v =
  if is_inexact (dtype v) then v else NL.astype v (default_float ())

let matrix_transpose_v x = NL.matrix_transpose x
let hconj x = UF.conj (matrix_transpose_v x)
let iota_idx n = NL.arange ~dtype:D.I32 (float_of_int n)

let cholesky ?(lower = false) a =
  let a = to_inexact a in
  let a = if lower then a else hconj a in
  let l = LL.cholesky a in
  if lower then l else hconj l

let cho_factor ?(lower = false) a = (cholesky ~lower a, lower)

let cho_solve (c, lower) b =
  let c = to_inexact c and b = to_inexact b in
  let sc = shape c in
  let n = sc.(Array.length sc - 1) in
  let bvec = ndim b = 1 in
  let bm = if bvec then NL.reshape b [| n; 1 |] else b in
  let x1 =
    LL.triangular_solve ~left_side:true ~lower ~transpose_a:(not lower)
      ~conjugate_a:(not lower) c bm
  in
  let x2 =
    LL.triangular_solve ~left_side:true ~lower ~transpose_a:lower
      ~conjugate_a:lower c x1
  in
  if bvec then NL.reshape x2 [| n |] else x2

let det a = LN.det a
let inv a = LN.inv a

let lu ?(permute_l = false) a =
  let a = to_inexact a in
  let dt = dtype a in
  let sa = shape a in
  let m = sa.(0) and n = sa.(1) in
  let k = min m n in
  let lu_, _piv, permutation = LL.lu a in
  let perm_row = NL.expand_dims permutation [| 0 |] in
  let iota = NL.arange ~dtype:(dtype permutation) (float_of_int m) in
  let iota_col = NL.expand_dims iota [| 1 |] in
  let p = NL.astype (UF.equal perm_row iota_col) dt in
  let kcols = iota_idx k in
  let l_lower = IDX.take ~axis:1 (NL.tril ~k:(-1) lu_) kcols in
  let l = UF.add l_lower (NL.eye ~m:k ~dtype:dt m) in
  let u = IDX.take ~axis:0 (NL.triu lu_) kcols in
  if permute_l then [ TC.matmul p l; u ] else [ p; l; u ]

let lu_factor a =
  let a = to_inexact a in
  let lu_, piv, _ = LL.lu a in
  (lu_, piv)

let lu_solve ?(trans = 0) (lu_, piv) b =
  let lu_ = to_inexact lu_ and b = to_inexact b in
  let sa = shape lu_ in
  let m = sa.(Array.length sa - 2) in
  let perm = LL.lu_pivots_to_permutation ~permutation_size:m piv in
  let bvec = ndim b = 1 in
  let x = if bvec then NL.reshape b [| m; 1 |] else b in
  let out =
    if trans = 0 then begin
      let x = IDX.take ~axis:0 x perm in
      let x =
        LL.triangular_solve ~left_side:true ~lower:true ~unit_diagonal:true lu_
          x
      in
      LL.triangular_solve ~left_side:true ~lower:false lu_ x
    end
    else if trans = 1 || trans = 2 then begin
      let conj = trans = 2 in
      let x =
        LL.triangular_solve ~left_side:true ~lower:false ~transpose_a:true
          ~conjugate_a:conj lu_ x
      in
      let x =
        LL.triangular_solve ~left_side:true ~lower:true ~unit_diagonal:true
          ~transpose_a:true ~conjugate_a:conj lu_ x
      in
      IDX.take ~axis:0 x (SORT.argsort perm)
    end
    else invalid_arg (Printf.sprintf "lu_solve: invalid trans value %d" trans)
  in
  if bvec then NL.reshape out [| m |] else out

let qr ?(mode = "full") ?(pivoting = false) a =
  if pivoting then
    invalid_arg "scipy.linalg.qr: pivoting=True not supported in ojax";
  let full_matrices =
    match mode with
    | "full" | "r" -> true
    | "economic" -> false
    | _ ->
        invalid_arg
          (Printf.sprintf "Unsupported QR decomposition mode '%s'" mode)
  in
  let a = to_inexact a in
  let q, r = LL.qr ~full_matrices a in
  if mode = "r" then [ r ] else [ q; r ]

let solve ?(lower = false) ?(assume_a = "gen") a b =
  (match assume_a with
  | "gen" | "sym" | "her" | "pos" -> ()
  | _ ->
      invalid_arg
        (Printf.sprintf
           "Expected assume_a to be one of ['gen', 'sym', 'her', 'pos']; got \
            '%s'"
           assume_a));
  if assume_a <> "pos" then LN.solve a b else cho_solve (cho_factor ~lower a) b

let solve_triangular ?(trans = 0) ?(lower = false) ?(unit_diagonal = false) a b
    =
  let transpose_a, conjugate_a =
    match trans with
    | 0 -> (false, false)
    | 1 -> (true, false)
    | 2 -> (true, true)
    | _ -> invalid_arg (Printf.sprintf "Invalid 'trans' value %d" trans)
  in
  let a = to_inexact a and b = to_inexact b in
  let sa = shape a in
  let n = sa.(Array.length sa - 1) in
  let bvec = ndim b = 1 in
  let bm = if bvec then NL.reshape b [| n; 1 |] else b in
  let x =
    LL.triangular_solve ~left_side:true ~lower ~transpose_a ~conjugate_a
      ~unit_diagonal a bm
  in
  if bvec then NL.reshape x [| n |] else x
