module C = Core
module T = Types
module D = Dtype
module NL = Lax_numpy

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let shape v = (get_aval v).T.shape
let ndim v = Array.length (shape v)
let bind1 = C.bind1

let convert v dt =
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let array ?dtype ?(ndmin = 0) v =
  let v = match dtype with Some d -> convert v d | None -> v in
  let n = ndim v in
  if ndmin > n then NL.expand_dims v (Array.init (ndmin - n) (fun i -> i))
  else v

let asarray ?dtype v = array ?dtype v
