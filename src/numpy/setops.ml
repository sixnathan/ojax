module Nd = Ndarray
module C = Core
module T = Types
module D = Dtype
module NL = Lax_numpy
module RED = Reductions
module AC = Array_creation

let get_aval = C.get_aval
let shape v = (get_aval v).T.shape
let dtype v = (get_aval v).T.dtype
let size v = Array.fold_left ( * ) 1 (shape v)
let bind1 = C.bind1
let ravel v = NL.reshape v [| size v |]
let astype v dt = if dtype v = dt then v else NL.astype v dt

let in1d_body ~invert ar1 ar2 =
  let dt = NL.result_type [ ar1; ar2 ] in
  let arr1 = ravel (astype ar1 dt) in
  let arr2 = ravel (astype ar2 dt) in
  let n1 = size arr1 and n2 = size arr2 in
  if n1 = 0 || n2 = 0 then
    if invert then AC.ones ~dtype:D.Bool [| n1 |]
    else AC.zeros ~dtype:D.Bool [| n1 |]
  else begin
    let a = NL.expand_dims arr1 [| 1 |] in
    let b = NL.expand_dims arr2 [| 0 |] in
    if invert then RED.all ~axis:[| -1 |] (Ufuncs.not_equal a b)
    else RED.any ~axis:[| -1 |] (Ufuncs.equal a b)
  end

let in1d ~invert ar1 ar2 =
  Jnp_util.jit1 "_in1d"
    ~key:(Printf.sprintf "%b,auto,false" invert)
    (function
      | [ ar1; ar2 ] -> in1d_body ~invert ar1 ar2
      | _ -> invalid_arg "in1d: arity")
    [ ar1; ar2 ]

let isin ?(assume_unique = false) ?(invert = false) ?(method_ = "auto") element
    test_elements =
  ignore assume_unique;
  (match method_ with
  | "auto" | "compare_all" -> ()
  | "binary_search" | "sort" ->
      Py_exn.not_implemented_error
        "setops.isin: only 'auto'/'compare_all' method is implemented"
  | m ->
      Py_exn.value_error
        (Printf.sprintf
           "method='%s' is not implemented; options are 'compare_all', \
            'binary_search', 'sort', and 'auto'"
           m));
  let result = in1d ~invert element test_elements in
  NL.reshape result (shape element)
