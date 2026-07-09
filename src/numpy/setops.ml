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

let in1d ~invert ar1 ar2 =
  let dt = NL.result_type [ ar1; ar2 ] in
  let arr1 = ravel (astype ar1 dt) in
  let arr2 = ravel (astype ar2 dt) in
  let n1 = size arr1 and n2 = size arr2 in
  if n1 = 0 || n2 = 0 then
    if invert then AC.ones ~dtype:D.Bool [| n1 |]
    else AC.zeros ~dtype:D.Bool [| n1 |]
  else begin
    let a = NL.broadcast_to (NL.reshape arr1 [| n1; 1 |]) [| n1; n2 |] in
    let b = NL.broadcast_to (NL.reshape arr2 [| 1; n2 |]) [| n1; n2 |] in
    if invert then RED.all ~axis:[| 1 |] (bind1 T.Ne [ a; b ])
    else RED.any ~axis:[| 1 |] (bind1 T.Eq [ a; b ])
  end

let isin ?(assume_unique = false) ?(invert = false) ?(method_ = "auto") element
    test_elements =
  ignore assume_unique;
  (match method_ with
  | "auto" | "compare_all" -> ()
  | _ -> failwith "setops.isin: only 'auto'/'compare_all' method is implemented");
  let result = in1d ~invert element test_elements in
  NL.reshape result (shape element)
