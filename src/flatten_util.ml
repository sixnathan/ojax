module T = Types
module C = Core
module D = Dtype
module LN = Numpy.Lax_numpy
module AC = Numpy.Array_creation

let aval v = C.get_aval v
let dtype_of v = (aval v).T.dtype
let shape_of v = (aval v).T.shape
let size_of v = Array.fold_left ( * ) 1 (shape_of v)

let unravel_single_dtype sizes shapes flat =
  let chunks = Lax_api.split flat (Array.of_list sizes) 0 in
  List.map2 (fun chunk shape -> LN.reshape chunk shape) chunks shapes

let unravel_multi sizes shapes from_dtypes flat =
  let chunks = Lax_api.split flat (Array.of_list sizes) 0 in
  List.map2
    (fun (chunk, shape) dt -> LN.astype (LN.reshape chunk shape) dt)
    (List.combine chunks shapes)
    from_dtypes

let ravel_list leaves =
  match leaves with
  | [] -> (AC.full ~dtype:D.F32 [| 0 |] 0.0, fun _ -> [])
  | _ ->
      let from_dtypes = List.map dtype_of leaves in
      let to_dtype = LN.result_type leaves in
      let sizes = List.map size_of leaves in
      let shapes = List.map shape_of leaves in
      let all_same = List.for_all (fun d -> d = to_dtype) from_dtypes in
      if all_same then
        let raveled =
          Lax_api.concatenate
            (List.map2 (fun e s -> Lax_api.reshape e [| s |]) leaves sizes)
            0
        in
        (raveled, unravel_single_dtype sizes shapes)
      else
        let raveled =
          Lax_api.concatenate
            (List.map2
               (fun e s -> LN.reshape (LN.astype e to_dtype) [| s |])
               leaves sizes)
            0
        in
        (raveled, unravel_multi sizes shapes from_dtypes)

let ravel_pytree tree =
  let leaves, treedef = Tree_util.tree_flatten tree in
  let flat, unravel_list = ravel_list leaves in
  (flat, fun v -> Tree_util.tree_unflatten treedef (unravel_list v))
