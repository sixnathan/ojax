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

let check_python_int v dt =
  match v with
  | T.Concrete nd
    when dt <> D.Bool
         && Dtypes.issubdtype dt `Integer
         && Dtypes.issubdtype (Ndarray.dtype nd) `Integer -> (
      match Ndarray.py_value nd with
      | Some x ->
          let info = Dtypes.iinfo dt in
          if x < info.imin || x > info.imax then
            raise
              (Py_exn.Overflow_error
                 (Printf.sprintf "Python integer %.0f out of bounds for %s" x
                    (D.numpy_name dt)))
      | None -> ())
  | _ -> ()

let array ?dtype ?(ndmin = 0) v =
  Option.iter (check_python_int v) dtype;
  let v = match dtype with Some d -> convert v d | None -> v in
  let n = ndim v in
  if ndmin > n then NL.expand_dims v (Array.init (ndmin - n) (fun i -> i))
  else v

let asarray ?dtype ?order v =
  NL.check_array_order order;
  array
    ?dtype:
      (Option.map
         (Dtypes.check_and_canonicalize_user_dtype ~fun_name:"asarray")
         dtype)
    v
