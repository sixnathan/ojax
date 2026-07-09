module C = Core
module T = Types
module D = Dtype

let get_aval = C.get_aval
let dtype v = (get_aval v).T.dtype
let bind1 = C.bind1

let convert v dt =
  if dtype v = dt then v else bind1 (T.Convert_element_type dt) [ v ]

let bool_ v = convert v D.Bool
let int32 v = convert v D.I32
let int64 v = convert v D.I64
let float32 v = convert v D.F32
let float64 v = convert v D.F64
let single = float32
let double = float64
let int_ = int64
let float_ = float64
