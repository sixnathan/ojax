open Types

type t = aval

let make ?(weak_type = false) shape dtype =
  { shape = Array.copy shape; dtype; weak_type }

let of_value = Core.get_aval
let like = Core.get_aval
let shape (s : t) = s.shape
let dtype (s : t) = s.dtype
let weak_type (s : t) = s.weak_type
let ndim (s : t) = Array.length s.shape
let size (s : t) = Array.fold_left ( * ) 1 s.shape

let update ?shape ?dtype ?weak_type (s : t) =
  make
    ?weak_type:(Some (Option.value weak_type ~default:s.weak_type))
    (Option.value shape ~default:s.shape)
    (Option.value dtype ~default:s.dtype)

let equal (a : t) (b : t) = a = b
let hash (s : t) = Hashtbl.hash (s.shape, s.dtype, s.weak_type)

let shape_string s =
  match Array.to_list s with
  | [] -> "()"
  | [ d ] -> Printf.sprintf "(%d,)" d
  | ds -> "(" ^ String.concat ", " (List.map string_of_int ds) ^ ")"

let to_string (s : t) =
  Printf.sprintf "ShapeDtypeStruct(shape=%s, dtype=%s%s)" (shape_string s.shape)
    (Ref_state.dtype_name s.dtype)
    (if s.weak_type then ", weak_type=True" else "")

let traced_aval (s : t) = { s with dtype = Dtypes.canonicalize_dtype s.dtype }
