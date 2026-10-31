open Types

type zero = { z_aval : aval }

let zeros_like_aval a =
  let n = Array.fold_left ( * ) 1 a.shape in
  Concrete (Ndarray.of_floats a.dtype a.shape (Array.make n 0.0))

let zeros_like_value v = zeros_like_aval (Core.get_aval v)
let symbolic_zeros : (Dtype.t * int array, value) Hashtbl.t = Hashtbl.create 16

let symbolic_zero_of_aval a =
  let key = (a.dtype, a.shape) in
  match Hashtbl.find_opt symbolic_zeros key with
  | Some z -> z
  | None ->
      let z = zeros_like_aval a in
      Hashtbl.replace symbolic_zeros key z;
      z

let symbolic_zero_like_value v = symbolic_zero_of_aval (Core.get_aval v)

let is_symbolic_zero = function
  | Concrete nd as v -> (
      match
        Hashtbl.find_opt symbolic_zeros (Ndarray.dtype nd, Ndarray.shape nd)
      with
      | Some z -> z == v
      | None -> false)
  | Device _ | Tracer _ -> false

let clear_symbolic_zeros () = Hashtbl.reset symbolic_zeros
let instantiate z = zeros_like_aval z.z_aval
let add_jaxvals x y = Core.bind1 Add [ x; y ]
