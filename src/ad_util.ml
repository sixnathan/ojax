open Types

type zero = { z_aval : aval }

let zeros_like_aval a =
  let n = Array.fold_left ( * ) 1 a.shape in
  Concrete (Ndarray.of_floats a.dtype a.shape (Array.make n 0.0))

let zeros_like_value v = zeros_like_aval (Core.get_aval v)
let instantiate z = zeros_like_aval z.z_aval
let add_jaxvals x y = Core.bind1 Add [ x; y ]
