open Types
module B = Interpreters.Batching

let collective kind axis_name v =
  ignore (B.find_axis axis_name);
  Core.bind1 (Collective { kind; axis_name }) [ v ]

let psum v ~axis_name = collective Psum axis_name v
let pmax v ~axis_name = collective Pmax axis_name v
let pmin v ~axis_name = collective Pmin axis_name v

let axis_size ~axis_name =
  let _, _, size = B.find_axis axis_name in
  size

let pmean v ~axis_name =
  let total = psum v ~axis_name in
  Lax_api.div total
    (Lax_api.full_like total (float_of_int (axis_size ~axis_name)))

let all_gather ?(axis = 0) v ~axis_name =
  collective (All_gather { axis; size = axis_size ~axis_name }) axis_name v

let pbroadcast ?(source = 0) v ~axis_name =
  collective (Pbroadcast source) axis_name v

let ppermute v ~axis_name ~perm =
  collective (Ppermute (Array.of_list perm)) axis_name v

let axis_index ~axis_name =
  ignore (B.find_axis axis_name);
  Core.bind1 (Collective { kind = Axis_index; axis_name }) []
