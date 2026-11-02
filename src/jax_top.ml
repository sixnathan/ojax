open Types

type array_ = value
type ref_ = Ref.t
type shape_dtype_struct = Shape_dtype_struct.t

let shape_dtype_struct = Shape_dtype_struct.make

type p = {
  partitions : Spmd.partition_spec;
  unreduced : string list;
  reduced : string list;
}

let partition ?(unreduced = []) ?(reduced = []) partitions =
  { partitions; unreduced; reduced }

let py_set names =
  "{" ^ String.concat ", " (List.map (Printf.sprintf "'%s'") names) ^ "}"

let p_to_string p =
  let axis = function
    | Spmd.Named name -> "'" ^ name ^ "'"
    | Spmd.Unconstrained -> "None"
    | Spmd.Axes [ one ] -> "('" ^ one ^ "',)"
    | Spmd.Axes names ->
        "(" ^ String.concat ", " (List.map (Printf.sprintf "'%s'") names) ^ ")"
  in
  let items = List.map axis p.partitions in
  let body =
    match items with [ only ] -> only ^ "," | _ -> String.concat ", " items
  in
  let extras =
    (if p.unreduced = [] then [] else [ "unreduced=" ^ py_set p.unreduced ])
    @ if p.reduced = [] then [] else [ "reduced=" ^ py_set p.reduced ]
  in
  match extras with
  | [] -> "P(" ^ body ^ ")"
  | _ -> "P(" ^ body ^ ", " ^ String.concat ", " extras ^ ")"

module Ad = struct
  type grad_value = Grad_value
  type grad_ref = Grad_ref
  type dont_want = Dont_want
  type didnt_want = Didnt_want
end

module Checkpoint_policies = struct
  let everything_saveable = Checkpoint.everything_saveable
  let nothing_saveable = Checkpoint.nothing_saveable
  let dots_saveable = Checkpoint.dots_saveable
  let checkpoint_dots = Checkpoint.checkpoint_dots

  let dots_with_no_batch_dims_saveable =
    Checkpoint.dots_with_no_batch_dims_saveable

  let checkpoint_dots_with_no_batch_dims =
    Checkpoint.checkpoint_dots_with_no_batch_dims

  let save_anything_except_these_names =
    Checkpoint.save_anything_except_these_names

  let save_any_names_but_these = Checkpoint.save_any_names_but_these
  let save_only_these_names = Checkpoint.save_only_these_names
  let save_from_both_policies = Checkpoint.save_from_both_policies
end

module Memory = struct
  type space = Device | Host | Any

  let to_string = function
    | Device -> "MemorySpace.Device"
    | Host -> "MemorySpace.Host"
    | Any -> "MemorySpace.Any"
end

let auto_pcast = Config.auto_pcast

let remove_size_one_mesh_axis_from_type =
  Config.remove_size_one_mesh_axis_from_type

let make_user_context = Config.make_user_context
let config_read = Config.read
let config_update = Config.update
let ds = Numpy.Indexing.ds
let io_callback = Custom_derivatives.io_callback

type primal_tangent_dtype = {
  primal_dtype : Dtype.t;
  tangent_dtype : Dtype.t;
  name : string;
}

let short_dtype_name d =
  let replace_prefix prefix by s =
    let n = String.length prefix in
    if String.length s >= n && String.sub s 0 n = prefix then
      by ^ String.sub s n (String.length s - n)
    else s
  in
  Ref_state.dtype_name d |> replace_prefix "float" "f"
  |> replace_prefix "uint" "u" |> replace_prefix "int" "i"
  |> replace_prefix "complex" "c"

let primal_tangent_dtype ?name primal_dtype tangent_dtype =
  let name =
    match name with
    | Some n when n <> "" -> n
    | _ ->
        Printf.sprintf "PrimalTangentDType{%s/%s}"
          (short_dtype_name primal_dtype)
          (short_dtype_name tangent_dtype)
  in
  { primal_dtype; tangent_dtype; name }

let cur_qdd (_ : Ref.t) =
  raise (Invalid_argument "'Ref' object has no attribute 'cur_qdd'")

type e_array = { e_aval : aval; e_data : value }

let e_array e_aval e_data = { e_aval; e_data }
let e_array_aval e = e.e_aval
let e_array_data e = e.e_data
let e_array_shape e = e.e_aval.shape
let e_array_dtype e = e.e_aval.dtype
let e_array_ndim e = Array.length e.e_aval.shape
let e_array_size e = Array.fold_left ( * ) 1 e.e_aval.shape

let e_array_to_string e =
  match e.e_data with
  | Concrete nd -> "E" ^ Numpy.Array_print.array_repr nd
  | Device b -> "E" ^ Numpy.Array_print.array_repr (Pjrt.Buffer.to_host b)
  | Tracer _ -> invalid_arg "EArray: a tracer has no printed value"
