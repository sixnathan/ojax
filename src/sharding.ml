open Types
module Tree = Tree_util

type axis_spec = Named of string | Unconstrained
type partition_spec = axis_spec list
type mesh = { axis_names : string list; axis_sizes : int list }
type t = Single_device | Named_sharding of mesh * partition_spec

let make_mesh axis_shapes axis_names =
  if List.length axis_shapes <> List.length axis_names then
    invalid_arg "Mesh axis names and shapes must have the same length";
  { axis_names; axis_sizes = axis_shapes }

let mesh_size m = List.fold_left ( * ) 1 m.axis_sizes
let single_device = Single_device

let named_sharding m spec =
  List.iter
    (function
      | Named name when not (List.mem name m.axis_names) ->
          invalid_arg ("Unknown mesh axis name: " ^ name)
      | _ -> ())
    spec;
  Named_sharding (m, spec)

let require_one_device m =
  if mesh_size m > Api.device_count () then
    invalid_arg
      (Printf.sprintf "Mesh of %d devices needs more than the %d available"
         (mesh_size m) (Api.device_count ()))

let device_put (x : value Tree.t) sharding =
  (match sharding with
  | Single_device -> ()
  | Named_sharding (m, _) -> require_one_device m);
  Api.device_put x

let reshard = device_put

let shard_map f ~mesh ~in_specs:_ ~out_specs:_ args =
  require_one_device mesh;
  f args

let is_fully_addressable _ = true

let num_devices = function
  | Single_device -> 1
  | Named_sharding (m, _) -> mesh_size m
