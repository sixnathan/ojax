open Types

type device = Pjrt.Client.device = {
  id : int;
  process_index : int;
  kind : string;
}

type axis_type = Auto | Explicit | Manual

type abstract_device = {
  device_kind : string;
  num_cores : int option;
  platform : string;
}

type mesh = {
  axis_names : string list;
  axis_sizes : int list;
  axis_types : axis_type list;
  devices : device array;
}

type abstract_mesh = {
  abstract_axis_names : string list;
  abstract_axis_sizes : int list;
  abstract_axis_types : axis_type list;
  abstract_device : abstract_device option;
}

type axis_spec = Named of string | Axes of string list | Unconstrained
type partition_spec = axis_spec list

type t =
  | Single_device of device option
  | Named_sharding of mesh * partition_spec
  | Tiled_sharding of {
      devices : device array;
      indices : (int * int) array array;
    }

type shard = {
  device : device;
  index : (int * int) array;
  replica_id : int;
  data : value;
}

val local_devices : unit -> device array
val device_count : unit -> int
val mesh_size : mesh -> int
val mesh_pairs : mesh -> (string * int) list
val device_ids : mesh -> int array

val make_mesh :
  ?axis_types:axis_type list ->
  ?devices:device array ->
  int list ->
  string list ->
  mesh

val named_sharding : mesh -> partition_spec -> t
val spec_array : partition_spec -> int -> string list array
val spec_of_array : string list array -> partition_spec
val spec_to_string : partition_spec -> string
val array_sharding : value -> t
val sharding_devices : t -> device array
val num_devices : t -> int
val is_fully_addressable : t -> bool
val is_fully_replicated : t -> bool
val devices_indices_map : t -> int array -> (device * (int * int) array) list
val shard_shape : t -> int array -> int array
val single_shard_value : Pjrt.Buffer.raw -> value
val addressable_shards : value -> shard list
val global_shards : value -> shard list
val addressable_data : value -> int -> value
val is_committed : value -> bool
val host_array : value -> Ndarray.t
val same_placement : value -> t -> bool
val device_put : value Tree_util.t -> t -> value Tree_util.t
val concrete_context : mesh option ref
val abstract_context : abstract_mesh option ref
val abstract_mesh_of : mesh -> abstract_mesh
val empty_abstract_mesh : abstract_mesh
val empty_mesh : mesh
val abstract_mesh_size : abstract_mesh -> int
val get_mesh : unit -> mesh
val get_abstract_mesh : unit -> abstract_mesh
val with_abstract_context : abstract_mesh -> (unit -> 'a) -> 'a

val update_axis_types :
  abstract_mesh -> (string * axis_type) list -> abstract_mesh

val context_mesh : value list -> mesh

val shard_map :
  ?mesh:mesh ->
  ?axis_names:string list ->
  ?in_specs:partition_spec list ->
  out_specs:partition_spec list ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val smap :
  ?in_axes:int option list ->
  out_axes:int option ->
  axis_name:string ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val pmap :
  ?axis_name:string ->
  ?in_axes:int option list ->
  ?out_axes:int ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val device_to_string : device -> string
val axis_type_to_string : axis_type -> string
val mesh_to_string : mesh -> string
val abstract_device_to_string : abstract_device -> string
val abstract_mesh_to_string : abstract_mesh -> string
val sharding_to_string : t -> string

val visualize_sharding_string :
  ?scale:float -> ?min_width:int -> ?max_width:int -> int array -> t -> string
