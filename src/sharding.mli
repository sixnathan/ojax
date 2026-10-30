open Types

type axis_spec = Named of string | Unconstrained
type partition_spec = axis_spec list
type mesh = { axis_names : string list; axis_sizes : int list }
type t

val make_mesh : int list -> string list -> mesh
val mesh_size : mesh -> int
val single_device : t
val named_sharding : mesh -> partition_spec -> t
val device_put : value Tree_util.t -> t -> value Tree_util.t
val reshard : value Tree_util.t -> t -> value Tree_util.t

val shard_map :
  ('a list -> 'b) ->
  mesh:mesh ->
  in_specs:partition_spec list ->
  out_specs:partition_spec list ->
  'a list ->
  'b

val is_fully_addressable : t -> bool
val num_devices : t -> int
