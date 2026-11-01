open Types

include module type of struct
  include Spmd
end

type manual_axis_type = {
  varying : string list;
  unreduced : string list;
  reduced : string list;
}

val manual_axis_type :
  ?varying:string list ->
  ?unreduced:string list ->
  ?reduced:string list ->
  unit ->
  manual_axis_type

val manual_axis_to_ct : manual_axis_type -> manual_axis_type
val single_device : t
val make_single_device_sharding : device -> t

val abstract_device :
  ?num_cores:int ->
  device_kind:string ->
  platform:string ->
  unit ->
  abstract_device

val abstract_mesh :
  ?axis_types:axis_type list ->
  ?abstract_device:abstract_device ->
  int list ->
  string list ->
  abstract_mesh

val get_mesh : unit -> mesh
val set_mesh : mesh -> unit
val with_mesh : mesh -> (unit -> 'a) -> 'a
val use_abstract_mesh : abstract_mesh -> (unit -> 'a) -> 'a
val of_spec : partition_spec -> t
val reshard : value Tree_util.t -> t -> value Tree_util.t
val with_sharding_constraint : value Tree_util.t -> t -> value Tree_util.t
val reshard_spec : value Tree_util.t -> partition_spec -> value Tree_util.t

val jit :
  ?in_shardings:t list ->
  ?out_shardings:t list ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val lower_text :
  ?out_shardings:t list ->
  name:string ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  string

val auto_axes :
  ?axes:string list ->
  out_sharding:t list ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val explicit_axes :
  ?axes:string list ->
  in_sharding:t list ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val make_array_from_single_device_arrays : int array -> t -> value list -> value

val make_array_from_process_local_data :
  ?global_shape:int array -> t -> value -> value

val make_array_from_callback :
  int array -> t -> ((int * int) array -> value) -> value
