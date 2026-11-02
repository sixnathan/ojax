open Types

type array_ = value
type ref_ = Ref.t
type shape_dtype_struct = Shape_dtype_struct.t

val shape_dtype_struct :
  ?weak_type:bool -> int array -> Dtype.t -> shape_dtype_struct

type p = {
  partitions : Spmd.partition_spec;
  unreduced : string list;
  reduced : string list;
}

val partition :
  ?unreduced:string list -> ?reduced:string list -> Spmd.partition_spec -> p

val p_to_string : p -> string

module Ad : sig
  type grad_value = Grad_value
  type grad_ref = Grad_ref
  type dont_want = Dont_want
  type didnt_want = Didnt_want
end

module Checkpoint_policies : sig
  val everything_saveable : Checkpoint.policy
  val nothing_saveable : Checkpoint.policy
  val dots_saveable : Checkpoint.policy
  val checkpoint_dots : Checkpoint.policy
  val dots_with_no_batch_dims_saveable : Checkpoint.policy
  val checkpoint_dots_with_no_batch_dims : Checkpoint.policy
  val save_anything_except_these_names : string list -> Checkpoint.policy
  val save_any_names_but_these : string list -> Checkpoint.policy
  val save_only_these_names : string list -> Checkpoint.policy
  val save_from_both_policies :
    Checkpoint.policy -> Checkpoint.policy -> Checkpoint.policy
end

module Memory : sig
  type space = Device | Host | Any

  val to_string : space -> string
end

val auto_pcast : bool Config.flag
val remove_size_one_mesh_axis_from_type : bool Config.flag
val make_user_context : ?default_value:'a -> unit -> 'a Config.user_context
val config_read : string -> Config.config_value
val config_update : string -> Config.config_value -> unit
val ds : ?size:int -> ?stride:int -> value option -> Numpy.Indexing.dslice

val io_callback :
  ?ordered:bool ->
  (value list -> value list) ->
  result_avals:aval list ->
  value list ->
  value list

type primal_tangent_dtype = {
  primal_dtype : Dtype.t;
  tangent_dtype : Dtype.t;
  name : string;
}

val primal_tangent_dtype :
  ?name:string -> Dtype.t -> Dtype.t -> primal_tangent_dtype

val cur_qdd : Ref.t -> unit

type e_array

val e_array : aval -> value -> e_array
val e_array_aval : e_array -> aval
val e_array_data : e_array -> value
val e_array_shape : e_array -> int array
val e_array_dtype : e_array -> Dtype.t
val e_array_ndim : e_array -> int
val e_array_size : e_array -> int
val e_array_to_string : e_array -> string
