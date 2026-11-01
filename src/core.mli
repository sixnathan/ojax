open Types

val fresh_id : unit -> int

type prim_rules = {
  mutable impl : primitive -> Ndarray.t list -> Ndarray.t list;
  mutable abstract_eval : primitive -> aval list -> aval list;
}

val rules : prim_rules

type interpreter = {
  i_pure : trace -> value -> value;
  i_lift : trace -> value -> value;
  i_full_lower : value -> value;
  i_process_primitive : trace -> primitive -> value list -> value list;
  i_process_custom_jvp :
    trace ->
    primal:(value list -> value list) ->
    jvp:(value list -> value list) ->
    value list ->
    value list;
  i_process_custom_vjp :
    trace ->
    primal:(value list -> value list) ->
    fwd:(value list -> value list) ->
    bwd:(value list -> value list) ->
    value list ->
    value list;
}

val register_interpreter : trace_kind -> interpreter -> unit
val current_level : unit -> int
val with_new_main : trace_kind -> global_data -> (trace -> 'a) -> 'a
val new_dynamic : trace -> (unit -> 'a) -> 'a
val find_top_trace : value list -> trace
val dynamic_trace : trace option ref
val linearize_scope : trace option option ref
val full_raise : trace -> value -> value
val full_lower : value -> value
val process_primitive : trace -> primitive -> value list -> value list

val process_custom_jvp :
  trace ->
  (value list -> value list) ->
  jvp:(value list -> value list) ->
  value list ->
  value list

val process_custom_vjp :
  trace ->
  (value list -> value list) ->
  fwd:(value list -> value list) ->
  bwd:(value list -> value list) ->
  value list ->
  value list

type custom_rule =
  | Jvp_rule of {
      primal : value list -> value list;
      jvp : value list -> value list -> value list * value list;
    }
  | Vjp_rule of {
      primal : value list -> value list;
      fwd : value list -> value list * value list;
      bwd : value list -> value list -> value list;
      fwd_perturbed :
        (bool list -> value list -> value list * value list) option;
    }
  | Linear_rule of {
      forward : value list -> value list -> value list;
      transpose : value list -> value list -> value list;
      res_differentiable : bool;
    }
  | Callback_rule of {
      host : value list -> value list;
      result_avals : aval list;
    }

val register_custom_rule : custom_rule -> int
val custom_rule : int -> custom_rule
val note_custom_call_jaxpr : int -> aval list -> closed_jaxpr -> unit
val find_custom_call_jaxpr : int -> aval list -> closed_jaxpr option
val note_custom_num_consts : int -> int -> unit
val custom_num_consts_of : int -> int option
val custom_rule_opt : primitive -> custom_rule option
val scan_lift_consts : num_consts:int -> closed_jaxpr -> closed_jaxpr
val note_custom_eqn_jaxpr : int -> closed_jaxpr -> unit
val find_custom_eqn_jaxpr : int -> closed_jaxpr option
val custom_call_jaxprs_of : int -> closed_jaxpr list
val with_eval_dynamic : (unit -> 'a) -> 'a
val ffi_impl : (primitive -> Ndarray.t list -> Ndarray.t list) ref
val named_axis_trace : (string -> trace option) ref
val bind : primitive -> value list -> value list
val bind1 : primitive -> value list -> value
val get_aval : value -> aval
val divide_shape_sizes : int array -> int array -> int

type shard_rule = {
  sr_mesh : (string * int) list;
  sr_devices : int array;
  sr_manual : string list;
  sr_in : string list array list;
  sr_out : string list array list;
  sr_body : closed_jaxpr;
}

type constraint_rule = {
  cr_mesh : (string * int) list;
  cr_devices : int array;
  cr_spec : string list array;
}

val shard_rule : int -> shard_rule option
val constraint_rule : int -> constraint_rule option
val register_shard_rule : int -> shard_rule -> unit
val register_constraint_rule : int -> constraint_rule -> unit
val mesh_axes : (string * int) list ref
val with_mesh_axes : (string * int) list -> (unit -> 'a) -> 'a
val placed_eval : (primitive -> value list -> value list option) ref
val tracing_dynamically : unit -> bool
val is_placed : value -> bool
val eager_eval : (primitive -> value list -> value list option) ref
val staging : unit -> bool
val splat : Dtype.t -> int array -> float -> value
val elementwise_shape : int array list -> int array
val is_elementwise : primitive -> bool
val call_inline : (closed_jaxpr -> value list -> value list) ref
val top_is_staging : value list -> bool
val register_custom_names : int -> name:string -> jvp_name:string -> unit
val custom_names_of : int -> (string * string) option
val py_scalar : Dtype.t -> float -> value
val weak_convert : bool ref
val convert_like : weak:bool -> value -> Dtype.t -> value
val mark_inline_call : int -> unit
val is_inline_call : int -> bool
val inherit_inline : from:int -> int -> unit
val call_direct : (closed_jaxpr -> value list -> value list) ref
val random_gamma_hook : (log_space:bool -> value -> value -> value) ref
val lowered_linalg_hook : (primitive -> value list -> value list) ref
val random_gamma_grad_hook : (log_space:bool -> value -> value -> value) ref
val register_decomposition : (value list -> value list) -> int
val decomposition : int -> value list -> value list
val gather_fill_hook : (primitive -> value list -> value list) ref
