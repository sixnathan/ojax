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
val with_eval_dynamic : (unit -> 'a) -> 'a
val named_axis_trace : (string -> trace option) ref
val bind : primitive -> value list -> value list
val bind1 : primitive -> value list -> value
val get_aval : value -> aval
val divide_shape_sizes : int array -> int array -> int
