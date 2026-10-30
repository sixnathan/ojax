open Types

val jit_flat : (value list -> value list) -> value list -> value list
val call : (value list -> value list) -> value list -> value list

val make_jaxpr :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  closed_jaxpr

val jit :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val device_put : value Tree_util.t -> value Tree_util.t
val to_host : value Tree_util.t -> value Tree_util.t

val jvp :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t list ->
  value Tree_util.t * value Tree_util.t

val linearize :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t * (value Tree_util.t list -> value Tree_util.t)

val vjp :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t * (value Tree_util.t -> value Tree_util.t list)

val grad :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val value_and_grad :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t * value Tree_util.t

val vmap :
  (value Tree_util.t list -> value Tree_util.t) ->
  int option list ->
  value Tree_util.t list ->
  value Tree_util.t

val value_and_grad_argnums :
  ?has_aux:bool ->
  argnums:int list ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value * value Tree_util.t list * value Tree_util.t

val jacfwd :
  ?argnums:int ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val jacrev :
  ?argnums:int ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val jacobian :
  ?argnums:int ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val hessian :
  ?argnums:int ->
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val eval_shape :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  aval Tree_util.t

val linear_transpose :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t ->
  value Tree_util.t list

val checkpoint : 'a -> 'a
val remat : 'a -> 'a
val named_call : 'a -> 'a
val named_scope : string -> (unit -> 'a) -> 'a
val device_get : value Tree_util.t -> value Tree_util.t
val block_until_ready : 'a -> 'a
val devices : unit -> string list
val local_devices : unit -> string list
val device_count : unit -> int
val local_device_count : unit -> int
val process_count : unit -> int
val process_index : unit -> int
val process_indices : unit -> int list
val host_count : unit -> int
val host_id : unit -> int
val host_ids : unit -> int list
val default_backend : unit -> string
val clear_caches : unit -> unit
val disable_jit : (unit -> 'a) -> 'a
val ensure_compile_time_eval : (unit -> 'a) -> 'a

val pmap :
  (value Tree_util.t list -> value Tree_util.t) ->
  value Tree_util.t list ->
  value Tree_util.t

val pure_callback :
  (value list -> value list) ->
  result_avals:aval list ->
  value list ->
  value list

val typeof : value -> aval
val effects_barrier : unit -> unit
val copy_to_host_async : 'a -> 'a
val default_device : unit -> string
val live_arrays : unit -> value list
val make_array_from_callback : int array -> ((int * int) array -> 'a) -> 'a
val print_environment_info : unit -> unit
