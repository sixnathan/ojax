open Types

val typecheck_jaxpr : jaxpr -> unit
val in_eqn_eval : bool ref
val eval_transparent : bool ref
val eval_jaxpr : jaxpr -> value list -> value list
val jaxpr_as_fun : jaxpr -> value list -> value list
val eval_closed_jaxpr : closed_jaxpr -> value list -> value list
val make_jaxpr : aval list -> (value list -> value list) -> closed_jaxpr

val make_jaxpr_for_lowering :
  aval list -> (value list -> value list) -> closed_jaxpr

val with_trace_debug : trace_debug -> (unit -> 'a) -> 'a
val origin_msg : value -> string
val stage_constant : value -> value

val lift_while :
  cond_nconsts:int ->
  body_nconsts:int ->
  closed_jaxpr ->
  closed_jaxpr ->
  aval list ->
  closed_jaxpr * closed_jaxpr

val make_jaxpr_stateful :
  aval list ->
  (value list -> value list) ->
  closed_jaxpr * value list * Ref_state.cell list

val make_jaxpr_forbidding_refs :
  message:string ->
  aval list ->
  (value list -> value list) ->
  closed_jaxpr * value list

val make_jaxpr_hoisted :
  aval list -> (value list -> value list) -> closed_jaxpr * value list

val barrier_calls : (int, unit) Hashtbl.t
val remat_body : (int -> closed_jaxpr option) ref
val remat_rebuild : (int -> closed_jaxpr -> int) ref
val install : unit -> unit
val eqn_has_effect : eqn -> bool
val dce_closed : closed_jaxpr -> bool list -> closed_jaxpr * bool list
val dce_keep_args : closed_jaxpr -> closed_jaxpr

val eager_jit :
  (string -> string -> (value list -> value list) -> value list -> value list)
  ref

val copy_call_name : source:closed_jaxpr -> closed_jaxpr -> unit
val clear_call_cache : unit -> unit

val named_jit :
  name:string ->
  ?key:string ->
  (value list -> value list) ->
  value list ->
  value list

val custom_call_jaxpr : int -> Types.aval list -> Types.closed_jaxpr option
