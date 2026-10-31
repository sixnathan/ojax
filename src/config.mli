type 'a flag

val value : 'a flag -> 'a
val name : 'a flag -> string
val set : 'a flag -> 'a -> unit
val with_value : 'a flag -> 'a -> (unit -> 'b) -> 'b
val enable_x64 : bool flag
val x64_enabled : unit -> bool
val disable_jit : bool flag
val debug_nans : bool flag
val debug_infs : bool flag
val log_compiles : bool flag
val no_tracing : bool flag
val no_execution : bool flag
val numpy_rank_promotion : string flag
val enable_checks : bool flag
val debug_key_reuse : bool flag
val check_tracer_leaks : bool flag
val explain_cache_misses : bool flag
val enable_custom_prng : bool flag
val softmax_custom_jvp : bool flag
val threefry_partitionable : bool flag
val allow_f16_reductions : bool flag
val thread_guard : bool flag
val jax2tf_associative_scan_reductions : bool flag
val numpy_dtype_promotion : string flag
val legacy_prng_key : string flag
val default_matmul_precision : string option flag
val matmul_level : unit -> string option
val matmul_preset : unit -> string option
val check_matmul_preset_on_cpu : unit -> unit
val transfer_guard_host_to_device : string option flag
val transfer_guard_device_to_device : string option flag
val transfer_guard_device_to_host : string option flag
val with_transfer_guard : string -> (unit -> 'a) -> 'a
val array_garbage_collection_guard : string option flag
val log : string -> unit
val set_log_sink : (string -> unit) -> string -> unit
val trace_context : unit -> (string * string) list
