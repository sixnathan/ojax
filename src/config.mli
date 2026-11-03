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
val auto_pcast : bool flag
val random_seed_offset : int flag
val high_dynamic_range_gumbel : bool flag
val check_static_indices : bool flag
val disable_most_optimizations : bool flag
val compiler_enable_remat_pass : bool flag

val compile_debug_options :
  unit -> (string * [ `Int of int | `Bool of bool | `String of string ]) list

val remove_size_one_mesh_axis_from_type : bool flag
val jax2tf_associative_scan_reductions : bool flag
val numpy_dtype_promotion : string flag
val legacy_prng_key : string flag
val explicit_x64_dtypes : string flag
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
val export_calling_convention_version : int flag
val platforms : string flag
val platform_resolver : (unit -> string) ref
val current_platform : unit -> string
val num_cpu_devices : int flag

type 'a user_context

val make_user_context : ?default_value:'a -> unit -> 'a user_context
val with_user_context : 'a user_context -> 'a -> (unit -> 'b) -> 'b
val user_context_value : 'a user_context -> 'a option

type config_value =
  | Config_bool of bool
  | Config_int of int
  | Config_string of string
  | Config_optional_string of string option

val read : string -> config_value
val update : string -> config_value -> unit
