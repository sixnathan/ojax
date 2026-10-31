type 'a flag = { name : string; mutable value : 'a }

let make name value = { name; value }
let value flag = flag.value
let name flag = flag.name
let set flag v = flag.value <- v

let with_value flag v thunk =
  let prev = flag.value in
  flag.value <- v;
  Fun.protect ~finally:(fun () -> flag.value <- prev) thunk

let enable_x64 = make "jax_enable_x64" false
let x64_enabled () = enable_x64.value
let disable_jit = make "jax_disable_jit" false
let debug_nans = make "jax_debug_nans" false
let debug_infs = make "jax_debug_infs" false
let log_compiles = make "jax_log_compiles" false
let no_tracing = make "jax_no_tracing" false
let no_execution = make "jax_no_execution" false
let numpy_rank_promotion = make "jax_numpy_rank_promotion" "allow"

let export_calling_convention_version =
  make "jax_export_calling_convention_version" 10
