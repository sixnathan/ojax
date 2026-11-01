exception Error of string

val env_var : string
val gpu_env_var : string
val pinned_sha256 : string list
val pinned_gpu_sha256 : string list
val pjrt_api_minor : int
val sha256_hex : string -> string
val sha256_file : string -> string
val validate_path : string option -> string
val resolve : unit -> string
val find : unit -> string option
val verify_at : string -> string
val preflight : unit -> string
val available : unit -> bool
val exact_sha256 : string
val conventional_path : unit -> string option
val is_exact : unit -> bool
val verify_gpu_at : string -> string
val gpu_configured : unit -> bool
val gpu_preflight : unit -> string
