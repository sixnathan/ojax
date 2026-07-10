exception Error of string

val env_var : string
val expected_sha256 : string
val pjrt_api_minor : int
val sha256_hex : string -> string
val sha256_file : string -> string
val validate_path : string option -> string
val resolve : unit -> string
val verify_at : string -> string
val preflight : unit -> string
