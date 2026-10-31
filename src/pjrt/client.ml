type t

type option_value =
  | String of string
  | Int64 of int64
  | Float of float
  | Bool of bool

external create_ : Abi.plugin -> (string * option_value) array -> t
  = "ojax_pjrt_client_create"

external destroy_ : t -> unit = "ojax_pjrt_client_destroy"
external platform_name_ : t -> string = "ojax_pjrt_client_platform_name"
external platform_version_ : t -> string = "ojax_pjrt_client_platform_version"
external device_count_ : t -> int = "ojax_pjrt_client_device_count"
external device_kind_ : t -> string = "ojax_pjrt_client_device_kind"

let create ?(options = []) plugin = create_ plugin (Array.of_list options)
let destroy client = destroy_ client
let platform_name client = platform_name_ client
let platform_version client = platform_version_ client
let device_count client = device_count_ client
let device_kind client = device_kind_ client
