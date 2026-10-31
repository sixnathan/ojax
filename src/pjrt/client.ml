type t
type device = { id : int; process_index : int; kind : string }

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
external devices_ : t -> (int * int * string) array = "ojax_pjrt_client_devices"

let create ?(options = []) ?cpu_devices plugin =
  let cpu_options =
    match cpu_devices with
    | Some n when n > 0 -> [ ("cpu_device_count", Int64 (Int64.of_int n)) ]
    | _ -> []
  in
  create_ plugin (Array.of_list (cpu_options @ options))

let destroy client = destroy_ client
let platform_name client = platform_name_ client
let platform client = platform_name_ client
let platform_version client = platform_version_ client
let device_count client = device_count_ client
let device_kind client = device_kind_ client

let devices client =
  Array.map
    (fun (id, process_index, kind) -> { id; process_index; kind })
    (devices_ client)
