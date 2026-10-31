type t
type device = { id : int; process_index : int; kind : string }

external create_ : Abi.plugin -> int -> t = "ojax_pjrt_client_create"
external destroy_ : t -> unit = "ojax_pjrt_client_destroy"
external platform_ : t -> string = "ojax_pjrt_client_platform"
external devices_ : t -> (int * int * string) array = "ojax_pjrt_client_devices"

let create ?cpu_devices plugin =
  create_ plugin (match cpu_devices with Some n -> n | None -> 0)

let destroy client = destroy_ client
let platform client = platform_ client

let devices client =
  Array.map
    (fun (id, process_index, kind) -> { id; process_index; kind })
    (devices_ client)
