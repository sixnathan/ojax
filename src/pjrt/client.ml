type t

external create_ : Abi.plugin -> t = "ojax_pjrt_client_create"
external destroy_ : t -> unit = "ojax_pjrt_client_destroy"

let create plugin = create_ plugin
let destroy client = destroy_ client
