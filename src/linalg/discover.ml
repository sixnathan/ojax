external available_ : unit -> bool = "ojax_lapack_available"
external abi_int_size_ : unit -> int = "ojax_lapack_abi_int_size"

let available = available_ ()
let abi_int_size = abi_int_size_ ()
let backend = if available then "accelerate" else "unavailable"
