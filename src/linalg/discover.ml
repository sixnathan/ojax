external available_ : unit -> bool = "ojax_lapack_available"
external abi_int_size_ : unit -> int = "ojax_lapack_abi_int_size"
external backend_ : unit -> string = "ojax_lapack_backend"
external library_path_ : unit -> string = "ojax_lapack_library_path"

let available = available_ ()
let abi_int_size = abi_int_size_ ()
let backend = backend_ ()
let library_path = library_path_ ()
