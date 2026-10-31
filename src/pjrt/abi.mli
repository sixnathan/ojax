exception Error of string

type plugin

val ffi_dlopen : string -> nativeint
val ffi_dlsym : nativeint -> string -> nativeint
val ffi_register_handler : plugin -> string -> string -> nativeint -> unit
val ffi_register_type : plugin -> string -> nativeint -> nativeint -> unit
val pjrt_api_major : int
val pjrt_api_minor : int
val pjrt_api_struct_size : int
val open_plugin : string -> plugin
val api_version : plugin -> int * int
val struct_size : plugin -> int
val close : plugin -> unit
val maxrss_bytes : unit -> int
val buffer_type : Dtype.t -> int
val dtype_of_buffer_type : int -> Dtype.t option
