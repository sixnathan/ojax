open Types

type layout = Default | Major_to_minor of int array

type attr = ffi_attr =
  | Ffi_bool of bool
  | Ffi_int of int64
  | Ffi_float of float
  | Ffi_string of string
  | Ffi_array of Ndarray.t
  | Ffi_dict of (string * ffi_attr) list
  | Ffi_seq of ffi_attr list
  | Ffi_dtype of Dtype.t

type library
type type_registration = { type_id : nativeint; type_info : nativeint option }

val dlopen : string -> library
val dlsym : library -> string -> nativeint

val register_ffi_target :
  ?platform:string -> ?api_version:int -> string -> nativeint -> unit

val register_ffi_type_id : ?platform:string -> string -> nativeint -> unit
val register_ffi_type : ?platform:string -> string -> type_registration -> unit

val ffi_call :
  ?has_side_effect:bool ->
  ?vmap_method:string ->
  ?input_layouts:layout list ->
  ?output_layouts:layout list ->
  ?input_output_aliases:(int * int) list ->
  ?custom_call_api_version:int ->
  ?legacy_backend_config:string ->
  string ->
  aval list ->
  ?attributes:(string * attr) list ->
  value list ->
  value list

val ffi_call1 :
  ?has_side_effect:bool ->
  ?vmap_method:string ->
  ?input_layouts:layout list ->
  ?output_layout:layout ->
  ?input_output_aliases:(int * int) list ->
  ?custom_call_api_version:int ->
  ?legacy_backend_config:string ->
  string ->
  aval ->
  ?attributes:(string * attr) list ->
  value list ->
  value
