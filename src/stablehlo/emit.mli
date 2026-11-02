type partition = {
  pt_mesh : (string * int) list;
  pt_devices : int array;
  pt_in : string list array list;
  pt_out : string list array option list;
}

val emit_closed_jaxpr : ?partition:partition -> Types.closed_jaxpr -> string
val shard_mesh_seen : unit -> ((string * int) list * int array) option
val lapack_ffi_ready : (unit -> bool) ref
val gpu_solver_ready : (unit -> bool) ref
val emitting : unit -> bool

type lowered = { text : string; kept : int list }

val lower :
  ?keep_unused:bool ->
  ?donated:bool list ->
  ?result_names:string list ->
  name:string ->
  Types.closed_jaxpr ->
  lowered

val warn_unusable_donations :
  ?keep_unused:bool -> donated:bool list -> Types.closed_jaxpr -> unit

val emit_jax_text :
  ?keep_unused:bool ->
  ?donated:bool list ->
  ?result_names:string list ->
  name:string ->
  Types.closed_jaxpr ->
  string

val lower_partitioned :
  name:string -> partition -> Types.closed_jaxpr -> lowered

val emit_export_text :
  ?platforms:string list ->
  ?result_names:string list ->
  name:string ->
  uses_shape_polymorphism:bool ->
  Types.closed_jaxpr ->
  lowered
