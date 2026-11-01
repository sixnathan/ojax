type partition = {
  pt_mesh : (string * int) list;
  pt_devices : int array;
  pt_in : string list array list;
  pt_out : string list array option list;
}

val emit_closed_jaxpr : ?partition:partition -> Types.closed_jaxpr -> string
val shard_mesh_seen : unit -> ((string * int) list * int array) option
val lapack_ffi_ready : (unit -> bool) ref
val emitting : unit -> bool

type lowered = { text : string; kept : int list }

val lower : name:string -> Types.closed_jaxpr -> lowered
val emit_jax_text : name:string -> Types.closed_jaxpr -> string

val lower_partitioned :
  name:string -> partition -> Types.closed_jaxpr -> lowered
