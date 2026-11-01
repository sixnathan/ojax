module type S = sig
  type buffer
  type compiled

  val of_host : Ndarray.t -> buffer
  val to_host : buffer -> Ndarray.t
  val compile : Types.closed_jaxpr -> compiled
  val execute : compiled -> buffer list -> buffer list
end

module Interpreter :
  S with type buffer = Ndarray.t and type compiled = Types.closed_jaxpr

module Xla :
  S with type buffer = Pjrt.Buffer.t and type compiled = Pjrt.Executable.t

val executor :
  ?options:(string * Pjrt.Executable.compiler_option) list ->
  Types.closed_jaxpr ->
  Types.value list ->
  Types.value list

val of_host_value : Types.value -> Types.value
val to_host_value : Types.value -> Types.value
val run_text : string -> Types.value list -> Types.value list
val run_text_forced : string -> Types.value list -> Types.value list
val xla_plugin : unit -> Pjrt.Abi.plugin
val xla_client : unit -> Pjrt.Client.t
val stats : unit -> int * string list

type eager_stats = {
  eager_compiled : int;
  eager_cache_hits : int;
  eager_executions : int;
  eager_fallbacks : (string * string * int) list;
}

val eager_stats : unit -> eager_stats

val jitted_keyed :
  ?inline:bool ->
  string * Obj.t ->
  (Types.value list -> Types.value list) ->
  Types.value list ->
  Types.value list

val current_platform : unit -> string
val default_backend : unit -> string
val devices : ?backend:string -> unit -> string list
val with_platform : string -> (unit -> 'a) -> 'a
val platform_version : unit -> string
val device_kind : unit -> string
val has_visible_nvidia_gpu : unit -> bool
val gpu_ready : unit -> bool
val value_device : Types.value -> string option

type plan = {
  mesh : (string * int) list;
  devices : int array;
  arg_specs : string list array list;
  out_specs : string list array option list;
}

val local_devices : unit -> Pjrt.Client.device array
val multi_device_available : unit -> bool
val platform : unit -> string

val place_named :
  Ndarray.t ->
  mesh:(string * int) list ->
  devices:int array ->
  spec:string list array ->
  Types.value

val place_on : Ndarray.t -> int -> Types.value

val placed_of :
  shape:int array ->
  dtype:Dtype.t ->
  devices:int array ->
  indices:(int * int) array array ->
  layout:Pjrt.Buffer.layout ->
  Pjrt.Buffer.raw array ->
  Types.value

val named_index :
  shape:int array ->
  mesh:(string * int) list ->
  spec:string list array ->
  int ->
  (int * int) array

val check_divisible :
  shape:int array -> mesh:(string * int) list -> spec:string list array -> unit

val run_plan :
  Types.closed_jaxpr -> plan -> Types.value list -> Types.value list option

val client : unit -> Pjrt.Client.t
val clear_placed_programs : unit -> unit
