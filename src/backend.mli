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

val executor : Types.closed_jaxpr -> Types.value list -> Types.value list
val of_host_value : Types.value -> Types.value
val to_host_value : Types.value -> Types.value
val run_text : string -> Types.value list -> Types.value list
val stats : unit -> int * string list

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
