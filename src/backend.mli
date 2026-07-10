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
