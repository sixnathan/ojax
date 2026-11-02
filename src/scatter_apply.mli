val mark :
  Types.primitive -> Types.closed_jaxpr -> (float -> float) -> Types.primitive

val find : Types.primitive -> (Types.closed_jaxpr * (float -> float)) option
