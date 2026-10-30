open Types

type lowered = {
  closed : closed_jaxpr;
  in_avals : aval list;
  out_avals : aval list;
}

type compiled

val lower : (value list -> value list) -> aval list -> lowered
val as_text : lowered -> string
val jaxpr_text : lowered -> string
val compile : lowered -> compiled
val call : compiled -> value list -> value list
val cost_analysis : compiled -> (string * float) list
