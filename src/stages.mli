open Types

type lowered = {
  closed : closed_jaxpr;
  in_avals : aval list;
  out_avals : aval list;
  module_name : string;
  result_paths : string list option;
}

type compiled
type compiler_option = Pjrt.Executable.compiler_option
type compiler_options = (string * compiler_option) list
type arg_info = { arg_aval : aval; donated : bool }

type traced = {
  jaxpr : closed_jaxpr;
  fun_name : string;
  args_info : arg_info Tree_util.t;
  in_tree : Tree_util.treedef;
  out_tree : Tree_util.treedef;
  out_avals : aval Tree_util.t;
  call : value Tree_util.t list -> value Tree_util.t;
  trace_lower : unit -> lowered;
}

type wrapped = {
  apply : value Tree_util.t list -> value Tree_util.t;
  trace : value Tree_util.t list -> traced;
  wrapped_lower : value Tree_util.t list -> lowered;
}

val arg_info_shape : arg_info -> int array
val arg_info_dtype : arg_info -> Dtype.t
val lower : (value list -> value list) -> aval list -> lowered
val as_text : lowered -> string
val raw_text : lowered -> string
val jaxpr_text : lowered -> string
val as_jax_text : ?name:string -> ?result_paths:string list -> lowered -> string
val jax_jaxpr_text : lowered -> string
val compile : ?compiler_options:compiler_options -> lowered -> compiled
val call : compiled -> value list -> value list
val cost_analysis : compiled -> (string * float) list

val jit :
  ?name:string -> (value Tree_util.t list -> value Tree_util.t) -> wrapped
