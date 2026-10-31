open Types

type disabled_safety_check

val platform_check : disabled_safety_check
val custom_call_check : string -> disabled_safety_check
val is_custom_call : disabled_safety_check -> string option
val disabled_check_to_string : disabled_safety_check -> string
val minimum_supported_calling_convention_version : int
val maximum_supported_calling_convention_version : int
val default_export_platform : unit -> string

type exported = {
  fun_name : string;
  in_tree : Tree_util.treedef;
  out_tree : Tree_util.treedef;
  mlir_module : string;
  in_avals : aval list;
  out_avals : aval list;
  platforms : string list;
  disabled_safety_checks : disabled_safety_check list;
  nr_devices : int;
  calling_convention_version : int;
  module_kept_var_idx : int list;
  uses_global_constants : bool;
  get_vjp : (exported -> exported) option;
}

val export :
  ?fun_name:string ->
  ?platforms:string list ->
  ?disabled_checks:disabled_safety_check list ->
  (value list -> value list) ->
  aval list ->
  exported

val export_tree :
  ?fun_name:string ->
  ?platforms:string list ->
  ?disabled_checks:disabled_safety_check list ->
  (value Tree_util.t -> value Tree_util.t) ->
  aval Tree_util.t ->
  exported

val register_pytree_node_serialization :
  serialized_name:string ->
  serialize_auxdata:(string -> string) ->
  deserialize_auxdata:(string -> string) ->
  ?from_children:(string -> unit Tree_util.t list -> unit Tree_util.t) ->
  string ->
  unit

val register_namedtuple_serialization :
  serialized_name:string -> fields:string list -> string -> unit

val serialize : ?vjp_order:int -> exported -> string
val deserialize : string -> exported
val has_vjp : exported -> bool
val vjp : exported -> exported
val call : exported -> value list -> value list
val call_tree : exported -> value Tree_util.t -> value Tree_util.t
val symbolic_scope : ?constraints:string list -> unit -> Shape_poly.scope

val symbolic_shape :
  ?constraints:string list ->
  ?scope:Shape_poly.scope ->
  ?like:int option array ->
  string option ->
  Shape_poly.dim array

val symbolic_args_specs :
  ?constraints:string list ->
  ?scope:Shape_poly.scope ->
  aval Tree_util.t ->
  string option Tree_util.t ->
  Shape_poly.spec Tree_util.t

val is_symbolic_dim : Shape_poly.dim -> bool

type symbolic_exported

val export_symbolic :
  ?fun_name:string ->
  ?platforms:string list ->
  ?disabled_checks:disabled_safety_check list ->
  (value list -> value list) ->
  Shape_poly.spec list ->
  symbolic_exported

val symbolic_in_specs : symbolic_exported -> Shape_poly.spec list
val specialize : symbolic_exported -> value list -> exported
val call_symbolic : symbolic_exported -> value list -> value list
