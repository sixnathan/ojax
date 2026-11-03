type aval = { shape : string list; dtype : int; memory_space : int }
type disabled_check = Platform | Custom_call of string

type t = {
  fun_name : string;
  in_tree : Tree_util.treedef;
  out_tree : Tree_util.treedef;
  in_avals : aval list;
  out_avals : aval list;
  nr_devices : int;
  in_shardings : int;
  out_shardings : int;
  platforms : string list;
  ordered_effects : string list;
  unordered_effects : string list;
  disabled_safety_checks : disabled_check list;
  mlir_module_serialized : string;
  calling_convention_version : int;
  module_kept_var_idx : int list;
  uses_global_constants : bool;
  vjp : t option;
}

val dtype_of_kind : int -> Dtype.t
val kind_of_dtype : Dtype.t -> int
val aval_of_shape : Dtype.t -> int array -> aval
val serialize : ?vjp_order:int -> t -> string
val deserialize : string -> t
val has_vjp : t -> bool
val describe : t -> string
val call : t -> Types.value Tree_util.t -> Types.value Tree_util.t
val of_exported : Export.exported -> t
