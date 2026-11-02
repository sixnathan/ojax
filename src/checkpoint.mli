open Types

type policy = primitive -> aval list -> bool

val everything_saveable : policy
val nothing_saveable : policy
val dots_saveable : policy
val checkpoint_dots : policy
val dots_with_no_batch_dims_saveable : policy
val checkpoint_dots_with_no_batch_dims : policy
val save_anything_except_these_names : string list -> policy
val save_any_names_but_these : string list -> policy
val save_only_these_names : string list -> policy
val save_from_both_policies : policy -> policy -> policy

type remat_case = Recompute | Saveable | Offloadable of string * string

val offload_dot_with_no_batch_dims :
  offload_src:string -> offload_dst:string -> policy

val save_and_offload_only_these_names :
  names_which_can_be_saved:string list ->
  names_which_can_be_offloaded:string list ->
  offload_src:string ->
  offload_dst:string ->
  policy

val remat_case : policy -> primitive -> aval list -> remat_case
val policy_repr : policy -> string
val checkpoint_name : value -> string -> value
val checkpoint_name_tree : value Tree_util.t -> string -> value Tree_util.t

val checkpoint :
  ?policy:policy -> (value list -> value list) -> value list -> value list

val saved_residuals :
  (value list -> value list) -> value list -> (aval * string) list
