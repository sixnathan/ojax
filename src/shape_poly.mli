exception Inconclusive_dimension_operation of string
exception Unexpected_dim_var of string

type scope
type dim
type spec = { shape : dim array; dtype : Dtype.t }
type precision = For_geq0_or_leq0 | For_geq0_or_lt0 | Best

val symbolic_scope : ?constraints:string list -> unit -> scope

val symbolic_shape :
  ?constraints:string list ->
  ?scope:scope ->
  ?like:int option array ->
  string option ->
  dim array

val symbolic_args_specs :
  ?constraints:string list ->
  ?scope:scope ->
  Types.aval Tree_util.t ->
  string option Tree_util.t ->
  spec Tree_util.t

val of_int : int -> dim
val to_constant : dim -> int option
val is_symbolic_dim : dim -> bool
val is_constant_dim : dim -> bool
val dim_to_string : dim -> string
val shape_to_string : dim array -> string
val int_shape_to_string : int array -> string
val add : dim -> dim -> dim
val sub : dim -> dim -> dim
val neg : dim -> dim
val mul : dim -> dim -> dim
val pow : dim -> int -> dim
val divmod : dim -> dim -> dim * dim
val floordiv_dim : dim -> dim -> dim
val mod_dim : dim -> dim -> dim
val max_dim : dim -> dim -> dim
val min_dim : dim -> dim -> dim
val equal : dim -> dim -> bool
val ge : dim -> dim -> bool
val le : dim -> dim -> bool
val gt : dim -> dim -> bool
val lt : dim -> dim -> bool
val bounds_decision : dim -> precision -> float * float
val vars_of_dim : dim -> string list
val evaluate : (string * int) list -> dim -> int
val all_dim_vars : spec list -> string list

val solve_dim_vars :
  ?arg_names:string list -> spec list -> int array list -> (string * int) list

val check_scopes : when_:string -> spec list -> unit
