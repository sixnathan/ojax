type optimize =
  | Auto
  | Optimal
  | Greedy
  | Branch_all
  | Branch_2
  | Branch_1
  | Unoptimized
  | Path of int list list

val optimize_of_string : string -> optimize

type contraction = {
  positions : int list;
  einsum_str : string;
  remaining : string list option;
  blas : string option;
}

type path_info = {
  contraction_list : contraction list;
  input_subscripts : string;
  output_subscript : string;
  eq : string;
  path : int list list;
  indices : string;
  scale_list : int list;
  naive_cost : int;
  opt_cost : int;
  speedup : float;
  size_list : int list;
  size_dict : (char * int) list;
  shapes : int array list;
  largest_intermediate : int;
}

val parse : string -> int array list -> string * string

val contract_path :
  ?optimize:optimize -> string -> int array list -> int list list * path_info

val path_info_to_string : path_info -> string
