open Types

type schedule = int -> float
type leaf_state = value list

type optimizer = {
  init : value Tree_util.t -> leaf_state Tree_util.t;
  update :
    int -> value Tree_util.t -> leaf_state Tree_util.t -> leaf_state Tree_util.t;
  get_params : leaf_state Tree_util.t -> value Tree_util.t;
}

type optimizer_state = {
  packed_state : value list list;
  tree_def : Tree_util.treedef;
  subtree_defs : Tree_util.treedef list;
}

type join_point = Join_point of value Tree_util.t

val optimizer_state_of : leaf_state Tree_util.t -> optimizer_state
val leaf_state_of : optimizer_state -> leaf_state Tree_util.t
val optimizer_state_leaves : optimizer_state -> value list
val unpack_optimizer_state : optimizer_state -> join_point Tree_util.t
val pack_optimizer_state : join_point Tree_util.t -> optimizer_state

val make_schedule :
  [ `Scalar of float | `Value of value | `Schedule of schedule ] -> schedule

val constant : float -> schedule
val exponential_decay : float -> int -> float -> schedule
val inverse_time_decay : ?staircase:bool -> float -> int -> float -> schedule
val polynomial_decay : ?power:float -> float -> int -> float -> schedule
val piecewise_constant : int list -> float list -> schedule
val sgd : schedule -> optimizer
val momentum : schedule -> mass:float -> optimizer
val nesterov : schedule -> mass:float -> optimizer
val adagrad : ?momentum:float -> schedule -> optimizer
val rmsprop : ?gamma:float -> ?eps:float -> schedule -> optimizer

val rmsprop_momentum :
  ?gamma:float -> ?eps:float -> ?momentum:float -> schedule -> optimizer

val adam : ?b1:float -> ?b2:float -> ?eps:float -> schedule -> optimizer
val adamax : ?b1:float -> ?b2:float -> ?eps:float -> schedule -> optimizer
val sm3 : ?momentum:float -> schedule -> optimizer
val l2_norm : value Tree_util.t -> value
val clip_grads : value Tree_util.t -> float -> value Tree_util.t
