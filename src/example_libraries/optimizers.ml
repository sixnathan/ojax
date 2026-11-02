open Types
module Tree = Tree_util
module L = Lax_api

type schedule = int -> float
type leaf_state = value list

type optimizer = {
  init : value Tree.t -> leaf_state Tree.t;
  update : int -> value Tree.t -> leaf_state Tree.t -> leaf_state Tree.t;
  get_params : leaf_state Tree.t -> value Tree.t;
}

let const_like v x = L.full_like v x
let zeros_like v = const_like v 0.0
let scale s v = L.mul (const_like v s) v
let add = L.add
let sub = L.sub
let mul = L.mul
let div = L.div
let sqrt_ = L.sqrt
let constant c _ = c

let exponential_decay step_size decay_steps decay_rate i =
  step_size *. (decay_rate ** (float_of_int i /. float_of_int decay_steps))

let inverse_time_decay ?(staircase = false) step_size decay_steps decay_rate i =
  let ratio = float_of_int i /. float_of_int decay_steps in
  let ratio = if staircase then Float.floor ratio else ratio in
  step_size /. (1.0 +. (decay_rate *. ratio))

let polynomial_decay ?(power = 1.0) step_size decay_steps final_step_size i =
  let step = Stdlib.min i decay_steps in
  let step_frac = 1.0 -. (float_of_int step /. float_of_int decay_steps) in
  ((step_size -. final_step_size) *. (step_frac ** power)) +. final_step_size

let piecewise_constant boundaries values i =
  let rec go bs vs =
    match (bs, vs) with
    | b :: brest, v :: vrest -> if i < b then v else go brest vrest
    | [], v :: _ -> v
    | _, [] -> invalid_arg "values must be one longer than boundaries"
  in
  go boundaries values

let of_leaf_rules ~init ~update ~get =
  {
    init = (fun params -> Tree.tree_map init params);
    update =
      (fun i grads state ->
        Tree.tree_map2 (fun g s -> update i g s) grads state);
    get_params = (fun state -> Tree.tree_map get state);
  }

let head = function x :: _ -> x | [] -> invalid_arg "optimizer: empty state"

module UF = Numpy.Ufuncs
module AC = Numpy.Array_creation
module RED = Numpy.Reductions
module NL = Numpy.Lax_numpy

let ( +! ) = UF.add
let ( -! ) = UF.subtract
let ( *! ) = UF.multiply
let ( /! ) = UF.true_divide
let lit v x = Concrete (Ndarray.of_floats (Core.get_aval v).dtype [||] [| x |])
let zl = AC.zeros_like

let sgd (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x ])
    ~update:(fun i g s ->
      let x = head s in
      let step = lit g (step_size i) *! g in
      [ x -! step ])
    ~get:head

let momentum (step_size : schedule) ~mass =
  of_leaf_rules
    ~init:(fun x -> [ x; zl x ])
    ~update:(fun i g s ->
      match s with
      | [ x; v ] ->
          let v' = (lit v mass *! v) +! g in
          let step = lit v' (step_size i) *! v' in
          [ x -! step; v' ]
      | _ -> invalid_arg "momentum: state")
    ~get:head

let nesterov (step_size : schedule) ~mass =
  of_leaf_rules
    ~init:(fun x -> [ x; zl x ])
    ~update:(fun i g s ->
      match s with
      | [ x; v ] ->
          let v' = (lit v mass *! v) +! g in
          let inner = (lit v' mass *! v') +! g in
          let step = lit inner (step_size i) *! inner in
          [ x -! step; v' ]
      | _ -> invalid_arg "nesterov: state")
    ~get:head

let adagrad ?(momentum = 0.9) (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x; zl x; zl x ])
    ~update:(fun i g s ->
      match s with
      | [ x; gsq; m ] ->
          let gsq' = gsq +! UF.square g in
          let positive = UF.greater gsq' (lit gsq' 0.0) in
          let inv = lit gsq' 1.0 /! UF.sqrt gsq' in
          let g_sq_inv_sqrt =
            Numpy.Jnp_util.where_weak ~y_weak:true positive inv
              (Numpy.Jnp_util.py_float 0.0)
          in
          let t1 = lit g (1.0 -. momentum) *! (g *! g_sq_inv_sqrt) in
          let m' = t1 +! (lit m momentum *! m) in
          let step = lit m' (step_size i) *! m' in
          [ x -! step; gsq'; m' ]
      | _ -> invalid_arg "adagrad: state")
    ~get:head

let rmsprop ?(gamma = 0.9) ?(eps = 1e-8) (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x; zl x ])
    ~update:(fun i g s ->
      match s with
      | [ x; avg ] ->
          let t1 = avg *! lit avg gamma in
          let avg' = t1 +! (UF.square g *! lit g (1.0 -. gamma)) in
          let num = lit g (step_size i) *! g in
          let denom = UF.sqrt (avg' +! lit avg' eps) in
          [ x -! (num /! denom); avg' ]
      | _ -> invalid_arg "rmsprop: state")
    ~get:head

let rmsprop_momentum ?(gamma = 0.9) ?(eps = 1e-8) ?(momentum = 0.9)
    (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x; zl x; zl x ])
    ~update:(fun i g s ->
      match s with
      | [ x; avg; mom ] ->
          let t1 = avg *! lit avg gamma in
          let avg' = t1 +! (UF.square g *! lit g (1.0 -. gamma)) in
          let t2 = lit mom momentum *! mom in
          let num = lit g (step_size i) *! g in
          let denom = UF.sqrt (avg' +! lit avg' eps) in
          let mom' = t2 +! (num /! denom) in
          [ x -! mom'; avg'; mom' ]
      | _ -> invalid_arg "rmsprop_momentum: state")
    ~get:head

let beta_power m b i =
  let base = lit m b in
  let raised =
    UF.power base
      (Concrete
         (Ndarray.of_floats
            (Dtypes.default_int_dtype ())
            [||]
            [| float_of_int (i + 1) |]))
  in
  lit m 1.0 -! raised

let adam ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x; zl x; zl x ])
    ~update:(fun i g s ->
      match s with
      | [ x; m; v ] ->
          let t1 = lit g (1.0 -. b1) *! g in
          let m' = t1 +! (lit m b1 *! m) in
          let t2 = lit g (1.0 -. b2) *! UF.square g in
          let v' = t2 +! (lit v b2 *! v) in
          let mhat = m' /! beta_power m' b1 i in
          let vhat = v' /! beta_power m' b2 i in
          let num = lit g (step_size i) *! mhat in
          let denom = UF.sqrt vhat +! lit vhat eps in
          [ x -! (num /! denom); m'; v' ]
      | _ -> invalid_arg "adam: state")
    ~get:head

let adamax ?(b1 = 0.9) ?(b2 = 0.999) ?(eps = 1e-8) (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x -> [ x; zl x; zl x ])
    ~update:(fun i g s ->
      match s with
      | [ x; m; u ] ->
          let t1 = lit g (1.0 -. b1) *! g in
          let m' = t1 +! (lit m b1 *! m) in
          let t2 = lit u b2 *! u in
          let u' = UF.maximum t2 (UF.abs g) in
          let num = lit g (step_size i) /! beta_power m' b1 i in
          let num = num *! m' in
          let denom = u' +! lit u' eps in
          [ x -! (num /! denom); m'; u' ]
      | _ -> invalid_arg "adamax: state")
    ~get:head

let sm3_shape_leaf x_shape =
  Concrete
    (Ndarray.of_floats Dtype.I32
       [| Array.length x_shape |]
       (Array.map float_of_int x_shape))

let sm3 ?(momentum = 0.9) (step_size : schedule) =
  of_leaf_rules
    ~init:(fun x0 ->
      let x_shape = (Core.get_aval x0).shape in
      let x = NL.atleast_1d x0 in
      let a = Core.get_aval x in
      let vs =
        List.map
          (fun size -> L.full [| size |] 0.0 a.dtype)
          (Array.to_list a.shape)
      in
      x :: zl x :: sm3_shape_leaf x_shape :: vs)
    ~update:(fun i g s ->
      match s with
      | x :: m :: x_shape :: vs ->
          let shape = (Core.get_aval g).shape in
          let rank = Array.length shape in
          let expand axis v =
            let sh = Array.make rank 1 in
            if rank > 0 then sh.(axis) <- shape.(axis);
            if rank <= 1 then v
            else
              Core.bind1
                (Types.Broadcast_in_dim { shape = sh; dims = [| axis |] })
                [ v ]
          in
          let smallest =
            match List.mapi expand vs with
            | first :: rest -> List.fold_left UF.minimum first rest
            | [] -> zl g
          in
          let accum = smallest +! UF.square g in
          let positive = UF.greater accum (lit accum 0.0) in
          let inverse =
            NL.where_ positive
              (lit accum 1.0 /! UF.sqrt accum)
              (Concrete
                 (Ndarray.of_floats
                    (Dtypes.default_int_dtype ())
                    [||] [| 0.0 |]))
          in
          let t1 = lit g (1.0 -. momentum) *! (g *! inverse) in
          let m' = t1 +! (lit m momentum *! m) in
          let x' = x -! (lit m' (step_size i) *! m') in
          let x_rank = Array.length (Core.get_aval x').shape in
          let vs' =
            List.init x_rank (fun j ->
                let axes =
                  Array.of_list
                    (List.filter (fun d -> d <> j) (List.init x_rank Fun.id))
                in
                RED.max ~axis:axes accum)
          in
          x' :: m' :: x_shape :: vs'
      | _ -> invalid_arg "sm3: state")
    ~get:(function
      | x :: _ :: Concrete x_shape :: _ ->
          NL.reshape x
            (Array.init
               (Ndarray.shape x_shape).(0)
               (fun d -> Int64.to_int (Ndarray.get_i64 x_shape [| d |])))
      | _ -> invalid_arg "sm3: state")

let l2_norm tree =
  let total =
    List.fold_left
      (fun acc g ->
        let rank = Array.length (Core.get_aval g).shape in
        let s = L.reduce_sum (mul g g) (Array.init rank Fun.id) in
        match acc with None -> Some s | Some a -> Some (add a s))
      None (Tree.tree_leaves tree)
  in
  match total with
  | Some t -> L.sqrt t
  | None -> invalid_arg "l2_norm: empty tree"

let clip_grads grad_tree max_norm =
  let norm = l2_norm grad_tree in
  let limit = const_like norm max_norm in
  let factor =
    L.select (L.lt norm limit) (const_like norm 1.0) (div limit norm)
  in
  Tree.tree_map
    (fun g ->
      let rank = Array.length (Core.get_aval g).shape in
      ignore rank;
      mul g (L.broadcast_in_dim factor (Core.get_aval g).shape [||]))
    grad_tree

type optimizer_state = {
  packed_state : value list list;
  tree_def : Tree.treedef;
  subtree_defs : Tree.treedef list;
}

type join_point = Join_point of value Tree.t

let optimizer_state_of state =
  let states_flat, tree_def = Tree.tree_flatten state in
  let subtrees =
    List.map
      (fun leaf ->
        Tree.tree_flatten (Tree.Tuple (List.map (fun x -> Tree.Leaf x) leaf)))
      states_flat
  in
  {
    packed_state = List.map fst subtrees;
    tree_def;
    subtree_defs = List.map snd subtrees;
  }

let leaf_state_of opt_state =
  Tree.tree_unflatten opt_state.tree_def opt_state.packed_state

let optimizer_state_leaves opt_state = List.concat opt_state.packed_state

let unpack_optimizer_state opt_state =
  let sentinels =
    List.map2
      (fun def leaves -> Join_point (Tree.tree_unflatten def leaves))
      opt_state.subtree_defs opt_state.packed_state
  in
  Tree.tree_unflatten opt_state.tree_def sentinels

let pack_optimizer_state marked =
  let sentinels, tree_def = Tree.tree_flatten marked in
  let subtrees =
    List.map (fun (Join_point subtree) -> Tree.tree_flatten subtree) sentinels
  in
  {
    packed_state = List.map fst subtrees;
    tree_def;
    subtree_defs = List.map snd subtrees;
  }

let make_schedule = function
  | `Schedule schedule -> schedule
  | `Scalar step_size -> constant step_size
  | `Value v -> (
      if Array.length (Core.get_aval v).shape <> 0 then
        invalid_arg "make_schedule: expected a scalar or a schedule";
      match v with
      | Concrete nd -> constant (Ndarray.get_f nd [||])
      | Device b -> constant (Ndarray.get_f (Pjrt.Buffer.to_host b) [||])
      | Tracer _ ->
          invalid_arg "make_schedule: a traced step size is not supported")
